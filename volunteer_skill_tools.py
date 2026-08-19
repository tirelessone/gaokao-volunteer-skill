"""Atomic tools exposed only while the volunteer Agent has an active Skill."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping

from agents import function_tool
from dotenv import load_dotenv
from openai import OpenAI
import requests

from event_queue import send_final_result
from userInfo import format_user_info_with_llm
from volunteer_planner.candidates import parse_candidate_groups, render_candidate_groups
from volunteer_planner.engine import load_university_cache
from volunteer_planner.models import MajorCandidate, MajorGroupCandidate
from volunteer_planner.service import recall_candidates, validate_user_info
from volunteer_workspace import (
    MAJOR_SORT_STAGE,
    PUBLISH_STAGE,
    RANK_STAGE,
    RECALL_STAGE,
    VolunteerPlanWorkspace,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_TOOL_REFERENCE = "references/tool-policy.md"


def _stage_log(context: Any, stage: str, event: str, **data: Any) -> None:
    details = ""
    if data:
        import json

        details = " " + json.dumps(data, ensure_ascii=False, default=str)
    session_id = getattr(context, "session_id", "unknown")
    print(f"[VOLUNTEER][session={session_id}][{stage}][{event}]{details}", flush=True)


def register_volunteer_skill_tools(registry: Any) -> None:
    registry.register("recall_and_filter_candidates", _build_recall_tool)
    registry.register("rank_major_groups", _build_group_ranking_tool)
    registry.register("sort_majors_within_groups", _build_major_sorting_tool)
    registry.register("publish_volunteer_plan", _build_publish_tool)


def _workspace(context: Any) -> VolunteerPlanWorkspace:
    value = context.artifacts.get("volunteer_plan")
    if value is None:
        value = VolunteerPlanWorkspace()
        context.artifacts["volunteer_plan"] = value
    return value


def _checkpoint(context: Any, workspace: VolunteerPlanWorkspace, stage: str) -> None:
    workspace.mark_completed(stage)
    _stage_log(
        context,
        stage,
        "CHECKPOINT",
        revision=workspace.revision,
        completed_stages=list(workspace.completed_stages),
    )
    persist = getattr(context, "persist_workspace", None)
    if callable(persist):
        persist()


def _build_recall_tool(context: Any):
    @function_tool(strict_mode=False)
    async def recall_and_filter_candidates(
        query: str,
        group_probability_min: int | None = None,
        group_probability_max: int | None = None,
        major_probability_min: int | None = None,
        major_probability_max: int | None = None,
    ) -> dict[str, Any]:
        """提取考生信息、调用真实 API 召回候选，并按可选概率阈值过滤。默认不改变原概率范围。"""
        workspace = _workspace(context)
        canonical_query = str(context.artifacts.get("skill_query") or query).strip()
        inferred_filters = _infer_probability_filters(canonical_query)
        if "group_probability_min" in inferred_filters:
            group_probability_min = inferred_filters["group_probability_min"]
        if "group_probability_max" in inferred_filters:
            group_probability_max = inferred_filters["group_probability_max"]
        if "major_probability_min" in inferred_filters:
            major_probability_min = inferred_filters["major_probability_min"]
        if "major_probability_max" in inferred_filters:
            major_probability_max = inferred_filters["major_probability_max"]
        _stage_log(
            context,
            RECALL_STAGE,
            "START",
            query=query[:500],
            probability_filters={
                "group_min": group_probability_min,
                "group_max": group_probability_max,
                "major_min": major_probability_min,
                "major_max": major_probability_max,
            },
            inferred_probability_filters=inferred_filters,
        )
        if workspace.recalled_groups:
            raise RuntimeError("召回工具已成功执行，不要重复调用")
        if (context.current_skill, REQUIRED_TOOL_REFERENCE) not in context.loaded_references:
            raise RuntimeError(f"请先读取 {REQUIRED_TOOL_REFERENCE}")
        _validate_probability_range(group_probability_min, group_probability_max, "专业组")
        _validate_probability_range(major_probability_min, major_probability_max, "专业")

        user_info = await asyncio.to_thread(format_user_info_with_llm, canonical_query)
        if not user_info:
            raise ValueError("无法提取完整考生信息")
        user_info = _repair_extracted_user_info(dict(user_info), canonical_query)
        validated = validate_user_info(user_info)
        _stage_log(
            context,
            RECALL_STAGE,
            "USER_INFO_EXTRACTED",
            user_profile=_bounded_profile(validated),
            excluded_majors=validated.get("not_major", []),
        )
        _stage_log(context, RECALL_STAGE, "API_RECALL_START", attempts=3)
        raw = await _recall_with_retry(validated)
        recalled = parse_candidate_groups(raw)
        _stage_log(
            context,
            RECALL_STAGE,
            "API_RECALL_DONE",
            raw_character_count=len(raw),
            recalled_groups=len(recalled),
            recalled_majors=sum(len(group.majors) for group in recalled),
        )
        excluded = [str(item) for item in validated.get("not_major", []) if str(item)]
        eligible: List[MajorGroupCandidate] = []
        for original in recalled:
            group = original.model_copy(deep=True)
            if not _within(group.probability, group_probability_min, group_probability_max):
                continue
            group.majors = [
                major for major in group.majors
                if _within(major.probability, major_probability_min, major_probability_max)
                and not any(keyword in major.name for keyword in excluded)
            ]
            if group.majors:
                eligible.append(group)

        workspace.query = canonical_query
        workspace.user_info = validated
        workspace.recalled_groups = recalled
        workspace.eligible_groups = eligible
        workspace.raw_character_count = len(raw)
        workspace.filters_applied = _filter_labels(
            group_probability_min,
            group_probability_max,
            major_probability_min,
            major_probability_max,
            bool(excluded),
        )
        _stage_log(
            context,
            RECALL_STAGE,
            "FILTER_DONE",
            recalled_groups=len(recalled),
            eligible_groups=len(eligible),
            eligible_majors=sum(len(group.majors) for group in eligible),
            filters_applied=workspace.filters_applied,
        )
        context.trace_tool(
            "recall_and_filter_candidates",
            "success",
            recalled_groups=len(recalled),
            eligible_groups=len(eligible),
            raw_character_count=len(raw),
        )
        _checkpoint(context, workspace, RECALL_STAGE)
        return {
            "status": "success",
            "user_profile": _bounded_profile(validated),
            "recalled_groups": len(recalled),
            "eligible_groups": len(eligible),
            "filters_applied": workspace.filters_applied,
            "data_boundary": "候选全集已保存在工具内部 workspace，未返回给 Agent",
        }

    return recall_and_filter_candidates


def _build_group_ranking_tool(context: Any):
    @function_tool(strict_mode=False)
    def rank_major_groups(
        total_groups: int = 45,
        chong_count: int = 15,
        wen_count: int = 15,
        bao_count: int = 15,
        university_level_weight: float = 0.9,
        city_level_weight: float = 0.4,
        university_preference_weight: float = 1.2,
        city_preference_weight: float = 0.8,
        major_probability_weight: float = 1.0,
        subject_grade_weight: float = 0.7,
        major_relevance_weight: float = 1.0,
        major_probability_order: str = "asc",
    ) -> dict[str, Any]:
        """按冲稳保配额和可调权重筛选、排序专业组；权重越大代表该因素越重要。"""
        workspace = _workspace(context)
        _stage_log(
            context,
            RANK_STAGE,
            "START",
            eligible_groups=len(workspace.eligible_groups),
            total_groups=total_groups,
            requested_distribution={
                "chong": chong_count,
                "wen": wen_count,
                "bao": bao_count,
            },
            major_probability_order=major_probability_order,
        )
        if not workspace.eligible_groups:
            raise RuntimeError("请先调用 recall_and_filter_candidates")
        if workspace.selected_groups:
            raise RuntimeError("组间排序工具已成功执行，不要重复调用")
        if not 1 <= total_groups <= 45:
            raise ValueError("total_groups 必须在 1 到 45 之间")
        counts = {"chong": chong_count, "wen": wen_count, "bao": bao_count}
        if any(value < 0 for value in counts.values()) or sum(counts.values()) > total_groups:
            raise ValueError("冲稳保数量必须非负且总和不能超过 total_groups")
        if major_probability_order not in {"asc", "desc"}:
            raise ValueError("major_probability_order 只能是 asc 或 desc")
        weights = {
            "university_level": university_level_weight,
            "city_level": city_level_weight,
            "university_preference": university_preference_weight,
            "city_preference": city_preference_weight,
            "major_probability": major_probability_weight,
            "subject_grade": subject_grade_weight,
            "major_relevance": major_relevance_weight,
        }
        if any(value < 0 for value in weights.values()) or not any(weights.values()):
            raise ValueError("组间排序权重必须非负，且至少有一个权重大于 0")
        _stage_log(context, RANK_STAGE, "WEIGHTS_READY", weights=weights)

        cache = load_university_cache(PROJECT_ROOT / "university_cache.json")
        buckets: Dict[str, List[tuple[float, MajorGroupCandidate]]] = {
            "chong": [], "wen": [], "bao": []
        }
        for original in workspace.eligible_groups:
            group = original.model_copy(deep=True)
            info = cache.get(group.university, {})
            group.city = str(info.get("city", "") or "")
            score = _group_score(group, workspace.user_info, info, weights, major_probability_order)
            buckets[_risk_bucket(group.probability)].append((score, group))
        for name in buckets:
            buckets[name].sort(key=lambda item: item[0], reverse=True)

        workspace.bucket_available = {name: len(values) for name, values in buckets.items()}
        selected_by_bucket = {
            name: [group for _, group in values[:counts[name]]]
            for name, values in buckets.items()
        }
        selected_keys = {
            (group.university, group.group_id)
            for groups in selected_by_bucket.values() for group in groups
        }
        deficit = total_groups - len(selected_keys)
        for name in ("wen", "bao", "chong"):
            if deficit <= 0:
                break
            remaining = [
                group for _, group in buckets[name]
                if (group.university, group.group_id) not in selected_keys
            ]
            additions = remaining[:deficit]
            selected_by_bucket[name].extend(additions)
            selected_keys.update((group.university, group.group_id) for group in additions)
            deficit -= len(additions)

        selected: List[MajorGroupCandidate] = []
        for name in ("chong", "wen", "bao"):
            selected.extend(selected_by_bucket[name])
        workspace.selected_groups = selected[:total_groups]
        workspace.bucket_selected = {
            name: sum(1 for group in workspace.selected_groups if _risk_bucket(group.probability) == name)
            for name in ("chong", "wen", "bao")
        }
        workspace.ranking_config = {
            "total_groups": total_groups,
            "requested_distribution": counts,
            "weights": weights,
            "major_probability_order": major_probability_order,
        }
        preview = [
            {"university": group.university, "group_id": group.group_id, "probability": group.probability}
            for group in workspace.selected_groups[:5]
        ]
        _stage_log(
            context,
            RANK_STAGE,
            "DONE",
            bucket_available=workspace.bucket_available,
            bucket_selected=workspace.bucket_selected,
            selected_groups=len(workspace.selected_groups),
            top_preview=preview,
        )
        context.trace_tool(
            "rank_major_groups",
            "success",
            selected_groups=len(workspace.selected_groups),
            bucket_selected=workspace.bucket_selected,
            weights=weights,
        )
        _checkpoint(context, workspace, RANK_STAGE)
        return {
            "status": "success",
            "selected_groups": len(workspace.selected_groups),
            "bucket_available": workspace.bucket_available,
            "bucket_selected": workspace.bucket_selected,
            "weights": weights,
            "top_preview": preview,
        }

    return rank_major_groups


def _build_major_sorting_tool(context: Any):
    @function_tool(strict_mode=False)
    async def sort_majors_within_groups(
        custom_instructions: str = "",
        max_majors_per_group: int = 6,
    ) -> dict[str, Any]:
        """逐组按原志愿提示词排序专业；用户有额外组内要求时通过 custom_instructions 追加。"""
        workspace = _workspace(context)
        _stage_log(
            context,
            MAJOR_SORT_STAGE,
            "START",
            selected_groups=len(workspace.selected_groups),
            selected_majors_before=sum(len(group.majors) for group in workspace.selected_groups),
            custom_instructions=custom_instructions.strip(),
            max_majors_per_group=max_majors_per_group,
        )
        if not workspace.selected_groups:
            raise RuntimeError("请先调用 rank_major_groups")
        if workspace.final_groups:
            raise RuntimeError("组内排序工具已成功执行，不要重复调用")
        if not 1 <= max_majors_per_group <= 6:
            raise ValueError("max_majors_per_group 必须在 1 到 6 之间")

        sorted_groups, llm_count, fallback_count = await asyncio.to_thread(
            _sort_groups_with_prompt,
            workspace.selected_groups,
            workspace.user_info,
            custom_instructions.strip(),
            max_majors_per_group,
        )
        workspace.final_groups = sorted_groups
        workspace.major_sort_config = {
            "custom_instructions": custom_instructions.strip(),
            "max_majors_per_group": max_majors_per_group,
            "llm_sorted_groups": llm_count,
            "fallback_groups": fallback_count,
        }
        _stage_log(
            context,
            MAJOR_SORT_STAGE,
            "DONE",
            processed_groups=len(sorted_groups),
            selected_majors=sum(len(group.majors) for group in sorted_groups),
            llm_sorted_groups=llm_count,
            fallback_groups=fallback_count,
        )
        context.trace_tool(
            "sort_majors_within_groups",
            "success",
            groups=len(sorted_groups),
            llm_sorted_groups=llm_count,
            fallback_groups=fallback_count,
        )
        _checkpoint(context, workspace, MAJOR_SORT_STAGE)
        return {
            "status": "success",
            "processed_groups": len(sorted_groups),
            "selected_majors": sum(len(group.majors) for group in sorted_groups),
            **workspace.major_sort_config,
        }

    return sort_majors_within_groups


def _build_publish_tool(context: Any):
    @function_tool(strict_mode=False)
    def publish_volunteer_plan() -> dict[str, Any]:
        """发布最终志愿表并结束 Skill。只有召回、组间排序、组内排序均成功后才能调用。"""
        workspace = _workspace(context)
        _stage_log(
            context,
            PUBLISH_STAGE,
            "START",
            final_groups=len(workspace.final_groups),
            final_majors=sum(len(group.majors) for group in workspace.final_groups),
        )
        if not workspace.final_groups:
            raise RuntimeError("必须先完成组内专业排序")
        if workspace.summary:
            raise RuntimeError("志愿表已经发布，不要重复调用")
        workspace.plan_text = render_candidate_groups(
            workspace.final_groups,
            include_internal_flags=False,
        )
        print(
            f"\n[VOLUNTEER][session={context.session_id}][FINAL_PLAN_BEGIN]\n"
            f"{workspace.plan_text}\n"
            f"[VOLUNTEER][session={context.session_id}][FINAL_PLAN_END]\n",
            flush=True,
        )
        info = workspace.user_info
        metadata = {
            "statistics": {
                "raw_character_count": workspace.raw_character_count,
                "recalled_groups": len(workspace.recalled_groups),
                "eligible_groups": len(workspace.eligible_groups),
                "selected_groups": len(workspace.final_groups),
                "selected_majors": sum(len(group.majors) for group in workspace.final_groups),
                "bucket_available": workspace.bucket_available,
                "bucket_selected": workspace.bucket_selected,
                "filters_applied": workspace.filters_applied,
            },
            "group_ranking": workspace.ranking_config,
            "major_sorting": workspace.major_sort_config,
            "execution": {
                "workspace_restored": bool(context.artifacts.get("workspace_restored")),
                "revision": workspace.revision,
                "resume_stage": context.artifacts.get("resume_stage", RECALL_STAGE),
                "resume_reason": context.artifacts.get("resume_reason", ""),
                "bound_skill_tools": context.artifacts.get("active_allowed_tools", []),
            },
        }
        send_final_result(
            context.session_id,
            workspace.plan_text,
            info["score"],
            info.get("firstChoice", ""),
            info.get("reselection", []),
            info.get("recruit", 0),
            info.get("city", []),
            info.get("not_city", []),
            info.get("university", []),
            info.get("major", []),
            info.get("not_major", []),
            info.get("matriculation", 1),
            info.get("target", 1),
            info.get("dimension", 0),
            info.get("foreign", 0),
            info.get("province", "四川省"),
            metadata=metadata,
        )
        workspace.summary = {
            "status": "success",
            "message": "志愿表已由志愿生成 Agent 的 Skill 模式按 SOP 生成并发布。",
            "selected_groups": metadata["statistics"]["selected_groups"],
            "selected_majors": metadata["statistics"]["selected_majors"],
            "bucket_selected": workspace.bucket_selected,
            "filters_applied": workspace.filters_applied,
            "group_weights": workspace.ranking_config.get("weights", {}),
            "major_sort_customized": bool(
                workspace.major_sort_config.get("custom_instructions")
            ),
            "revision": workspace.revision,
            "resume_stage": context.artifacts.get("resume_stage", RECALL_STAGE),
        }
        context.session_state["generate_plan"] = True
        context.trace_tool(
            "publish_volunteer_plan",
            "success",
            selected_groups=workspace.summary["selected_groups"],
            selected_majors=workspace.summary["selected_majors"],
        )
        _stage_log(
            context,
            PUBLISH_STAGE,
            "DONE",
            summary=workspace.summary,
            execution=metadata["execution"],
        )
        _checkpoint(context, workspace, PUBLISH_STAGE)
        return workspace.summary

    return publish_volunteer_plan


def _validate_probability_range(minimum: int | None, maximum: int | None, label: str) -> None:
    for value in (minimum, maximum):
        if value is not None and not 0 <= value <= 100:
            raise ValueError(f"{label}概率阈值必须在 0 到 100 之间")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{label}最低概率不能大于最高概率")


async def _recall_with_retry(user_info: Mapping[str, Any], attempts: int = 3) -> str:
    last_error: requests.RequestException | None = None
    for index in range(attempts):
        try:
            return await asyncio.to_thread(recall_candidates, user_info)
        except requests.RequestException as exc:
            last_error = exc
            if index + 1 < attempts:
                await asyncio.sleep(0.5 * (index + 1))
    assert last_error is not None
    raise last_error


def _infer_probability_filters(query: str) -> Dict[str, int]:
    """Translate natural-language probability limits into inclusive integer bounds."""
    text = re.sub(r"\s+", "", query)
    result: Dict[str, int] = {}
    patterns = (
        re.compile(
            r"(?:录取)?概率(?:为|在)?(\d{1,3})%?"
            r"(及以上|以上|及以下|以下|不低于|不高于|高于|低于|超过|不超过|至少|至多)"
        ),
        re.compile(
            r"(?:录取)?概率(不低于|不高于|高于|低于|超过|不超过|至少|至多)"
            r"(\d{1,3})%?"
        ),
    )
    removal_words = ("去掉", "删除", "排除", "不要", "过滤掉", "剔除")
    lower_inclusive = {"以上", "及以上", "不低于", "至少"}
    lower_exclusive = {"高于", "超过"}
    upper_inclusive = {"以下", "及以下", "不高于", "不超过", "至多"}
    upper_exclusive = {"低于"}

    for pattern_index, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            if pattern_index == 0:
                threshold = int(match.group(1))
                operator = match.group(2)
            else:
                operator = match.group(1)
                threshold = int(match.group(2))
            if not 0 <= threshold <= 100:
                continue

            window_start = max(0, match.start() - 12)
            window_end = min(len(text), match.end() + 16)
            window = text[window_start:window_end]
            remove_matching = any(word in window for word in removal_words)
            if "专业组" in window or "学校" in window or "院校" in window:
                prefix = "group_probability"
            elif "专业" in window:
                prefix = "major_probability"
            else:
                # 用户只说“学校录取概率”时通常指专业组概率。
                prefix = "group_probability"

            if remove_matching:
                if operator in lower_inclusive:
                    result[f"{prefix}_max"] = max(0, threshold - 1)
                elif operator in lower_exclusive:
                    result[f"{prefix}_max"] = threshold
                elif operator in upper_inclusive:
                    result[f"{prefix}_min"] = min(100, threshold + 1)
                elif operator in upper_exclusive:
                    result[f"{prefix}_min"] = threshold
            else:
                if operator in lower_inclusive:
                    result[f"{prefix}_min"] = threshold
                elif operator in lower_exclusive:
                    result[f"{prefix}_min"] = min(100, threshold + 1)
                elif operator in upper_inclusive:
                    result[f"{prefix}_max"] = threshold
                elif operator in upper_exclusive:
                    result[f"{prefix}_max"] = max(0, threshold - 1)
    return result


def _within(value: int | None, minimum: int | None, maximum: int | None) -> bool:
    if value is None:
        return minimum is None and maximum is None
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _filter_labels(
    group_min: int | None,
    group_max: int | None,
    major_min: int | None,
    major_max: int | None,
    excluded: bool,
) -> List[str]:
    values = []
    for name, operator, value in (
        ("group_probability", ">=", group_min),
        ("group_probability", "<=", group_max),
        ("major_probability", ">=", major_min),
        ("major_probability", "<=", major_max),
    ):
        if value is not None:
            values.append(f"{name}{operator}{value}")
    if excluded:
        values.append("excluded_major_keywords")
    return values


def _bounded_profile(info: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: info.get(key)
        for key in (
            "province", "score", "firstChoice", "reselection", "major", "not_major",
            "city", "not_city", "university", "dimension", "foreign",
        )
    }


def _repair_extracted_user_info(info: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Repair stable profile fields that a model may omit from compact subject combinations."""
    if not info.get("score"):
        score_match = re.search(r"(?<!\d)([3-7]\d{2})\s*分", query)
        if score_match:
            info["score"] = int(score_match.group(1))
    if not info.get("firstChoice"):
        if "物" in query or "物理" in query:
            info["firstChoice"] = "物"
        elif "史" in query or "历史" in query:
            info["firstChoice"] = "历"
    reselection = info.get("reselection")
    existing = [] if reselection is None else (
        list(reselection) if isinstance(reselection, (list, tuple)) else [str(reselection)]
    )
    selected = [subject for subject in ("化", "生", "政", "地") if subject in query]
    if len([item for item in existing if item != "不限"]) < 2 and len(selected) >= 2:
        info["reselection"] = ["不限", *selected]
    return info


def _risk_bucket(probability: int) -> str:
    if probability <= 50:
        return "chong"
    if probability <= 80:
        return "wen"
    return "bao"


def _group_score(
    group: MajorGroupCandidate,
    user_info: Mapping[str, Any],
    university_info: Mapping[str, Any],
    weights: Mapping[str, float],
    probability_order: str,
) -> float:
    probabilities = [
        major.probability for major in group.majors
        if major.intentional and major.probability is not None
    ] or [major.probability for major in group.majors if major.probability is not None]
    average_probability = sum(probabilities) / len(probabilities) if probabilities else group.probability
    probability_score = average_probability / 100
    if probability_order == "asc":
        probability_score = 1 - probability_score
    intentional_ratio = sum(major.intentional for major in group.majors) / max(1, len(group.majors))
    grade_score = max((_grade_score(major.grade) for major in group.majors), default=0)
    university_level = _university_level_score(str(university_info.get("tag", "") or ""))
    city_level = _city_level_score(group.city)
    university_preference = _ordered_score(
        group.university, user_info.get("university", []), exact=True
    )
    city_preference = _ordered_score(group.city, user_info.get("city", []), exact=False)
    features = {
        "university_level": university_level,
        "city_level": city_level,
        "university_preference": university_preference,
        "city_preference": city_preference,
        "major_probability": probability_score,
        "subject_grade": grade_score,
        "major_relevance": intentional_ratio,
    }
    total_weight = sum(weights.values())
    return sum(features[name] * weights[name] for name in features) / total_weight


def _grade_score(grade: str | None) -> float:
    values = {"A+": 1.0, "A": 0.95, "A-": 0.9, "B+": 0.8, "B": 0.7, "B-": 0.6,
              "C+": 0.5, "C": 0.4, "C-": 0.3}
    return values.get(str(grade or ""), 0.0)


def _university_level_score(tag: str) -> float:
    for label, score in (("C9", 1.0), ("985", 0.9), ("211", 0.8), ("双一流", 0.7), ("省重点", 0.6)):
        if label in tag:
            return score
    return 0.3


def _city_level_score(city: str) -> float:
    tier_one = {"北京", "上海", "广州", "深圳"}
    new_tier_one = {"成都", "杭州", "重庆", "武汉", "苏州", "西安", "南京", "长沙", "郑州", "天津", "合肥", "青岛", "东莞", "宁波", "佛山"}
    provincial = {"哈尔滨", "长春", "沈阳", "呼和浩特", "石家庄", "乌鲁木齐", "兰州", "西宁", "银川", "太原", "济南", "贵阳", "昆明", "南宁", "拉萨", "南昌", "福州", "海口"}
    if any(value in city for value in tier_one):
        return 1.0
    if any(value in city for value in new_tier_one):
        return 0.85
    if any(value in city for value in provincial):
        return 0.7
    return 0.4


def _ordered_score(text: str, values: Iterable[Any], *, exact: bool) -> float:
    preferences = [str(value) for value in values if str(value)]
    for index, preference in enumerate(preferences):
        if (text == preference) if exact else (preference in text):
            return (len(preferences) - index) / len(preferences)
    return 0.0


def _sort_groups_with_prompt(
    groups: List[MajorGroupCandidate],
    user_info: Mapping[str, Any],
    custom_instructions: str,
    max_majors: int,
) -> tuple[List[MajorGroupCandidate], int, int]:
    load_dotenv()
    api_key = os.getenv("AGENT_API_KEY")
    base_url = os.getenv("AGENT_BASE_URL")
    model = os.getenv("MAJOR_SORT_MODEL") or os.getenv("AGENT_MODEL_NAME")
    client = OpenAI(api_key=api_key, base_url=base_url) if api_key and base_url and model else None
    results: List[MajorGroupCandidate | None] = [None] * len(groups)
    llm_count = 0
    fallback_count = 0

    def process(index: int, group: MajorGroupCandidate) -> tuple[int, MajorGroupCandidate, bool]:
        if client is not None:
            try:
                ordered = _sort_one_group_with_llm(
                    client, model, group, user_info, custom_instructions, max_majors
                )
                return index, ordered, True
            except Exception as exc:
                print(f"组内排序回退: {group.university}/{group.group_id}: {exc}")
        return index, _fallback_major_sort(
            group, user_info, max_majors, custom_instructions
        ), False

    workers = min(9, max(1, len(groups)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, index, group) for index, group in enumerate(groups)]
        for future in as_completed(futures):
            index, group, used_llm = future.result()
            results[index] = group
            if used_llm:
                llm_count += 1
            else:
                fallback_count += 1
    return [group for group in results if group is not None], llm_count, fallback_count


def _sort_one_group_with_llm(
    client: OpenAI,
    model: str,
    group: MajorGroupCandidate,
    user_info: Mapping[str, Any],
    custom_instructions: str,
    max_majors: int,
) -> MajorGroupCandidate:
    system_prompt = """你是精通中国新高考志愿填报的专家，请严格按规则处理专业组数据。
1. 专业组是录取基本单位，每个专业组最多填6个专业，并尽量填满。
2. 专业名称和概率必须完整保留，禁止拆分、编造或篡改。
3. 只做专业选择和排序，仅输出处理后的原格式数据，不输出解释或思考过程。"""
    if int(user_info.get("dimension", 0) or 0) == 0:
        default_rules = """1. 标记为(true)的意向专业放在最前面。
2. 与用户意向相关度高的其他专业放在中间。
3. 与用户意向相关度低的专业放在最后。"""
    else:
        default_rules = """1. 录取概率低于50%的专业放在最前面。
2. 录取概率50%-80%的专业放在中间。
3. 录取概率80%以上的专业放在最后。"""
    custom_section = ""
    if custom_instructions:
        custom_section = f"\n用户补充排序要求（与默认规则冲突时优先执行）：\n{custom_instructions}\n"
    prompt = f"""# 任务
对组内专业排序，最多保留 {max_majors} 个专业。

# 默认规则
{default_rules}
{custom_section}
用户信息：{_bounded_profile(user_info)}

需要处理的专业组：
{render_candidate_groups([group], include_internal_flags=True)}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    content = response.choices[0].message.content.strip()
    parsed = parse_candidate_groups(content)
    if not parsed:
        raise ValueError("模型未返回可解析的专业组")
    returned_names = [major.name for major in parsed[0].majors]
    originals = {major.name: major for major in group.majors}
    ordered = [originals[name].model_copy(deep=True) for name in returned_names if name in originals]
    if not ordered:
        raise ValueError("模型返回的专业均不在原始候选中")
    used = {major.name for major in ordered}
    fallback = _fallback_major_sort(
        group, user_info, max_majors, custom_instructions
    ).majors
    ordered.extend(major.model_copy(deep=True) for major in fallback if major.name not in used)
    ordered.sort(key=lambda major: int(_should_deprioritize_major(major, custom_instructions)))
    output = group.model_copy(deep=True)
    output.majors = ordered[:max_majors]
    return output


def _fallback_major_sort(
    group: MajorGroupCandidate,
    user_info: Mapping[str, Any],
    max_majors: int,
    custom_instructions: str = "",
) -> MajorGroupCandidate:
    output = group.model_copy(deep=True)
    keywords = [str(value) for value in user_info.get("major", []) if str(value)]
    if int(user_info.get("dimension", 0) or 0) == 0:
        output.majors.sort(
            key=lambda major: (
                int(_should_deprioritize_major(major, custom_instructions)),
                -int(major.intentional),
                -int(any(keyword in major.name for keyword in keywords)),
                -(major.probability if major.probability is not None else -1),
            )
        )
    else:
        output.majors.sort(key=lambda major: (
            int(_should_deprioritize_major(major, custom_instructions)),
            major.probability if major.probability is not None else 101,
        ))
    output.majors = output.majors[:max_majors]
    return output


def _should_deprioritize_major(major: MajorCandidate, instructions: str) -> bool:
    if not instructions or major.name not in instructions:
        return False
    return any(
        marker in instructions
        for marker in ("不要选", "不选", "排除", "放在最后", "放最后", "最后考虑")
    )

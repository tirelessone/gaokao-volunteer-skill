from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from volunteer_planner.candidates import render_candidate_groups
from volunteer_planner.models import (
    MajorCandidate,
    MajorGroupCandidate,
    PlanPolicy,
    PlanResult,
    PlanStatistics,
    SortRule,
)


def build_plan(
    groups: Sequence[MajorGroupCandidate],
    policy: PlanPolicy,
    university_cache: Mapping[str, Mapping[str, Any]] | None = None,
    raw_character_count: int = 0,
) -> PlanResult:
    cache = university_cache or {}
    filters_applied: List[str] = []
    eligible: List[MajorGroupCandidate] = []

    for original in groups:
        group = original.model_copy(deep=True)
        if not _within(group.probability, policy.probability_filter.group_min, policy.probability_filter.group_max):
            continue
        majors = [
            major for major in group.majors
            if _major_is_eligible(major, policy)
        ]
        if not majors:
            continue
        group.majors = _sort_majors(majors, policy)[:policy.max_majors_per_group]
        eligible.append(group)

    if policy.probability_filter.group_min is not None:
        filters_applied.append(f"group_probability>={policy.probability_filter.group_min}")
    if policy.probability_filter.group_max is not None:
        filters_applied.append(f"group_probability<={policy.probability_filter.group_max}")
    if policy.probability_filter.major_min is not None:
        filters_applied.append(f"major_probability>={policy.probability_filter.major_min}")
    if policy.probability_filter.major_max is not None:
        filters_applied.append(f"major_probability<={policy.probability_filter.major_max}")
    if policy.excluded_major_keywords:
        filters_applied.append("excluded_major_keywords")

    buckets: Dict[str, List[MajorGroupCandidate]] = defaultdict(list)
    for group in eligible:
        info = cache.get(group.university, {})
        group.city = str(info.get("city", "") or "")
        buckets[_risk_bucket(group.probability)].append(group)

    for bucket_name, bucket_groups in buckets.items():
        buckets[bucket_name] = _sort_groups(bucket_groups, policy, cache)

    available = {name: len(buckets[name]) for name in ("chong", "wen", "bao")}
    targets = policy.risk_distribution.as_dict()
    selected_by_bucket = {
        name: list(buckets[name][:min(targets[name], len(buckets[name]))])
        for name in ("chong", "wen", "bao")
    }
    selected_keys = {
        (group.university, group.group_id)
        for values in selected_by_bucket.values()
        for group in values
    }
    deficit = policy.total_groups - len(selected_keys)

    for bucket_name in policy.deficit_fill_order:
        if deficit <= 0:
            break
        remaining = [
            group for group in buckets[bucket_name]
            if (group.university, group.group_id) not in selected_keys
        ]
        additions = remaining[:deficit]
        selected_by_bucket[bucket_name].extend(additions)
        selected_keys.update((group.university, group.group_id) for group in additions)
        deficit -= len(additions)

    selected: List[MajorGroupCandidate] = []
    for bucket_name in policy.risk_order:
        selected.extend(selected_by_bucket[bucket_name])
    selected = selected[:policy.total_groups]

    selected_counts = {
        name: sum(1 for group in selected if _risk_bucket(group.probability) == name)
        for name in ("chong", "wen", "bao")
    }
    statistics = PlanStatistics(
        raw_character_count=raw_character_count,
        recalled_groups=len(groups),
        eligible_groups=len(eligible),
        selected_groups=len(selected),
        selected_majors=sum(len(group.majors) for group in selected),
        bucket_available=available,
        bucket_selected=selected_counts,
        filters_applied=filters_applied,
    )
    if not selected:
        return PlanResult(
            status="error",
            policy=policy,
            statistics=statistics,
            message="当前条件下没有可用的专业组，请放宽概率或专业限制。",
        )

    return PlanResult(
        status="success",
        plan_text=render_candidate_groups(selected, include_internal_flags=False),
        groups=selected,
        policy=policy,
        statistics=statistics,
        message="志愿表已生成并通过规则校验。",
    )


def load_university_cache(path: Path | str) -> Dict[str, Dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.is_file():
        return {}
    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _major_is_eligible(major: MajorCandidate, policy: PlanPolicy) -> bool:
    if not _within(major.probability, policy.probability_filter.major_min, policy.probability_filter.major_max):
        return False
    return not any(keyword in major.name for keyword in policy.excluded_major_keywords)


def _within(value: int | None, minimum: int | None, maximum: int | None) -> bool:
    if value is None:
        return minimum is None and maximum is None
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def _risk_bucket(probability: int) -> str:
    if probability <= 50:
        return "chong"
    if probability <= 80:
        return "wen"
    return "bao"


def _sort_majors(majors: Iterable[MajorCandidate], policy: PlanPolicy) -> List[MajorCandidate]:
    values = list(majors)
    accessors: Dict[str, Callable[[MajorCandidate], Any]] = {
        "intentional": lambda major: int(major.intentional),
        "major_preference": lambda major: _ordered_match_score(major.name, policy.preferred_major_keywords),
        "admission_probability": lambda major: major.probability if major.probability is not None else -1,
        "major_name": lambda major: major.name,
    }
    return _multi_sort(values, policy.major_sort, accessors)


def _sort_groups(
    groups: Iterable[MajorGroupCandidate],
    policy: PlanPolicy,
    cache: Mapping[str, Mapping[str, Any]],
) -> List[MajorGroupCandidate]:
    values = list(groups)

    def intent_probability(group: MajorGroupCandidate) -> float:
        probabilities = [
            major.probability for major in group.majors
            if major.intentional and major.probability is not None
        ]
        return sum(probabilities) / len(probabilities) if probabilities else group.probability

    accessors: Dict[str, Callable[[MajorGroupCandidate], Any]] = {
        "admission_probability": lambda group: group.probability,
        "intent_major_probability": intent_probability,
        "intentional_major_count": lambda group: sum(major.intentional for major in group.majors),
        "major_count": lambda group: len(group.majors),
        "university_level": lambda group: _university_level(cache.get(group.university, {})),
        "university_preference": lambda group: _ordered_exact_score(group.university, policy.preferred_university_order),
        "city_level": lambda group: _city_level(str(cache.get(group.university, {}).get("city", "") or "")),
        "city_preference": lambda group: _ordered_match_score(
            str(cache.get(group.university, {}).get("city", "") or ""),
            policy.preferred_city_order,
        ),
    }
    return _multi_sort(values, policy.group_sort, accessors)


def _multi_sort(values: List[Any], rules: Sequence[SortRule], accessors: Mapping[str, Callable[[Any], Any]]) -> List[Any]:
    output = list(values)
    for rule in reversed(rules):
        output.sort(key=accessors[rule.field], reverse=rule.direction == "desc")
    return output


def _ordered_match_score(text: str, preferences: Sequence[str]) -> int:
    count = len(preferences)
    for index, preference in enumerate(preferences):
        if preference and preference in text:
            return count - index
    return 0


def _ordered_exact_score(text: str, preferences: Sequence[str]) -> int:
    count = len(preferences)
    for index, preference in enumerate(preferences):
        if preference and preference == text:
            return count - index
    return 0


def _university_level(info: Mapping[str, Any]) -> int:
    tag = str(info.get("tag", "") or "")
    for label, score in (("C9", 6), ("985", 5), ("211", 4), ("双一流", 3), ("省重点", 2)):
        if label in tag:
            return score
    return 1


def _city_level(city: str) -> int:
    tier_one = {"北京", "上海", "广州", "深圳"}
    new_tier_one = {"成都", "杭州", "重庆", "武汉", "苏州", "西安", "南京", "长沙", "郑州", "天津", "合肥", "青岛", "东莞", "宁波", "佛山"}
    provincial_capitals = {"哈尔滨", "长春", "沈阳", "呼和浩特", "石家庄", "乌鲁木齐", "兰州", "西宁", "银川", "太原", "济南", "贵阳", "昆明", "南宁", "拉萨", "南昌", "福州", "海口"}
    if any(name in city for name in tier_one):
        return 4
    if any(name in city for name in new_tier_one):
        return 3
    if any(name in city for name in provincial_capitals):
        return 2
    return 1

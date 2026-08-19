from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import requests

from event_queue import send_end, send_error, send_final_result
from volunteer_planner.candidates import parse_candidate_groups
from volunteer_planner.engine import build_plan, load_university_cache
from volunteer_planner.models import PlanResult
from volunteer_planner.policy import parse_plan_policy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAJOR_API_URL = "https://rest.nextgoo.cn/nextgoo/v1/aimodel/querySelectedMajorGroup"
CITY_API_URL = "https://rest.nextgoo.cn/nextgoo/v1/aimodel/querySelectedMajorGroup1"


class CandidateRecallError(RuntimeError):
    pass


def generate_plan(
    user_info: Mapping[str, Any],
    session_id: str,
    plan_policy: Mapping[str, Any] | str | None = None,
    *,
    emit_events: bool = True,
    request_session: requests.Session | None = None,
) -> PlanResult:
    """Run the complete skill pipeline without exposing recalled candidates to a model."""
    try:
        validated = validate_user_info(user_info)
        policy = parse_plan_policy(plan_policy, validated)
        raw_candidates = recall_candidates(validated, request_session=request_session)
        groups = parse_candidate_groups(raw_candidates)
        cache = load_university_cache(PROJECT_ROOT / "university_cache.json")
        result = build_plan(
            groups,
            policy,
            cache,
            raw_character_count=len(raw_candidates),
        )
        if emit_events:
            if result.status == "success":
                _emit_final_result(session_id, result, validated)
            else:
                send_error(session_id, result.message)
            send_end(session_id)
        return result
    except Exception as exc:
        if emit_events:
            send_error(session_id, str(exc))
            send_end(session_id)
        raise


def validate_user_info(user_info: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(user_info, Mapping):
        raise ValueError("user_info must be an object")
    data = dict(user_info)
    score = data.get("score")
    if score is None:
        raise ValueError("缺少必需信息：高考分数")
    try:
        data["score"] = int(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("高考分数必须是整数") from exc
    if data["score"] < 425:
        raise ValueError("目前系统仅支持本科B段，分数需不低于425分")
    if not data.get("firstChoice"):
        raise ValueError("缺少必需信息：首选科目")
    reselection = _as_list(data.get("reselection"))
    if len([item for item in reselection if item != "不限"]) < 2 and len(reselection) < 3:
        raise ValueError("缺少完整选科信息")
    data["reselection"] = reselection
    data["major"] = _as_list(data.get("major"))
    data["not_major"] = _as_list(data.get("not_major"))
    data["university"] = _as_list(data.get("university"))
    data["city"] = _as_list(data.get("city"))
    data["not_city"] = _as_list(data.get("not_city"))
    if int(data.get("dimension", 0) or 0) == 0 and not data["major"]:
        raise ValueError("专业优先模式缺少意向专业")
    return data


def recall_candidates(
    user_info: Mapping[str, Any],
    *,
    request_session: requests.Session | None = None,
) -> str:
    """Fetch the complete recall set. The returned payload must stay in the code layer."""
    client = request_session or requests
    dimension = int(user_info.get("dimension", 0) or 0)
    url = MAJOR_API_URL if dimension == 0 else CITY_API_URL
    payload = {
        "majorSecondDesc": _join(user_info.get("major")),
        "firstChoice": user_info.get("firstChoice", "物"),
        "score": user_info.get("score"),
        "recruit": user_info.get("recruit", 0) or 0,
        "province": user_info.get("province", "四川省"),
        "foreign": user_info.get("foreign", 0),
        "reselection": _join(user_info.get("reselection")),
        "matriculation": user_info.get("matriculation", 1),
        "target": user_info.get("target", 1),
        "healthCheckupLimit": user_info.get("healthCheckupLimit", "000"),
        "notMajorSecondDesc": _join(user_info.get("not_major")),
        "universitySecondDesc": _join(user_info.get("university")),
        "citySecondDesc": None,
        "notCitySecondDesc": _join(user_info.get("not_city")),
        "nature": user_info.get("nature"),
    }
    response = client.get(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 200:
        raise CandidateRecallError(body.get("msg") or "志愿候选数据召回失败")
    data = body.get("data", "")
    if not isinstance(data, str):
        raise CandidateRecallError("候选数据格式异常：API data 字段必须是字符串")
    if not data.strip():
        raise CandidateRecallError("当前条件下未召回到可用专业组")
    return data


def _emit_final_result(session_id: str, result: PlanResult, user_info: Mapping[str, Any]) -> None:
    send_final_result(
        session_id,
        result.plan_text,
        user_info["score"],
        user_info.get("firstChoice", ""),
        user_info.get("reselection", []),
        user_info.get("recruit", 0),
        user_info.get("city", []),
        user_info.get("not_city", []),
        user_info.get("university", []),
        user_info.get("major", []),
        user_info.get("not_major", []),
        user_info.get("matriculation", 1),
        user_info.get("target", 1),
        user_info.get("dimension", 0),
        user_info.get("foreign", 0),
        user_info.get("province", "四川省"),
        metadata={
            "policy": result.policy.model_dump(),
            "statistics": result.statistics.model_dump(),
        },
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        import re
        return [item.strip() for item in re.split(r"[,，、]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _join(value: Any) -> str:
    return ",".join(_as_list(value))

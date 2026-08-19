from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping

from pydantic import ValidationError

from volunteer_planner.models import PlanPolicy


class PlanPolicyError(ValueError):
    pass


def _strip_json_fence(value: str) -> str:
    text = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def parse_plan_policy(
    value: PlanPolicy | Mapping[str, Any] | str | None,
    user_info: Mapping[str, Any] | None = None,
) -> PlanPolicy:
    if isinstance(value, PlanPolicy):
        policy = value
    elif value is None or value == "":
        policy = PlanPolicy()
    else:
        payload: Dict[str, Any]
        if isinstance(value, str):
            try:
                decoded = json.loads(_strip_json_fence(value))
            except json.JSONDecodeError as exc:
                raise PlanPolicyError(f"plan_policy_json is not valid JSON: {exc.msg}") from exc
            if not isinstance(decoded, dict):
                raise PlanPolicyError("plan_policy_json must contain a JSON object")
            payload = decoded
        else:
            payload = dict(value)
        payload = _normalize_agent_payload(payload)
        if "total_groups" in payload and "risk_distribution" not in payload:
            try:
                total_groups = int(payload["total_groups"])
            except (TypeError, ValueError) as exc:
                raise PlanPolicyError("total_groups must be an integer") from exc
            base, remainder = divmod(total_groups, 3)
            payload["risk_distribution"] = {
                "chong": base + (1 if remainder > 0 else 0),
                "wen": base + (1 if remainder > 1 else 0),
                "bao": base,
            }
        try:
            policy = PlanPolicy.model_validate(payload)
        except ValidationError as exc:
            raise PlanPolicyError(str(exc)) from exc

    if user_info is None:
        return policy

    updates: Dict[str, Any] = {}
    if not policy.preferred_city_order:
        city = user_info.get("city", [])
        updates["preferred_city_order"] = _as_string_list(city)
    if not policy.preferred_university_order:
        university = user_info.get("university", [])
        updates["preferred_university_order"] = _as_string_list(university)
    if not policy.preferred_major_keywords:
        major = user_info.get("major", [])
        updates["preferred_major_keywords"] = _as_string_list(major)
    if not policy.excluded_major_keywords:
        excluded = user_info.get("not_major", [])
        updates["excluded_major_keywords"] = _as_string_list(excluded)
    return policy.model_copy(update=updates)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,，、]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common model aliases while keeping the executable schema strict."""
    normalized = dict(payload)
    risk_distribution = normalized.get("risk_distribution")
    if isinstance(risk_distribution, dict):
        aliases = {
            "aggressive": "chong",
            "moderate": "wen",
            "conservative": "bao",
        }
        normalized["risk_distribution"] = {
            aliases.get(key, key): value
            for key, value in risk_distribution.items()
        }

    for sort_field in ("group_sort", "major_sort"):
        rules = normalized.get(sort_field)
        if isinstance(rules, dict):
            if "field" in rules:
                rules = [rules]
            else:
                rules = [
                    {"field": field, "direction": direction}
                    for field, direction in rules.items()
                ]
        if isinstance(rules, list):
            normalized[sort_field] = [
                _normalize_sort_rule(rule)
                for rule in rules
            ]

    # These describe UserProfile and are already carried by the query argument.
    for profile_field in (
        "batch",
        "score",
        "subjects",
        "province",
        "excluded_conditions",
        "admission_exclusion",
    ):
        normalized.pop(profile_field, None)
    return normalized


def _normalize_sort_rule(rule: Any) -> Any:
    if not isinstance(rule, str):
        return rule
    parts = rule.replace(":", " ").split()
    if not parts:
        return rule
    direction = parts[-1].lower() if parts[-1].lower() in {"asc", "desc"} else "desc"
    field_parts = parts[:-1] if parts[-1].lower() in {"asc", "desc"} else parts
    return {"field": "_".join(field_parts), "direction": direction}

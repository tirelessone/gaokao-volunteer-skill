"""Serializable volunteer-plan workspace and stage dependency rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List

from volunteer_planner.models import MajorGroupCandidate


RECALL_STAGE = "recall_and_filter_candidates"
RANK_STAGE = "rank_major_groups"
MAJOR_SORT_STAGE = "sort_majors_within_groups"
PUBLISH_STAGE = "publish_volunteer_plan"
STAGE_ORDER = (RECALL_STAGE, RANK_STAGE, MAJOR_SORT_STAGE, PUBLISH_STAGE)


@dataclass
class VolunteerPlanWorkspace:
    query: str = ""
    user_info: Dict[str, Any] = field(default_factory=dict)
    recalled_groups: List[MajorGroupCandidate] = field(default_factory=list)
    eligible_groups: List[MajorGroupCandidate] = field(default_factory=list)
    selected_groups: List[MajorGroupCandidate] = field(default_factory=list)
    final_groups: List[MajorGroupCandidate] = field(default_factory=list)
    raw_character_count: int = 0
    filters_applied: List[str] = field(default_factory=list)
    bucket_available: Dict[str, int] = field(default_factory=dict)
    bucket_selected: Dict[str, int] = field(default_factory=dict)
    ranking_config: Dict[str, Any] = field(default_factory=dict)
    major_sort_config: Dict[str, Any] = field(default_factory=dict)
    plan_text: str = ""
    summary: Dict[str, Any] | None = None
    revision: int = 0
    completed_stages: List[str] = field(default_factory=list)

    def mark_completed(self, stage: str) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.completed_stages.sort(key=STAGE_ORDER.index)

    def invalidate_from(self, stage: str) -> None:
        start = STAGE_ORDER.index(stage)
        invalidated = set(STAGE_ORDER[start:])
        self.completed_stages = [item for item in self.completed_stages if item not in invalidated]
        if start <= STAGE_ORDER.index(RECALL_STAGE):
            self.user_info = {}
            self.recalled_groups = []
            self.eligible_groups = []
            self.raw_character_count = 0
            self.filters_applied = []
        if start <= STAGE_ORDER.index(RANK_STAGE):
            self.selected_groups = []
            self.bucket_available = {}
            self.bucket_selected = {}
            self.ranking_config = {}
        if start <= STAGE_ORDER.index(MAJOR_SORT_STAGE):
            self.final_groups = []
            self.major_sort_config = {}
        if start <= STAGE_ORDER.index(PUBLISH_STAGE):
            self.plan_text = ""
            self.summary = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field_name in ("recalled_groups", "eligible_groups", "selected_groups", "final_groups"):
            payload[field_name] = [group.model_dump(mode="json") for group in getattr(self, field_name)]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VolunteerPlanWorkspace":
        data = dict(payload)
        for field_name in ("recalled_groups", "eligible_groups", "selected_groups", "final_groups"):
            data[field_name] = [
                MajorGroupCandidate.model_validate(item)
                for item in data.get(field_name, [])
            ]
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})


def workspace_key(session_id: str, skill_name: str) -> str:
    return f"{session_id}:{skill_name}"


def infer_resume_stage(query: str, workspace: VolunteerPlanWorkspace | None) -> tuple[str, str]:
    """Return the earliest stage invalidated by the current user change."""
    if workspace is None:
        return RECALL_STAGE, "workspace_missing"

    text = re.sub(r"\s+", "", query)
    major_sort_markers = (
        "组内", "专业顺序", "专业排序", "放在最后", "放最后", "优先放",
        "不要选", "不选", "专业筛选不对", "调剂专业",
    )
    rank_markers = (
        "冲稳保", "多一点冲", "少一点稳", "少一些稳", "组间", "院校层次",
        "学校层次", "城市最重要", "院校最重要", "权重", "专业组顺序",
    )
    recall_markers = (
        "概率低于", "概率高于", "概率阈值", "过滤", "意向专业改", "换专业",
        "新增专业", "不考虑地区", "中外合作改", "省份改", "批次改", "选科改",
    )
    analysis_markers = ("只分析", "重新分析", "分析重点", "解读")

    candidates: list[tuple[str, str]] = []
    profile_change = bool(re.search(r"(?:分数|选科|批次|省份).{0,8}(?:改|换|调整)", text))
    score_change = bool(re.search(r"(?:改成|调整到|变成)\d{3}分", text))
    numeric_probability_change = bool(
        re.search(
            r"(?:概率.{0,8}\d{1,3}%?.{0,4}(?:以上|以下|高于|低于|超过|不超过|不低于)"
            r"|(?:去掉|删除|排除|过滤).{0,12}概率)",
            text,
        )
    )
    if (
        any(marker in text for marker in recall_markers)
        or profile_change
        or score_change
        or numeric_probability_change
    ):
        candidates.append((RECALL_STAGE, "recall_input_change"))
    if any(marker in text for marker in rank_markers):
        candidates.append((RANK_STAGE, "group_ranking_change"))
    if any(marker in text for marker in major_sort_markers):
        candidates.append((MAJOR_SORT_STAGE, "major_sort_change"))
    if any(marker in text for marker in analysis_markers):
        candidates.append((PUBLISH_STAGE, "analysis_only_change"))
    if candidates:
        desired, reason = min(candidates, key=lambda item: STAGE_ORDER.index(item[0]))
    else:
        incomplete_stage = next(
            (stage for stage in STAGE_ORDER if stage not in workspace.completed_stages),
            None,
        )
        if incomplete_stage is not None:
            desired, reason = incomplete_stage, "continue_incomplete_workspace"
        else:
            desired, reason = RECALL_STAGE, "ambiguous_change_fallback"

    if desired == PUBLISH_STAGE and not workspace.final_groups:
        desired, reason = MAJOR_SORT_STAGE, "missing_final_groups"
    if desired == MAJOR_SORT_STAGE and not workspace.selected_groups:
        desired, reason = RANK_STAGE, "missing_selected_groups"
    if desired == RANK_STAGE and not workspace.eligible_groups:
        desired, reason = RECALL_STAGE, "missing_eligible_groups"
    return desired, reason

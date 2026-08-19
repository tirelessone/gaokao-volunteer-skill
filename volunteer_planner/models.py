from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


RiskBucket = Literal["chong", "wen", "bao"]
SortDirection = Literal["asc", "desc"]


class RiskDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chong: int = Field(default=15, ge=0, le=45)
    wen: int = Field(default=15, ge=0, le=45)
    bao: int = Field(default=15, ge=0, le=45)

    def as_dict(self) -> Dict[str, int]:
        return self.model_dump()


class ProbabilityFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_min: Optional[int] = Field(default=None, ge=0, le=100)
    group_max: Optional[int] = Field(default=None, ge=0, le=100)
    major_min: Optional[int] = Field(default=None, ge=0, le=100)
    major_max: Optional[int] = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.group_min is not None and self.group_max is not None:
            if self.group_min > self.group_max:
                raise ValueError("group_min cannot be greater than group_max")
        if self.major_min is not None and self.major_max is not None:
            if self.major_min > self.major_max:
                raise ValueError("major_min cannot be greater than major_max")
        return self


class SortRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: SortDirection = "desc"


class PlanPolicy(BaseModel):
    """Executable policy compiled from the user's natural-language request."""

    model_config = ConfigDict(extra="forbid")

    total_groups: int = Field(default=45, ge=1, le=45)
    max_majors_per_group: int = Field(default=6, ge=1, le=6)
    risk_distribution: RiskDistribution = Field(default_factory=RiskDistribution)
    probability_filter: ProbabilityFilter = Field(default_factory=ProbabilityFilter)
    risk_order: List[RiskBucket] = Field(default_factory=lambda: ["chong", "wen", "bao"])
    deficit_fill_order: List[RiskBucket] = Field(default_factory=lambda: ["wen", "bao", "chong"])
    group_sort: List[SortRule] = Field(
        default_factory=lambda: [
            SortRule(field="intent_major_probability", direction="asc"),
            SortRule(field="university_level", direction="desc"),
            SortRule(field="city_preference", direction="desc"),
        ]
    )
    major_sort: List[SortRule] = Field(
        default_factory=lambda: [
            SortRule(field="intentional", direction="desc"),
            SortRule(field="major_preference", direction="desc"),
            SortRule(field="admission_probability", direction="desc"),
        ]
    )
    preferred_major_keywords: List[str] = Field(default_factory=list)
    excluded_major_keywords: List[str] = Field(default_factory=list)
    preferred_city_order: List[str] = Field(default_factory=list)
    preferred_university_order: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_policy(self):
        if len(set(self.risk_order)) != 3:
            raise ValueError("risk_order must contain chong, wen and bao exactly once")
        if set(self.risk_order) != {"chong", "wen", "bao"}:
            raise ValueError("risk_order must contain chong, wen and bao exactly once")
        if len(set(self.deficit_fill_order)) != 3:
            raise ValueError("deficit_fill_order must contain chong, wen and bao exactly once")
        if set(self.deficit_fill_order) != {"chong", "wen", "bao"}:
            raise ValueError("deficit_fill_order must contain chong, wen and bao exactly once")
        if sum(self.risk_distribution.as_dict().values()) > self.total_groups:
            raise ValueError("risk_distribution cannot exceed total_groups")

        group_fields = {
            "admission_probability",
            "intent_major_probability",
            "intentional_major_count",
            "major_count",
            "university_level",
            "university_preference",
            "city_level",
            "city_preference",
        }
        major_fields = {
            "intentional",
            "major_preference",
            "admission_probability",
            "major_name",
        }
        unsupported_group = [rule.field for rule in self.group_sort if rule.field not in group_fields]
        unsupported_major = [rule.field for rule in self.major_sort if rule.field not in major_fields]
        if unsupported_group:
            raise ValueError(f"Unsupported group_sort fields: {unsupported_group}")
        if unsupported_major:
            raise ValueError(f"Unsupported major_sort fields: {unsupported_major}")
        return self


class MajorCandidate(BaseModel):
    name: str
    probability: Optional[int] = None
    intentional: bool = False
    grade: Optional[str] = None


class MajorGroupCandidate(BaseModel):
    group_id: str
    university: str = ""
    probability: int = 0
    city: str = ""
    metadata_lines: List[str] = Field(default_factory=list)
    majors: List[MajorCandidate] = Field(default_factory=list)


class PlanStatistics(BaseModel):
    raw_character_count: int = 0
    recalled_groups: int = 0
    eligible_groups: int = 0
    selected_groups: int = 0
    selected_majors: int = 0
    bucket_available: Dict[str, int] = Field(default_factory=dict)
    bucket_selected: Dict[str, int] = Field(default_factory=dict)
    filters_applied: List[str] = Field(default_factory=list)


class PlanResult(BaseModel):
    status: Literal["success", "error"]
    plan_text: str = ""
    groups: List[MajorGroupCandidate] = Field(default_factory=list)
    policy: PlanPolicy = Field(default_factory=PlanPolicy)
    statistics: PlanStatistics = Field(default_factory=PlanStatistics)
    message: str = ""

    def model_summary(self) -> Dict[str, Any]:
        """Return the bounded payload that may safely be exposed to the model."""
        return {
            "status": self.status,
            "message": self.message,
            "selected_groups": self.statistics.selected_groups,
            "selected_majors": self.statistics.selected_majors,
            "bucket_selected": self.statistics.bucket_selected,
            "filters_applied": self.statistics.filters_applied,
        }

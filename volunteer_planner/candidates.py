from __future__ import annotations

import re
from typing import Iterable, List

from volunteer_planner.models import MajorCandidate, MajorGroupCandidate


_GROUP_RE = re.compile(r"=+\s*专业组:(\d+)=+\s*(.*?)\s*(?==+\s*专业组:|\Z)", re.S)
_PROBABILITY_RE = re.compile(r"\(概率:(-?\d+)%\)")
_INTENT_RE = re.compile(r"\((true|false)\)", re.IGNORECASE)
_GRADE_RE = re.compile(r"\((A\+|A-|A|B\+|B-|B|C\+|C-|C)\)\s*$")


def parse_candidate_groups(raw: str) -> List[MajorGroupCandidate]:
    groups: List[MajorGroupCandidate] = []
    for match in _GROUP_RE.finditer(raw or ""):
        group_id = match.group(1).strip()
        content = match.group(2).strip()
        metadata_lines: List[str] = []
        majors: List[MajorCandidate] = []
        university = ""
        group_probability = 0

        for original_line in content.splitlines():
            line = original_line.strip()
            if not line:
                continue
            if line.startswith("院校:"):
                university = line.split(":", 1)[1].strip()
                metadata_lines.append(line)
                continue
            if line.startswith("专业组概率:"):
                probability_match = re.search(r"专业组概率:(-?\d+)%", line)
                if probability_match:
                    group_probability = int(probability_match.group(1))
                metadata_lines.append(line)
                continue
            if line.startswith("包含专业:"):
                continue
            if line.startswith("- "):
                majors.append(_parse_major(line[2:].strip()))
                continue
            metadata_lines.append(line)

        for major in majors:
            if major.probability is None or major.probability < 0:
                major.probability = group_probability

        groups.append(
            MajorGroupCandidate(
                group_id=group_id,
                university=university,
                probability=group_probability,
                metadata_lines=metadata_lines,
                majors=majors,
            )
        )
    return groups


def _parse_major(text: str) -> MajorCandidate:
    probability_match = _PROBABILITY_RE.search(text)
    intent_match = _INTENT_RE.search(text)
    grade_match = _GRADE_RE.search(text)
    probability = int(probability_match.group(1)) if probability_match else None
    intentional = bool(intent_match and intent_match.group(1).lower() == "true")
    grade = grade_match.group(1) if grade_match else None

    name = _PROBABILITY_RE.sub("", text)
    name = _INTENT_RE.sub("", name)
    if grade_match:
        name = _GRADE_RE.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    return MajorCandidate(
        name=name,
        probability=probability,
        intentional=intentional,
        grade=grade,
    )


def render_candidate_groups(
    groups: Iterable[MajorGroupCandidate],
    include_internal_flags: bool = False,
) -> str:
    blocks: List[str] = []
    for group in groups:
        lines = [f"========== 专业组:{group.group_id}=========="]
        metadata = _normalized_metadata(group)
        lines.extend(metadata)
        lines.append("包含专业:")
        for major in group.majors:
            suffix = f" (概率:{major.probability}%)" if major.probability is not None else ""
            if include_internal_flags:
                suffix += f" ({str(major.intentional).lower()})"
                if major.grade:
                    suffix += f" ({major.grade})"
            lines.append(f"- {major.name}{suffix}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _normalized_metadata(group: MajorGroupCandidate) -> List[str]:
    output: List[str] = []
    has_university = False
    has_probability = False
    for line in group.metadata_lines:
        if line.startswith("院校:"):
            if not has_university:
                output.append(f"院校:{group.university}")
                has_university = True
        elif line.startswith("专业组概率:"):
            if not has_probability:
                output.append(f"专业组概率:{group.probability}%")
                has_probability = True
        else:
            output.append(line)
    if not has_university:
        output.insert(0, f"院校:{group.university}")
    if not has_probability:
        output.insert(1, f"专业组概率:{group.probability}%")
    return output

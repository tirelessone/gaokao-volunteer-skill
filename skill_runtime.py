"""Project-local Agent Skill discovery with three-level progressive loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Any, Dict, Iterable, List, Optional

import yaml


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillConfigurationError(ValueError):
    """Raised when a project Skill or its runtime configuration is malformed."""


class SkillNotLoadedError(RuntimeError):
    """Raised when a caller skips the required progressive-loading step."""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    directory: Path
    version: str = "1.0.0"
    tags: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "directory": str(self.directory),
            "version": self.version,
            "tags": list(self.tags),
            "allowed_tools": list(self.allowed_tools),
        }


@dataclass(frozen=True)
class SkillDefinition:
    metadata: SkillMetadata
    instructions: str
    references: tuple[str, ...]
    scripts: tuple[str, ...]


class SkillRegistry:
    """Load metadata, instructions, references, and runtime config only when needed."""

    def __init__(self, roots: Iterable[Path | str]):
        self._roots = tuple(Path(root).resolve() for root in roots)
        self._metadata: Dict[str, SkillMetadata] = {}
        self._definitions: Dict[str, SkillDefinition] = {}
        self._reference_cache: Dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        self.reload()

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    def reload(self) -> int:
        """Scan only SKILL.md frontmatter; do not read any instruction body."""
        discovered: Dict[str, SkillMetadata] = {}
        with self._lock:
            for root in self._roots:
                if not root.exists():
                    continue
                for directory in sorted(path for path in root.iterdir() if path.is_dir()):
                    skill_file = directory / "SKILL.md"
                    if not skill_file.is_file():
                        continue
                    metadata = self._read_metadata_frontmatter(skill_file)
                    discovered[metadata.name] = metadata
            self._metadata = discovered
            # A manual reload must also pick up body/reference edits whose frontmatter is unchanged.
            self._definitions.clear()
            self._reference_cache.clear()
            return len(discovered)

    def list_metadata(self) -> List[SkillMetadata]:
        with self._lock:
            return list(self._metadata.values())

    def catalog_prompt(self) -> str:
        rows = []
        for item in self.list_metadata():
            tags = f"；标签：{', '.join(item.tags)}" if item.tags else ""
            rows.append(f"- {item.name}: {item.description}{tags}")
        return "\n".join(rows) if rows else "- No project skills are configured."

    def recall_metadata(self, query: str, limit: int = 5) -> List[SkillMetadata]:
        """Return a small metadata-only candidate set for the current specialist Agent."""
        normalized = str(query or "").lower()
        scored: List[tuple[int, SkillMetadata]] = []
        for item in self.list_metadata():
            terms = [item.name, item.description, *item.tags]
            score = sum(3 if term.lower() in normalized else 0 for term in terms if term)
            score += sum(1 for char in normalized if char.strip() and char in item.description)
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [item for _, item in scored[:max(1, limit)]]

    def load_skill(self, name: str) -> SkillDefinition:
        """Level 2: load and cache the complete SKILL.md body after selection."""
        with self._lock:
            if name in self._definitions:
                return self._definitions[name]
            metadata = self._metadata.get(name)
            if metadata is None:
                roots = ", ".join(str(root) for root in self._roots)
                raise SkillConfigurationError(f"Skill '{name}' was not found under: {roots}")
            definition = self._read_definition(metadata)
            self._definitions[name] = definition
            return definition

    def read_reference(self, name: str, reference_path: str) -> str:
        """Level 3: load and cache one explicitly requested reference file."""
        with self._lock:
            definition = self._definitions.get(name)
            if definition is None:
                raise SkillNotLoadedError(f"Call load_skill('{name}') before reading references")
            normalized = self._normalize_reference_path(reference_path)
            if normalized not in definition.references:
                raise SkillConfigurationError(
                    f"Reference '{normalized}' is not declared by Skill '{name}'"
                )
            cache_key = (name, normalized)
            if cache_key in self._reference_cache:
                return self._reference_cache[cache_key]
            path = self._resolve_inside(definition.metadata.directory, normalized)
            content = path.read_text(encoding="utf-8")
            self._reference_cache[cache_key] = content
            return content

    def is_skill_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._definitions

    def is_reference_cached(self, name: str, reference_path: str) -> bool:
        normalized = self._normalize_reference_path(reference_path)
        with self._lock:
            return (name, normalized) in self._reference_cache

    @staticmethod
    def _read_frontmatter_only(skill_file: Path) -> dict:
        lines: List[str] = []
        with skill_file.open("r", encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                raise SkillConfigurationError(f"Missing YAML frontmatter: {skill_file}")
            for line in handle:
                if line.strip() == "---":
                    break
                lines.append(line)
            else:
                raise SkillConfigurationError(f"Unclosed YAML frontmatter: {skill_file}")
        try:
            frontmatter = yaml.safe_load("".join(lines)) or {}
        except yaml.YAMLError as exc:
            raise SkillConfigurationError(f"Invalid YAML frontmatter: {skill_file}") from exc
        if not isinstance(frontmatter, dict):
            raise SkillConfigurationError(f"Frontmatter must be a mapping: {skill_file}")
        return frontmatter

    @classmethod
    def _read_metadata_frontmatter(cls, skill_file: Path) -> SkillMetadata:
        frontmatter = cls._read_frontmatter_only(skill_file)
        supported = {
            "name", "description", "version", "tags", "allowed-tools",
            "disable-model-invocation", "metadata",
        }
        unsupported = set(frontmatter) - supported
        if unsupported:
            raise SkillConfigurationError(
                f"Unsupported frontmatter fields in {skill_file}: {sorted(unsupported)}"
            )
        name = str(frontmatter.get("name", "")).strip()
        description = str(frontmatter.get("description", "")).strip()
        if not name or not _SKILL_NAME_RE.fullmatch(name):
            raise SkillConfigurationError(f"Invalid skill name '{name}' in {skill_file}")
        if name != skill_file.parent.name:
            raise SkillConfigurationError(
                f"Skill name '{name}' must match directory '{skill_file.parent.name}'"
            )
        if not description:
            raise SkillConfigurationError(f"Missing skill description: {skill_file}")
        custom_metadata = frontmatter.get("metadata", {}) or {}
        if not isinstance(custom_metadata, dict):
            raise SkillConfigurationError(f"metadata must be a mapping in {skill_file}")
        tags = cls._as_string_tuple(
            frontmatter.get("tags", custom_metadata.get("tags", ())), "tags", skill_file
        )
        allowed_tools = cls._as_string_tuple(
            frontmatter.get("allowed-tools", ()), "allowed-tools", skill_file
        )
        version = str(
            frontmatter.get("version", custom_metadata.get("version", "1.0.0"))
        ).strip() or "1.0.0"
        return SkillMetadata(
            name=name,
            description=description,
            directory=skill_file.parent,
            version=version,
            tags=tags,
            allowed_tools=allowed_tools,
        )

    @classmethod
    def _read_definition(cls, metadata: SkillMetadata) -> SkillDefinition:
        content = (metadata.directory / "SKILL.md").read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(content)
        if match is None:
            raise SkillConfigurationError(f"Missing YAML frontmatter: {metadata.directory / 'SKILL.md'}")
        instructions = content[match.end():].strip()
        if not instructions:
            raise SkillConfigurationError(f"Skill instructions are empty: {metadata.directory}")
        references_dir = metadata.directory / "references"
        references = tuple(
            f"references/{path.relative_to(references_dir).as_posix()}"
            for path in sorted(references_dir.rglob("*"))
            if path.is_file()
        ) if references_dir.is_dir() else ()
        scripts_dir = metadata.directory / "scripts"
        scripts = tuple(
            f"scripts/{path.relative_to(scripts_dir).as_posix()}"
            for path in sorted(scripts_dir.rglob("*"))
            if path.is_file()
        ) if scripts_dir.is_dir() else ()
        return SkillDefinition(
            metadata=metadata,
            instructions=instructions,
            references=references,
            scripts=scripts,
        )

    @staticmethod
    def _as_string_tuple(value: Any, field_name: str, skill_file: Path) -> tuple[str, ...]:
        if value is None:
            return ()
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, (list, tuple)):
            raise SkillConfigurationError(f"{field_name} must be a list in {skill_file}")
        return tuple(str(item).strip() for item in values if str(item).strip())

    @staticmethod
    def _normalize_reference_path(reference_path: str) -> str:
        normalized = reference_path.strip().replace("\\", "/")
        if not normalized.startswith("references/"):
            normalized = f"references/{normalized}"
        return normalized

    @staticmethod
    def _resolve_inside(skill_dir: Path, relative_path: str) -> Path:
        candidate = (skill_dir / relative_path).resolve()
        root = skill_dir.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SkillConfigurationError("Reference path escapes the Skill directory") from exc
        if not candidate.is_file():
            raise SkillConfigurationError(f"Reference does not exist: {relative_path}")
        return candidate


_registry: Optional[SkillRegistry] = None
_registry_lock = threading.Lock()


def get_skill_registry() -> SkillRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            project_root = Path(__file__).resolve().parent
            _registry = SkillRegistry([project_root / "skills"])
        return _registry

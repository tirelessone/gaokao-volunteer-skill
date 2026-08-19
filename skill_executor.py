"""Progressively activate Skills for a domain Agent without creating a SubAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, MutableMapping, Optional

from agents import function_tool

from skill_runtime import SkillConfigurationError, SkillRegistry, get_skill_registry
from workspace_store import FileWorkspaceStore


ToolBuilder = Callable[["SkillExecutionContext"], Any]


def _skill_log(session_id: str, event: str, **data: Any) -> None:
    details = ""
    if data:
        details = " " + json.dumps(data, ensure_ascii=False, default=str)
    print(f"[SKILL][session={session_id}][{event}]{details}", flush=True)


@dataclass
class SkillExecutionContext:
    session_id: str
    session_state: MutableMapping[str, Any]
    activated_skills: set[str] = field(default_factory=set)
    loaded_references: set[tuple[str, str]] = field(default_factory=set)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    current_skill: str = ""
    workspace_store: FileWorkspaceStore | None = None
    workspace_store_key: str = ""

    def trace(self, event: str, **data: Any) -> None:
        self.session_state.setdefault("skill_trace", []).append({"event": event, **data})

    def trace_tool(self, tool_name: str, status: str, **data: Any) -> None:
        self.trace("skill_tool", tool_name=tool_name, status=status, **data)

    def persist_workspace(self) -> None:
        workspace = self.artifacts.get("volunteer_plan")
        if self.workspace_store is None or not self.workspace_store_key or workspace is None:
            return
        serializer = getattr(workspace, "to_dict", None)
        if callable(serializer):
            self.workspace_store.set(self.workspace_store_key, serializer())
            _skill_log(
                self.session_id,
                "WORKSPACE_SAVED",
                key=self.workspace_store_key,
                revision=getattr(workspace, "revision", 0),
                completed_stages=list(getattr(workspace, "completed_stages", [])),
                ttl_seconds=self.workspace_store.ttl_seconds,
            )


@dataclass
class ActivatedSkill:
    """Per-request capability lease attached to the same logical domain Agent."""

    name: str
    query: str
    instructions: str
    tools: list[Any]
    allowed_tools: tuple[str, ...]
    references: tuple[str, ...]
    scripts: tuple[str, ...]
    trace_start: int

    def get_tool(self, name: str) -> Any | None:
        return next((tool for tool in self.tools if getattr(tool, "name", "") == name), None)


class SkillToolRegistry:
    """Registry of atomic tools from which each Skill receives an allow-listed subset."""

    def __init__(self):
        self._builders: Dict[str, ToolBuilder] = {}
        self._lock = threading.RLock()

    def register(self, name: str, builder: ToolBuilder) -> None:
        with self._lock:
            if name in self._builders:
                raise SkillConfigurationError(f"Skill tool '{name}' is already registered")
            self._builders[name] = builder

    def build_allowed(self, names: tuple[str, ...], context: SkillExecutionContext) -> list[Any]:
        with self._lock:
            missing = [name for name in names if name not in self._builders]
            if missing:
                raise SkillConfigurationError(f"Skill declares unknown tools: {missing}")
            return [self._builders[name](context) for name in names]

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._builders)


class SkillExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        tools: SkillToolRegistry,
        workspace_store: FileWorkspaceStore | None = None,
    ):
        self.registry = registry
        self.tools = tools
        self.workspace_store = workspace_store

    def get_workspace_status(self, session_id: str, skill_name: str) -> dict[str, Any]:
        from volunteer_workspace import VolunteerPlanWorkspace, workspace_key

        if self.workspace_store is None:
            return {"available": False, "ttl_seconds": 0, "completed_stages": [], "revision": 0}
        key = workspace_key(session_id, skill_name)
        metadata = self.workspace_store.metadata(key)
        if not metadata["available"]:
            return {**metadata, "completed_stages": [], "revision": 0}
        payload = self.workspace_store.get(key)
        try:
            workspace = VolunteerPlanWorkspace.from_dict(payload or {})
        except Exception:
            self.workspace_store.delete(key)
            return {"available": False, "ttl_seconds": 0, "completed_stages": [], "revision": 0}
        return {
            **metadata,
            "completed_stages": list(workspace.completed_stages),
            "revision": workspace.revision,
        }

    def activate_skill(
        self,
        skill_name: str,
        query: str,
        context: SkillExecutionContext,
    ) -> ActivatedSkill:
        """Load one Skill and return a temporary prompt/tool lease for the domain Agent."""
        _skill_log(
            context.session_id,
            "ACTIVATE_START",
            skill_name=skill_name,
            query=query[:500],
        )
        trace_start = len(context.session_state.get("skill_trace", []))
        cache_hit = self.registry.is_skill_loaded(skill_name)
        definition = self.registry.load_skill(skill_name)
        _skill_log(
            context.session_id,
            "SKILL_LOADED",
            skill_name=skill_name,
            cache_hit=cache_hit,
            version=definition.metadata.version,
            declared_tools=list(definition.metadata.allowed_tools),
        )
        from volunteer_workspace import (
            STAGE_ORDER,
            VolunteerPlanWorkspace,
            infer_resume_stage,
            workspace_key,
        )

        context.current_skill = skill_name
        context.artifacts["skill_query"] = query
        context.artifacts["current_change_query"] = query
        context.activated_skills.add(skill_name)
        context.session_state.pop("volunteer_plan_analysis", None)
        context.artifacts.pop("volunteer_plan_analysis", None)
        context.trace("load_skill", skill_name=skill_name, cache_hit=cache_hit)

        declared_tools = definition.metadata.allowed_tools
        if not declared_tools:
            raise SkillConfigurationError(f"Skill '{skill_name}' does not declare allowed-tools")
        context.workspace_store = context.workspace_store or self.workspace_store
        context.workspace_store_key = workspace_key(context.session_id, skill_name)
        stored_payload = (
            context.workspace_store.get(context.workspace_store_key)
            if context.workspace_store is not None else None
        )
        workspace = None
        if stored_payload:
            try:
                workspace = VolunteerPlanWorkspace.from_dict(stored_payload)
            except Exception:
                _skill_log(
                    context.session_id,
                    "WORKSPACE_CORRUPT",
                    key=context.workspace_store_key,
                )
                if context.workspace_store is not None:
                    context.workspace_store.delete(context.workspace_store_key)
        resume_stage, resume_reason = infer_resume_stage(query, workspace)
        workspace_restored = workspace is not None
        if workspace is None:
            workspace = VolunteerPlanWorkspace(revision=1)
            resume_stage = STAGE_ORDER[0]
            resume_reason = "workspace_missing_or_expired"
        else:
            workspace.revision = max(1, workspace.revision + 1)
        _skill_log(
            context.session_id,
            "RESUME_DECISION",
            workspace_restored=workspace_restored,
            workspace_revision=workspace.revision,
            resume_stage=resume_stage,
            resume_reason=resume_reason,
            completed_before_resume=list(workspace.completed_stages),
        )
        effective_query = query
        if workspace_restored and workspace.query and workspace.query not in query:
            effective_query = f"{workspace.query}\n\n用户本轮补充或调整：{query}"
        context.artifacts["skill_query"] = effective_query
        completed_before_resume = list(workspace.completed_stages)
        workspace.invalidate_from(resume_stage)
        context.artifacts["volunteer_plan"] = workspace
        context.artifacts["resume_stage"] = resume_stage
        context.artifacts["workspace_restored"] = workspace_restored
        context.artifacts["resume_reason"] = resume_reason
        context.persist_workspace()

        start_index = STAGE_ORDER.index(resume_stage)
        remaining_stage_tools = set(STAGE_ORDER[start_index:])
        allowed = tuple(
            name for name in declared_tools
            if name == "read_skill_reference" or name in remaining_stage_tools
        )
        context.artifacts["active_allowed_tools"] = list(allowed)
        skill_tools = self.tools.build_allowed(allowed, context)
        _skill_log(
            context.session_id,
            "TOOLS_BOUND",
            skill_name=skill_name,
            allowed_tools=list(allowed),
        )
        context.trace(
            "workspace_restored" if workspace_restored else "workspace_initialized",
            skill_name=skill_name,
            revision=workspace.revision,
            completed_stages=completed_before_resume,
        )
        context.trace(
            "resume_stage_selected",
            skill_name=skill_name,
            resume_stage=resume_stage,
            reason=resume_reason,
        )
        context.trace(
            "bind_skill_tools",
            skill_name=skill_name,
            allowed_tools=list(allowed),
        )

        previous = context.session_state.get("previous_skill_execution")
        previous_section = ""
        if previous:
            previous_section = (
                "\n<previous_skill_execution>\n"
                + json.dumps(previous, ensure_ascii=False)
                + "\n</previous_skill_execution>\n"
            )

        reference_lines = "\n".join(f"- {item}" for item in definition.references) or "- 无"
        script_lines = "\n".join(f"- {item}" for item in definition.scripts) or "- 无"
        system_prompt = f"""
你仍然是志愿生成 Agent `volunteer_plan_specialist`，现在已进入 Skill `{skill_name}` 执行模式。
这是同一个领域 Agent 的临时能力激活，不是向其他 Agent 转交任务。执行期间只负责严格完成 SOP。

<execution_rules>
- 只能调用当前工具池中的工具，不能假设存在其他工具。
- 必须按照 SKILL.md 的阶段顺序执行，不得跳过前置阶段。
- 工具返回成功后不要用相同参数重复调用。
- 禁止编造外部数据；工具失败时不得猜测结果。
- 候选专业组全集保存在工具内部 workspace，不得要求工具返回完整候选数据。
- 完成 Skill 阶段后必须调用 `publish_volunteer_plan`，随后调用 `analyze_volunteer_plan`。
- 不要自行撰写志愿表或分析正文，这两项内容由工具通过事件发布。
- `resume_stage` 是 Runtime 根据 workspace 和依赖关系校验后的结果，禁止重复调用它之前的阶段。
</execution_rules>

<resume_context>
workspace_restored: {str(workspace_restored).lower()}
workspace_revision: {workspace.revision}
completed_before_resume: {json.dumps(completed_before_resume, ensure_ascii=False)}
resume_stage: {resume_stage}
resume_reason: {resume_reason}
remaining_tools: {json.dumps(list(allowed), ensure_ascii=False)}
如果 workspace_restored=false，必须从召回阶段开始；如果为 true，只执行 resume_stage 及其下游阶段。
</resume_context>

<skill_dir>{definition.metadata.directory}</skill_dir>
<available_references>
{reference_lines}
</available_references>
<available_scripts>
{script_lines}
</available_scripts>
<skill_content>
{definition.instructions}
</skill_content>
<skill_query>
{effective_query}
</skill_query>
<current_change_query>
{query}
</current_change_query>
{previous_section}
""".strip()
        activated = ActivatedSkill(
            name=skill_name,
            query=effective_query,
            instructions=system_prompt,
            tools=skill_tools,
            allowed_tools=allowed,
            references=definition.references,
            scripts=definition.scripts,
            trace_start=trace_start,
        )
        context.artifacts["activated_skill"] = activated
        context.trace("skill_activated", skill_name=skill_name)
        _skill_log(
            context.session_id,
            "ACTIVATE_DONE",
            skill_name=skill_name,
            resume_stage=resume_stage,
        )
        return activated

    def finalize_skill(
        self,
        activated: ActivatedSkill,
        context: SkillExecutionContext,
        *,
        final_output: Any = "",
    ) -> dict[str, Any]:
        """Validate postconditions, persist a compact execution record, and revoke the lease."""
        workspace = context.artifacts.get("volunteer_plan")
        summary = getattr(workspace, "summary", None) if workspace is not None else None
        if not summary:
            _skill_log(
                context.session_id,
                "FINALIZE_FAILED",
                skill_name=activated.name,
                reason="missing_publish_result",
            )
            context.trace(
                "skill_failed",
                skill_name=activated.name,
                error_type="MissingPublishResult" if getattr(workspace, "final_groups", None) else "SubAgentIncomplete",
                message=f"Skill mode final output: {final_output}",
            )
            raise RuntimeError(
                f"Volunteer Agent did not publish the active Skill result; final output: {final_output}"
            )

        current_trace = context.session_state.get("skill_trace", [])[activated.trace_start:]
        context.session_state["active_skill"] = activated.name
        context.session_state["previous_skill_execution"] = {
            "skill_name": activated.name,
            "query": activated.query,
            "summary": summary,
            "tool_trace": [
                item for item in current_trace
                if item.get("event") == "skill_tool"
            ],
            "workspace_revision": getattr(workspace, "revision", 0),
        }
        context.persist_workspace()
        context.trace("skill_completed", skill_name=activated.name)
        context.trace("revoke_skill_tools", skill_name=activated.name)
        context.artifacts.pop("activated_skill", None)
        context.current_skill = ""
        _skill_log(
            context.session_id,
            "FINALIZE_DONE",
            skill_name=activated.name,
            summary=summary,
        )
        return summary

    def fail_skill(
        self,
        activated: ActivatedSkill,
        context: SkillExecutionContext,
        exc: Exception,
    ) -> None:
        _skill_log(
            context.session_id,
            "EXECUTION_FAILED",
            skill_name=activated.name,
            error_type=type(exc).__name__,
            message=str(exc),
        )
        context.trace(
            "skill_failed",
            skill_name=activated.name,
            error_type=type(exc).__name__,
            message=str(exc),
        )
        context.trace("revoke_skill_tools", skill_name=activated.name)
        context.artifacts.pop("activated_skill", None)
        context.current_skill = ""


def build_reference_tool(registry: SkillRegistry) -> ToolBuilder:
    def builder(context: SkillExecutionContext):
        @function_tool(strict_mode=False)
        def read_skill_reference(reference_path: str) -> dict[str, Any]:
            """读取当前 Skill 声明的一份 reference。只在 SOP 明确要求时按需调用。"""
            skill_name = context.current_skill
            if not skill_name:
                raise SkillConfigurationError("No active Skill")
            normalized = registry._normalize_reference_path(reference_path)
            cache_hit = registry.is_reference_cached(skill_name, normalized)
            _skill_log(
                context.session_id,
                "REFERENCE_READ_START",
                skill_name=skill_name,
                reference_path=normalized,
                cache_hit=cache_hit,
            )
            content = registry.read_reference(skill_name, normalized)
            context.loaded_references.add((skill_name, normalized))
            context.trace_tool(
                "read_skill_reference",
                "success",
                reference_path=normalized,
                cache_hit=cache_hit,
            )
            _skill_log(
                context.session_id,
                "REFERENCE_READ_DONE",
                skill_name=skill_name,
                reference_path=normalized,
                characters=len(content),
            )
            return {"reference_path": normalized, "content": content}

        return read_skill_reference

    return builder


_executor: Optional[SkillExecutor] = None
_executor_lock = threading.Lock()


def get_skill_executor() -> SkillExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            from volunteer_skill_tools import register_volunteer_skill_tools

            registry = get_skill_registry()
            tool_registry = SkillToolRegistry()
            tool_registry.register("read_skill_reference", build_reference_tool(registry))
            register_volunteer_skill_tools(tool_registry)
            workspace_store = FileWorkspaceStore(
                Path(__file__).resolve().parent / "runtime_data" / "skill_workspace_store.txt",
                ttl_seconds=30 * 60,
            )
            _executor = SkillExecutor(registry, tool_registry, workspace_store)
        return _executor

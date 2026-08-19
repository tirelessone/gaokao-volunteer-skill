"""Agent tool that analyzes the volunteer plan produced by the generation Skill."""

from __future__ import annotations

import asyncio
from typing import Any

from agents import function_tool

from analyze_groups import create_groups_analysis
from event_queue import send_end, send_error


async def execute_volunteer_plan_analysis(
    context: Any,
    analysis_focus: str = "",
) -> dict[str, Any]:
    """Analyze the private Skill workspace without asking the model to relay plan data."""
    workspace = context.artifacts.get("volunteer_plan")
    if workspace is None or not getattr(workspace, "summary", None):
        return {
            "status": "error",
            "message": "尚未生成并发布志愿表，不能执行志愿表分析。",
        }

    previous = context.session_state.get("volunteer_plan_analysis")
    if previous and previous.get("status") == "success":
        return {
            **previous,
            "message": "当前志愿表已经完成分析，无需重复调用。",
        }

    context.trace_tool(
        "analyze_volunteer_plan",
        "started",
        analysis_focus=analysis_focus.strip(),
    )
    try:
        analysis = await asyncio.to_thread(
            create_groups_analysis,
            workspace.user_info,
            workspace.plan_text,
            context.session_id,
            int(workspace.user_info.get("dimension", 0) or 0),
            analysis_focus,
        )
        if analysis.startswith("生成分析时出错:"):
            raise RuntimeError(analysis)

        result = {
            "status": "success",
            "message": "志愿表分析已生成并通过 analysis 事件发布。",
            "analysis_characters": len(analysis),
            "analysis_focus": analysis_focus.strip(),
        }
        context.session_state["volunteer_plan_analysis"] = result
        context.artifacts["volunteer_plan_analysis"] = analysis
        context.trace_tool(
            "analyze_volunteer_plan",
            "success",
            analysis_characters=len(analysis),
        )
        send_end(context.session_id)
        return result
    except Exception as exc:
        message = f"志愿表分析失败：{exc}"
        result = {"status": "error", "message": message}
        context.session_state["volunteer_plan_analysis"] = result
        context.trace_tool(
            "analyze_volunteer_plan",
            "error",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        send_error(context.session_id, message)
        send_end(context.session_id)
        return result


def build_volunteer_analysis_tool(context: Any):
    @function_tool(strict_mode=False)
    async def analyze_volunteer_plan(analysis_focus: str = "") -> dict[str, Any]:
        """分析刚由 Skill 生成的志愿表。不得传入志愿表正文；可传入用户特别关注的分析方向。"""
        return await execute_volunteer_plan_analysis(context, analysis_focus)

    return analyze_volunteer_plan

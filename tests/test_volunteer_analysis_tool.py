import unittest
from types import SimpleNamespace
from unittest.mock import patch

from event_queue import get_or_create_queue, remove_queue, send_analysis
from skill_executor import SkillExecutionContext
from volunteer_analysis_tool import execute_volunteer_plan_analysis


class VolunteerAnalysisToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_consumes_workspace_and_finishes_event_stream(self):
        session_id = "analysis-tool-test"
        context = SkillExecutionContext(session_id, {})
        context.artifacts["volunteer_plan"] = SimpleNamespace(
            summary={"status": "success"},
            user_info={"score": 600, "dimension": 0, "major": ["计算机"]},
            plan_text="专业组概率:60%\n院校:测试大学",
        )
        get_or_create_queue(session_id)

        def fake_analysis(user_info, plan_text, called_session_id, dimension, focus):
            self.assertEqual(user_info["score"], 600)
            self.assertEqual(called_session_id, session_id)
            self.assertEqual(focus, "重点分析冲稳保结构")
            send_analysis(session_id, "分析正文")
            return "分析正文"

        with patch("volunteer_analysis_tool.create_groups_analysis", side_effect=fake_analysis):
            result = await execute_volunteer_plan_analysis(
                context,
                "重点分析冲稳保结构",
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(context.artifacts["volunteer_plan_analysis"], "分析正文")
        queue = get_or_create_queue(session_id)
        event_types = []
        while not queue.empty():
            event_types.append(queue.get_nowait()["type"])
        self.assertEqual(event_types, ["analysis", "end"])
        remove_queue(session_id)

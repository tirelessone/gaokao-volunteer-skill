import unittest
from unittest.mock import patch

from agents import Agent

import multi_agent
from skill_executor import SkillExecutionContext, get_skill_executor
from volunteer_analysis_tool import build_volunteer_analysis_tool


class FakeRunResult:
    new_items = []

    async def stream_events(self):
        if False:
            yield None


class MultiAgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_skill_mode_keeps_generation_agent_identity_and_binds_tools_dynamically(self):
        context = SkillExecutionContext("dynamic-agent-test", {})
        activated = get_skill_executor().activate_skill(
            "generate-gaokao-volunteer-plan",
            "完整考生信息",
            context,
        )
        agent = Agent(name="volunteer_plan_specialist", instructions="base", tools=[])
        returned_agent = multi_agent.activate_generation_agent_mode(
            agent,
            activated,
            build_volunteer_analysis_tool(context),
        )

        self.assertIs(returned_agent, agent)
        self.assertEqual(agent.name, "volunteer_plan_specialist")
        self.assertFalse(agent.name.startswith("skill_"))
        self.assertEqual(
            [tool.name for tool in agent.tools],
            [*activated.allowed_tools, "analyze_volunteer_plan"],
        )
        self.assertEqual(agent.model_settings.tool_choice, "required")
        self.assertFalse(agent.reset_tool_choice)

    async def test_generation_agent_starts_with_activation_only_and_consulting_uses_mcp(self):
        captured = {}

        def fake_run_streamed(agent, *args, **kwargs):
            captured["triage"] = agent
            return FakeRunResult()

        with patch.object(multi_agent.Runner, "run_streamed", side_effect=fake_run_streamed):
            chunks = []
            async for chunk in multi_agent.multi_agent_chat("普通咨询", "skill-integration-test"):
                chunks.append(chunk)
        self.assertEqual(chunks, [])
        triage = captured["triage"]
        self.assertNotIn("generate-gaokao-volunteer-plan", triage.instructions)
        generation_agent = triage.handoffs[0]
        self.assertIn("generate-gaokao-volunteer-plan", generation_agent.instructions)
        self.assertNotIn("# 高考志愿生成", generation_agent.instructions)
        self.assertEqual(
            [tool.name for tool in generation_agent.tools],
            ["activate_skill"],
        )
        self.assertEqual(
            generation_agent.tool_use_behavior,
            "run_llm_again",
        )
        self.assertNotIn("recall_and_filter_candidates", generation_agent.instructions)
        self.assertNotIn("analyze_volunteer_plan", [tool.name for tool in generation_agent.tools])
        normal_agent = triage.handoffs[1]
        self.assertEqual(normal_agent.tools, [])
        self.assertEqual(len(normal_agent.mcp_servers), 1)
        self.assertEqual(normal_agent.mcp_servers[0].name, "gaokao-consulting-mcp")


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skill_executor import SkillExecutionContext, SkillExecutor, SkillToolRegistry
from skill_runtime import SkillConfigurationError, SkillRegistry
from volunteer_planner.models import MajorCandidate, MajorGroupCandidate
from volunteer_workspace import VolunteerPlanWorkspace, workspace_key
from workspace_store import FileWorkspaceStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SkillRegistryTests(unittest.TestCase):
    def test_project_skill_is_discovered_and_loaded_progressively(self):
        registry = SkillRegistry([PROJECT_ROOT / "skills"])

        metadata = registry.list_metadata()
        self.assertEqual([item.name for item in metadata], ["generate-gaokao-volunteer-plan"])
        self.assertEqual(metadata[0].version, "2.2.0")
        self.assertIn("rank_major_groups", metadata[0].allowed_tools)
        self.assertFalse(registry.is_skill_loaded("generate-gaokao-volunteer-plan"))
        self.assertFalse(registry.is_reference_cached(
            "generate-gaokao-volunteer-plan", "references/tool-policy.md"
        ))

        definition = registry.load_skill("generate-gaokao-volunteer-plan")
        self.assertIn("高考志愿表生成 SOP", definition.instructions)
        self.assertEqual(definition.references, ("references/tool-policy.md",))
        self.assertTrue(registry.is_skill_loaded("generate-gaokao-volunteer-plan"))
        self.assertFalse(registry.is_reference_cached(
            "generate-gaokao-volunteer-plan", "references/tool-policy.md"
        ))

        reference = registry.read_reference(
            "generate-gaokao-volunteer-plan", "references/tool-policy.md"
        )
        self.assertIn("rank_major_groups", reference)
        self.assertTrue(registry.is_reference_cached(
            "generate-gaokao-volunteer-plan", "references/tool-policy.md"
        ))

    def test_unknown_allowed_tool_is_rejected(self):
        tools = SkillToolRegistry()
        with self.assertRaises(SkillConfigurationError):
            tools.build_allowed(("not_registered",), SkillExecutionContext("x", {}))


class SkillActivationTests(unittest.TestCase):
    def test_executor_activates_restricted_tools_without_creating_subagent(self):
        registry = SkillRegistry([PROJECT_ROOT / "skills"])
        definition = registry.load_skill("generate-gaokao-volunteer-plan")
        tools = SkillToolRegistry()
        for name in definition.metadata.allowed_tools:
            tools.register(name, lambda context, tool_name=name: type("Tool", (), {"name": tool_name})())
        executor = SkillExecutor(registry, tools)
        context = SkillExecutionContext("activation-test", {})
        activated = executor.activate_skill(
            "generate-gaokao-volunteer-plan",
            "完整考生信息",
            context,
        )

        self.assertEqual(activated.name, "generate-gaokao-volunteer-plan")
        self.assertEqual(
            [tool.name for tool in activated.tools],
            list(definition.metadata.allowed_tools),
        )
        self.assertIn("高考志愿表生成 SOP", activated.instructions)
        self.assertIn("同一个领域 Agent", activated.instructions)
        self.assertNotIn("专用执行 SubAgent", activated.instructions)
        self.assertNotIn("not_registered", [tool.name for tool in activated.tools])
        self.assertIs(context.artifacts["activated_skill"], activated)
        trace_events = [item["event"] for item in context.session_state["skill_trace"]]
        self.assertIn("bind_skill_tools", trace_events)
        self.assertIn("skill_activated", trace_events)

    def test_major_sort_change_restores_workspace_and_skips_recall_and_ranking(self):
        registry = SkillRegistry([PROJECT_ROOT / "skills"])
        definition = registry.load_skill("generate-gaokao-volunteer-plan")
        tools = SkillToolRegistry()
        for name in definition.metadata.allowed_tools:
            tools.register(name, lambda context, tool_name=name: type("Tool", (), {"name": tool_name})())

        with TemporaryDirectory() as directory:
            store = FileWorkspaceStore(Path(directory) / "redis.txt", ttl_seconds=1800)
            executor = SkillExecutor(registry, tools, store)
            group = MajorGroupCandidate(
                group_id="101",
                university="测试大学",
                probability=70,
                majors=[
                    MajorCandidate(name="计算机科学与技术", probability=60),
                    MajorCandidate(name="土木工程", probability=90),
                ],
            )
            workspace = VolunteerPlanWorkspace(
                query="四川考生600分物化生，本科B段，想学计算机",
                user_info={"score": 600},
                recalled_groups=[group],
                eligible_groups=[group],
                selected_groups=[group],
                final_groups=[group],
                summary={"status": "success"},
                revision=1,
                completed_stages=list(definition.metadata.allowed_tools[1:]),
            )
            store.set(
                workspace_key("resume-test", definition.metadata.name),
                workspace.to_dict(),
            )
            context = SkillExecutionContext("resume-test", {}, workspace_store=store)
            activated = executor.activate_skill(
                definition.metadata.name,
                "组内筛选不对，不要选土木工程，如果实在没有就放在最后",
                context,
            )

            self.assertEqual(
                activated.allowed_tools,
                (
                    "read_skill_reference",
                    "sort_majors_within_groups",
                    "publish_volunteer_plan",
                ),
            )
            restored = context.artifacts["volunteer_plan"]
            self.assertEqual(len(restored.selected_groups), 1)
            self.assertEqual(restored.final_groups, [])
            self.assertIsNone(restored.summary)
            self.assertIn("resume_stage: sort_majors_within_groups", activated.instructions)

    def test_missing_workspace_forces_full_pipeline(self):
        registry = SkillRegistry([PROJECT_ROOT / "skills"])
        definition = registry.load_skill("generate-gaokao-volunteer-plan")
        tools = SkillToolRegistry()
        for name in definition.metadata.allowed_tools:
            tools.register(name, lambda context, tool_name=name: type("Tool", (), {"name": tool_name})())
        with TemporaryDirectory() as directory:
            store = FileWorkspaceStore(Path(directory) / "redis.txt", ttl_seconds=1800)
            executor = SkillExecutor(registry, tools, store)
            context = SkillExecutionContext("empty-test", {}, workspace_store=store)
            activated = executor.activate_skill(
                definition.metadata.name,
                "组内筛选不对，不要土木工程",
                context,
            )
            self.assertEqual(activated.allowed_tools, definition.metadata.allowed_tools)
            self.assertIn("workspace_restored: false", activated.instructions)


if __name__ == "__main__":
    unittest.main()

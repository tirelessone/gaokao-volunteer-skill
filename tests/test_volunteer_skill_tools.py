import json
import unittest
from unittest.mock import patch

from agents.tool_context import ToolContext

from event_queue import get_or_create_queue, remove_queue
from skill_executor import SkillExecutionContext, get_skill_executor
from skill_runtime import get_skill_registry
from volunteer_planner.models import MajorCandidate, MajorGroupCandidate
from volunteer_skill_tools import _fallback_major_sort, _infer_probability_filters


RAW = """========== 专业组:101==========
院校:测试大学
专业组概率:45%
包含专业:
- 计算机科学与技术 (概率:35%) (true) (A)
- 软件工程 (概率:65%) (true) (B+)
- 土木工程 (概率:90%) (false)

========== 专业组:102==========
院校:示例大学
专业组概率:75%
包含专业:
- 计算机科学与技术 (概率:60%) (true) (A-)
- 人工智能 (概率:70%) (true) (A)
"""

RAW_WITH_HIGH_PROBABILITY_GROUP = """========== 专业组:201==========
院校:保留大学
专业组概率:93%
包含专业:
- 口腔医学 (概率:90%) (true) (A)

========== 专业组:202==========
院校:删除大学
专业组概率:95%
包含专业:
- 临床医学 (概率:96%) (true) (A)
"""


async def invoke(tool, payload):
    raw = json.dumps(payload, ensure_ascii=False)
    tool_context = ToolContext(
        None,
        tool_name=tool.name,
        tool_call_id=f"test-{tool.name}",
        tool_arguments=raw,
    )
    return await tool.on_invoke_tool(tool_context, raw)


class VolunteerSkillToolTests(unittest.IsolatedAsyncioTestCase):
    def test_probability_filter_language_uses_correct_inclusive_boundaries(self):
        self.assertEqual(
            _infer_probability_filters("去掉录取概率94%以上的学校"),
            {"group_probability_max": 93},
        )
        self.assertEqual(
            _infer_probability_filters("删除录取概率高于94%的专业组"),
            {"group_probability_max": 94},
        )
        self.assertEqual(
            _infer_probability_filters("去掉录取概率低于40%的全部专业"),
            {"major_probability_min": 40},
        )

    async def test_recall_tool_enforces_inferred_group_probability_limit(self):
        executor = get_skill_executor()
        definition = get_skill_registry().load_skill("generate-gaokao-volunteer-plan")
        context = SkillExecutionContext("probability-filter-test", {})
        context.current_skill = definition.metadata.name
        context.loaded_references.add((definition.metadata.name, "references/tool-policy.md"))
        context.artifacts["skill_query"] = "完整考生信息，去掉录取概率94%以上的学校"
        tool = {
            item.name: item
            for item in executor.tools.build_allowed(definition.metadata.allowed_tools, context)
        }["recall_and_filter_candidates"]
        user_info = {
            "province": "四川省", "score": 661, "firstChoice": "物",
            "reselection": ["不限", "化", "生"], "major": ["医学"],
            "not_major": [], "university": [], "city": [], "not_city": [],
            "dimension": 0, "foreign": 0, "recruit": 0,
            "matriculation": 1, "target": 1,
        }

        with patch("volunteer_skill_tools.format_user_info_with_llm", return_value=user_info), \
             patch(
                 "volunteer_skill_tools.recall_candidates",
                 return_value=RAW_WITH_HIGH_PROBABILITY_GROUP,
             ):
            result = await invoke(tool, {"query": context.artifacts["skill_query"]})

        workspace = context.artifacts["volunteer_plan"]
        self.assertEqual(result["eligible_groups"], 1)
        self.assertEqual(result["filters_applied"], ["group_probability<=93"])
        self.assertEqual([group.university for group in workspace.eligible_groups], ["保留大学"])

    def test_deprioritized_major_is_excluded_when_alternatives_exist(self):
        group = MajorGroupCandidate(
            group_id="civil-test",
            majors=[
                MajorCandidate(name="土木工程", probability=99, intentional=True),
                *[
                    MajorCandidate(name=f"计算机方向{i}", probability=80 - i, intentional=True)
                    for i in range(6)
                ],
            ],
        )
        result = _fallback_major_sort(
            group,
            {"dimension": 0, "major": ["计算机"]},
            6,
            "不要选土木工程，如果实在没有选的，就把土木工程放在最后",
        )
        self.assertNotIn("土木工程", [major.name for major in result.majors])

        short_group = group.model_copy(deep=True)
        short_group.majors = short_group.majors[:2]
        short_result = _fallback_major_sort(
            short_group,
            {"dimension": 0, "major": ["计算机"]},
            6,
            "不要选土木工程，如果实在没有选的，就把土木工程放在最后",
        )
        self.assertEqual(short_result.majors[-1].name, "土木工程")

    async def test_atomic_tools_share_private_workspace_without_returning_raw_candidates(self):
        executor = get_skill_executor()
        definition = get_skill_registry().load_skill("generate-gaokao-volunteer-plan")
        state = {}
        context = SkillExecutionContext("atomic-tools-test", state)
        context.current_skill = definition.metadata.name
        context.activated_skills.add(definition.metadata.name)
        context.loaded_references.add((definition.metadata.name, "references/tool-policy.md"))
        tools = {
            tool.name: tool
            for tool in executor.tools.build_allowed(definition.metadata.allowed_tools, context)
        }
        rank_params = tools["rank_major_groups"].params_json_schema["properties"]
        major_params = tools["sort_majors_within_groups"].params_json_schema["properties"]
        publish_params = tools["publish_volunteer_plan"].params_json_schema["properties"]
        self.assertNotIn("groups", rank_params)
        self.assertNotIn("user_info", rank_params)
        self.assertNotIn("groups", major_params)
        self.assertEqual(publish_params, {})
        user_info = {
            "province": "四川省", "score": 600, "firstChoice": "物",
            "reselection": ["不限", "化", "生"], "major": ["计算机"],
            "not_major": [], "university": [], "city": [], "not_city": [],
            "dimension": 0, "foreign": 0, "recruit": 0,
            "matriculation": 1, "target": 1,
        }

        with patch("volunteer_skill_tools.format_user_info_with_llm", return_value=user_info), \
             patch("volunteer_skill_tools.recall_candidates", return_value=RAW):
            recalled = await invoke(tools["recall_and_filter_candidates"], {
                "query": "完整考生信息", "major_probability_min": 40
            })
        self.assertNotIn("测试大学", str(recalled))
        self.assertNotIn(RAW, str(recalled))
        self.assertEqual(recalled["eligible_groups"], 2)

        ranked = await invoke(tools["rank_major_groups"], {
            "total_groups": 2, "chong_count": 1, "wen_count": 1, "bao_count": 0,
            "university_level_weight": 3.0,
        })
        self.assertEqual(ranked["selected_groups"], 2)
        self.assertEqual(ranked["weights"]["university_level"], 3.0)

        def fake_sort(groups, user_info, custom, maximum):
            return [group.model_copy(deep=True) for group in groups], 2, 0

        with patch("volunteer_skill_tools._sort_groups_with_prompt", side_effect=fake_sort):
            sorted_result = await invoke(tools["sort_majors_within_groups"], {
                "custom_instructions": "软件工程优先", "max_majors_per_group": 6
            })
        self.assertEqual(sorted_result["custom_instructions"], "软件工程优先")

        get_or_create_queue(context.session_id)
        published = await invoke(tools["publish_volunteer_plan"], {})
        self.assertEqual(published["status"], "success")
        self.assertTrue(published["major_sort_customized"])
        event_types = []
        queue = get_or_create_queue(context.session_id)
        while not queue.empty():
            event_types.append(queue.get_nowait()["type"])
        self.assertEqual(event_types, ["final_result"])
        remove_queue(context.session_id)


if __name__ == "__main__":
    unittest.main()

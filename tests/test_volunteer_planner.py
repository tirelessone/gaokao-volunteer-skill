import unittest

from volunteer_planner.candidates import parse_candidate_groups
from volunteer_planner.engine import build_plan
from volunteer_planner.models import PlanPolicy
from volunteer_planner.policy import parse_plan_policy
from volunteer_planner.service import generate_plan


def make_group(group_id, university, group_probability, majors):
    major_lines = []
    for name, probability, intentional in majors:
        major_lines.append(
            f"- {name} (概率:{probability}%) ({str(intentional).lower()})"
        )
    return "\n".join([
        f"========== 专业组:{group_id}==========",
        f"院校:{university}",
        f"专业组概率:{group_probability}%",
        "包含专业:",
        *major_lines,
    ])


RAW_CANDIDATES = "\n\n".join([
    make_group("101", "冲刺大学A", 20, [("土木工程", 30, False), ("计算机科学与技术", 45, True)]),
    make_group("102", "冲刺大学B", 40, [("软件工程", -1, True), ("护理学", 70, False)]),
    make_group("103", "冲刺大学C", 50, [("人工智能", 60, True)]),
    make_group("201", "稳妥大学A", 55, [("计算机类", 55, True)]),
    make_group("202", "稳妥大学B", 65, [("电子信息工程", 65, False)]),
    make_group("203", "稳妥大学C", 75, [("软件工程", 75, True)]),
    make_group("301", "保底大学A", 85, [("计算机科学与技术", 85, True)]),
    make_group("302", "保底大学B", 90, [("信息安全", 90, False)]),
    make_group("303", "保底大学C", 95, [("数据科学与大数据技术", 95, True)]),
])


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"code": 200, "data": RAW_CANDIDATES}


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


class VolunteerPlannerTests(unittest.TestCase):
    def test_common_agent_policy_aliases_are_normalized(self):
        policy = parse_plan_policy({
            "risk_distribution": {
                "aggressive": 20,
                "moderate": 13,
                "conservative": 12,
            },
            "score": 600,
            "subjects": ["物理", "化学", "生物"],
            "province": "四川省",
            "batch": "本科批次B段",
            "excluded_conditions": {"sino_foreign_cooperation": True},
            "admission_exclusion": {"sino_foreign": True},
            "probability_filter": {"major_min": 40},
            "major_sort": ["major_preference desc", "admission_probability asc"],
        })

        self.assertEqual(policy.risk_distribution.as_dict(), {"chong": 20, "wen": 13, "bao": 12})
        self.assertEqual(policy.probability_filter.major_min, 40)
        self.assertEqual(policy.major_sort[0].field, "major_preference")
        self.assertEqual(policy.major_sort[1].direction, "asc")

        compact_policy = parse_plan_policy({
            "major_sort": {"major_preference": "desc"},
        })
        self.assertEqual(compact_policy.major_sort[0].model_dump(), {
            "field": "major_preference",
            "direction": "desc",
        })

    def test_parser_repairs_negative_major_probability(self):
        groups = parse_candidate_groups(RAW_CANDIDATES)
        group = next(item for item in groups if item.group_id == "102")
        self.assertEqual(group.majors[0].probability, 40)

    def test_policy_filters_and_allocates_without_model_access(self):
        groups = parse_candidate_groups(RAW_CANDIDATES)
        policy = PlanPolicy.model_validate({
            "total_groups": 6,
            "risk_distribution": {"chong": 3, "wen": 2, "bao": 1},
            "probability_filter": {"major_min": 40},
            "preferred_major_keywords": ["计算机", "软件"],
            "excluded_major_keywords": ["护理", "土木"],
        })

        result = build_plan(groups, policy, raw_character_count=len(RAW_CANDIDATES))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.statistics.bucket_selected, {"chong": 3, "wen": 2, "bao": 1})
        self.assertEqual(result.statistics.selected_groups, 6)
        self.assertNotIn("护理学", result.plan_text)
        self.assertNotIn("土木工程", result.plan_text)
        self.assertNotIn("(true)", result.plan_text)
        summary = result.model_summary()
        self.assertNotIn("plan_text", summary)
        self.assertNotIn("groups", summary)

    def test_service_keeps_recall_payload_inside_code_layer(self):
        session = FakeSession()
        user_info = {
            "score": 600,
            "firstChoice": "物",
            "reselection": ["不限", "化", "生"],
            "major": ["计算机类"],
            "dimension": 0,
            "province": "四川省",
        }
        policy = {
            "total_groups": 3,
            "risk_distribution": {"chong": 1, "wen": 1, "bao": 1},
        }

        result = generate_plan(
            user_info,
            "unit-test",
            policy,
            emit_events=False,
            request_session=session,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.statistics.recalled_groups, 9)
        self.assertEqual(result.statistics.selected_groups, 3)
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn(RAW_CANDIDATES, str(result.model_summary()))


if __name__ == "__main__":
    unittest.main()

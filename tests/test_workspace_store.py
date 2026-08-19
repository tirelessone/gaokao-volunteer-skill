from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from workspace_store import FileWorkspaceStore
from volunteer_planner.models import MajorGroupCandidate
from volunteer_workspace import (
    MAJOR_SORT_STAGE,
    RANK_STAGE,
    RECALL_STAGE,
    VolunteerPlanWorkspace,
    infer_resume_stage,
)


class WorkspaceStoreTests(unittest.TestCase):
    def test_resume_stage_uses_earliest_invalidated_dependency(self):
        group = MajorGroupCandidate(group_id="101")
        workspace = VolunteerPlanWorkspace(
            eligible_groups=[group],
            selected_groups=[group],
            final_groups=[group],
        )
        self.assertEqual(
            infer_resume_stage("组内不要选土木工程，实在没得选就放最后", workspace)[0],
            MAJOR_SORT_STAGE,
        )
        self.assertEqual(
            infer_resume_stage("把冲增加到20个，院校层次最重要", workspace)[0],
            RANK_STAGE,
        )
        self.assertEqual(
            infer_resume_stage("把分数改成610分，同时组内不要土木工程", workspace)[0],
            RECALL_STAGE,
        )
        self.assertEqual(
            infer_resume_stage(
                "录取概率94%以上的学校去掉，每组有口腔时把口腔放第一位",
                workspace,
            )[0],
            RECALL_STAGE,
        )
        interrupted = VolunteerPlanWorkspace(
            eligible_groups=[group],
            completed_stages=[RECALL_STAGE],
        )
        self.assertEqual(infer_resume_stage("继续刚才的生成", interrupted)[0], RANK_STAGE)

    def test_value_expires_after_thirty_minutes(self):
        clock = [1000.0]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "redis.txt"
            store = FileWorkspaceStore(
                path,
                ttl_seconds=1800,
                now=lambda: clock[0],
            )
            store.set("session:skill", {"stage": "rank"})
            self.assertEqual(store.get("session:skill"), {"stage": "rank"})
            self.assertEqual(store.metadata("session:skill")["ttl_seconds"], 1800)

            clock[0] += 1801
            self.assertIsNone(store.get("session:skill"))
            self.assertFalse(store.metadata("session:skill")["available"])


if __name__ == "__main__":
    unittest.main()

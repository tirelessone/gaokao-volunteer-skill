"""Exercise the live Agent handoff, project Skill instructions, and planner tool."""

from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent import multi_agent_chat, session_data


MESSAGE = (
    "请生成志愿表。我是四川考生，600分，物化生，本科批次B段，想学计算机类，"
    "不考虑中外合作。去掉专业录取概率40%以下的专业，冲20个、稳13个、保12个。"
)


async def run(message: str = MESSAGE) -> dict[str, object]:
    chunks: list[str] = []
    session_id = f"live-agent-skill-test-{uuid.uuid4().hex[:8]}"
    async for chunk in multi_agent_chat(message, session_id):
        chunks.append(chunk)
    response = "".join(chunks)
    return {
        "session_id": session_id,
        "response_characters": len(response),
        "response": response,
        "skill_trace": session_data.get(session_id, {}).get("skill_trace", []),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default=MESSAGE)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.message)), ensure_ascii=False, indent=2))

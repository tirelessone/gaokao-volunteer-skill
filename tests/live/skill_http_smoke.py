"""Verify the live HTTP multi-Agent, dynamic Skill mode, SSE, and final plan pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import uuid

import requests


DEFAULT_MESSAGE = (
    "帮我生成志愿表。我是四川考生，600分，物化生，本科批次B段，想学计算机类，"
    "不考虑中外合作，去掉专业录取概率40%以下的专业，冲20个、稳13个、保12个。"
    "院校层次是最重要的考虑因素，组内同等相关度时录取概率高的专业优先。"
)


def run(base_url: str, message: str) -> tuple[dict, str]:
    session_id = f"live-skill-http-{uuid.uuid4().hex[:8]}"
    plan_events: list[dict] = []
    stream_ready = threading.Event()
    stream_errors: list[str] = []

    def consume_plan_stream() -> None:
        try:
            with requests.get(
                f"{base_url}/api/stream",
                params={"session_id": session_id},
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=(10, 300),
            ) as response:
                response.raise_for_status()
                response.encoding = "utf-8"
                stream_ready.set()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    plan_events.append(event)
                    if event.get("type") == "end":
                        return
        except Exception as exc:
            stream_errors.append(str(exc))
            stream_ready.set()

    thread = threading.Thread(target=consume_plan_stream, daemon=True)
    thread.start()
    if not stream_ready.wait(timeout=10):
        raise RuntimeError("SSE plan stream did not become ready")
    if stream_errors:
        raise RuntimeError(stream_errors[0])

    response_chunks: list[str] = []
    with requests.post(
        f"{base_url}/api/chat",
        json={"session_id": session_id, "message": message},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=(10, 300),
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "content":
                response_chunks.append(event.get("content", ""))
            if event.get("type") in {"end", "error"}:
                break

    thread.join(timeout=30)
    if stream_errors:
        raise RuntimeError(stream_errors[0])
    final_event = next(
        (event for event in plan_events if event.get("type") == "final_result"),
        None,
    )
    if final_event is None:
        raise RuntimeError(
            f"Missing final_result; received events: {[event.get('type') for event in plan_events]}"
        )
    event_types = [event.get("type") for event in plan_events]
    if event_types[-3:] != ["final_result", "analysis", "end"]:
        raise RuntimeError(f"Unexpected plan event order: {event_types}")
    final_data = final_event.get("data", {})
    metadata = final_data.get("metadata", {})
    summary = {
        "session_id": session_id,
        "agent_response": "".join(response_chunks),
        "event_types": event_types,
        "statistics": metadata.get("statistics", {}),
        "group_ranking": metadata.get("group_ranking", {}),
        "major_sorting": metadata.get("major_sorting", {}),
        "plan_characters": len(final_data.get("content", "")),
    }
    return summary, final_data.get("content", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary, plan = run(args.base_url.rstrip("/"), args.message)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(plan, encoding="utf-8")
        print(f"\n完整志愿表已保存到: {args.output.resolve()}")
    if args.print_plan:
        print("\n========== 完整志愿表 ==========")
        print(plan)


if __name__ == "__main__":
    main()

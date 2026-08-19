"""Run two HTTP turns and verify that the second resumes from major sorting."""

from __future__ import annotations

import json
import threading
import uuid

import requests


FIRST_MESSAGE = (
    "帮我生成志愿表。我是四川考生，600分，物化生，本科批次B段，想学计算机类，"
    "不考虑中外合作，去掉专业录取概率40%以下的专业，冲20个、稳13个、保12个。"
    "院校层次是最重要的考虑因素。"
)
SECOND_MESSAGE = (
    "我觉得组内筛选不对，我不要选土木工程的专业，如果实在没有选的，"
    "就把土木工程放在最后。"
)


def run_turn(base_url: str, session_id: str, message: str) -> tuple[list[dict], str]:
    events: list[dict] = []
    ready = threading.Event()
    errors: list[str] = []

    def consume_events() -> None:
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
                ready.set()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    if event.get("type") == "end":
                        return
        except Exception as exc:
            errors.append(str(exc))
            ready.set()

    thread = threading.Thread(target=consume_events, daemon=True)
    thread.start()
    if not ready.wait(10):
        raise RuntimeError("SSE stream did not become ready")

    chunks: list[str] = []
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
                chunks.append(event.get("content", ""))
            if event.get("type") in {"end", "error"}:
                break
    thread.join(30)
    if errors:
        raise RuntimeError(errors[0])
    return events, "".join(chunks)


def final_execution(events: list[dict]) -> dict:
    final = next((item for item in events if item.get("type") == "final_result"), None)
    if final is None:
        raise RuntimeError(f"Missing final_result: {[item.get('type') for item in events]}")
    return final.get("data", {}).get("metadata", {}).get("execution", {})


def main() -> None:
    base_url = "http://127.0.0.1:5001"
    session_id = f"live-resume-{uuid.uuid4().hex[:8]}"
    first_events, first_response = run_turn(base_url, session_id, FIRST_MESSAGE)
    second_events, second_response = run_turn(base_url, session_id, SECOND_MESSAGE)
    first = final_execution(first_events)
    second = final_execution(second_events)

    if first.get("resume_stage") != "recall_and_filter_candidates":
        raise RuntimeError(f"First turn did not start from recall: {first}")
    if second.get("resume_stage") != "sort_majors_within_groups":
        raise RuntimeError(f"Second turn did not resume from major sorting: {second}")
    forbidden = {"recall_and_filter_candidates", "rank_major_groups"}
    if forbidden.intersection(second.get("bound_skill_tools", [])):
        raise RuntimeError(f"Second turn exposed repeated tools: {second}")

    print(json.dumps({
        "session_id": session_id,
        "first_response": first_response,
        "second_response": second_response,
        "first_execution": first,
        "second_execution": second,
        "first_event_types": [item.get("type") for item in first_events],
        "second_event_types": [item.get("type") for item in second_events],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

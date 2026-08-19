"""Exercise the live Flask, NextGoo recall API, planner, and SSE pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
from typing import Any

import requests


def run(base_url: str) -> tuple[dict[str, Any], str]:
    session_id = "live-smoke-test"
    events: list[dict[str, Any]] = []
    stream_ready = threading.Event()
    stream_error: list[str] = []

    def consume_events() -> None:
        try:
            with requests.get(
                f"{base_url}/api/stream",
                params={"session_id": session_id},
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()
                response.encoding = "utf-8"
                stream_ready.set()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    if event.get("type") == "end":
                        return
        except Exception as exc:
            stream_error.append(str(exc))
            stream_ready.set()

    stream_thread = threading.Thread(target=consume_events, daemon=True)
    stream_thread.start()
    if not stream_ready.wait(timeout=10):
        raise RuntimeError("SSE stream did not become ready")
    if stream_error:
        raise RuntimeError(stream_error[0])

    payload = {
        "session_id": session_id,
        "major_list": ["计算机类"],
        "firstChoice": "物",
        "score": 600,
        "recruit": 0,
        "province": "四川省",
        "foreign": 0,
        "reselection": ["不限", "化", "生"],
        "matriculation": 1,
        "target": 1,
        "healthCheckupLimit": "000",
        "dimension": 0,
        "plan_policy": {
            "total_groups": 45,
            "risk_distribution": {"chong": 15, "wen": 15, "bao": 15},
            "probability_filter": {"major_min": 40},
            "preferred_major_keywords": ["计算机", "软件工程", "人工智能"],
        },
    }
    response = requests.post(
        f"{base_url}/api/generate_plan_from_form",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    stream_thread.join(timeout=20)
    if stream_thread.is_alive():
        raise RuntimeError("SSE stream did not finish")
    if stream_error:
        raise RuntimeError(stream_error[0])

    final_event = next((event for event in events if event.get("type") == "final_result"), None)
    if final_event is None:
        raise RuntimeError(f"Missing final_result event; received: {[event.get('type') for event in events]}")
    final_data = final_event.get("data", {})
    statistics = body.get("metadata", {}).get("statistics", {})
    summary = {
        "http_status": response.status_code,
        "api_status": body.get("status"),
        "event_types": [event.get("type") for event in events],
        "final_plan_characters": len(final_data.get("content", "")),
        "recalled_groups": statistics.get("recalled_groups"),
        "eligible_groups": statistics.get("eligible_groups"),
        "selected_groups": statistics.get("selected_groups"),
        "selected_majors": statistics.get("selected_majors"),
        "bucket_selected": statistics.get("bucket_selected"),
        "filters_applied": statistics.get("filters_applied"),
    }
    return summary, final_data.get("content", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the complete volunteer plan after the smoke-test summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the complete volunteer plan to a UTF-8 text file.",
    )
    args = parser.parse_args()
    summary, plan_text = run(args.base_url.rstrip("/"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(plan_text, encoding="utf-8")
        print(f"\n完整志愿表已保存到: {args.output.resolve()}")

    if args.print_plan:
        print("\n========== 完整志愿表 ==========")
        print(plan_text)


if __name__ == "__main__":
    main()

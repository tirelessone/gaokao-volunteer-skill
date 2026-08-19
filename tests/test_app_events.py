import json
import unittest

from app import event_generator
from event_queue import get_or_create_queue, remove_queue


class EventStreamTests(unittest.TestCase):
    def test_final_result_is_not_blocked_when_thinking_events_are_absent(self):
        session_id = "event-stream-test"
        remove_queue(session_id)
        event_queue = get_or_create_queue(session_id)
        event_queue.put({"type": "final_result", "data": {"content": "plan", "order": 5}})
        event_queue.put({"type": "end", "data": {"content": "done"}})

        events = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in event_generator(session_id)
            if chunk.startswith("data: ")
        ]

        self.assertEqual([event["type"] for event in events], ["final_result", "end"])


if __name__ == "__main__":
    unittest.main()

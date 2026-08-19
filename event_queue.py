from queue import Queue

# 全局事件队列：session_id -> Queue
event_queues = {}
_pending_counts = {}


# === 队列管理 ===
def get_or_create_queue(session_id):
    if session_id not in event_queues:
        event_queues[session_id] = Queue()
    return event_queues[session_id]


def remove_queue(session_id):
    if session_id in event_queues:
        del event_queues[session_id]
    # 清理并行任务计数
    if session_id in _pending_counts:
        del _pending_counts[session_id]


# === 发送事件 ===
def send_event(session_id, event_type: str, data: dict):
    """
    向指定 session 的队列发送事件
    event_type: "thinking" / "final_result" / "error" / "end"
    data: 事件数据（字典）
    """
    q = event_queues.get(session_id)
    if q:
        q.put({
            "type": event_type,
            "data": data
        })


# === 常用包装函数 ===
def send_thinking(session_id, content: str, order: int):
    send_event(session_id, "thinking", {
        "order": order,
        "content": content
    })


def send_final_result(session_id, final_result: str, score: int,subject: str, reselection: list,
                      recruit: int, city_list, not_city_list,university_list, major_list,not_major_list,matriculation,
                      target,dimension, foreign=None, province="四川省", metadata=None):
    payload = {
        "content": final_result,
        "score": score,
        "firstChoice": subject,
        "reselection": reselection,
        "recruit": recruit,
        "foreign": foreign,
        "province": province,
        "citySecondDesc": city_list,
        "notCitySecondDesc": not_city_list,
        "universitySecondDesc": university_list,
        "majorSecondDesc": major_list,
        "notMajorSecondDesc": not_major_list,
        "matriculation": matriculation,
        "target": target,
        "dimension": dimension,
        "order": 5
    }
    if metadata is not None:
        payload["metadata"] = metadata
    send_event(session_id, "final_result", payload)

def send_analysis(session_id, analysis: str):
    send_event(session_id, "analysis", {
        "content": analysis,
        "order": 6
        })

def send_error(session_id, error: str):
    send_event(session_id, "error", {"content": error})


def send_end(session_id):
    send_event(session_id, "end", {"content": "流程结束"})


# === 并行阶段完成度跟踪 ===
def register_pending(session_id, n: int):
    """为某个 session 登记 n 个并行待完成任务。"""
    if n <= 0:
        return
    _pending_counts[session_id] = _pending_counts.get(session_id, 0) + n


def notify_done(session_id):
    """并行任务完成一次，计数归零后自动发送 end。"""
    if session_id not in _pending_counts:
        return
    _pending_counts[session_id] -= 1
    if _pending_counts[session_id] <= 0:
        del _pending_counts[session_id]
        send_end(session_id)

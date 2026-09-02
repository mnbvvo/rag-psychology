"""准入调度指标与结构化日志（总稿 §4.3.10）。

不引入外部依赖：计数器 + 最近 N 次耗时环形缓冲；日志只记录
request ID 截断、用户哈希、状态、等待/运行时长与终态原因，
不记录 JWT、密钥、问题/回答全文。
"""
import hashlib
import time
from collections import Counter, deque
from typing import Deque, Dict


def _user_hash(user_id: str, length: int = 8) -> str:
    return hashlib.sha1((user_id or "").encode("utf-8")).hexdigest()[:length]


class AdmissionMetrics:
    """进程内指标：拒绝/终态计数 + 等待/运行耗时环形缓冲（最近 200 次）。"""

    def __init__(self, ring_capacity: int = 200):
        self.ring_capacity = ring_capacity
        self.submit_total = 0
        self.started_total = 0
        self.queued_total = 0
        self.rejected_total: Counter = Counter()          # reason -> count
        self.terminal_total: Counter = Counter()          # TerminalReason -> count
        self.cancel_total = 0
        self.queue_wait_ms: Deque[float] = deque(maxlen=ring_capacity)
        self.run_ms: Deque[float] = deque(maxlen=ring_capacity)

    def record_submit(self, code_value: str) -> None:
        self.submit_total += 1
        if code_value == "started":
            self.started_total += 1
        elif code_value == "queued":
            self.queued_total += 1
        else:
            self.rejected_total[code_value] += 1

    def record_run_start(self, wait_ms: float) -> None:
        self.queue_wait_ms.append(wait_ms)

    def record_terminal(
        self,
        request_id: str,
        user_id: str,
        reason: str,
        wait_ms: float,
        run_ms: float,
        backend: str,
    ) -> None:
        self.terminal_total[reason] += 1
        if reason == "cancelled":
            self.cancel_total += 1
        if run_ms >= 0:
            self.run_ms.append(run_ms)
        # 结构化日志：不含敏感正文/令牌/密钥
        run_txt = f"{run_ms:.0f}" if run_ms >= 0 else "-"
        print(
            "[admission] backend={} req={} user={} reason={} wait_ms={:.0f} run_ms={}".format(
                backend,
                request_id[:10],
                _user_hash(user_id),
                reason,
                wait_ms,
                run_txt,
            ),
            flush=True,
        )

    def summary(self) -> Dict:
        def _avg(ring: Deque[float]) -> float:
            return (sum(ring) / len(ring)) if ring else 0.0

        return {
            "submit_total": self.submit_total,
            "started_total": self.started_total,
            "queued_total": self.queued_total,
            "rejected_total": dict(self.rejected_total),
            "terminal_total": dict(self.terminal_total),
            "cancel_total": self.cancel_total,
            "avg_queue_wait_ms": round(_avg(self.queue_wait_ms), 1),
            "avg_run_ms": round(_avg(self.run_ms), 1),
        }

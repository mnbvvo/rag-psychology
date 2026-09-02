"""统一准入服务入口：路由层只依赖本模块的 admission 单例。

对应总稿 §4.3.1。后端按 settings.AI_ADMISSION_BACKEND 构建：
- memory：Phase 1 单进程（当前实现）；
- redis：Phase 2 多实例（尚未实现，配置即拒绝启动，防止静默错配）。
"""
import time

from config.settings import settings

from modules.concurrency.base import AdmissionBackend
from modules.concurrency.memory_backend import MemoryAdmissionBackend
from modules.concurrency.metrics import AdmissionMetrics
from modules.concurrency.models import (
    CancelResult,
    CapacitySnapshot,
    QueueUpdate,
    ReleaseResult,
    SubmitResult,
    Ticket,
    WaitResult,
)


class AdmissionService:
    """对路由层暴露的门面：提交 → 等待 → 心跳 → 释放 → 取消 → 快照 + 指标。"""

    def __init__(self, backend: AdmissionBackend, metrics: AdmissionMetrics):
        self.backend = backend
        self.metrics = metrics
        self.backend_name = backend.backend_name
        # request_id -> (wait_ms, run_start_monotonic)：供 release 汇总耗时
        self._run_ctx: dict = {}

    # ---------- 生命周期 ----------
    async def submit(self, user_id: str, request_id: str) -> SubmitResult:
        res = await self.backend.submit(user_id, request_id)
        self.metrics.record_submit(res.code.value)
        return res

    async def wait_until_running(self, request_id: str) -> WaitResult:
        res = await self.backend.wait_until_running(request_id)
        if res.code.value == "started":
            self.metrics.record_run_start(res.wait_ms)
        return res

    def note_started(self, ticket: Ticket, wait_ms: float) -> None:
        """RUNNING 开始时记录等待耗时，供 release 计算运行耗时。"""
        self._run_ctx[ticket.request_id] = (wait_ms, time.monotonic())

    async def release(
        self, ticket: Ticket, terminal: str = "completed"
    ) -> ReleaseResult:
        ctx = self._run_ctx.pop(ticket.request_id, None)
        res = await self.backend.release(ticket.request_id, ticket.owner_token, terminal)
        if res.code.value == "released":
            wait_ms, t0 = ctx if ctx else (0.0, None)
            run_ms = ((time.monotonic() - t0) * 1000) if t0 else -1.0
            self.metrics.record_terminal(
                ticket.request_id, ticket.user_id, terminal,
                wait_ms=wait_ms, run_ms=run_ms, backend=self.backend_name,
            )
        else:
            # 非 RELEASED 时也要清上下文
            self._run_ctx.pop(ticket.request_id, None)
        return res

    def record_dropped(self, ticket: Ticket, wait_ms: float, reason: str) -> None:
        """未进入 RUNNING 即终态（queue_timeout / cancelled）：补记指标。"""
        self._run_ctx.pop(ticket.request_id, None)
        self.metrics.record_terminal(
            ticket.request_id, ticket.user_id, reason,
            wait_ms=wait_ms, run_ms=-1.0, backend=self.backend_name,
        )

    async def cancel(self, user_id: str, request_id: str) -> CancelResult:
        return await self.backend.cancel(user_id, request_id)

    async def heartbeat(self, ticket: Ticket) -> None:
        await self.backend.heartbeat(ticket.request_id, ticket.owner_token)

    # ---------- 查询 ----------
    async def snapshot(self) -> CapacitySnapshot:
        return await self.backend.snapshot()

    def queue_update(self, request_id: str) -> QueueUpdate:
        return self.backend.queue_update(request_id)

    def is_cancelling(self, request_id: str) -> bool:
        return self.backend.is_cancelling(request_id)

    def resolve_user_of(self, request_id: str):
        return self.backend.resolve_user_of(request_id)

    def metrics_summary(self) -> dict:
        return self.metrics.summary()


def _build_backend() -> AdmissionBackend:
    backend_name = (settings.AI_ADMISSION_BACKEND or "memory").strip().lower()
    if backend_name == "redis":
        raise RuntimeError(
            "AI_ADMISSION_BACKEND=redis 属于总稿 Phase 2（多实例共享调度），尚未实现。"
            "Phase 1 请使用 memory（单 worker）。"
        )
    if backend_name != "memory":
        raise RuntimeError(f"未知的 AI_ADMISSION_BACKEND: {backend_name!r}（可选 memory / redis）")
    return MemoryAdmissionBackend(
        max_active=settings.AI_MAX_ACTIVE_REQUESTS,
        max_queue=settings.AI_MAX_QUEUED_REQUESTS,
        queue_wait_timeout_seconds=settings.AI_QUEUE_WAIT_TIMEOUT_SECONDS,
        one_inflight_per_user=settings.AI_ONE_INFLIGHT_PER_USER,
    )


# 进程级单例：同步与 SSE 共用同一个 service 实例和后端
admission = AdmissionService(_build_backend(), AdmissionMetrics())

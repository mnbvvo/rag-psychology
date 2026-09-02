"""Phase 1 准入后端：进程内 asyncio 实现（单 worker 专用）。

对应总稿 Phase 1（memory 模式）。约束：
- 只运行在单个事件循环上（uvicorn 单 worker），状态机切换点之间不跨 await，
  因此无需加锁，靠「无 await 的检查-置位」保证原子性；
- 只有工作协程真实退出（release）后才释放活跃槽位 → 防超卖；
- release/cancel 幂等；owner_token 防旧请求释放新租约（AC-12）；
- 多 worker 部署由 settings.validate() 拒绝（见 WEB_CONCURRENCY 守卫）。
"""
import asyncio
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from modules.concurrency.base import AdmissionBackend
from modules.concurrency.models import (
    AI_REQUEST_CANCELLED,
    CancelCode,
    CancelResult,
    CapacitySnapshot,
    QueueUpdate,
    ReleaseCode,
    ReleaseResult,
    SubmitCode,
    SubmitResult,
    TerminalReason,
    Ticket,
    WaitCode,
    WaitResult,
)

# 最近终态缓存容量（cancel 幂等 / ALREADY_FINISHED 判定用）
_RECENT_TERMINAL_CAP = 1000


@dataclass
class _Entry:
    """一个待准入请求的内部状态（queued / running / cancelling）。"""
    request_id: str
    user_id: str
    owner_token: str
    state: str = "queued"            # queued | running | cancelling
    terminal: Optional[str] = None   # 终态 TerminalReason（置位后不可变）
    cancel_requested: bool = False   # 运行中收到取消 → True
    wake: asyncio.Event = field(default_factory=asyncio.Event)  # 状态变化唤醒等待者
    queued_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class MemoryAdmissionBackend(AdmissionBackend):
    backend_name = "memory"

    def __init__(
        self,
        max_active: int = 20,
        max_queue: int = 40,
        queue_wait_timeout_seconds: float = 30.0,
        one_inflight_per_user: bool = True,
    ):
        self.max_active = max(1, max_active)
        self.max_queue = max(0, max_queue)
        self.queue_wait_timeout_seconds = max(0.0, queue_wait_timeout_seconds)
        self.one_inflight_per_user = one_inflight_per_user

        self._active: Dict[str, _Entry] = {}
        self._queue: Deque[_Entry] = deque()
        self._entries: Dict[str, _Entry] = {}          # queued + running，按 request_id
        self._user_req: Dict[str, str] = {}            # user_id -> request_id（单 in-flight 去重）
        self._recent_terminals: "OrderedDict[str, tuple[str, str]]" = OrderedDict()  # rid -> (user, reason)

    # ---------------- submit ----------------
    async def submit(self, user_id: str, request_id: str) -> SubmitResult:
        if self.one_inflight_per_user and user_id in self._user_req:
            return SubmitResult(
                SubmitCode.REJECTED_DUPLICATE,
                message="同一用户已有未完成请求（AI_REQUEST_IN_PROGRESS）",
            )
        owner_token = secrets.token_hex(16)
        # 1) 有槽位 → 直接运行
        if len(self._active) < self.max_active:
            entry = _Entry(request_id, user_id, owner_token, state="running")
            entry.started_at = time.monotonic()
            self._active[request_id] = entry
            self._entries[request_id] = entry
            if self.one_inflight_per_user:
                self._user_req[user_id] = request_id
            return SubmitResult(
                SubmitCode.STARTED,
                Ticket(
                    request_id=request_id, user_id=user_id, owner_token=owner_token,
                    code=SubmitCode.STARTED, queued=False, position=None,
                    queued_total=None, active=len(self._active),
                ),
            )
        # 2) 无槽位 → 入队（有界）
        if self.max_queue <= 0 or len(self._queue) >= self.max_queue:
            return SubmitResult(
                SubmitCode.REJECTED_FULL,
                message="当前排队已满，请稍后重试（AI_QUEUE_FULL）",
            )
        entry = _Entry(request_id, user_id, owner_token, state="queued")
        self._queue.append(entry)
        self._entries[request_id] = entry
        if self.one_inflight_per_user:
            self._user_req[user_id] = request_id
        return SubmitResult(
            SubmitCode.QUEUED,
            Ticket(
                request_id=request_id, user_id=user_id, owner_token=owner_token,
                code=SubmitCode.QUEUED, queued=True, position=len(self._queue) - 1,
                queued_total=len(self._queue), active=len(self._active),
            ),
        )

    # ---------------- wait ----------------
    async def wait_until_running(self, request_id: str) -> WaitResult:
        """阻塞等待放行；排队超时 / 被取消时由本方法完成清理后返回。"""
        entry = self._entries.get(request_id)
        if entry is None:
            # submit 后立刻被移除的极端竞态：按已清理处理
            return WaitResult(WaitCode.CANCELLED)
        t0 = time.monotonic()
        deadline = t0 + self.queue_wait_timeout_seconds
        while True:
            if entry.state == "running":
                return WaitResult(WaitCode.STARTED, wait_ms=(time.monotonic() - t0) * 1000)
            if entry.terminal is not None:
                code = (
                    WaitCode.QUEUE_TIMEOUT
                    if entry.terminal == TerminalReason.QUEUE_TIMEOUT.value
                    else WaitCode.CANCELLED
                )
                return WaitResult(code, wait_ms=(time.monotonic() - t0) * 1000)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._finalize_queued(entry, TerminalReason.QUEUE_TIMEOUT.value)
                return WaitResult(
                    WaitCode.QUEUE_TIMEOUT,
                    wait_ms=(time.monotonic() - t0) * 1000,
                    queue_wait_timeout_seconds=self.queue_wait_timeout_seconds,
                )
            entry.wake.clear()
            try:
                await asyncio.wait_for(entry.wake.wait(), timeout=min(remaining, 0.5))
            except asyncio.TimeoutError:
                pass

    # ---------------- heartbeat（Phase 1 空操作） ----------------
    async def heartbeat(self, request_id: str, owner_token: str) -> None:
        return None

    # ---------------- release ----------------
    async def release(
        self, request_id: str, owner_token: str, terminal: str = "completed"
    ) -> ReleaseResult:
        entry = self._entries.get(request_id)
        if entry is None or request_id not in self._active:
            return ReleaseResult(ReleaseCode.NOT_RUNNING, message="幂等：请求不在活跃集合")
        if entry.owner_token != owner_token:
            return ReleaseResult(ReleaseCode.TOKEN_MISMATCH, message="owner token 不匹配")
        # 取消优先：运行中被取消 → 终态 cancelled
        final_reason = TerminalReason.CANCELLED.value if entry.cancel_requested else terminal
        entry.terminal = final_reason
        entry.finished_at = time.monotonic()
        self._active.pop(request_id, None)
        self._entries.pop(request_id, None)
        self._clear_user_placeholder(request_id)
        self._remember_terminal(entry)
        self._promote()
        return ReleaseResult(ReleaseCode.RELEASED, message="已释放槽位")

    # ---------------- cancel ----------------
    async def cancel(self, user_id: str, request_id: str) -> CancelResult:
        """调用方须先用 resolve_user_of 完成越权校验（403 在 API 层抛）。"""
        entry = self._entries.get(request_id)
        if entry is None:
            return CancelResult(CancelCode.ALREADY_FINISHED, message="请求不存在或已结束")
        if entry.state == "queued":
            # 原子移除队列并释放占位，唤醒等待者让其走 CANCELLED 分支
            self._finalize_queued(entry, TerminalReason.CANCELLED.value)
            return CancelResult(CancelCode.CANCELLED, message="已从队列取消")
        if entry.state in ("running", "cancelling"):
            entry.state = "cancelling"
            entry.cancel_requested = True
            return CancelResult(CancelCode.CANCELLING, message="已请求取消，等待真实退出")
        return CancelResult(CancelCode.ALREADY_FINISHED, message="请求已处于终态")

    # ---------------- snapshot / 查询 ----------------
    async def snapshot(self) -> CapacitySnapshot:
        active = len(self._active)
        queued = len(self._queue)
        return CapacitySnapshot(
            backend=self.backend_name,
            max_active=self.max_active,
            active=active,
            max_queue=self.max_queue,
            queued=queued,
            accepting=(active < self.max_active or queued < self.max_queue),
        )

    def queue_update(self, request_id: str) -> QueueUpdate:
        for pos, entry in enumerate(self._queue):
            if entry.request_id == request_id:
                return QueueUpdate(position=pos, queued=len(self._queue), active=len(self._active))
        return QueueUpdate(position=-1, queued=len(self._queue), active=len(self._active))

    def is_cancelling(self, request_id: str) -> bool:
        entry = self._entries.get(request_id)
        return entry is not None and entry.state == "cancelling"

    def resolve_user_of(self, request_id: str) -> Optional[str]:
        entry = self._entries.get(request_id)
        if entry is not None:
            return entry.user_id
        recent = self._recent_terminals.get(request_id)
        return recent[0] if recent else None

    # ---------------- 内部工具 ----------------
    def _finalize_queued(self, entry: _Entry, reason: str) -> None:
        """把仍处于队列的 entry 移除并置终态（cancel / queue_timeout 共用）。"""
        if entry in self._queue:
            self._queue.remove(entry)
        self._entries.pop(entry.request_id, None)
        entry.terminal = reason
        entry.finished_at = time.monotonic()
        self._clear_user_placeholder(entry.request_id)
        self._remember_terminal(entry)
        entry.wake.set()  # 唤醒 wait_until_running

    def _promote(self) -> None:
        """释放槽位后提升队首：队列非空且仍有槽位时逐个放行。"""
        while self._queue and len(self._active) < self.max_active:
            entry = self._queue.popleft()
            if entry.terminal is not None:  # 防御：跳过已被清理的条目
                continue
            entry.state = "running"
            entry.started_at = time.monotonic()
            self._active[entry.request_id] = entry
            entry.wake.set()

    def _clear_user_placeholder(self, request_id: str) -> None:
        if not self.one_inflight_per_user:
            return
        for uid, rid in list(self._user_req.items()):
            if rid == request_id:
                self._user_req.pop(uid, None)
                break

    def _remember_terminal(self, entry: _Entry) -> None:
        self._recent_terminals[entry.request_id] = (entry.user_id, entry.terminal)
        if len(self._recent_terminals) > _RECENT_TERMINAL_CAP:
            self._recent_terminals.popitem(last=False)

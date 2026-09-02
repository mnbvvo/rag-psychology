"""AdmissionBackend 协议：不同实现（memory / redis）必须满足的接口。

内存实现对应总稿 Phase 1（单 worker），Redis 实现对应 Phase 2（多实例）。
所有实现必须保证：submit 原子完成用户去重/槽位检查/入队；release/cancel 幂等；
终态不泄漏用户占位；只有真实退出后才释放活跃槽位（防超卖）。
"""
from abc import ABC, abstractmethod

from modules.concurrency.models import (
    CancelResult,
    CapacitySnapshot,
    QueueUpdate,
    ReleaseResult,
    SubmitResult,
    WaitResult,
)


class AdmissionBackend(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    async def submit(self, user_id: str, request_id: str) -> SubmitResult:
        """原子完成：用户去重 → 槽位检查 → 直接运行或入队。"""

    @abstractmethod
    async def wait_until_running(self, request_id: str) -> WaitResult:
        """阻塞等待放行；排队超时 / 被取消时由实现完成清理后返回。"""

    @abstractmethod
    async def heartbeat(self, request_id: str, owner_token: str) -> None:
        """续租（Phase 1 memory 后端为空操作；Phase 2 Redis 实现心跳）。"""

    @abstractmethod
    async def release(
        self, request_id: str, owner_token: str, terminal: str = "completed"
    ) -> ReleaseResult:
        """请求真实退出后释放活跃槽位，并按需提升队首。必须幂等。"""

    @abstractmethod
    async def cancel(self, user_id: str, request_id: str) -> CancelResult:
        """取消：QUEUED 原子移除并释放占位；RUNNING 标记 CANCELLING 等待真实退出。"""

    @abstractmethod
    async def snapshot(self) -> CapacitySnapshot:
        """聚合快照。"""

    @abstractmethod
    def queue_update(self, request_id: str) -> QueueUpdate:
        """SSE queue 事件现场快照；请求已不在队列时 position=-1。"""

    @abstractmethod
    def is_cancelling(self, request_id: str) -> bool:
        """RUNNING 请求是否已被请求取消（工作协程每轮检查）。"""

    @abstractmethod
    def resolve_user_of(self, request_id: str):
        """返回活跃/排队请求归属的 user_id（取消接口越权校验用）；不存在返回 None。"""

"""AI 问答并发准入：数据模型与契约。

对应《并发能力方案总稿》§4（AdmissionController）。本模块只定义
Ticket / 状态 / 错误码 / 返回码等纯数据契约，不依赖任何具体后端实现。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------- 错误码（对外契约，HTTP 语义见总稿 §4.3.3） ----------------
AI_REQUEST_IN_PROGRESS = "AI_REQUEST_IN_PROGRESS"      # 409：同一用户已有未完成请求
AI_QUEUE_FULL = "AI_QUEUE_FULL"                        # 429 + Retry-After：等待队列已满
AI_QUEUE_TIMEOUT = "AI_QUEUE_TIMEOUT"                  # 503 / SSE error：排队超时
AI_REQUEST_ALREADY_FINISHED = "AI_REQUEST_ALREADY_FINISHED"  # 409：请求已终止再取消
AI_REQUEST_CANCELLED = "AI_REQUEST_CANCELLED"          # 409 / SSE error：排队中或运行中被用户取消


class SubmitCode(str, Enum):
    """submit() 的结果码。"""
    STARTED = "started"          # 立即获得槽位，直接运行
    QUEUED = "queued"            # 进入等待队列
    REJECTED_DUPLICATE = "rejected_duplicate"  # 同用户已有未完成请求
    REJECTED_FULL = "rejected_full"            # 队列已满


class WaitCode(str, Enum):
    """wait_until_running() 的结果码。"""
    STARTED = "started"          # 已放行（含立即运行）
    QUEUE_TIMEOUT = "queue_timeout"  # 排队超时（后端已清理占位）
    CANCELLED = "cancelled"      # 排队期间被取消（后端已清理占位）


class CancelCode(str, Enum):
    """cancel() 的结果码。"""
    CANCELLED = "cancelled"      # QUEUED：已从队列原子移除并释放占位
    CANCELLING = "cancelling"    # RUNNING：已标记取消，等待工作协程/线程真实退出
    ALREADY_FINISHED = "already_finished"  # 已处于终态或不存在


class ReleaseCode(str, Enum):
    """release() 的结果码。"""
    RELEASED = "released"        # 已释放槽位并可能提升队首
    TOKEN_MISMATCH = "token_mismatch"  # owner token 不匹配（AC-12，不误释放）
    NOT_RUNNING = "not_running"  # 幂等：该请求不在活跃集合


class TerminalReason(str, Enum):
    """终态原因（可观测性/审计用）。"""
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISCONNECTED = "disconnected"
    QUEUE_TIMEOUT = "queue_timeout"


# ---------------- Ticket ----------------
@dataclass
class Ticket:
    """submit 返回的通行凭证：owner_token 仅服务端内部持有。"""
    request_id: str
    user_id: str
    owner_token: str = field(repr=False)  # 防旧请求释放新租约（AC-12）
    code: SubmitCode = SubmitCode.STARTED
    queued: bool = False
    position: Optional[int] = None       # 入队时的近似位置
    queued_total: Optional[int] = None   # 入队时的队列长度
    active: int = 0


@dataclass
class SubmitResult:
    code: SubmitCode
    ticket: Optional[Ticket] = None
    message: str = ""


@dataclass
class WaitResult:
    code: WaitCode
    wait_ms: float = 0.0
    queue_wait_timeout_seconds: float = 0.0


@dataclass
class CancelResult:
    code: CancelCode
    message: str = ""


@dataclass
class ReleaseResult:
    code: ReleaseCode
    message: str = ""


@dataclass
class CapacitySnapshot:
    """聚合快照（GET /api/concurrency/status 返回）。"""
    backend: str
    max_active: int
    active: int
    max_queue: int
    queued: int
    accepting: bool


@dataclass
class QueueUpdate:
    """SSE queue 事件所需的实时状态（每次推送时现场计算）。"""
    position: int
    queued: int
    active: int

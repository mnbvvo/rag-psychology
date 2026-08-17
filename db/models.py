"""ORM 模型：用户 / 会话 / 消息 / 危机审计。

持久化。六张表：
- users          用户账号（登录认证 + RBAC 角色）
- sessions      一次完整对话（前端一个 tab 对应一个）
- messages      单条消息（人类提问 / AI 回答），按会话外键聚合
- crisis_audit  危机命中审计（心理类产品的合规可追溯留痕）
- prompts       系统提示词模板（按用户隔离）
- compare_history AB 测试记录（按用户隔离）
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """统一使用 UTC，避免服务器时区不同导致审计时间错乱。"""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)  # 登录名
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)  # bcrypt 哈希，绝不存明文
    display_name: Mapped[str] = mapped_column(String(64), default="")
    role: Mapped[str] = mapped_column(String(20), default="user")  # user / admin（RBAC）
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # 禁用后无法登录与访问
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)  # 前端传入的会话 id（如 session-<timestamp>）；不传时由服务端生成
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)  # 归属用户（legacy 历史数据可空，由迁移归入 legacy 账号）
    title: Mapped[str] = mapped_column(String(255), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # human / ai
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped["Session"] = relationship(back_populates="messages")


class CrisisAudit(Base):
    __tablename__ = "crisis_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)  # 归属用户（合规留痕可追溯）
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    crisis_level: Mapped[str] = mapped_column(String(20), nullable=False)  # high / medium / low
    keywords_found: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 编码的命中列表
    question: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)  # 实际返回的安全话术
    is_crisis_response: Mapped[bool] = mapped_column(Boolean, default=False)
    detect_method: Mapped[str | None] = mapped_column(String(20), nullable=True)  # keyword / semantic / keyword+semantic
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # 语义距离（越小越贴近高危意图原型）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)  # 归属用户（提示词按用户隔离）
    name: Mapped[str] = mapped_column(String(255), default="未命名提示词")
    content: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)  # 是否作为 RAG 默认提示词
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class CompareHistory(Base):
    __tablename__ = "compare_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)  # 归属用户（对比历史按用户隔离）
    input: Mapped[str] = mapped_column(Text, nullable=False)  # 对比用的测试问题
    result_a: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 编码的 A 侧结果
    result_b: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 编码的 B 侧结果
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

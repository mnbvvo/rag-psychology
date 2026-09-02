"""ORM 模型：用户 / 会话 / 消息 / 危机审计 / 长期记忆。

持久化。五张表：
- users          用户账号（登录认证 + RBAC 角色）
- sessions      一次完整对话（前端一个 tab 对应一个）
- messages      单条消息（人类提问 / AI 回答），按会话外键聚合
- crisis_audit  危机命中审计（心理类产品的合规可追溯留痕）
- user_chat_history 长期记忆（每轮问答 + embedding，向量检索相似历史）
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, BigInteger, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from config.settings import settings


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


class UserChatHistory(Base):
    """长期记忆：用户每轮问答 + 双向量（向量检索相似历史注入上下文）。

    与 sessions/messages 的区别：sessions/messages 是「完整历史留痕」（前端可翻看），
    本表是「语义记忆」——每轮 query+answer 落库，提问时用当前问题向量检索该用户
    相似历史 top_k 条注入 prompt，成本恒定、不随历史总量线性增长。

    双向量（qa_embedding 为主）：
    - embedding：仅 query 的向量（兼容存量数据，保留回退用）
    - qa_embedding：query + answer 拼接后的向量，检索主用——匹配语义从
      「问题↔问题」升级为「问题↔问答内容」，用户换措辞也能靠 answer 语义召回。
    存量行 qa_embedding 为 NULL，由 SQL 函数 COALESCE 回退到 embedding。
    维度由 settings.VECTOR_DIMENSION 决定（当前 .env 为 text-embedding-v3 → 1024）。
    """

    __tablename__ = "user_chat_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list | None] = mapped_column(Vector(settings.VECTOR_DIMENSION), nullable=True)
    qa_embedding: Mapped[list | None] = mapped_column(Vector(settings.VECTOR_DIMENSION), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

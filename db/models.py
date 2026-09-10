"""ORM 模型：用户 / 会话 / 消息 / 危机审计 / 长期记忆 / 家庭档案。

持久化。七张表：
- users          用户账号（登录认证 + RBAC 角色；小程序通道为外部 userId 的镜像行）
- sessions       一次完整对话（前端一个 tab 对应一个）
- messages       单条消息（人类提问 / AI 回答），按会话外键聚合
- crisis_audit   危机命中审计（心理类产品的合规可追溯留痕）
- user_chat_history 长期记忆（每轮问答 + embedding，向量检索相似历史）
- user_family_profile 用户家庭档案主表（小程序 register/update 落库，1 用户 1 行）
- user_children  家庭档案的孩子子表（多孩子，随档案全量替换维护）
"""
import uuid
from datetime import date, datetime, timezone
from sqlalchemy import (
    String,
    Integer,
    BigInteger,
    DateTime,
    Boolean,
    Text,
    Float,
    ForeignKey,
    Date,
)
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

    # 注意：id 在 Web 端为服务端 uuid4().hex；在微信小程序通道（/api/mp/*）为外部
    # userId 原样落库（镜像行，见 crud_async.get_or_create_mp_user）→ 列宽 64 以容纳
    # openid / UUID / 第三方系统长 id。
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
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

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex)  # 前端传入的会话 id（如 session-<timestamp>）；不传时由服务端生成。小程序端生成 id 请控制在 36 字符内
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # 归属用户（legacy 历史数据可空，由迁移归入 legacy 账号；小程序通道为外部 userId）
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
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # 归属用户（合规留痕可追溯）
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


class UserFamilyProfile(Base):
    """用户家庭档案主表（微信小程序通道 /api/mp/register|update 落库，1 用户 1 行）。

    存储家长本人信息（昵称/生日/家长角色）+ 若干孩子（user_children 子表）。
    每次 register/update 以「整份档案」幂等 upsert：主行 INSERT ... ON CONFLICT DO UPDATE，
    children 先 upsert 提交项、再删除不在提交列表里的（合起来 = 全量替换，事务内，
    见 crud_async.upsert_family_profile）。不用「先删后插」是因为并发下会撞唯一索引。

    敏感个人信息（家长生日、孩子生日/性别）：仅用于对话个性化与审计留痕。
    注入 LLM prompt 前必须经 modules/family_profile.py 加工（生日换算为年龄等
    相对表述，昵称/角色/性别按需保留），原始生日等字段不得直接进 prompt。
    """

    __tablename__ = "user_family_profile"

    user_id: Mapped[str] = mapped_column(
        String(64), primary_key=True
    )  # 对应 users.id（mp 通道 = 外部 userId 原值）
    user_nickname: Mapped[str] = mapped_column(String(64), default="")  # 家长昵称
    birthday: Mapped[date | None] = mapped_column(Date, nullable=True)  # 家长生日
    parent_role: Mapped[str] = mapped_column(
        String(20), default=""
    )  # 家长角色（爸爸/妈妈/其他照护者，前端原值）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    children: Mapped[list["UserChild"]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        order_by="UserChild.child_id",
        passive_deletes=True,
    )


class UserChild(Base):
    """家庭档案的孩子子表：一个家长档案下可有多个孩子（复合主键 user_id+child_id）。"""

    __tablename__ = "user_children"

    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("user_family_profile.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    child_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # 小程序侧生成的稳定 id
    child_nickname: Mapped[str] = mapped_column(String(64), default="")
    child_birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str] = mapped_column(String(8), default="")  # 前端原值（男/女/…），注入前归一化
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    profile: Mapped["UserFamilyProfile"] = relationship(back_populates="children")

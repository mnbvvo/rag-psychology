"""异步数据层：async 请求路径（async def 端点 + 请求级 AsyncSession）专用。

与 db/crud.py（同步）的关系：
- crud.py 供线程路径使用（bg Worker 后台落库、长期记忆检索、auth 等 sync def 端点）；
- 本模块供事件循环上的 async def 端点使用（sessions 管理、admin 查询、越权预检）。
AsyncSession 不能在 asyncio.to_thread / 线程池中运行，反之同步 Session 也不应在
事件循环上被 async 端点直接使用 —— 两条路径用各自的实现，互不混用。

读取均显式 selectinload(messages)，避免 AsyncSession 惰性加载抛 MissingGreenlet。
"""
import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import CrisisAudit, Message, Session as ConvSession, User


# ---------------- 用户 ----------------
async def get_user(db: AsyncSession, user_id: str) -> User | None:
    return await db.get(User, user_id)


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    res = await db.execute(select(User).where(User.username == username))
    return res.scalars().first()


async def create_user(
    db: AsyncSession,
    username: str,
    password_hash: str,
    display_name: str = "",
    role: str = "user",
    is_active: bool = True,
) -> User:
    user = User(
        id=uuid.uuid4().hex,
        username=username,
        password_hash=password_hash,
        display_name=display_name or username,
        role=role,
        is_active=is_active,
    )
    db.add(user)
    await db.flush()
    return user


async def list_users(db: AsyncSession, limit: int = 100) -> list[User]:
    res = await db.execute(select(User).order_by(User.created_at).limit(limit))
    return list(res.scalars().all())


# ---------------- 会话（显式 eager load messages，防 MissingGreenlet） ----------------
async def list_sessions(db: AsyncSession, user_id: str, limit: int = 50) -> list[ConvSession]:
    res = await db.execute(
        select(ConvSession)
        .options(selectinload(ConvSession.messages))
        .where(ConvSession.user_id == user_id)
        .order_by(desc(ConvSession.updated_at))
        .limit(limit)
    )
    return list(res.scalars().all())


async def get_session_with_messages(db: AsyncSession, session_id: str) -> ConvSession | None:
    res = await db.execute(
        select(ConvSession)
        .options(selectinload(ConvSession.messages))
        .where(ConvSession.id == session_id)
    )
    return res.scalars().first()


async def session_belongs_to(db: AsyncSession, session_id: str, user_id: str) -> bool:
    """水平越权校验：存在→必须本人；不存在（待新建）→放行（与同步版语义一致）。"""
    if not session_id or not user_id:
        return False
    s = await db.get(ConvSession, session_id)
    if s is None:
        return True
    return s.user_id == user_id


async def create_session(db: AsyncSession, session_id: str, name: str, user_id: str) -> ConvSession:
    sess = ConvSession(id=session_id, title=name[:255], user_id=user_id)
    db.add(sess)
    await db.flush()
    return sess


# ---------------- 危机审计（admin） ----------------
async def list_crisis_audits(db: AsyncSession, limit: int = 100) -> list[CrisisAudit]:
    res = await db.execute(
        select(CrisisAudit).order_by(desc(CrisisAudit.created_at)).limit(limit)
    )
    return list(res.scalars().all())

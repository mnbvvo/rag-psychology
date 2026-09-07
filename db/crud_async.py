"""异步数据层：async 请求路径（async def 端点 + 请求级 AsyncSession）专用。

与 db/crud.py（同步）的关系：
- crud.py 供线程路径使用（bg Worker 后台落库、长期记忆检索、auth 等 sync def 端点）；
- 本模块供事件循环上的 async def 端点使用（sessions 管理、admin 查询、越权预检、
  小程序通道 /api/mp/* 的镜像用户与家庭档案读写）。
AsyncSession 不能在 asyncio.to_thread / 线程池中运行，反之同步 Session 也不应在
事件循环上被 async 端点直接使用 —— 两条路径用各自的实现，互不混用。

读取原则：**只在需要返回消息内容的查询上用 selectinload(messages)**（如会话消息端点）；
归属校验/列表计数走轻量路径（get_session 单行、COUNT 聚合），避免把整会话消息
读进内存只为数个数或验归属。
"""
import hashlib
import uuid
from datetime import date

from sqlalchemy import delete, desc, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from modules.security import LEGACY_PASSWORD_HASH
from .models import (
    CrisisAudit,
    Message,
    Session as ConvSession,
    User,
    UserChild,
    UserFamilyProfile,
)


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


# ---------------- 会话（归属校验用轻量路径；仅取消息的端点才 selectinload） ----------------
async def list_sessions(
    db: AsyncSession, user_id: str, limit: int = 200
) -> list[tuple]:
    """列出当前用户的最近会话（含消息数）。

    返回 [(Session, message_count), ...]：message_count 用 **COUNT 聚合**
    （outerjoin + group_by），而非 selectinload(messages) 后 len()——后者会把
    每个会话的全部消息读进内存只为数个数，会话多/消息长时内存放大。
    user_id 隔离由调用方携带；limit 上限由 API 层 Query(le=200) 保证。
    """
    res = await db.execute(
        select(ConvSession, func.count(Message.id))
        .outerjoin(Message, Message.session_id == ConvSession.id)
        .where(ConvSession.user_id == user_id)
        .group_by(ConvSession.id)
        .order_by(desc(ConvSession.updated_at))
        .limit(limit)
    )
    return [(s, int(cnt)) for s, cnt in res.all()]


async def get_session(db: AsyncSession, session_id: str) -> ConvSession | None:
    """按 id 取会话行（**不**载入消息）——归属校验/改名/删除等轻量路径。

    避免为一条归属校验用 get_session_with_messages 把整会话消息全载进内存。
    """
    return await db.get(ConvSession, session_id)


async def get_session_with_messages(db: AsyncSession, session_id: str) -> ConvSession | None:
    """按 id 取会话 + 全部消息——仅「需要返回消息内容」的端点使用
    （GET /api/sessions/{id}/messages）；归属校验请用 get_session 轻量路径。"""
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


# ---------------- 小程序通道（/api/mp/*）：外部 userId 镜像 + 家庭档案 ----------------
def mp_username(external_id: str) -> str:
    """镜像账号登录名：mp_ 前缀 + sha1 截断 → 3-32 位字母数字下划线、全局唯一、不可登录。

    username 唯一约束只防 Web 端同登录名注册；镜像号由外部 id 确定性派生，
    同一外部 id 幂等命中同一行，不同外部 id 碰撞概率可忽略（sha1 前 24 hex）。
    """
    return "mp_" + hashlib.sha1(external_id.encode("utf-8")).hexdigest()[:24]


def _guard_mp_mirror(user: User) -> User:
    """镜像守卫：mp 通道只允许以「本服务创建的小程序镜像号」复用 users 行。

    若外部 userId 恰好等于既有**内部账号** id（JWT 注册生成的 uuid / legacy /
    admin 等，username 非 mp_ 前缀），复用会让外部请求把档案/会话挂到内部账号
    名下（数据串号 + 以内部账号口径记账）。此时拒绝而非复用（404 语义偏软，
    用 400 明确是 userId 冲突，让小程序换用其生态自己的 id）。
    """
    if not (user.username or "").startswith("mp_"):
        raise ValueError("userId 与系统内部账号冲突，请更换 userId")
    return user


async def get_or_create_mp_user(
    db: AsyncSession,
    external_id: str,
    nickname: str = "",
) -> User:
    """宽容建号：users 镜像行不存在则创建（幂等 + 并发安全）。

    users.id = 外部 userId 原值（≤ settings.MP_USER_ID_MAX_LEN，api 层已校验），
    password_hash 用不可登录占位、role=user、is_active=True。
    宽容语义：小程序未调 register 直接 query 也能让会话/记忆/危机审计挂到归属行。

    并发安全：不依赖「先查后插」（两个并发首见同一 userId 会主键冲突），改用
    PG INSERT ... ON CONFLICT DO NOTHING，冲突（他人已插/自插）后统一再读，幂等。
    命中既有内部账号（username 非 mp_ 前缀）→ ValueError（api 层转 400）。
    注意：本函数只服务 PostgreSQL 下的 async 请求路径（mp 通道要求 PG，见 api/deps）。
    """
    user = await db.get(User, external_id)
    if user is not None:
        return _guard_mp_mirror(user)
    stmt = (
        pg_insert(User)
        .values(
            id=external_id,
            username=mp_username(external_id),
            password_hash=LEGACY_PASSWORD_HASH,  # 不可登录（镜像账号，密码体系不属于本服务）
            display_name=(nickname or external_id)[:64],
            role="user",
            is_active=True,
        )
        .on_conflict_do_nothing(index_elements=[User.id])
    )
    await db.execute(stmt)
    await db.flush()
    # 自插成功或与他人并发冲突都走到这里；读到的是已存在/刚提交的镜像行
    user = await db.get(User, external_id)
    if user is None:
        # 理论不可达（ON CONFLICT DO NOTHING 后行必存在）；防御性兜底
        raise RuntimeError("小程序用户镜像创建失败，请重试")
    return _guard_mp_mirror(user)


async def upsert_family_profile(
    db: AsyncSession,
    user_id: str,
    *,
    user_nickname: str,
    birthday: date | None,
    parent_role: str,
    children: list[dict],
) -> UserFamilyProfile:
    """整份档案幂等 upsert：主行覆盖 + children 全量替换（事务内，由请求级 session 提交）。

    children 每项含 child_id/child_nickname/child_birth_date/gender（api 层已完成
    字段清洗与截断），此处仅落库。全量替换符合小程序"资料编辑页整份提交"的形态。
    """
    row = await db.get(UserFamilyProfile, user_id)
    if row is None:
        row = UserFamilyProfile(user_id=user_id)
        db.add(row)
    row.user_nickname = (user_nickname or "")[:64]
    row.birthday = birthday
    row.parent_role = (parent_role or "")[:20]
    # children 全量替换：先删后插（同事务）
    await db.execute(delete(UserChild).where(UserChild.user_id == user_id))
    for c in children:
        db.add(
            UserChild(
                user_id=user_id,
                child_id=(c.get("child_id") or "")[:64],
                child_nickname=(c.get("child_nickname") or "")[:64],
                child_birth_date=c.get("child_birth_date"),
                gender=(c.get("gender") or "")[:8],
            )
        )
    await db.flush()
    return row


async def get_family_profile(db: AsyncSession, user_id: str) -> dict | None:
    """读取整份家庭档案；无档案返回 None。

    返回 dict：{user_nickname, birthday(YYYY-MM-DD|None), parent_role,
    children:[{child_id, child_nickname, child_birth_date, gender}]}
    （供 /api/mp/profile 回显与 modules/family_profile 加工注入共用）。
    """
    row = await db.get(UserFamilyProfile, user_id)
    if row is None:
        return None
    kids = await db.execute(
        select(UserChild)
        .where(UserChild.user_id == user_id)
        .order_by(UserChild.child_id)
    )
    return {
        "user_nickname": row.user_nickname,
        "birthday": row.birthday.isoformat() if row.birthday else None,
        "parent_role": row.parent_role,
        "children": [
            {
                "child_id": k.child_id,
                "child_nickname": k.child_nickname,
                "child_birth_date": k.child_birth_date.isoformat()
                if k.child_birth_date
                else None,
                "gender": k.gender,
            }
            for k in kids.scalars().all()
        ],
    }

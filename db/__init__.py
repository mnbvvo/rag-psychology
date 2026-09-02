"""关系型数据库初始化（SQLAlchemy，双后端：SQLite / PostgreSQL）。

与向量库互补：本模块负责结构化持久化（用户 / 会话 / 消息 / 危机审计 /
提示词 / 长期记忆），向量检索由 pgvector（或 Chroma）负责。
使用 SQLAlchemy 抽象，切换数据库仅需修改 settings.DB_URL，业务代码无需改动。

- SQLite：单文件、零部署，本地原型默认；
- PostgreSQL：生产 / 多 worker 推荐（DB_BACKEND=postgres，.env 配置 PG_*）。
"""
import os
import sys
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings

# Windows 兼容：psycopg(async) 要求 SelectorEventLoop，而 Windows 默认 ProactorEventLoop
# 不兼容（sqlalchemy psycopg async / greenlet 会抛 InterfaceError）。模块 import 时统一
# 切换到 Selector 策略，保证 uvicorn 与脚本里创建的 asyncio 循环都可跑 async engine。
if sys.platform == "win32":
    import asyncio as _asyncio

    try:
        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:  # 非 Windows 或极老版本无此策略
        pass


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


# SQLite 单文件连接在多线程（FastAPI 线程池跑同步 DB 调用）下需关闭跨线程检查
_connect_args = {"check_same_thread": False} if _is_sqlite(settings.DB_URL) else {}

if _is_sqlite(settings.DB_URL):
    # SQLite 本地原型：默认池即可（单文件、零部署）
    engine = create_engine(settings.DB_URL, connect_args=_connect_args, future=True)
else:
    # PostgreSQL：同步 engine 只服务「子线程路径」（bg Worker 落库/记忆检索/auth 等 sync
    # def 端点——AsyncSession 不能在 to_thread/线程池使用）。池给小的 Worker 池即可；
    # 请求路径的 async 主池见下方 async_engine（AsyncSession 只能在事件循环使用）。
    engine = create_engine(
        settings.DB_URL,
        connect_args=_connect_args,
        future=True,
        pool_size=settings.DB_WORKER_POOL_SIZE,
        max_overflow=settings.DB_WORKER_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,  # 借出前校验连接存活，防陈旧连接
    )
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


# ============ 异步数据层（请求路径专用；总稿 AsyncEngine + 请求级 AsyncSession） ============
# 线程路径（bg Worker 持久化、长期记忆检索、auth/me 等 sync def 端点由 FastAPI 自动丢线程池）
# 继续用上面的同步 engine/SessionLocal —— AsyncSession 不能跨 asyncio.to_thread/线程池使用。
# SQLite 本地原型不提供 async 支持（异步请求路径端点仅在 PostgreSQL 下可用）。
# 连接预算（同库多池需合并计算，见总稿 §3.4）：
#   async 主池（请求路径 AsyncSession）= DB_POOL_SIZE + DB_MAX_OVERFLOW
#   sync Worker 池（后台落库/记忆/auth 等线程路径）= DB_WORKER_POOL_SIZE + DB_WORKER_MAX_OVERFLOW
#   PGVector(langchain_postgres) 自带池
async_engine = None
async_session_factory = None
if not _is_sqlite(settings.DB_URL):
    from sqlalchemy.ext.asyncio import async_sessionmaker as _async_sessionmaker
    from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine

    # psycopg3 同一条连接串可驱动 async engine（无需 asyncpg），pgvector SQL 兼容
    async_engine = _create_async_engine(
        settings.DB_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
    )
    async_session_factory = _async_sessionmaker(
        async_engine, autoflush=False, expire_on_commit=False
    )

# 历史数据归属账号（所有 user_id 为空的历史行在启动时归入此账号，不可登录）
LEGACY_USERNAME = "legacy"
# 需要加 user_id 归属列的表（新增列由轻量迁移补齐；users 表由 create_all 新建）。
# 注意：prompts 不在其中——提示词全局共享、无用户归属，不参与 legacy 归并。
_USER_TABLES = ("sessions", "crisis_audit")


def _table_columns(table: str) -> set[str]:
    """数据库无关的表列名查询（SQLite 用 PRAGMA，PostgreSQL 用 information_schema）。"""
    with engine.connect() as conn:
        if _is_sqlite(settings.DB_URL):
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            return {str(row[1]) for row in rows}
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :t"
            ),
            {"t": table.lower()},
        ).fetchall()
        return {str(row[0]) for row in rows}


def _migrate_crisis_audit_columns() -> None:
    """轻量迁移：老库为 crisis_audit 补充 detect_method / confidence 列。

    create_all 只会建新表、不会给已存在的表加列，因此对老库需要显式 ALTER。
    """
    with engine.connect() as conn:
        cols = _table_columns("crisis_audit")
        if "detect_method" not in cols:
            conn.execute(text("ALTER TABLE crisis_audit ADD COLUMN detect_method VARCHAR(20)"))
        if "confidence" not in cols:
            conn.execute(text("ALTER TABLE crisis_audit ADD COLUMN confidence FLOAT"))
        conn.commit()


def _migrate_user_columns() -> None:
    """轻量迁移：为 4 张业务表补齐 user_id 归属列（幂等）。

    user_id 可空：历史行由 ensure_bootstrap_users 统一归入 legacy 账号，
    新写入的数据由 crud / API 层强制携带当前用户 id。
    """
    with engine.connect() as conn:
        for table in _USER_TABLES:
            cols = _table_columns(table)
            if "user_id" not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN user_id VARCHAR(36)"))
        conn.commit()


def ensure_bootstrap_users() -> None:
    """引导账号（幂等）：legacy（历史数据归属）+ 初始管理员。

    1. 创建 legacy 账号（不可登录，is_active=False），并把所有 user_id 为空的
       历史行归入它 —— 保证历史数据有归属且对新用户不可见（数据隔离）。
    2. users 表为空时按 settings.INIT_ADMIN_USERNAME/PASSWORD 创建管理员，
       供垂直越权（admin 接口）测试与运维使用；生产环境请通过 .env 覆盖密码。
    """
    from modules.security import hash_password, LEGACY_PASSWORD_HASH

    from . import crud

    with SessionLocal.begin() as db:
        legacy = crud.get_user_by_username(db, LEGACY_USERNAME)
        if legacy is None:
            legacy = crud.create_user(
                db,
                username=LEGACY_USERNAME,
                password_hash=LEGACY_PASSWORD_HASH,
                display_name="历史数据迁移账号",
                role="admin",
                is_active=False,
            )
        # 无归属的历史行统一归入 legacy
        for table in _USER_TABLES:
            db.execute(
                text(f"UPDATE {table} SET user_id=:uid WHERE user_id IS NULL"),
                {"uid": legacy.id},
            )
        # 初始管理员（users 表为空时创建）
        admin = crud.get_user_by_username(db, settings.INIT_ADMIN_USERNAME)
        if admin is None:
            crud.create_user(
                db,
                username=settings.INIT_ADMIN_USERNAME,
                password_hash=hash_password(settings.INIT_ADMIN_PASSWORD),
                display_name="管理员",
                role="admin",
                is_active=True,
            )


def init_db() -> None:
    """幂等创建所有表；SQLite 额外确保文件目录存在。"""
    if _is_sqlite(settings.DB_URL):
        db_path = settings.DB_URL.replace("sqlite:///", "", 1)
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    # 触发模型注册（必须在 create_all 之前导入）
    from . import models  # noqa: F401
    models.Base.metadata.create_all(bind=engine)
    # 老库轻量迁移（补列，幂等）
    try:
        _migrate_crisis_audit_columns()
        _migrate_user_columns()
    except Exception as e:
        print(f"[db][WARN] 列迁移失败（不影响启动）: {e}", flush=True)
    # 引导账号：legacy + 初始管理员（失败不影响启动，仅告警）
    try:
        ensure_bootstrap_users()
    except Exception as e:
        print(f"[db][WARN] 引导账号创建失败（不影响启动）: {e}", flush=True)

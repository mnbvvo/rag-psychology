"""关系型数据库初始化（SQLAlchemy，双后端：SQLite / PostgreSQL）。

与向量库互补：本模块负责结构化持久化（用户 / 会话 / 消息 / 危机审计 /
提示词库 / 对比历史），向量检索由 pgvector（或 Chroma）负责。
使用 SQLAlchemy 抽象，切换数据库仅需修改 settings.DB_URL，业务代码无需改动。

- SQLite：单文件、零部署，本地原型默认；
- PostgreSQL：生产 / 多 worker 推荐（DB_BACKEND=postgres，.env 配置 PG_*）。
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


# SQLite 单文件连接在多线程（FastAPI 线程池跑同步 DB 调用）下需关闭跨线程检查
_connect_args = {"check_same_thread": False} if _is_sqlite(settings.DB_URL) else {}

engine = create_engine(settings.DB_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

# 历史数据归属账号（所有 user_id 为空的历史行在启动时归入此账号，不可登录）
LEGACY_USERNAME = "legacy"
# 需要加 user_id 归属列的表（新增列由轻量迁移补齐；users 表由 create_all 新建）
_USER_TABLES = ("sessions", "prompts", "compare_history", "crisis_audit")


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

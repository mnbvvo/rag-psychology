"""关系型数据库初始化（SQLAlchemy + SQLite）。

与 Chroma 向量库互补：Chroma 负责语义检索（向量），本模块负责结构化
持久化（会话 / 消息 / 危机审计）。使用 SQLAlchemy 抽象，未来切换到
MySQL 仅需修改 settings.DB_URL，业务代码无需改动。
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


# SQLite 单文件连接在多线程（FastAPI 线程池跑同步 DB 调用）下需关闭跨线程检查
_connect_args = {"check_same_thread": False} if _is_sqlite(settings.DB_URL) else {}

engine = create_engine(settings.DB_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _migrate_crisis_audit_columns() -> None:
    """SQLite 轻量迁移：老库为 crisis_audit 补充 detect_method / confidence 列。

    create_all 只会建新表、不会给已存在的表加列，因此对老库需要显式 ALTER。
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(crisis_audit)"))}
        if "detect_method" not in cols:
            conn.execute(text("ALTER TABLE crisis_audit ADD COLUMN detect_method VARCHAR(20)"))
        if "confidence" not in cols:
            conn.execute(text("ALTER TABLE crisis_audit ADD COLUMN confidence FLOAT"))
        conn.commit()


def init_db() -> None:
    """幂等创建所有表，并确保 SQLite 文件所在目录存在。"""
    if _is_sqlite(settings.DB_URL):
        db_path = settings.DB_URL.replace("sqlite:///", "", 1)
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    # 触发模型注册（必须在 create_all 之前导入）
    from . import models  # noqa: F401
    models.Base.metadata.create_all(bind=engine)
    # 老库轻量迁移（仅 SQLite 支持 PRAGMA / ALTER 语义）
    if _is_sqlite(settings.DB_URL):
        try:
            _migrate_crisis_audit_columns()
        except Exception as e:
            print(f"[db][WARN] crisis_audit 列迁移失败（不影响启动）: {e}", flush=True)

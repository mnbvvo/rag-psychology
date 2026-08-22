"""数据迁移：SQLite + Chroma → PostgreSQL + pgvector（全量，含向量复用）。

把现有系统数据完整迁到新数据库：
  关系数据：users / sessions / messages / crisis_audit / prompts / compare_history
  向量数据：Chroma 的 257 条知识卡片 → pgvector（复用原向量，不重新调用 embedding API）

前置条件：
  1) PostgreSQL 已安装 pgvector 扩展（Windows 需管理员复制 vector.dll 到 PG lib/ 并重启服务）；
  2) 数据库 rag_psychology 存在（脚本可自动创建）。

用法：
    python scripts/migrate_to_postgres.py
    python scripts/migrate_to_postgres.py --old-sqlite data/rag_psychology.sqlite3

校验：迁移后逐表对比新旧行数，向量数对比，全部一致才输出 MIGRATION OK。
"""
import argparse
import sys
from datetime import timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, select, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from config.settings import settings  # noqa: E402


def _to_aware(dt):
    """SQLite 读出的 naive datetime → UTC aware（PG timestamptz 需要 aware）。"""
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        try:
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return dt
    return dt


def _dt_columns(model):
    from sqlalchemy import DateTime

    return {
        col.key
        for col in model.__table__.columns
        if isinstance(col.type, DateTime)
    }


def ensure_database(pg_admin_url: str, db_name: str) -> None:
    """连接 postgres 管理库，若目标库不存在则创建（AUTOCOMMIT，CREATE DATABASE 不能在事务中）。"""
    engine = create_engine(pg_admin_url.replace(f"/{db_name}", "/postgres"), future=True)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": db_name}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            print(f"[pg] 已创建数据库 {db_name}")
        else:
            print(f"[pg] 数据库 {db_name} 已存在")
    engine.dispose()


def check_pgvector(engine) -> bool:
    """检查 pgvector 扩展是否可用（未安装时给出安装指引）。"""
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except Exception as e:
        print(f"[pg] ✗ pgvector 扩展不可用: {e}")
        print("[pg] Windows 安装指引：以管理员运行 PowerShell 执行：")
        print("    Stop-Service postgresql-x64-16 -Force")
        print('    Copy-Item -Force "$env:TEMP\\pgv\\vector.dll" "C:\\software\\PostgreSQL\\lib\\vector.dll"')
        print('    Copy-Item -Force "$env:TEMP\\pgv\\vector*" "C:\\software\\PostgreSQL\\share\\extension\\"')
        print("    Start-Service postgresql-x64-16")
        return False


def migrate_relations(old_engine, new_engine) -> dict:
    """关系数据：旧库逐表 → 新库（保留 id，datetime 转 aware）。"""
    from db.models import CompareHistory, CrisisAudit, Message, Prompt, Session, User

    models = [User, Session, Message, CrisisAudit, Prompt, CompareHistory]
    result = {}
    Old = sessionmaker(bind=old_engine, future=True)
    New = sessionmaker(bind=new_engine, future=True)

    with Old() as old_db, New.begin() as new_db:
        for model in models:
            rows = old_db.execute(select(model)).scalars().all()
            dt_cols = _dt_columns(model)
            for r in rows:
                kwargs = {}
                for col in model.__table__.columns:
                    val = getattr(r, col.key)
                    if col.key in dt_cols:
                        val = _to_aware(val)
                    kwargs[col.key] = val
                new_db.add(model(**kwargs))
            result[model.__tablename__] = len(rows)
            print(f"  [pg] {model.__tablename__}: {len(rows)} 行")

    # 同步自增序列（显式插入 id 后需要 setval，否则后续插入冲突）
    with new_engine.begin() as conn:
        for table in ("messages", "crisis_audit", "compare_history"):
            conn.execute(
                text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
                )
            )
    print("  [pg] 自增序列已同步")
    return result


def migrate_vectors(old_chroma_path: str, new_url: str, collection: str, dimension: int) -> int:
    """向量数据：Chroma → pgvector（复用原向量，不重新 embed）。"""
    import chromadb
    from langchain_postgres import PGVector

    from modules.vector_store import TimedOpenAIEmbeddings

    client = chromadb.PersistentClient(path=old_chroma_path)
    col = client.get_collection(collection)
    data = col.get(include=["documents", "metadatas", "embeddings"])
    texts = data.get("documents") or []
    metas = data.get("metadatas") or []
    raw_vecs = data.get("embeddings")
    vecs = [list(v) for v in raw_vecs] if raw_vecs is not None else []
    ids = data.get("ids") or []
    print(f"  [chroma] 读取 {len(texts)} 条文档（维度 {len(vecs[0]) if vecs else '?'}）")

    embeddings = TimedOpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=new_url,
        embedding_length=dimension,
        use_jsonb=True,
        create_extension=True,
    )
    if store.get_collection_count() if hasattr(store, "get_collection_count") else False:
        pass
    store.add_embeddings(texts=texts, embeddings=vecs, metadatas=metas, ids=ids)
    print(f"  [pg] pgvector 写入 {len(texts)} 条向量")
    return len(texts)


def main():
    ap = argparse.ArgumentParser(description="SQLite+Chroma → PostgreSQL+pgvector 全量迁移")
    ap.add_argument("--old-sqlite", default=str(Path(settings._DB_PATH).resolve() if hasattr(settings, "_DB_PATH") else PROJECT_ROOT / "data/rag_psychology.sqlite3"))
    ap.add_argument("--new-url", default="", help="PG 连接串（默认取 settings.DB_URL）")
    ap.add_argument("--vector-dim", type=int, default=settings.VECTOR_DIMENSION)
    ap.add_argument("--skip-vector", action="store_true", help="只迁关系数据，跳过向量")
    args = ap.parse_args()

    old_sqlite = str(Path(args.old_sqlite).resolve())
    if not Path(old_sqlite).is_file():
        raise FileNotFoundError(f"旧 SQLite 库不存在: {old_sqlite}")

    new_url = args.new_url or settings.DB_URL
    if not new_url.startswith("postgresql"):
        raise SystemExit(f"目标必须是 PostgreSQL（当前 DB_URL={new_url}），请先配置 .env 的 DB_BACKEND=postgres")
    print(f"旧库(SQLite): {old_sqlite}")
    print(f"新库(PostgreSQL): {new_url}")

    db_name = new_url.rsplit("/", 1)[-1].split("?")[0]
    ensure_database(new_url, db_name)

    # 建表（模型）前先清空，保证从零迁移、幂等
    from db import models as db_models  # noqa: F401

    engine_new = create_engine(new_url, future=True)
    if not check_pgvector(engine_new):
        sys.exit(1)
    with engine_new.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS messages CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS sessions CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS crisis_audit CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS prompts CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS compare_history CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE"))
    db_models.Base.metadata.create_all(bind=engine_new)
    print("[pg] 表结构已创建")

    old_engine = create_engine(f"sqlite:///{old_sqlite}", future=True)
    print("\n== 1/2 关系数据迁移 ==")
    relation_counts = migrate_relations(old_engine, engine_new)

    vec_count = 0
    if not args.skip_vector:
        print("\n== 2/2 向量数据迁移（Chroma → pgvector） ==")
        from config.settings import settings as _s

        vec_count = migrate_vectors(
            str(PROJECT_ROOT / "data/chroma"),
            new_url,
            _s.COLLECTION_NAME,
            args.vector_dim,
        )

    # 校验
    print("\n== 校验 ==")
    Old = sessionmaker(bind=old_engine, future=True)
    New = sessionmaker(bind=engine_new, future=True)
    ok = True
    with Old() as odb, New() as ndb:
        for table, expected in relation_counts.items():
            got = ndb.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            status = "OK" if got == expected else "MISMATCH"
            if got != expected:
                ok = False
            print(f"  {table}: 旧={expected} 新={got} {status}")
        if not args.skip_vector:
            got_v = ndb.execute(
                text(
                    "SELECT COUNT(*) FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON e.collection_id=c.uuid "
                    "WHERE c.name=:n"
                ),
                {"n": settings.COLLECTION_NAME},
            ).scalar()
            status = "OK" if got_v == vec_count else "MISMATCH"
            if got_v != vec_count:
                ok = False
            print(f"  pgvector: 旧(chroma)={vec_count} 新={got_v} {status}")
    old_engine.dispose()
    engine_new.dispose()

    print("\n" + "=" * 46)
    if ok:
        print("MIGRATION OK ✓  数据已全部迁入 PostgreSQL + pgvector")
    else:
        print("MIGRATION FAILED ✗  存在数量不一致，请检查")
        sys.exit(1)


if __name__ == "__main__":
    main()

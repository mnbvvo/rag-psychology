"""PostgreSQL 全量备份脚本（含 pgvector 向量表）。

用 pg_dump 自定义格式导出整个 rag_psychology 库（关系表 + pgvector 向量表都在
同一库中，全库导出天然覆盖），并额外保存一份「备份前行数快照」JSON 供
一致性核对与监控使用。

用法：
  python scripts/backup_pg.py                    # 全量备份到 backups/
  python scripts/backup_pg.py --outdir D:\\backups  # 指定备份目录
  python scripts/backup_pg.py --keep 7           # 保留最近 7 份（默认全留）

定时备份（Windows）：任务计划程序 → 创建任务 → 触发器选每天 → 操作：
  python C:\\Users\\Thunderobot\\Desktop\\rag-psychology\\scripts\\backup_pg.py

恢复（演练/灾难恢复）：
  pg_restore -h 127.0.0.1 -U postgres -d rag_psychology --clean --if-exists backups/rag_psychology_YYYYMMDD_HHMMSS.dump
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402


def _pg_args_from_url(url: str) -> dict:
    """从 SQLAlchemy URL 解析 pg 连接参数（不暴露密码到命令行日志）。"""
    import urllib.parse

    rest = url.split("://", 1)[1]
    cred, hostpart = rest.rsplit("@", 1)
    user, _, pw = cred.partition(":")
    host, _, db = hostpart.partition("/")
    db = db.split("?")[0]
    h, _, port = host.partition(":")
    return {"host": h or "127.0.0.1", "port": port or "5432", "user": user, "password": pw, "db": db}


def snapshot_counts(pg: dict) -> dict:
    """备份前各表行数快照（一致性核对用）。"""
    import psycopg2

    conn = psycopg2.connect(host=pg["host"], port=pg["port"], user=pg["user"],
                            password=pg["password"], dbname=pg["db"])
    try:
        with conn.cursor() as cur:
            tables = ["users", "sessions", "messages", "crisis_audit", "prompts",
                      "compare_history", "user_chat_history", "langchain_pg_embedding",
                      "langchain_pg_collection"]
            counts = {}
            for t in tables:
                cur.execute(f'SELECT COUNT(*) FROM "{t}"')
                counts[t] = cur.fetchone()[0]
            return counts
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="PostgreSQL 全量备份（含 pgvector）")
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "backups"), help="备份目录（默认项目 backups/）")
    ap.add_argument("--keep", type=int, default=0, help="仅保留最近 N 份（0=全留）")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pg = _pg_args_from_url(settings.DB_URL)
    if not settings.DB_URL.startswith("postgresql"):
        raise SystemExit(f"当前 DB_BACKEND 不是 postgres（{settings.DB_URL}），备份仅支持 PostgreSQL 后端")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = outdir / f"rag_psychology_{stamp}.dump"

    # 1) pg_dump 全量（自定义格式，含 vector 类型表）
    # 注意：Windows 下裸命令 pg_dump 可能命中 shim（报 "could not find a pg_dump"），
    # 必须用绝对路径（shutil.which 解析 PATH 中的真实 pg_dump.exe）
    import os
    import shutil

    pg_dump_bin = shutil.which("pg_dump") or ""
    if not pg_dump_bin:
        raise SystemExit("未找到 pg_dump，请确认 PostgreSQL bin 目录在 PATH 中")
    # 必须继承完整环境（Windows 下仅传 PGPASSWORD 会丢 PATH，pg_dump 内部找不到辅助程序）
    env = {**os.environ, "PGPASSWORD": pg["password"]}
    cmd = [pg_dump_bin, "-h", pg["host"], "-p", pg["port"], "-U", pg["user"],
           "-F", "c", "-f", str(dump_path), pg["db"]]
    print(f"备份中：{dump_path}（{pg_dump_bin}）")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[backup] pg_dump 失败，stderr={e.stderr!r}", flush=True)
        raise

    # 2) 行数快照
    counts = snapshot_counts(pg)
    snap_path = dump_path.with_suffix(".snapshot.json")
    snap_path.write_text(json.dumps({"time": stamp, "counts": counts}, ensure_ascii=False, indent=2), encoding="utf-8")

    size_mb = dump_path.stat().st_size / 1024 / 1024
    print(f"备份完成：{dump_path.name}（{size_mb:.1f} MB）")
    print(f"行数快照：{snap_path.name} → {counts}")

    # 3) 清理旧备份（按 --keep）
    if args.keep > 0:
        dumps = sorted(outdir.glob("rag_psychology_*.dump"))
        for old in dumps[:-args.keep]:
            old.unlink()
            old.with_suffix(".snapshot.json").unlink(missing_ok=True)
            print(f"已清理旧备份：{old.name}")


if __name__ == "__main__":
    main()

"""测试点 6：数据备份与恢复 —— 灾难恢复演练（RPO / RTO / 一致性）。

流程：
  1. 取最近一份全量备份（dump + 行数快照）；
  2. 记录当前行数；随后【备份后】写入带标记的 RPO 测试数据（模拟备份完成后的新写入）；
  3. 故意破坏：DROP 全部表（关系表 + pgvector 向量表）；
  4. 从 dump 恢复（计时 → RTO）；
  5. 核对：恢复后行数 vs 备份快照必须完全一致；RPO 期间新写入的数据确实丢失（记录丢失量）；
  6. 应用可用性：init_db 幂等 + 管理员登录。

运行（会真的删库并恢复，请确保已存在可用备份）：
  python tests/test_backup_restore.py
  python tests/test_backup_restore.py --dump-file backups/rag_psychology_xxx.dump
  python tests/test_backup_restore.py --report tests/results/backup-restore.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = ""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} {extra}")


def _pg(settings_url: str) -> dict:
    import urllib.parse

    rest = settings_url.split("://", 1)[1]
    cred, hostpart = rest.rsplit("@", 1)
    user, _, pw = cred.partition(":")
    host, _, db = hostpart.partition("/")
    db = db.split("?")[0]
    h, _, port = host.partition(":")
    return {"host": h, "port": port, "user": user, "password": pw, "db": db}


def table_counts(pg: dict) -> dict:
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
                try:
                    cur.execute(f'SELECT COUNT(*) FROM "{t}"')
                    counts[t] = cur.fetchone()[0]
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()  # 表不存在 → 终止失败事务，后续查询继续
                    counts[t] = 0  # 表不存在视为空（删库后校验用）
            return counts
    finally:
        conn.close()


ALL_TABLES = [
    "messages", "sessions", "crisis_audit", "prompts", "compare_history", "users",
    "user_chat_history", "langchain_pg_embedding", "langchain_pg_collection",
]


def drop_all_tables(pg: dict) -> None:
    import psycopg2

    conn = psycopg2.connect(host=pg["host"], port=pg["port"], user=pg["user"],
                            password=pg["password"], dbname=pg["db"])
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for t in ALL_TABLES:
                cur.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
            # 扩展保留（vector 类型依赖，恢复时会 CREATE EXTENSION IF NOT EXISTS）
            print("  已 DROP 全部表：", ", ".join(ALL_TABLES))
    finally:
        conn.close()


def write_rpo_data(pg: dict, suffix: str) -> int:
    """备份后写入带标记的新数据（模拟备份完成后的新写入），返回写入的 users 数。"""
    import psycopg2
    from modules.security import hash_password

    conn = psycopg2.connect(host=pg["host"], port=pg["port"], user=pg["user"],
                            password=pg["password"], dbname=pg["db"])
    try:
        with conn.cursor() as cur:
            n = 3
            for i in range(n):
                u = f"rpo_{i}_{suffix}"
                uid = str(uuid.uuid4()).replace("-", "")
                cur.execute(
                    "INSERT INTO users (id, username, password_hash, display_name, role, is_active, created_at, updated_at) "
                    "VALUES (%s,%s,%s,%s,'user',true,now(),now())",
                    (uid, u, hash_password("Rpo@123456"), f"RPO用户{i}"),
                )
            conn.commit()
            return n
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="数据备份与恢复灾难演练")
    ap.add_argument("--dump-file", default="", help="指定备份 dump（默认取 backups/ 最近一份）")
    ap.add_argument("--report", default="tests/results/backup-restore.json")
    args = ap.parse_args()

    if not settings.DB_URL.startswith("postgresql"):
        raise SystemExit("演练仅支持 PostgreSQL 后端（DB_BACKEND=postgres）")
    pg = _pg(settings.DB_URL)

    # 1) 定位备份文件
    backups_dir = PROJECT_ROOT / "backups"
    if args.dump_file:
        dump_path = Path(args.dump_file)
    else:
        dumps = sorted(backups_dir.glob("rag_psychology_*.dump"))
        if not dumps:
            raise SystemExit(f"backups/ 下没有备份，请先运行 scripts/backup_pg.py")
        dump_path = dumps[-1]
    snap_path = dump_path.with_suffix(".snapshot.json")
    backup_snapshot = json.loads(snap_path.read_text(encoding="utf-8"))["counts"] if snap_path.exists() else None
    print(f"使用备份：{dump_path.name}（{dump_path.stat().st_size/1024/1024:.1f} MB）")
    if backup_snapshot:
        print(f"备份快照行数：{backup_snapshot}")

    # 2) 当前状态 + 写入 RPO 数据
    print("\n== 1/5 备份后写入 RPO 测试数据 ==")
    cur_before = table_counts(pg)
    suffix = uuid.uuid4().hex[:6]
    rpo_users = write_rpo_data(pg, suffix)
    print(f"  RPO 期间写入 {rpo_users} 个用户（rpo_*_{suffix}），预期恢复后丢失")

    # 3) 破坏
    print("\n== 2/5 故意删库（DROP 全部表） ==")
    drop_all_tables(pg)
    after_drop = table_counts(pg)
    check("删库后所有表为空（破坏成功）", all(v == 0 for v in after_drop.values()), f"{after_drop}")

    # 4) 恢复 + 服务重启 + 健康检查（完整 RTO）
    print("\n== 3/5 从备份恢复 + 服务重启（完整 RTO） ==")
    import shutil as _sh
    pg_restore_bin = _sh.which("pg_restore")
    if not pg_restore_bin:
        raise SystemExit("未找到 pg_restore")
    env = {**os.environ, "PGPASSWORD": pg["password"]}
    cmd = [pg_restore_bin, "-h", pg["host"], "-p", pg["port"], "-U", pg["user"],
           "-d", pg["db"], "--no-owner", str(dump_path)]
    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True)
        restore_s = time.time() - t0
        print(f"  数据恢复完成：{restore_s:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"[restore] pg_restore 失败：{e.stderr[:400]}")
        raise

    # 服务重启：启动 uvicorn（同进程）→ 等待 /api/health 200 → 完整 RTO
    import asyncio
    import httpx
    import uvicorn
    from api.main import app as fastapi_app

    async def _wait_health(base: str, timeout: float = 60.0):
        t0w = time.time()
        async with httpx.AsyncClient(timeout=5) as c:
            while time.time() - t0w < timeout:
                try:
                    r = await c.get(f"{base}/api/health")
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.3)
        raise RuntimeError("服务健康检查超时")

    async def _serve_and_wait(port: int):
        config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        t_srv = time.time()
        await _wait_health(f"http://127.0.0.1:{port}")
        serve_s = time.time() - t_srv
        server.should_exit = True
        await task
        return serve_s

    serve_s = asyncio.run(_serve_and_wait(8028))
    rto_full = (time.time() - t0)  # 从 pg_restore 开始 → 服务健康
    print(f"  服务重启至 /api/health 200：{serve_s:.1f}s")
    print(f"  完整 RTO（数据恢复 + 服务重启）= {rto_full:.1f}s")

    # 5) 一致性核对
    print("\n== 4/5 一致性核对 ==")
    cur_after = table_counts(pg)
    if backup_snapshot:
        for table, expected in backup_snapshot.items():
            got = cur_after.get(table)
            check(f"[一致性] {table}: 备份={expected} 恢复后={got}", got == expected, f"got={got} expected={expected}")
    else:
        print("  （无备份快照，跳过逐表对照；仅检查关键表非空）")
        check("users 恢复非空", cur_after["users"] > 0, f"{cur_after['users']}")
        check("pgvector 向量恢复", cur_after["langchain_pg_embedding"] > 0, f"{cur_after['langchain_pg_embedding']}")
    # RPO：备份后写入的 rpo_* 用户应不存在
    import psycopg2

    conn = psycopg2.connect(host=pg["host"], port=pg["port"], user=pg["user"],
                            password=pg["password"], dbname=pg["db"])
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users WHERE username LIKE %s", (f"rpo_%_{suffix}",))
        rpo_left = cur.fetchone()[0]
    conn.close()
    check(f"[RPO] 备份后写入的 {rpo_users} 个用户已丢失（未混入恢复数据）", rpo_left == 0, f"left={rpo_left}")

    # 6) 应用可用性
    print("\n== 5/5 应用可用性 ==")
    from db import SessionLocal, init_db
    from db import crud

    init_db()
    with SessionLocal() as db:
        admin = crud.get_user_by_username(db, settings.INIT_ADMIN_USERNAME)
        check("init_db 幂等 + 管理员账号存在", admin is not None, "admin 缺失")
    rpo_report = {
        "dump": dump_path.name,
        "rpo_seconds_window": "备份时刻→破坏时刻（本次约 0s 后写入即删）",
        "rpo_lost_users": rpo_users,
        "rpo_lost_rows": rpo_users,
        "rto_data_restore_seconds": round(restore_s, 2),
        "rto_service_start_seconds": round(serve_s, 2),
        "rto_full_seconds": round(rto_full, 2),
        "tables_consistent": len(PASS) - 1,  # 占位，真实一致性以逐表 PASS 为准
        "passed": PASS,
        "failed": FAIL,
    }
    print(f"\n{'=' * 56}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    print(f"RPO = 备份完成后新写入 {rpo_users} 个用户（恢复后丢失，符合预期）")
    print(f"完整 RTO（数据恢复 + 服务重启）= {rto_full:.1f}s"
          f"（数据恢复 {restore_s:.1f}s + 服务重启 {serve_s:.1f}s，备份 {dump_path.stat().st_size/1024/1024:.1f} MB）")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
    else:
        print("全部通过 ✓  数据备份与恢复演练成功")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rpo_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告：{args.report}")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()

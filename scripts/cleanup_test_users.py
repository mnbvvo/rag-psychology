"""清理压测/联调产生的测试账号及其数据（幂等，安全范围 = 明确前缀匹配）。

用法：
  # 干跑：只统计不删除
  python scripts/cleanup_test_users.py
  # 实际删除（默认前缀集合为历年压测账号前缀）
  python scripts/cleanup_test_users.py --commit
  # 自定义前缀
  python scripts/cleanup_test_users.py --prefixes adm_,loadtest_,e2e_ --commit

安全约束：
- 只删除 username 命中前缀（已用下划线结尾区分，避免误伤如 legacy/admin/真实账号）；
- 级联清理其 sessions/messages、crisis_audit、user_chat_history；
- 默认 --dry-run；必须显式 --commit 才落库。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from db import SessionLocal

DEFAULT_PREFIXES = (
    "adm_",       # 准入 AdmissionController 冒烟
    "loadtest_",  # scripts/concurrency_test.py --num-users
    "bg_",        # 后台 Worker 冒烟
    "e2e_",       # 端到端冒烟
    "w3_",        # Wave3 数据层 async 冒烟
    "w3b_",
    "probe_",     # 诊断探针
    "dbg_",       # 诊断
    "p1async_",   # P1 非流式 async 冒烟
)


def _collect(db, prefixes):
    """前缀匹配查询（必须转义下划线：SQL LIKE 中 _ 是单字符通配符，
    否则 'adm_%' 会误匹配 'admin' 等真实账号）。"""
    conds = []
    params = {}
    for i, p in enumerate(prefixes):
        # 转义 _ 为字面量，再加 % 前缀后缀匹配
        conds.append(f"username LIKE :p{i} ESCAPE '\\'")
        params[f"p{i}"] = p.replace("_", "\\_") + "%"
    sql = (
        "SELECT id, username, role, is_active, created_at FROM users WHERE "
        + " OR ".join(conds)
        + " ORDER BY username"
    )
    return db.execute(text(sql), params).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES), help="逗号分隔的用户名前缀")
    ap.add_argument("--commit", action="store_true", help="实际删除（默认干跑）")
    args = ap.parse_args()
    prefixes = tuple(p for p in args.prefixes.split(",") if p.strip())

    # —— 收集（只读 session）——
    with SessionLocal() as db:
        users = _collect(db, prefixes)
    print(f"前缀集合: {prefixes}")
    print(f"命中用户: {len(users)}")
    for u in users:
        print(f"  - {u.username} (id={u.id[:8]}…, role={u.role}, active={u.is_active})")
    if not users:
        print("无需清理。")
        return 0
    if not args.commit:
        print("\n[dry-run] 未删除任何数据；加 --commit 实际清理。")
        return 0

    # —— 删除（独立 session + begin，避免与收集事务嵌套）——
    ids = [u.id for u in users]
    counts = {}
    with SessionLocal() as db:
        with db.begin():
            counts["messages"] = db.execute(
                text(
                    "DELETE FROM messages WHERE session_id IN "
                    "(SELECT id FROM sessions WHERE user_id = ANY(:ids))"
                ),
                {"ids": ids},
            ).rowcount
            counts["sessions"] = db.execute(
                text("DELETE FROM sessions WHERE user_id = ANY(:ids)"), {"ids": ids}
            ).rowcount
            counts["crisis_audit"] = db.execute(
                text("DELETE FROM crisis_audit WHERE user_id = ANY(:ids)"), {"ids": ids}
            ).rowcount
            counts["user_chat_history"] = db.execute(
                text("DELETE FROM user_chat_history WHERE user_id = ANY(:ids)"), {"ids": ids}
            ).rowcount
            counts["users"] = db.execute(
                text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": ids}
            ).rowcount
    print(f"\n已删除（commit）: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

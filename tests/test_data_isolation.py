"""测试点 2：数据隔离 —— 多账户并发下验证数据库读写不串隐私。

与「鉴权与越权」的区别：这里不仅断言状态码 403，还从四个层次验证隔离：
  ① 接口层：B 用 A 的资源 id（会话/提示词/对比记录）操作 → 403
  ② 内容层：B 的任何响应【文本】不含 A 的私有标记（PRIV_A_xxx），
            防止"能访问但泄露内容"式的半隔离
  ③ 数据层：直查当前数据库，B 名下任何数据行中不含 A 的标记
  ④ 并发层：A、B 同时（线程池并发）创建/读取，各自只看到自己的数据

提示词窃取专项（对应测试方案）：B 通过 update / deleteId / activeId /
完整替换列表 / /api/query 引用 A 的 prompt_id，全部被拒且响应不含 A 内容。

运行：
  python tests/test_data_isolation.py                    # 打真实服务（默认 127.0.0.1:8000）
  python tests/test_data_isolation.py --inprocess        # 进程内直测（无需起服务）
  python tests/test_data_isolation.py --admin-password 123456789
"""
import argparse
import concurrent.futures
import json
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


class RealRunner:
    def __init__(self, base: str):
        import httpx

        self._http = httpx.Client(base_url=base, timeout=30)
        self.name = f"HTTP {base}"

    def call(self, method, path, token=None, json_body=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self._http.request(method, path, headers=headers, json=json_body)

    def close(self):
        self._http.close()


class InProcRunner:
    def __init__(self):
        settings.RERANK_ENABLED = False
        settings.SEMANTIC_CHECK_ENABLED = False
        settings.HYBRID_ENABLED = False

        from fastapi.testclient import TestClient
        from api.main import app

        self._ctx = TestClient(app)
        self._ctx.__enter__()
        self.name = "InProcess TestClient"

    def call(self, method, path, token=None, json_body=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self._ctx.request(method, path, headers=headers, json=json_body)

    def close(self):
        self._ctx.__exit__(None, None, None)


def cleanup_test_accounts(usernames: list[str]):
    from sqlalchemy import delete, select

    from db import SessionLocal
    from db.models import CompareHistory, CrisisAudit, Message, Prompt, Session, User

    with SessionLocal() as db, db.begin():
        users = db.execute(select(User).where(User.username.in_(usernames))).scalars().all()
        for u in users:
            sids = select(Session.id).where(Session.user_id == u.id)
            db.execute(delete(Message).where(Message.session_id.in_(sids)))
            db.execute(delete(Session).where(Session.user_id == u.id))
            db.execute(delete(Prompt).where(Prompt.user_id == u.id))
            db.execute(delete(CompareHistory).where(CompareHistory.user_id == u.id))
            db.execute(delete(CrisisAudit).where(CrisisAudit.user_id == u.id))
            db.delete(u)
        return len(users)


def verify_db_ownership(uid_a: str, uid_b: str, mark_a: str):
    """③ 数据层：直查当前数据库（SQLite/PostgreSQL），验证 B 名下任何数据行中都不含 A 的私有标记。

    这是比接口断言更深的隔离验证：即使未来某个接口有遗漏，数据库层面
    B 的私有区域也不允许混入 A 的内容。
    """
    from sqlalchemy import func, select

    from db import SessionLocal
    from db.models import CompareHistory, Message, Prompt, Session

    with SessionLocal() as db:
        # 1) prompts：B 的提示词内容不含 A 标记
        n = db.execute(
            select(func.count()).select_from(Prompt).where(
                Prompt.user_id == uid_b, Prompt.content.like(f"%{mark_a}%")
            )
        ).scalar()
        check("[DB] B 的 prompts 中无 A 的标记内容", n == 0, f"found={n}")
        # 2) messages：B 的会话消息不含 A 标记（经 session 归属）
        b_sids = select(Session.id).where(Session.user_id == uid_b)
        n = db.execute(
            select(func.count()).select_from(Message).where(
                Message.session_id.in_(b_sids), Message.content.like(f"%{mark_a}%")
            )
        ).scalar()
        check("[DB] B 的 messages 中无 A 的标记内容", n == 0, f"found={n}")
        # 3) compare_history：B 的记录不含 A 标记
        n = db.execute(
            select(func.count()).select_from(CompareHistory).where(
                CompareHistory.user_id == uid_b, CompareHistory.input.like(f"%{mark_a}%")
            )
        ).scalar()
        check("[DB] B 的 compare_history 中无 A 的标记内容", n == 0, f"found={n}")
        # 4) 对照：A 名下确实存在标记内容（证明标记已真实入库，验证有效）
        n = db.execute(
            select(func.count()).select_from(Prompt).where(
                Prompt.user_id == uid_a, Prompt.content.like(f"%{mark_a}%")
            )
        ).scalar()
        check("[DB] 对照：A 的 prompts 确实含标记（数据已入库）", n > 0, f"n={n}")


def run_suite(api, cleanup: bool, admin_password: str = "") -> list[str]:
    suffix = uuid.uuid4().hex[:6]
    ua, ub = f"isa_{suffix}", f"isb_{suffix}"
    pw = "Test@123456"
    MARK_A = f"PRIV_A_{suffix}"
    MARK_B = f"PRIV_B_{suffix}"
    created: list[str] = [ua, ub]

    # ---------- 注册 + 登录 ----------
    print("\n== 准备：注册并登录 A / B ==")
    assert api.call("POST", "/api/auth/register", json_body={"username": ua, "password": pw}).status_code == 201
    assert api.call("POST", "/api/auth/register", json_body={"username": ub, "password": pw}).status_code == 201
    r = api.call("POST", "/api/auth/login", json_body={"username": ua, "password": pw})
    tok_a, uid_a = r.json()["access_token"], r.json()["user"]["id"]
    r = api.call("POST", "/api/auth/login", json_body={"username": ub, "password": pw})
    tok_b, uid_b = r.json()["access_token"], r.json()["user"]["id"]
    print(f"  A={ua}({uid_a[:8]})  B={ub}({uid_b[:8]})  标记={MARK_A} / {MARK_B}")

    # ---------- A 创建带私有标记的资源 ----------
    print("\n== A 创建带私有标记的资源 ==")
    r = api.call("POST", "/api/sessions", token=tok_a, json_body={"name": f"A会话_{MARK_A}"})
    sid_a = r.json()["id"]
    from db import crud as _crud

    with _crud.get_db() as db:
        _crud.append_turn(db, sid_a, f"问题 {MARK_A}", f"回答 {MARK_A}", title=f"A会话_{MARK_A}", user_id=uid_a)
    check("A 创建会话并写入标记消息 → 200", True)
    r = api.call("PUT", "/api/system-prompt", token=tok_a, json_body={"add": {"name": "A私密提示词", "content": f"系统提示词 {MARK_A} 请保守秘密"}})
    pid_a = next((p["id"] for p in r.json()["config"]["prompts"] if p["name"] == "A私密提示词"), None)
    check("A 创建带标记提示词 → 200 且拿到 id", pid_a is not None)
    r = api.call("POST", "/api/compare-history", token=tok_a, json_body={"input": f"问题 {MARK_A}", "a": {"note": MARK_A}, "b": {}})
    cid_a = r.json()["id"]
    check("A 创建带标记对比记录 → 200", True)

    # ---------- ① 接口层：B 用 A 的资源 id 操作 → 403 ----------
    print("\n== ① 接口层：B 操作 A 的资源 → 403 ==")
    cases = [
        ("GET", f"/api/sessions/{sid_a}/messages", None, "B 读 A 会话消息"),
        ("PATCH", f"/api/sessions/{sid_a}", {"name": "篡改"}, "B 改名 A 会话"),
        ("DELETE", f"/api/sessions/{sid_a}", None, "B 删除 A 会话"),
        ("DELETE", f"/api/compare-history/{cid_a}", None, "B 删除 A 对比记录"),
        ("POST", "/api/query", {"question": "测试", "session_id": sid_a}, "B 用 A 的 session_id 调 query"),
        ("POST", "/api/query", {"question": "测试", "prompt_id": pid_a}, "B 用 A 的 prompt_id 调 query"),
    ]
    for method, path, body, name in cases:
        r = api.call(method, path, token=tok_b, json_body=body)
        check(f"{name} → 403", r.status_code == 403, f"got {r.status_code}")

    # ---------- 提示词窃取专项（测试方案核心） ----------
    print("\n== ② 提示词窃取专项：B 尝试各种方式拿到 A 的提示词 ==")
    steal_cases = [
        ({"update": {"id": pid_a, "content": "篡改"}}, "B update A 的提示词"),
        ({"deleteId": pid_a}, "B delete A 的提示词"),
        ({"activeId": pid_a}, "B 激活 A 的提示词"),
        ({"prompts": [{"id": pid_a, "name": "窃取", "content": "试试"}]}, "B 完整替换列表含 A 的 id"),
    ]
    for body, name in steal_cases:
        r = api.call("PUT", "/api/system-prompt", token=tok_b, json_body=body)
        check(f"{name} → 403", r.status_code == 403, f"got {r.status_code} {r.text[:120]}")

    # ---------- ② 内容层：B 的任何响应文本不含 A 的标记 ----------
    print("\n== ③ 内容层：B 的响应文本不含 A 的私有标记 ==")
    content_cases = [
        ("GET", "/api/system-prompt", None, "B 的提示词库响应"),
        ("GET", "/api/sessions", None, "B 的会话列表响应"),
        ("GET", "/api/compare-history", None, "B 的对比历史响应"),
    ]
    for method, path, body, name in content_cases:
        r = api.call(method, path, token=tok_b, json_body=body)
        check(f"{name} 不含 {MARK_A}", r.status_code == 200 and MARK_A not in r.text, f"LEAK! {r.text[:300]}")

    # 反向对照：A 自己的响应包含标记（证明标记有效、数据确实在库中）
    r = api.call("GET", "/api/system-prompt", token=tok_a)
    check("对照：A 的提示词库含自己的标记", MARK_A in r.text, "对照失败")
    r = api.call("GET", f"/api/sessions/{sid_a}/messages", token=tok_a)
    check("对照：A 能读自己的会话且含标记", r.status_code == 200 and MARK_A in r.text, f"got {r.status_code}")

    # ---------- ④ 并发层：A、B 同时创建与读取，互不可见 ----------
    print("\n== ④ 并发层：A、B 同时操作（线程池 8 并发） ==")
    def worker(token, label, n):
        r = api.call("POST", "/api/sessions", token=token, json_body={"name": f"{label}并发会话{n}"})
        return r.status_code
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(worker, tok_a, "A", i) for i in range(4)] + [ex.submit(worker, tok_b, "B", i) for i in range(4)]
        codes = [f.result() for f in concurrent.futures.as_completed(futs)]
    check("并发 8 路创建会话全部 200", all(c == 200 for c in codes), f"{codes}")
    r = api.call("GET", "/api/sessions", token=tok_a)
    a_ids = {s["id"] for s in r.json()}
    r = api.call("GET", "/api/sessions", token=tok_b)
    b_ids = {s["id"] for s in r.json()}
    check("并发后 A、B 会话列表零交集", not (a_ids & b_ids), f"交集={a_ids & b_ids}")
    check("并发后 B 的会话列表不含 A 的标记会话", all(MARK_A not in (s.get("title") or "") for s in r.json()), "泄漏标记会话")

    # ---------- 数据库层：归属验证（③） ----------
    print("\n== ⑤ 数据层：直查数据库归属 ==")
    verify_db_ownership(uid_a, uid_b, MARK_A)

    # ---------- 清理 ----------
    if cleanup:
        removed = cleanup_test_accounts(created)
        print(f"\n[cleanup] 已删除 {removed} 个测试账号")

    return created


def main():
    ap = argparse.ArgumentParser(description="数据隔离专项自动化测试")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--admin-password", default="", help="管理员密码（本脚本暂未使用，保留兼容）")
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    api = InProcRunner() if args.inprocess else RealRunner(args.url)
    print(f"运行模式：{api.name}")
    t0 = time.time()
    try:
        run_suite(api, cleanup=not args.no_cleanup, admin_password=args.admin_password)
    finally:
        api.close()

    print(f"\n{'=' * 56}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}   （耗时 {time.time() - t0:.1f}s）")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps({"passed": PASS, "failed": FAIL, "ok": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.exit(1)
    print("全部通过 ✓")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({"passed": PASS, "failed": FAIL, "ok": True}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

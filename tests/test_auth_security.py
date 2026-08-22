"""测试点 1：鉴权与越权 —— 自动化验收测试。

两种运行模式（行为等价 Postman）：
  1) 默认：打真实运行中的服务（httpx）
        python tests/test_auth_security.py [--url http://127.0.0.1:8000]
  2) 进程内：FastAPI TestClient 直测（无需起服务；自动关闭重排/语义预热）
        python tests/test_auth_security.py --inprocess

覆盖断言：
  - 无 token 访问受保护接口 → 401
  - 水平越权：A 的 token 访问 B 的会话/提示词/对比记录 → 403
  - 垂直越权：普通 token 访问 /api/admin/* → 403；管理员 → 200
  - 请求体篡改 user_id → 403
  - 数据隔离：A 写入的数据 B 查不到

默认测试结束自动清理测试账号（--no-cleanup 关闭）。
"""
import argparse
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
    """打真实服务（httpx），等价 Postman 行为。"""

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
    """进程内直测（TestClient），无需起服务；自动关闭重排/语义预热。"""

    def __init__(self):
        settings.RERANK_ENABLED = False
        settings.SEMANTIC_CHECK_ENABLED = False
        settings.HYBRID_ENABLED = False

        from fastapi.testclient import TestClient
        from api.main import app

        self._ctx = TestClient(app)
        self._ctx.__enter__()  # 触发 startup（建表 / 迁移 / 引导账号）
        self.name = "InProcess TestClient"

    def call(self, method, path, token=None, json_body=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self._ctx.request(method, path, headers=headers, json=json_body)

    def close(self):
        self._ctx.__exit__(None, None, None)


def _bearer(tok):  # 兼容语义（runner 内已处理，保留占位）
    return tok


def cleanup_test_accounts(usernames: list[str]):
    """删除测试账号及其业务数据（幂等；不影响 admin / legacy）。"""
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


def run_suite(api, cleanup: bool, admin_password: str = "") -> list[str]:
    suffix = uuid.uuid4().hex[:6]
    ua, ub = f"user_a_{suffix}", f"user_b_{suffix}"
    pw = "Test@123456"
    admin_pw = admin_password or settings.INIT_ADMIN_PASSWORD
    created: list[str] = []

    # ============ 1. 注册 ============
    print("\n== 注册 ==")
    r = api.call("POST", "/api/auth/register", json_body={"username": ua, "password": pw, "display_name": "A用户"})
    check("注册 A → 201", r.status_code == 201, f"got {r.status_code} {r.text[:120]}")
    created.append(ua)
    r = api.call("POST", "/api/auth/register", json_body={"username": ub, "password": pw, "display_name": "B用户"})
    check("注册 B → 201", r.status_code == 201, f"got {r.status_code}")
    created.append(ub)
    r = api.call("POST", "/api/auth/register", json_body={"username": ua, "password": pw})
    check("重复注册 A → 409", r.status_code == 409, f"got {r.status_code}")
    r = api.call("POST", "/api/auth/register", json_body={"username": "ab", "password": pw})
    check("非法用户名 → 400", r.status_code == 400, f"got {r.status_code}")
    r = api.call("POST", "/api/auth/register", json_body={"username": f"weak_{suffix}", "password": "123"})
    check("弱密码 → 400", r.status_code == 400, f"got {r.status_code}")

    # ============ 2. 无 token → 401 ============
    print("\n== 无 token 访问受保护接口 → 401 ==")
    noauth_cases = [
        ("GET", "/api/sessions"),
        ("GET", "/api/system-prompt"),
        ("GET", "/api/compare-history"),
        ("GET", "/api/auth/me"),
        ("GET", f"/api/sessions/{uuid.uuid4().hex}/messages"),
        ("GET", "/api/admin/users"),
        ("POST", "/api/query"),
    ]
    for method, path in noauth_cases:
        body = {"question": "测试"} if method == "POST" and path == "/api/query" else None
        r = api.call(method, path, json_body=body)
        check(f"无 token {method} {path} → 401", r.status_code == 401, f"got {r.status_code}")

    # ============ 3. 登录 ============
    print("\n== 登录 ==")
    r = api.call("POST", "/api/auth/login", json_body={"username": ua, "password": pw})
    check("登录 A → 200", r.status_code == 200, f"got {r.status_code}")
    tok_a = r.json()["access_token"]
    r = api.call("POST", "/api/auth/login", json_body={"username": ub, "password": pw})
    check("登录 B → 200", r.status_code == 200, f"got {r.status_code}")
    tok_b = r.json()["access_token"]
    r = api.call("POST", "/api/auth/login", json_body={"username": ua, "password": "wrong-pass"})
    check("错误密码 → 401", r.status_code == 401, f"got {r.status_code}")
    r = api.call("GET", "/api/auth/me", token=tok_a)
    check("A 的 me → 200 且用户名匹配", r.status_code == 200 and r.json()["username"] == ua, f"got {r.status_code}")
    uid_a = r.json()["id"]
    r = api.call("GET", "/api/auth/me", token=tok_b)
    uid_b = r.json()["id"]

    # ============ 4. A 创建资源（会话/对比/提示词） ============
    print("\n== A 创建资源 ==")
    r = api.call("POST", "/api/sessions", token=tok_a, json_body={"name": "A的会话"})
    check("A 新建会话 → 200", r.status_code == 200, f"got {r.status_code}")
    sid_a = r.json()["id"]
    r = api.call("POST", "/api/compare-history", token=tok_a, json_body={"input": "测试问题", "a": {"x": 1}, "b": {"x": 2}})
    check("A 新建对比记录 → 200", r.status_code == 200, f"got {r.status_code}")
    cid_a = r.json()["id"]
    r = api.call("PUT", "/api/system-prompt", token=tok_a, json_body={"add": {"name": "A的提示词", "content": "A 私密内容"}})
    check("A 新增提示词 → 200", r.status_code == 200, f"got {r.status_code}")
    pid_a = next((p["id"] for p in r.json()["config"]["prompts"] if p["name"] == "A的提示词"), None)
    check("A 的提示词 id 已返回", pid_a is not None, f"config={r.text[:200]}")

    # ============ 5. 水平越权：B 访问 A 的资源 → 403 ============
    print("\n== 水平越权（B 访问 A 的资源 → 403） ==")
    r = api.call("GET", f"/api/sessions/{sid_a}/messages", token=tok_b)
    check("B 读 A 的会话消息 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("PATCH", f"/api/sessions/{sid_a}", token=tok_b, json_body={"name": "篡改"})
    check("B 改名 A 的会话 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("DELETE", f"/api/sessions/{sid_a}", token=tok_b)
    check("B 删除 A 的会话 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("DELETE", f"/api/compare-history/{cid_a}", token=tok_b)
    check("B 删除 A 的对比记录 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("PUT", "/api/system-prompt", token=tok_b, json_body={"update": {"id": pid_a, "content": "篡改"}})
    check("B 更新 A 的提示词 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("PUT", "/api/system-prompt", token=tok_b, json_body={"deleteId": pid_a})
    check("B 删除 A 的提示词 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("PUT", "/api/system-prompt", token=tok_b, json_body={"activeId": pid_a})
    check("B 激活 A 的提示词 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("POST", "/api/query", token=tok_b, json_body={"question": "测试", "session_id": sid_a})
    check("B 用 A 的 session_id 调 query → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("POST", "/api/query", token=tok_b, json_body={"question": "测试", "prompt_id": pid_a})
    check("B 用 A 的 prompt_id 调 query → 403", r.status_code == 403, f"got {r.status_code}")

    # ============ 6. 垂直越权：普通用户访问 admin → 403 ============
    print("\n== 垂直越权（admin 接口） ==")
    r = api.call("GET", "/api/admin/users", token=tok_b)
    check("普通用户查用户列表 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("GET", "/api/admin/crisis-audit", token=tok_b)
    check("普通用户查危机审计 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("POST", "/api/auth/login", json_body={"username": settings.INIT_ADMIN_USERNAME, "password": admin_pw})
    check("管理员登录 → 200", r.status_code == 200, f"got {r.status_code} (若密码不同请用 --admin-password 指定)")
    tok_admin = r.json().get("access_token") if r.status_code == 200 else None
    if not tok_admin:
        check("管理员查用户列表 → 200", False, "（跳过：管理员未登录成功）")
        check("管理员查危机审计 → 200", False, "（跳过：管理员未登录成功）")
    else:
        r = api.call("GET", "/api/admin/users", token=tok_admin)
        names = {u["username"] for u in r.json()}
        check("管理员查用户列表 → 200 且含 A/B", r.status_code == 200 and ua in names and ub in names, f"got {r.status_code}")
        r = api.call("GET", "/api/admin/crisis-audit", token=tok_admin)
        check("管理员查危机审计 → 200", r.status_code == 200, f"got {r.status_code}")

    # ============ 7. 请求体篡改 user_id → 403 ============
    print("\n== 请求体篡改 user_id ==")
    r = api.call("POST", "/api/query", token=tok_a, json_body={"question": "测试", "user_id": uid_b})
    check("A 以 B 的 user_id 调 query → 403", r.status_code == 403, f"got {r.status_code}")

    # ============ 8. 数据隔离：A 写入的数据 B 查不到 ============
    print("\n== 数据隔离 ==")
    r = api.call("GET", "/api/sessions", token=tok_b)
    check("B 的会话列表不含 A 的会话", sid_a not in [s["id"] for s in r.json()], f"leak: {r.text[:200]}")
    r = api.call("GET", "/api/system-prompt", token=tok_b)
    check("B 的提示词库不含 A 的提示词", pid_a not in [p["id"] for p in r.json()["current"]["prompts"]], f"leak: {r.text[:200]}")
    r = api.call("GET", "/api/system-prompt", token=tok_a)
    check("A 的提示词库含自己的提示词", pid_a in [p["id"] for p in r.json()["current"]["prompts"]], f"missing: {r.text[:200]}")
    r = api.call("GET", "/api/compare-history", token=tok_b)
    check("B 的对比历史不含 A 的记录", cid_a not in [h["id"] for h in r.json()], f"leak: {r.text[:200]}")
    # A 写入一轮消息后，B 依然读不到；A 自己能读
    from db import crud as _crud

    with _crud.get_db() as db:
        _crud.append_turn(db, sid_a, "A 的私密问题", "A 的私密回答", title="A的会话", user_id=uid_a)
    r = api.call("GET", f"/api/sessions/{sid_a}/messages", token=tok_b)
    check("A 写入消息后 B 读取仍 → 403", r.status_code == 403, f"got {r.status_code}")
    r = api.call("GET", f"/api/sessions/{sid_a}/messages", token=tok_a)
    check("A 读取自己的会话消息 → 200", r.status_code == 200 and any("A 的私密问题" in m["content"] for m in r.json()), f"got {r.status_code}")

    # ============ 9. 登录失败锁定 ============
    print("\n== 登录失败锁定 ==")
    ulock = f"lock_{suffix}"
    api.call("POST", "/api/auth/register", json_body={"username": ulock, "password": pw})
    created.append(ulock)
    for _ in range(settings.LOGIN_MAX_FAILS):
        api.call("POST", "/api/auth/login", json_body={"username": ulock, "password": "bad"})
    r = api.call("POST", "/api/auth/login", json_body={"username": ulock, "password": pw})
    check("连续失败后锁定 → 429", r.status_code == 429, f"got {r.status_code}")

    # ============ 清理 ============
    if cleanup:
        removed = cleanup_test_accounts(created)
        print(f"\n[cleanup] 已删除 {removed} 个测试账号")

    return created


def main():
    ap = argparse.ArgumentParser(description="鉴权与越权自动化验收测试")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="服务地址（默认 http://127.0.0.1:8000）")
    ap.add_argument("--inprocess", action="store_true", help="进程内 TestClient 直测（无需起服务）")
    ap.add_argument("--admin-password", default="123456789", help="管理员密码（默认取 settings.INIT_ADMIN_PASSWORD，若已改过 admin 密码请显式传入）")
    ap.add_argument("--no-cleanup", action="store_true", help="测试结束不清理测试账号")
    ap.add_argument("--report", default="", help="把结果写入 JSON 报告文件（可选）")
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

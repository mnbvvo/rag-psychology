"""登录系统自动化验收测试：鉴权 / 越权 / 数据隔离。

覆盖用户的两组测试方案：
  1. 鉴权与越权：无 token 401、A 访问 B 资源 403（水平越权）、
     普通 token 访问 admin 接口 403（垂直越权）、请求体篡改 user_id 403；
  2. 数据隔离：A 写入的数据 B 查不到；提示词越权窃取 403。

运行方式（测试环境会自动关闭重排/语义预热，避免加载 2.27GB 模型）：
    python scripts/test_auth.py
    （用 FastAPI TestClient 直测 ASGI 应用，等价 Postman 行为，无需起服务）
"""
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

# 测试环境关闭重排/语义/混合预热，避免加载大模型拖慢测试（与业务无关）
settings.RERANK_ENABLED = False
settings.SEMANTIC_CHECK_ENABLED = False
settings.HYBRID_ENABLED = False

from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402
from db import crud  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = ""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} {extra}")


def bearer(tok: str):
    return {"Authorization": f"Bearer {tok}"}


def main():
    suffix = uuid.uuid4().hex[:6]
    ua, ub = f"user_a_{suffix}", f"user_b_{suffix}"
    pw = "Test@123456"

    with TestClient(app) as c:
        # ================= 注册 =================
        print("\n== 注册 ==")
        r = c.post("/api/auth/register", json={"username": ua, "password": pw, "display_name": "A用户"})
        check("注册 A → 201", r.status_code == 201, f"got {r.status_code} {r.text[:100]}")
        r = c.post("/api/auth/register", json={"username": ub, "password": pw, "display_name": "B用户"})
        check("注册 B → 201", r.status_code == 201, f"got {r.status_code}")
        r = c.post("/api/auth/register", json={"username": ua, "password": pw})
        check("重复注册 A → 409", r.status_code == 409, f"got {r.status_code}")
        r = c.post("/api/auth/register", json={"username": "ab", "password": pw})
        check("非法用户名 → 400", r.status_code == 400, f"got {r.status_code}")
        r = c.post("/api/auth/register", json={"username": "short_pw_" + suffix, "password": "123"})
        check("弱密码 → 400", r.status_code == 400, f"got {r.status_code}")

        # ================= 无 token → 401 =================
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
            r = c.request(method, path, json={"question": "测试"} if method == "POST" and path == "/api/query" else None)
            check(f"无 token {method} {path} → 401", r.status_code == 401, f"got {r.status_code}")

        # ================= 登录 =================
        print("\n== 登录 ==")
        r = c.post("/api/auth/login", json={"username": ua, "password": pw})
        check("登录 A → 200", r.status_code == 200, f"got {r.status_code}")
        tok_a = r.json()["access_token"]
        r = c.post("/api/auth/login", json={"username": ub, "password": pw})
        check("登录 B → 200", r.status_code == 200, f"got {r.status_code}")
        tok_b = r.json()["access_token"]
        r = c.post("/api/auth/login", json={"username": ua, "password": "wrong-pass"})
        check("错误密码 → 401", r.status_code == 401, f"got {r.status_code}")
        r = c.get("/api/auth/me", headers=bearer(tok_a))
        check("A 的 me → 200 且用户名匹配", r.status_code == 200 and r.json()["username"] == ua, f"got {r.status_code}")
        uid_a = r.json()["id"]
        r = c.get("/api/auth/me", headers=bearer(tok_b))
        uid_b = r.json()["id"]

        # ================= A 创建资源（会话/对比/提示词） =================
        print("\n== A 创建资源 ==")
        r = c.post("/api/sessions", json={"name": "A的会话"}, headers=bearer(tok_a))
        check("A 新建会话 → 200", r.status_code == 200, f"got {r.status_code}")
        sid_a = r.json()["id"]
        r = c.post("/api/compare-history", json={"input": "测试问题", "a": {"x": 1}, "b": {"x": 2}}, headers=bearer(tok_a))
        check("A 新建对比记录 → 200", r.status_code == 200, f"got {r.status_code}")
        cid_a = r.json()["id"]
        r = c.put("/api/system-prompt", json={"add": {"name": "A的提示词", "content": "A 私密内容"}}, headers=bearer(tok_a))
        check("A 新增提示词 → 200", r.status_code == 200, f"got {r.status_code}")
        pid_a = None
        for p in r.json()["config"]["prompts"]:
            if p["name"] == "A的提示词":
                pid_a = p["id"]
        check("A 的提示词 id 已返回", pid_a is not None, f"config={r.text[:200]}")

        # ================= 水平越权：B 访问 A 的资源 → 403 =================
        print("\n== 水平越权（B 访问 A 的资源 → 403） ==")
        r = c.get(f"/api/sessions/{sid_a}/messages", headers=bearer(tok_b))
        check("B 读 A 的会话消息 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.patch(f"/api/sessions/{sid_a}", json={"name": "篡改"}, headers=bearer(tok_b))
        check("B 改名 A 的会话 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.delete(f"/api/sessions/{sid_a}", headers=bearer(tok_b))
        check("B 删除 A 的会话 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.delete(f"/api/compare-history/{cid_a}", headers=bearer(tok_b))
        check("B 删除 A 的对比记录 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.put("/api/system-prompt", json={"update": {"id": pid_a, "content": "篡改"}}, headers=bearer(tok_b))
        check("B 更新 A 的提示词 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.put("/api/system-prompt", json={"deleteId": pid_a}, headers=bearer(tok_b))
        check("B 删除 A 的提示词 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.put("/api/system-prompt", json={"activeId": pid_a}, headers=bearer(tok_b))
        check("B 激活 A 的提示词 → 403", r.status_code == 403, f"got {r.status_code}")
        # 提示词越权窃取：A 的请求里带 B 的提示词 id / 会话 id（应在调用 LLM 前被 403 拦截）
        r = c.post("/api/query", json={"question": "测试", "session_id": sid_a}, headers=bearer(tok_b))
        check("B 用 A 的 session_id 调 query → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.post("/api/query", json={"question": "测试", "prompt_id": pid_a}, headers=bearer(tok_b))
        check("B 用 A 的 prompt_id 调 query → 403", r.status_code == 403, f"got {r.status_code}")

        # ================= 垂直越权：普通用户访问 admin → 403 =================
        print("\n== 垂直越权（admin 接口） ==")
        r = c.get("/api/admin/users", headers=bearer(tok_b))
        check("普通用户查用户列表 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.get("/api/admin/crisis-audit", headers=bearer(tok_b))
        check("普通用户查危机审计 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.post("/api/auth/login", json={"username": settings.INIT_ADMIN_USERNAME, "password": settings.INIT_ADMIN_PASSWORD})
        check("管理员登录 → 200", r.status_code == 200, f"got {r.status_code}")
        tok_admin = r.json()["access_token"]
        r = c.get("/api/admin/users", headers=bearer(tok_admin))
        names = {u["username"] for u in r.json()}
        check("管理员查用户列表 → 200 且含 A/B", r.status_code == 200 and ua in names and ub in names, f"got {r.status_code}")
        r = c.get("/api/admin/crisis-audit", headers=bearer(tok_admin))
        check("管理员查危机审计 → 200", r.status_code == 200, f"got {r.status_code}")

        # ================= 请求体篡改 user_id → 403 =================
        print("\n== 请求体篡改 user_id ==")
        r = c.post("/api/query", json={"question": "测试", "user_id": uid_b}, headers=bearer(tok_a))
        check("A 以 B 的 user_id 调 query → 403", r.status_code == 403, f"got {r.status_code}")

        # ================= 数据隔离：A 写入的数据 B 查不到 =================
        print("\n== 数据隔离 ==")
        r = c.get("/api/sessions", headers=bearer(tok_b))
        ids_b = [s["id"] for s in r.json()]
        check("B 的会话列表不含 A 的会话", sid_a not in ids_b, f"leak: {ids_b}")
        r = c.get("/api/system-prompt", headers=bearer(tok_b))
        ids_bp = [p["id"] for p in r.json()["current"]["prompts"]]
        check("B 的提示词库不含 A 的提示词", pid_a not in ids_bp, f"leak: {ids_bp}")
        r = c.get("/api/system-prompt", headers=bearer(tok_a))
        ids_ap = [p["id"] for p in r.json()["current"]["prompts"]]
        check("A 的提示词库含自己的提示词", pid_a in ids_ap, f"missing: {ids_ap}")
        r = c.get("/api/compare-history", headers=bearer(tok_b))
        ids_bc = [h["id"] for h in r.json()]
        check("B 的对比历史不含 A 的记录", cid_a not in ids_bc, f"leak: {ids_bc}")
        # A 的会话经内部 crud 写入一轮消息后，B 依然读不到（模拟 A 已产生对话数据）
        with crud.get_db() as db:
            crud.append_turn(db, sid_a, "A 的私密问题", "A 的私密回答", title="A的会话", user_id=uid_a)
        r = c.get(f"/api/sessions/{sid_a}/messages", headers=bearer(tok_b))
        check("A 写入消息后 B 读取仍 → 403", r.status_code == 403, f"got {r.status_code}")
        r = c.get(f"/api/sessions/{sid_a}/messages", headers=bearer(tok_a))
        check("A 读取自己的会话消息 → 200", r.status_code == 200 and any("A 的私密问题" in m["content"] for m in r.json()), f"got {r.status_code}")

        # ================= 登录失败锁定 =================
        print("\n== 登录失败锁定 ==")
        ulock = f"lock_{suffix}"
        c.post("/api/auth/register", json={"username": ulock, "password": pw})
        codes = []
        for _ in range(settings.LOGIN_MAX_FAILS):
            codes.append(c.post("/api/auth/login", json={"username": ulock, "password": "bad"}).status_code)
        r = c.post("/api/auth/login", json={"username": ulock, "password": pw})
        check("连续失败后锁定 → 429", r.status_code == 429 and codes[-1] == 401, f"got {r.status_code}")

    print(f"\n{'=' * 56}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("全部通过 ✓")


if __name__ == "__main__":
    main()

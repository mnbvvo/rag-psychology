"""测试点 3：并发安全 —— 进程内共享状态在并发下不丢更新、不串号、不损坏。

测试方案（对应需求）：
  用 mock 对同一资源并发操作；模拟多用户同时登录与问答时，验证
  进程内共享状态（token 校验 / 会话归属 / 提示词库 / 审计写入）在并发下：
    - 不丢更新（并发创建的每条数据都在）
    - 不串号（A 的操作永远落在 A 名下）
    - 不损坏（并发写同一资源后数据完整、归属校验不破）

用例：
  A. 并发登录：8 用户同时登录，每个 token 的 me 必须对应当前用户（不串号）
  B. 并发创建会话：A/B 各并发建 5 个，总数不丢、列表零交集
  C. 并发写提示词：A 并发新增 5 条，全部保留（不丢更新）
  D. 并发越权写：B 并发 10 次操作 A 的提示词，全部 403 且 A 数据不损坏
  E. 并发问答（mock LLM/embedding，仅 --inprocess）：并发 /api/query 后
     会话消息/审计归属不串（A 的问题只进 A 的会话）
  F. 数据库层校验：并发后各表按 user 计数与预期一致

运行：
  python tests/test_concurrency.py --inprocess   # 推荐：并发问答需同进程 mock
  python tests/test_concurrency.py                # 真实服务：A/B/C/D/F，E 自动跳过
"""
import argparse
import concurrent.futures
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
SKIP: list[str] = []


def check(name: str, cond: bool, extra: str = ""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} {extra}")


def skip(name: str):
    SKIP.append(name)
    print(f"  [SKIP] {name}")


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
        settings.SEMANTIC_CHECK_ENABLED = False  # 并发问答 mock embedding 时避免语义拦截干扰
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


def run_suite(api, allow_mock: bool, cleanup: bool) -> list[str]:
    suffix = uuid.uuid4().hex[:6]
    pw = "Test@123456"
    created: list[str] = []

    def reg(u):
        created.append(u)
        return api.call("POST", "/api/auth/register", json_body={"username": u, "password": pw}).status_code

    def login(u):
        return api.call("POST", "/api/auth/login", json_body={"username": u, "password": pw})

    # ================= A. 并发登录（不串号） =================
    print("\n== A. 并发登录：8 用户同时登录，token 不串号 ==")
    ua_list = [f"ccA_{i}_{suffix}" for i in range(8)]
    for u in ua_list:
        assert reg(u) == 201

    def do_login(u):
        r = login(u)
        if r.status_code != 200:
            return u, None
        me = api.call("GET", "/api/auth/me", token=r.json()["access_token"])
        return u, (me.json().get("username") if me.status_code == 200 else None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(do_login, ua_list))
    mismatch = [u for u, got in results if got != u]
    check("并发登录 8 用户全部 token 对应正确账号（不串号）", not mismatch and len(results) == 8, f"mismatch={mismatch}")

    # ================= B. 并发创建会话（不丢更新 / 不串号） =================
    print("\n== B. 并发创建会话：A/B 各并发建 5 个，不丢、零交集 ==")
    ub_a, ub_b = f"ccB_a_{suffix}", f"ccB_b_{suffix}"
    assert reg(ub_a) == 201 and reg(ub_b) == 201
    tok_a = login(ub_a).json()["access_token"]
    tok_b = login(ub_b).json()["access_token"]

    def create_sess(args):
        tok, label, i = args
        r = api.call("POST", "/api/sessions", token=tok, json_body={"name": f"{label}并发{i}"})
        return r.status_code, (r.json().get("id") if r.status_code == 200 else None)

    tasks = [(tok_a, "A", i) for i in range(5)] + [(tok_b, "B", i) for i in range(5)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        codes = [c for c, _ in ex.map(create_sess, tasks)]
    check("并发 10 路创建会话全部 200（不损坏）", all(c == 200 for c in codes), f"{codes}")
    r = api.call("GET", "/api/sessions", token=tok_a)
    a_ids = {s["id"] for s in r.json()}
    r = api.call("GET", "/api/sessions", token=tok_b)
    b_ids = {s["id"] for s in r.json()}
    check("A 的会话数 = 5（不丢更新）", len(a_ids) == 5, f"got {len(a_ids)}")
    check("B 的会话数 = 5（不丢更新）", len(b_ids) == 5, f"got {len(b_ids)}")
    check("A、B 会话列表零交集（不串号）", not (a_ids & b_ids), f"交集={a_ids & b_ids}")

    # ================= C. 并发写提示词（不丢更新） =================
    print("\n== C. 并发写提示词：A 并发新增 5 条，全部保留 ==")
    def add_prompt(args):
        tok, i = args
        r = api.call("PUT", "/api/system-prompt", token=tok, json_body={"add": {"name": f"并发提示词{i}", "content": f"content-{i}"}})
        return r.status_code
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        codes = list(ex.map(add_prompt, [(tok_a, i) for i in range(5)]))
    check("并发新增 5 条提示词全部 200", all(c == 200 for c in codes), f"{codes}")
    r = api.call("GET", "/api/system-prompt", token=tok_a)
    names = {p["name"] for p in r.json()["current"]["prompts"]}
    check("5 条并发新增提示词全部保留（不丢更新）", all(f"并发提示词{i}" in names for i in range(5)), f"missing={[f'并发提示词{i}' for i in range(5) if f'并发提示词{i}' not in names]}")

    # ================= D. 并发越权写（不损坏） =================
    print("\n== D. 并发越权写：B 并发 10 次操作 A 的提示词，全部 403 且不损坏 ==")
    r = api.call("GET", "/api/system-prompt", token=tok_a)
    victim = next((p for p in r.json()["current"]["prompts"] if p["name"] == "并发提示词0"), None)
    check("取得目标提示词（受害资源）", victim is not None)
    victim_content = victim["content"] if victim else ""
    if victim:
        def steal(args):
            mode, pid = args
            body = {"update": {"id": pid, "content": "被篡改"}} if mode == "update" else {"deleteId": pid}
            return api.call("PUT", "/api/system-prompt", token=tok_b, json_body=body).status_code
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            codes = list(ex.map(steal, [("update" if i % 2 else "deleteId", victim["id"]) for i in range(10)]))
        check("B 并发 10 次越权操作全部 403（归属校验并发下不破）", all(c == 403 for c in codes), f"{codes}")
        r = api.call("GET", "/api/system-prompt", token=tok_a)
        now = next((p for p in r.json()["current"]["prompts"] if p["id"] == victim["id"]), None)
        check("A 的提示词仍在且内容未损坏", now is not None and now["content"] == victim_content, f"content={now['content'][:20] if now else 'DELETED'}")

    # ================= E. 并发问答（mock，仅 --inprocess） =================
    print("\n== E. 并发问答：A/B 并发 /api/query，会话/审计归属不串（mock LLM/embedding） ==")
    if not allow_mock:
        skip("并发问答需同进程 mock（请用 --inprocess）")
    else:
        from unittest.mock import patch

        import numpy as np

        from modules import rag_system
        from modules.rag_core import build_sources

        fixed_vec = [0.01] * settings.VECTOR_DIMENSION

        def mock_generate(question, context=None, system_prompt_override=None, prompt_id=None,
                          timings=None, messages=None, user_id=None):
            # 必须返回与真实 generate 相同的结构（answer/timings/sources）
            return {
                "answer": f"MOCK-ANS:{question[:24]}",
                "timings": {"llm": 0.0},
                "sources": build_sources(context or []),
            }

        uc_a, uc_b = f"ccQ_a_{suffix}", f"ccQ_b_{suffix}"
        assert reg(uc_a) == 201 and reg(uc_b) == 201
        tq_a = login(uc_a).json()["access_token"]
        tq_b = login(uc_b).json()["access_token"]
        rq_a = api.call("POST", "/api/sessions", token=tq_a, json_body={"name": "问答A"})
        sid_a = rq_a.json()["id"]
        rq_b = api.call("POST", "/api/sessions", token=tq_b, json_body={"name": "问答B"})
        sid_b = rq_b.json()["id"]

        def ask(args):
            tok, sid, label, i = args
            r = api.call("POST", "/api/query", token=tok, json_body={
                "question": f"{label}的问题{i}", "session_id": sid, "persist": True, "title": f"{label}并发问答{i}",
            })
            return r.status_code

        with patch("modules.vector_store.TimedOpenAIEmbeddings.embed_query", return_value=fixed_vec), \
             patch("modules.vector_store.TimedOpenAIEmbeddings.embed_documents", return_value=[fixed_vec]), \
             patch.object(rag_system.rag, "generate", side_effect=mock_generate):
            tasks = ([(tq_a, sid_a, "QA", i) for i in range(3)] + [(tq_b, sid_b, "QB", i) for i in range(3)])
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                codes = list(ex.map(ask, tasks))
        check("并发 6 路问答全部 200", all(c == 200 for c in codes), f"{codes}")
        r = api.call("GET", f"/api/sessions/{sid_a}/messages", token=tq_a)
        msg_a = [m["content"] for m in r.json()]
        r = api.call("GET", f"/api/sessions/{sid_b}/messages", token=tq_b)
        msg_b = [m["content"] for m in r.json()]
        check("A 的会话消息只含 A 的问题（不串号）", all("QB" not in m for m in msg_a), f"leak: {msg_a}")
        check("B 的会话消息只含 B 的问题（不串号）", all("QA" not in m for m in msg_b), f"leak: {msg_b}")
        check("A 的问题确实入库（3 条 QA）", sum(1 for m in msg_a if m.startswith("QA的问题")) == 3, f"count={sum(1 for m in msg_a if m.startswith('QA的问题'))}")

    # ================= F. 数据库层校验 =================
    print("\n== F. 数据库层：并发后按用户计数与预期一致 ==")
    from sqlalchemy import func, select

    from db import SessionLocal
    from db.models import Prompt, Session as DbSession, User

    with SessionLocal() as db:
        uid_a = db.execute(select(User.id).where(User.username == ub_a)).scalar()
        uid_b = db.execute(select(User.id).where(User.username == ub_b)).scalar()
        n_a = db.execute(
            select(func.count()).select_from(DbSession).where(DbSession.user_id == uid_a)
        ).scalar()
        check("数据库层：A 名下会话数 = 5（不丢更新）", n_a == 5, f"n={n_a}")
        n_b = db.execute(
            select(func.count()).select_from(DbSession).where(DbSession.user_id == uid_b)
        ).scalar()
        check("数据库层：B 名下会话数 = 5（不丢更新）", n_b == 5, f"n={n_b}")
        cross = db.execute(
            select(func.count()).select_from(DbSession).where(
                DbSession.title.like("A并发%"), DbSession.user_id == uid_b
            )
        ).scalar()
        check("数据库层：B 名下无 A 的并发会话（不串号）", cross == 0, f"cross={cross}")
        n_prompts_a = db.execute(
            select(func.count()).select_from(Prompt).where(Prompt.user_id == uid_a)
        ).scalar()
        check("数据库层：A 的提示词数 = 5（并发新增全部入库，不丢）", n_prompts_a == 5, f"n={n_prompts_a}")

    # ================= 清理 =================
    if cleanup:
        removed = cleanup_test_accounts(created)
        print(f"\n[cleanup] 已删除 {removed} 个测试账号")

    return created


def main():
    ap = argparse.ArgumentParser(description="并发安全自动化测试")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--inprocess", action="store_true", help="进程内直测（并发问答需此模式，mock 同进程生效）")
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    api = InProcRunner() if args.inprocess else RealRunner(args.url)
    print(f"运行模式：{api.name}")
    t0 = time.time()
    try:
        run_suite(api, allow_mock=args.inprocess, cleanup=not args.no_cleanup)
    finally:
        api.close()

    print(f"\n{'=' * 56}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}   （耗时 {time.time() - t0:.1f}s）")
    if SKIP:
        print(f"跳过 {len(SKIP)} 项：{[s for s in SKIP]}")
    if args.report:
        import json as _json

        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            _json.dumps({"passed": PASS, "failed": FAIL, "skipped": SKIP, "ok": not FAIL}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("全部通过 ✓")


if __name__ == "__main__":
    main()

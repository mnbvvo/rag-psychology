"""测试点 7：超时与慢响应 —— mock LLM 延迟，验证超时放弃 / 兜底 / 重试 / 不拖垮线程池。

测试方案（对应需求）：
  用 mock 把 LLM 响应延迟调大（如 60s），断言：
  ① 请求在超时阈值内主动放弃（不无限挂起）；
  ② 返回友好兜底（如"网络不佳稍后再试"）；
  ③ 慢请求不长期占用线程拖垮其他用户（并发处理而非串行阻塞）；
  ④ 超时后的重试正常工作。

实现：测试进程内起一个本地 HTTP 服务器（ThreadingHTTPServer）模拟 LLM API，
  - /slow  挂起 60s（无响应字节）→ 客户端 httpx read-timeout 真实触发；
  - /flaky 首次挂起 60s、重试时立即成功 → 验证 SDK 自动重试；
  - /fast  立即返回 → 验证链路正常。
  ChatOpenAI 指向该服务器（base_url），timeout/max_retries 走真实 httpx 机制。
  仅支持 --inprocess（mock 需与 API 同进程生效）。

运行：
  python tests/test_timeout_slow.py --inprocess
"""
import argparse
import concurrent.futures
import http.server
import json
import socketserver
import sys
import threading
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


# ---------- 本地 mock LLM HTTP 服务器 ----------
def llm_json(content="mock answer"):
    return {
        "id": "mock-1",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-slow",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


class MockLLMState:
    """控制 mock 服务器行为（线程安全）。"""
    flaky_calls = 0
    lock = threading.Lock()


class MockLLMServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """按 mode 区分行为的 mock LLM 服务器：
       slow  → 每次请求挂起 60s（触发客户端 read-timeout）
       flaky → 首次挂起 60s（超时），重试时立即成功
       fast  → 立即返回
    注意：openai SDK 请求路径固定为 /v1/chat/completions，故用不同服务器实例区分模式。"""
    daemon_threads = True

    def __init__(self, mode: str, *args, **kw):
        self.mode = mode
        super().__init__(*args, **kw)


class MockLLMHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        server: MockLLMServer = self.server
        if server.mode == "slow":
            time.sleep(60)
        elif server.mode == "flaky":
            with MockLLMState.lock:
                MockLLMState.flaky_calls += 1
                n = MockLLMState.flaky_calls
            if n == 1:
                time.sleep(60)
        content = "RETRY-OK" if server.mode == "flaky" else "FAST-OK"
        body = json.dumps(llm_json(content)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_mock_llm_server(mode: str = "fast") -> str:
    """启动 mock LLM HTTP 服务器（mode: slow/flaky/fast），返回 base_url。"""
    server = MockLLMServer(mode, ("127.0.0.1", 0), MockLLMHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def make_llm(base_url: str, *, timeout_s=8.0, max_retries=1):
    """构造指向 mock 服务器的 ChatOpenAI（真实 httpx 超时/重试机制）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="mock-slow",
        openai_api_key="test-key",
        base_url=f"{base_url}/v1",
        temperature=0.7,
        max_tokens=100,
        timeout=timeout_s,
        max_retries=max_retries,
    )


def run_suite(api, cleanup: bool) -> None:
    from modules import rag_system

    suffix = uuid.uuid4().hex[:6]
    pw = "Test@123456"
    username = f"timeout_{suffix}"
    r = api.call("POST", "/api/auth/register", json_body={"username": username, "password": pw})
    assert r.status_code in (201, 409)
    tok = api.call("POST", "/api/auth/login", json_body={"username": username, "password": pw}).json()["access_token"]

    original_llm = rag_system.rag.llm
    base = start_mock_llm_server()

    def ask(question, persist=False):
        return api.call("POST", "/api/query", token=tok, json_body={"question": question, "persist": persist})

    try:
        # ================= ① 慢响应：超时主动放弃 + 兜底 =================
        print("\n== ① 慢响应（上游挂起 60s > 超时 8s）→ 超时主动放弃 + 兜底 ==")
        rag_system.rag.llm = make_llm(start_mock_llm_server("slow"), timeout_s=8, max_retries=1)
        t0 = time.time()
        resp = ask("慢响应测试")
        elapsed = time.time() - t0
        check("在超时窗口内主动放弃（约 8s 超时 × 2 次尝试，<30s）", elapsed < 30, f"elapsed={elapsed:.1f}s")
        check("返回 500 兜底（未崩溃、未挂死）", resp.status_code == 500, f"status={resp.status_code}")
        check("兜底文案存在（含『稍后重试』）", "稍后重试" in resp.text, f"body={resp.text[:120]}")

        # ================= ② 首次超时后 SDK 自动重试 → 最终成功 =================
        print("\n== ② 首次超时后 SDK 自动重试 → 最终成功 ==")
        MockLLMState.flaky_calls = 0
        rag_system.rag.llm = make_llm(start_mock_llm_server("flaky"), timeout_s=8, max_retries=1)
        t0 = time.time()
        resp = ask("重试测试")
        elapsed = time.time() - t0
        check("重试后返回 200", resp.status_code == 200, f"status={resp.status_code} body={resp.text[:150]}")
        check("回答来自重试成功（RETRY-OK）", "RETRY-OK" in resp.text, f"body={resp.text[:150]}")
        check("总耗时 ≈ 首次超时(8s) + 重试成功（<20s）", elapsed < 20, f"elapsed={elapsed:.1f}s")

        # ================= ③ 慢请求不拖垮其他用户（并发处理） =================
        print("\n== ③ 并发：8 个慢请求并发处理，不串行阻塞、不拖垮 ==")
        # 慢请求使用独立 mock 服务器（sleep 15s < timeout 30，最终正常返回）
        base_slow = start_mock_llm_server()

        class SlowHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                time.sleep(15)
                body = json.dumps(llm_json("SLOW-OK")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        class SlowServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        slow_srv = SlowServer(("127.0.0.1", 0), SlowHandler)
        threading.Thread(target=slow_srv.serve_forever, daemon=True).start()
        rag_system.rag.llm = make_llm(f"http://127.0.0.1:{slow_srv.server_address[1]}",
                                      timeout_s=30, max_retries=0)

        def slow_ask(i):
            return ask(f"慢请求{i}", persist=False).status_code

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            codes = list(ex.map(slow_ask, range(8)))
        elapsed = time.time() - t0
        check("8 个慢请求全部完成（200）", all(c == 200 for c in codes), f"{codes}")
        check("并发完成而非串行阻塞（总耗时 < 40s，8×15s 串行=120s）", elapsed < 40, f"elapsed={elapsed:.1f}s")
        slow_srv.shutdown()

        # ================= ④ 恢复正常 LLM → 链路照常工作 =================
        print("\n== ④ 恢复正常 LLM → 链路照常工作 ==")
        rag_system.rag.llm = original_llm
        resp = ask("恢复正常后的测试问题", persist=False)
        check("恢复后 /api/query 正常返回（200）", resp.status_code == 200, f"status={resp.status_code} body={resp.text[:100]}")
    finally:
        rag_system.rag.llm = original_llm

    if cleanup:
        from sqlalchemy import delete, select

        from db import SessionLocal
        from db.models import User

        with SessionLocal() as db, db.begin():
            users = db.execute(select(User).where(User.username == username)).scalars().all()
            for u in users:
                db.delete(u)
            print(f"[cleanup] 已删除 {len(users)} 个测试账号")


def main():
    ap = argparse.ArgumentParser(description="超时与慢响应自动化测试")
    ap.add_argument("--inprocess", action="store_true", help="进程内直测（mock 需同进程，必需）")
    ap.add_argument("--no-cleanup", action="store_true")
    args = ap.parse_args()
    if not args.inprocess:
        raise SystemExit("本测试依赖同进程 mock，必须使用 --inprocess")

    settings.RERANK_ENABLED = False
    settings.SEMANTIC_CHECK_ENABLED = False
    settings.HYBRID_ENABLED = False
    settings.ENABLE_THINKING = False  # 简化：走非流式 llm.invoke（测试超时/重试机制不依赖思考模式）

    from fastapi.testclient import TestClient
    from api.main import app

    class Api:
        def __init__(self, client):
            self._c = client

        def call(self, method, path, token=None, json_body=None):
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            return self._c.request(method, path, headers=headers, json=json_body)

    t0 = time.time()
    with TestClient(app) as client:
        api = Api(client)
        print("运行模式：InProcess TestClient（本地 mock LLM 服务器生效）")
        run_suite(api, cleanup=not args.no_cleanup)

    print(f"\n{'=' * 56}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}   （耗时 {time.time() - t0:.1f}s）")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("全部通过 ✓")


if __name__ == "__main__":
    main()

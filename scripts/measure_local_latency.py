"""本地业务链路延迟测量（不调用上游 LLM / embedding API，支持并发）。

用途：把上游 API 耗时剥掉，量化「用户请求 → 框架处理 → 模拟生成 → 真实落库 → 返回」的本地纯开销。
对比口径：
- 单用户：完整链路（真实上游）P50 ≈ 3.9s vs 本地链路（mock）P50 ≈ 8.6ms
- 40 并发：完整链路（真实上游）P50 ≈ 16.3s vs 本地链路（mock）→ 本脚本测这个

方法：
- mock LLM 生成：rag_system.query 替换为本地函数（零上游调用，立即返回固定答案）；
- mock embedding：memory_service.embed 替换为本地函数（返回全 1 向量，不调 API）；
- 其余全部真实：FastAPI 路由、线程池、JWT 鉴权、append_turn 写 sessions/messages、
  save_turn 写 user_chat_history、响应序列化。

运行：
  python scripts/measure_local_latency.py                        # 单用户 20 轮（默认）
  python scripts/measure_local_latency.py --total 40 --concurrency 40 --num-users 40   # 40 用户并发
"""
import argparse
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402
from modules import rag_system  # noqa: E402
from modules.memory import memory_service  # noqa: E402


def _mock_query(**kwargs):
    """模拟 LLM 生成：不调用上游，立即返回固定答案。"""
    return {
        "question": kwargs.get("question") or "测试",
        "answer": "这是一个模拟答案，用于测量本地业务链路耗时，不调用任何上游 API。"
                  "（此回答由 mock 生成，无真实 LLM 参与）" * 2,
        "sources": [],
        "timings": {"llm": 0, "total": 0},
    }


def _mock_embed(text):
    """模拟 embedding：不调用 API，返回全 1 向量。"""
    return [0.01] * settings.VECTOR_DIMENSION


_LLM_DELAY_MS = 0  # 模拟 LLM 生成耗时（毫秒），由 --llm-delay-ms 设置


def _mock_query(**kwargs):
    """模拟 LLM 生成：不调用上游，按 _LLM_DELAY_MS 延时后返回固定答案。"""
    if _LLM_DELAY_MS > 0:
        time.sleep(_LLM_DELAY_MS / 1000.0)
    return {
        "question": kwargs.get("question") or "测试",
        "answer": "这是一个模拟答案，用于测量本地业务链路耗时，不调用任何上游 API。"
                  "（此回答由 mock 生成，无真实 LLM 参与）" * 2,
        "sources": [],
        "timings": {"llm": _LLM_DELAY_MS, "total": _LLM_DELAY_MS},
    }


def _register(client, prefix: str) -> str:
    u = f"{prefix}_{uuid.uuid4().hex[:10]}"
    r = client.post("/api/auth/register", json={"username": u, "password": "Test@123456"})
    assert r.status_code in (201, 409), f"注册失败 {r.status_code} {r.text[:120]}"
    tok = client.post("/api/auth/login", json={"username": u, "password": "Test@123456"}).json()["access_token"]
    return tok


def main():
    ap = argparse.ArgumentParser(description="本地业务链路延迟测量（mock 上游，真实落库）")
    ap.add_argument("--total", type=int, default=20, help="总请求数")
    ap.add_argument("--concurrency", type=int, default=1, help="并发数")
    ap.add_argument("--num-users", type=int, default=1, help="独立账号数（请求按账号轮换）")
    ap.add_argument("--llm-delay-ms", type=int, default=0, help="模拟 LLM 生成耗时（毫秒），0=立即返回")
    args = ap.parse_args()

    global _LLM_DELAY_MS
    _LLM_DELAY_MS = args.llm_delay_ms
    rag_system.query = _mock_query
    memory_service.embed = _mock_embed

    with TestClient(app) as c:
        tokens = [_register(c, "localtest") for _ in range(args.num_users)]
        headers = [{"Authorization": f"Bearer {t}"} for t in tokens]

        # 预热一轮
        r = c.post("/api/query", json={"question": "预热"}, headers=headers[0])
        assert r.status_code == 200, r.text[:120]

        def measure(i):
            h = headers[i % len(headers)]
            t0 = time.perf_counter()
            resp = c.post("/api/query", json={"question": f"本地链路并发测试 {i}"}, headers=h)
            dt = (time.perf_counter() - t0) * 1000
            return resp.status_code, dt

        times = []
        t_wall0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for st, dt in ex.map(measure, range(args.total)):
                assert st == 200, f"非 200: {st}"
                times.append(dt)
        t_wall = time.perf_counter() - t_wall0

        times.sort()
        p95 = times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]
        print(f"\n=== 本地业务链路耗时（mock 上游，真实落库）===")
        print(f"  total={args.total}  concurrency={args.concurrency}  账号={args.num_users}")
        print(f"  min = {times[0]:.1f} ms")
        print(f"  p50 = {statistics.median(times):.1f} ms")
        print(f"  p95 = {p95:.1f} ms")
        print(f"  max = {times[-1]:.1f} ms")
        print(f"  mean= {statistics.mean(times):.1f} ms")
        print(f"  墙钟= {t_wall:.2f} s")
        print("\n对照：完整链路（真实上游）40 并发 P50 ≈ 16.3s / 单用户 P50 ≈ 3.9s")


if __name__ == "__main__":
    main()

"""测试点 4：性能压测 —— 阶梯并发下测量系统自身上限（Mock 环境）。

对齐目标（参考外部压测示例）：
  | 并发 | 接口 | 请求 | 失败 | Avg | P95 | P99 | RPS |

环境（Mock · 系统自身上限）：
  - embedding API / LLM 均 mock（固定向量 + 固定回答），剔除外部网络与费用噪音；
  - 本地重排 / 语义检测 / BM25 预热关闭（RERANK/SEMANTIC/HYBRID=False）；
  - 单进程 uvicorn（与功能测试一致的单进程语义），压测脚本同进程跑服务使 mock 生效；
  - 关系库/向量库走真实 PostgreSQL + pgvector。

接口：
  1) /api/system-prompt（GET）      纯 DB 读 + 提示词隔离查询（最轻）
  2) /api/query（POST）             完整同步 RAG：安全检测 + PGVector 检索 + mock 生成 + 落库
  3) /api/query/stream（POST,SSE）  流式：首 token 时延(TTFT) + 完整时延

运行：
  python tests/test_performance.py                    # 默认 1/10/50/100 并发，每级 6s
  python tests/test_performance.py --levels 1,10,50,100 --duration 6 --port 8029
  python tests/test_performance.py --report tests/results/performance.json
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402

# ---- 压测环境（mock 外部依赖，测系统自身上限） ----
settings.RERANK_ENABLED = False
settings.SEMANTIC_CHECK_ENABLED = False
settings.HYBRID_ENABLED = False
settings.RATE_LIMIT_TIMES = 1_000_000  # 放宽单 IP 限流，避免压测被拦截


async def wait_healthy(base: str, timeout: float = 60.0):
    import httpx

    t0 = time.time()
    async with httpx.AsyncClient(timeout=5) as c:
        while time.time() - t0 < timeout:
            try:
                r = await c.get(f"{base}/api/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("服务在超时时间内未就绪")


async def setup_users(base: str, n: int = 10, suffix: str = "") -> list[str]:
    import httpx

    pw = "Test@123456"
    async with httpx.AsyncClient(timeout=15) as c:
        tokens = []
        for i in range(n):
            u = f"perf_{i}_{suffix}"
            r = await c.post(f"{base}/api/auth/register", json={"username": u, "password": pw})
            if r.status_code not in (201, 409):
                raise RuntimeError(f"注册失败 {u}: {r.status_code} {r.text[:120]}")
            r = await c.post(f"{base}/api/auth/login", json={"username": u, "password": pw})
            if r.status_code != 200:
                raise RuntimeError(f"登录失败 {u}: {r.status_code}")
            tokens.append(r.json()["access_token"])
        return tokens


async def run_level(base: str, tokens: list[str], level: int, duration: float,
                    endpoint: str, mode: str) -> dict:
    """单并发级压测。mode: sync(普通 POST) | sse(流式)。

    返回 {requests, errors, avg, p95, p99, rps, ttft}；ttft 仅 sse 模式。
    """
    import httpx

    latencies: list[float] = []   # 完整时延 ms
    ttfts: list[float] = []       # 首 token 时延 ms（仅 sse）
    errors = 0
    count = 0

    def q(i: int) -> str:
        return f"压测问题{i}-孩子的情绪管理方法"  # 固定题库，缓存友好

    async def worker(wid: int):
        nonlocal errors, count
        token = tokens[wid % len(tokens)]
        headers = {"Authorization": f"Bearer {token}"}
        method = "GET" if endpoint == "/api/system-prompt" else "POST"
        async with httpx.AsyncClient(timeout=120) as client:
            deadline = asyncio.get_event_loop().time() + duration
            i = 0
            while asyncio.get_event_loop().time() < deadline:
                i += 1
                t0 = time.perf_counter()
                try:
                    if mode == "sse":
                        async with client.stream("POST", f"{base}{endpoint}",
                                                 headers=headers, json={"question": q(i), "persist": False}) as resp:
                            first = None
                            async for chunk in resp.aiter_bytes():
                                if first is None and chunk:
                                    first = time.perf_counter()
                            if first is None:
                                errors += 1
                            else:
                                ttfts.append((first - t0) * 1000)
                            if resp.status_code != 200:
                                errors += 1
                    elif method == "GET":
                        resp = await client.get(f"{base}{endpoint}", headers=headers)
                        if resp.status_code != 200:
                            errors += 1
                    else:
                        resp = await client.post(f"{base}{endpoint}", headers=headers,
                                                 json={"question": q(i), "persist": False})
                        if resp.status_code != 200:
                            errors += 1
                except Exception:
                    errors += 1
                latencies.append((time.perf_counter() - t0) * 1000)
                count += 1

    await asyncio.gather(*[worker(w) for w in range(level)])

    n = len(latencies)
    if n == 0:
        return {"requests": 0, "errors": errors, "avg": 0, "p95": 0, "p99": 0, "rps": 0, "ttft": 0}
    lat = sorted(latencies)
    p95 = lat[min(int(n * 0.95), n - 1)]
    p99 = lat[min(int(n * 0.99), n - 1)]
    ttft = round(statistics.median(ttfts), 1) if ttfts else 0
    return {
        "requests": n,
        "errors": errors,
        "avg": round(statistics.mean(lat), 1),
        "p95": round(p95, 1),
        "p99": round(p99, 1),
        "rps": round(n / duration, 1),
        "ttft": ttft,
    }


def fmt_row(level: int, ep: str, r: dict) -> str:
    if ep == "/api/query/stream":
        return f"| {level} | {ep} | {r['requests']} | {r['errors']} | TTFT {r['ttft']}ms / {r['avg']}ms | {r['p95']}ms | {r['p99']}ms | {r['rps']} |"
    return f"| {level} | {ep} | {r['requests']} | {r['errors']} | {r['avg']}ms | {r['p95']}ms | {r['p99']}ms | {r['rps']} |"


async def main():
    ap = argparse.ArgumentParser(description="性能压测（Mock / 真实链路）")
    ap.add_argument("--port", type=int, default=8029)
    ap.add_argument("--levels", default="", help="并发阶梯，逗号分隔（默认：Mock=1,10,50,100 / live=1,2,5,10）")
    ap.add_argument("--duration", type=float, default=6.0, help="每级持续秒数")
    ap.add_argument("--report", default="tests/results/performance.json")
    ap.add_argument("--boundary", action="store_true",
                    help="边界自动探测：并发按 1→2→4→…倍增，直到错误率>0（硬边界）或 P99 超阈值（软边界）")
    ap.add_argument("--boundary-max", type=int, default=1024, help="边界探测的最大并发（默认 1024）")
    ap.add_argument("--boundary-p99", type=float, default=10000.0,
                    help="软边界 P99 阈值 ms（默认 10000=10s）")
    ap.add_argument("--live", action="store_true",
                    help="真实链路压测：不 mock embedding/LLM，走真实 API（注意：消耗 token 费用）")
    args = ap.parse_args()

    live = args.live
    default_levels = "1,2,5,10" if live else "1,10,50,100"
    levels_arg = args.levels or default_levels

    if not live:
        # ---- Mock 环境：替换 embedding/LLM，测系统自身上限 ----
        from unittest.mock import patch

        from modules import rag_system
        from modules.rag_core import build_sources

        fixed_vec = [0.01] * settings.VECTOR_DIMENSION

        async def mock_stream_generate(question, context=None, system_prompt_override=None,
                                       prompt_id=None, messages=None, user_id=None):
            text = f"MOCK-STREAM:{question[:20]}"
            for i in range(0, len(text), 4):
                yield text[i:i + 4]
                await asyncio.sleep(0.001)

        def mock_generate(question, context=None, system_prompt_override=None, prompt_id=None,
                          timings=None, messages=None, user_id=None):
            return {"answer": f"MOCK-ANS:{question[:20]}", "timings": {"llm": 0.0},
                    "sources": build_sources(context or [])}

        # mock 必须在 uvicorn 服务启动前生效（同进程，服务处理请求时读到 patch 后的对象）
        patchers = [
            patch("modules.vector_store.TimedOpenAIEmbeddings.embed_query", return_value=fixed_vec),
            patch("modules.vector_store.TimedOpenAIEmbeddings.embed_documents", return_value=[fixed_vec]),
            patch.object(rag_system.rag, "generate", side_effect=mock_generate),
            patch.object(rag_system.rag, "stream_generate", side_effect=mock_stream_generate),
        ]
        for p in patchers:
            p.start()
    else:
        print("[live] 真实链路模式：embedding/LLM 走真实 API（注意 token 费用）")

    # 同进程启动 uvicorn（单 worker；mock 与业务代码同进程才能生效）
    import uvicorn
    from api.main import app as fastapi_app

    port = args.port
    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    base = f"http://127.0.0.1:{port}"
    await wait_healthy(base)

    suffix = uuid.uuid4().hex[:6]
    print("准备压测用户池（10 个）...")
    tokens = await setup_users(base, n=10, suffix=suffix)
    print(f"服务就绪：{base}（Mock 环境 · 单进程 uvicorn · 关闭重排/语义/预热）")

    results: list[dict] = []
    print("\n| 并发 | 接口 | 请求 | 失败 | Avg/TTFT | P95 | P99 | RPS |")
    print("|---|---|---|---|---|---|---|---|")

    if args.boundary:
        # ---- 边界自动探测：倍增加压，只测代表接口 /api/query ----
        print("\n== 边界探测模式：并发 1→2→4→…倍增，测 /api/query ==")
        levels = []
        v = 1
        while v <= args.boundary_max:
            levels.append(v)
            v *= 2
        boundary_hit = None
        for level in levels:
            print(f"\n-- 并发 {level} --")
            r = await run_level(base, tokens, level, args.duration, "/api/query", "sync")
            print(fmt_row(level, "/api/query", r))
            results.append({"concurrency": level, "endpoint": "/api/query", "boundary_probe": True, **r})
            if r["errors"] > 0:
                boundary_hit = {"type": "硬边界（错误率 > 0）", "concurrency": level, **r}
                break
            if r["p99"] > args.boundary_p99:
                boundary_hit = {"type": "软边界（P99 超阈值）", "concurrency": level, **r}
                break
        print("\n== 边界结论 ==")
        if boundary_hit:
            print(f"  {boundary_hit['type']} @ 并发 {boundary_hit['concurrency']}"
                  f"（avg {boundary_hit['avg']}ms / P99 {boundary_hit['p99']}ms / 失败 {boundary_hit['errors']} / RPS {boundary_hit['rps']}）")
            print(f"  上一档（可接受）并发：{levels[levels.index(boundary_hit['concurrency']) - 1] if levels.index(boundary_hit['concurrency']) > 0 else '无（1 并发即超阈值）'}")
        else:
            print(f"  达到最大并发 {args.boundary_max} 仍未触发边界（错误率 0 且 P99 ≤ {args.boundary_p99}ms），可调大 --boundary-max 继续加压")
    else:
        levels = [int(x) for x in levels_arg.split(",") if x.strip()]
        endpoints = [
            ("/api/system-prompt", "sync"),
            ("/api/query", "sync"),
            ("/api/query/stream", "sse"),
        ]
        for level in levels:
            print(f"\n-- 并发 {level} --")
            for ep, mode in endpoints:
                r = await run_level(base, tokens, level, args.duration, ep, mode)
                print(fmt_row(level, ep, r))
                results.append({"concurrency": level, "endpoint": ep, **r})

    server.should_exit = True
    await server_task

    # 清理压测账号（uvicorn 已停，直接连库删）
    def _cleanup():
        from sqlalchemy import delete, select

        from db import SessionLocal
        from db.models import CompareHistory, CrisisAudit, Message, Prompt, Session, User

        with SessionLocal() as db, db.begin():
            users = db.execute(select(User).where(User.username.like(f"perf_%_{suffix}"))).scalars().all()
            for u in users:
                sids = select(Session.id).where(Session.user_id == u.id)
                db.execute(delete(Message).where(Message.session_id.in_(sids)))
                db.execute(delete(Session).where(Session.user_id == u.id))
                db.execute(delete(Prompt).where(Prompt.user_id == u.id))
                db.execute(delete(CompareHistory).where(CompareHistory.user_id == u.id))
                db.execute(delete(CrisisAudit).where(CrisisAudit.user_id == u.id))
                db.delete(u)
            print(f"[cleanup] 已清理 {len(users)} 个压测账号")

    _cleanup()

    # 输出
    print("\n" + "=" * 66)
    print("压测完成（真实链路）" if live else "压测完成（Mock · 系统自身上限）")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告：{args.report}")


if __name__ == "__main__":
    asyncio.run(main())

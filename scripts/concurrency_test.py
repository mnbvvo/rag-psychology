"""
RAG 心理系统并发压测脚本
========================

用法示例：
 # 基础：200 个请求，常驻并发 20，打 /api/query（默认 persist=true，会压 DB 写）
 python scripts/concurrency_test.py --total 200 --concurrency 20

 # 单账号认证（登录失败会自动注册）
 python scripts/concurrency_test.py --total 200 --concurrency 20 --username loadtest --password 'Test@123456'

 # 多账号（自动注册 N 个随机 loadtest 账号，轮流使用；QA-2 用 5/20/40 独立账号）
 python scripts/concurrency_test.py --total 100 --concurrency 20 --num-users 20

 # 只压 RAG 检索+生成，不落库（persist=false，隔离 LLM/embedding 瓶颈）
 python scripts/concurrency_test.py --total 200 --concurrency 20 --no-persist

 # 压 SSE 流式接口
 python scripts/concurrency_test.py --total 100 --concurrency 10 --endpoint stream

 # 自定义服务地址 / 问题文件
 python scripts/concurrency_test.py --url http://127.0.0.1:8000 --questions my_questions.txt

说明：
 - 项目除 /api/health、/api/auth/* 外全部接口需登录，压测前必须先准备账号；
 - --num-users N 会通过 /api/auth/register 自动注册 N 个随机账号（写入 users 表，压测后如需清理请自行删除）；
 - 默认问题池覆盖常规 + 中/高危安全路径，便于同时压到不同分支。
"""

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from pathlib import Path

import httpx

# 默认问题池（覆盖常规问答 + 中/高危安全路径，便于同时压到不同分支）
DEFAULT_QUESTIONS = [
    "孩子总是情绪低落、没兴趣，家长该怎么做？",
    "青少年厌学怎么办？",
    "孩子晚上睡不着、失眠严重，有什么办法？",
    "青春期孩子和家长总是吵架，怎么改善沟通？",
    "孩子被同学霸凌了，家长应该如何处理？",
    "高中生考试焦虑，考前紧张到手抖，怎么缓解？",
    "孩子有自伤倾向怎么办？",          # 高危求助型（会被安全检测拦截/关怀）
    "孩子说想用脑袋和房梁比赛",          # 隐喻高危（语义检测路径）
    "怎么判断孩子是不是抑郁症？",
    "二胎家庭大宝突然变得很乖，是不是有问题？",
]


def percentile(values, p):
    """p 为 0~100 的分位数（线性插值）。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


async def hit_query(client, sem, question, token, persist, results, stats):
    async with sem:
        t0 = time.perf_counter()
        try:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = await client.post(
                "/api/query",
                json={"question": question, "persist": persist},
                headers=headers,
                timeout=120.0,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            stats[r.status_code] = stats.get(r.status_code, 0) + 1
            results.append((r.status_code, elapsed))
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            stats["EXCEPTION"] = stats.get("EXCEPTION", 0) + 1
            results.append(("EXC", elapsed, repr(e)))


async def hit_stream(client, sem, question, token, persist, results, stats):
    async with sem:
        t0 = time.perf_counter()
        try:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            async with client.stream(
                "POST",
                "/api/query/stream",
                json={"question": question, "persist": persist},
                headers=headers,
                timeout=120.0,
            ) as resp:
                # 读取 SSE，直到收到 done / error 事件
                last_event = None
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event:"):
                        last_event = line.split(":", 1)[1].strip()
                    if line.startswith("data:") and last_event in ("done", "error"):
                        break
                elapsed = (time.perf_counter() - t0) * 1000
                stats[resp.status_code] = stats.get(resp.status_code, 0) + 1
                results.append((resp.status_code, elapsed))
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            stats["EXCEPTION"] = stats.get("EXCEPTION", 0) + 1
            results.append(("EXC", elapsed, repr(e)))


# ---------------- 认证准备（PRE-1：登录 / 自动注册，返回 token 列表） ----------------
async def _login_or_register(client, username: str, password: str) -> str:
    """注册或登录一个账号，返回 access_token。

    策略：先注册（新账号 201；已存在 409），随后登录一次。
    每个账号只发 1 次登录请求，避免触发服务端 IP 级登录限流。
    """
    r = await client.post("/api/auth/register", json={"username": username, "password": password})
    if r.status_code not in (201, 409):
        raise RuntimeError(f"注册失败 {username}: {r.status_code} {r.text[:120]}")
    r = await client.post("/api/auth/login", json={"username": username, "password": password})
    if r.status_code == 200:
        return r.json()["access_token"]
    raise RuntimeError(f"登录失败 {username}: {r.status_code} {r.text[:120]}")


async def prepare_tokens(base: str, username: str | None, password: str, num_users: int) -> list[str]:
    """压测前准备账号 token 列表（单账号 + 多账号随机注册）。"""
    tokens: list[str] = []
    async with httpx.AsyncClient(base_url=base, timeout=30.0) as c:
        if username:
            tokens.append(await _login_or_register(c, username, password))
            print(f"[auth] 单账号 {username} 认证成功（当前 {len(tokens)} 个 token）")
        for _ in range(num_users):
            u = f"loadtest_{uuid.uuid4().hex[:10]}"
            tok = await _login_or_register(c, u, "LoadTest@2026")
            tokens.append(tok)
        if num_users:
            print(f"[auth] 已就绪 {len(tokens)} 个压测账号 token（含自动注册 {num_users} 个 loadtest 账号）")
    if not tokens:
        raise RuntimeError("未提供任何账号：请用 --username/--password 或 --num-users N")
    return tokens


async def main():
    ap = argparse.ArgumentParser(description="RAG 心理系统并发压测")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="服务地址")
    ap.add_argument("--total", type=int, default=200, help="总请求数")
    ap.add_argument("--concurrency", type=int, default=20, help="最大并发数（同时 in-flight 的请求数）")
    ap.add_argument("--endpoint", choices=["query", "stream"], default="query")
    ap.add_argument("--no-persist", action="store_true", help="persist=false，不落库（隔离 DB 写压力）")
    ap.add_argument("--questions", help="问题文件路径（每行一条；不填用内置池）")
    ap.add_argument("--ramp", type=float, default=0.0, help="每发一个请求之间的间隔秒（0=立即全部投入）")
    ap.add_argument("--output", help="把每次请求的 {status,latency_ms} 与汇总写入该 JSON 文件（便于留档/对比）")
    ap.add_argument("--username", default="", help="压测账号用户名（登录失败会自动注册）")
    ap.add_argument("--password", default="LoadTest@2026", help="压测账号密码（默认 LoadTest@2026）")
    ap.add_argument("--num-users", type=int, default=0, help="自动注册 N 个随机 loadtest 账号并轮换（QA-2 独立账号场景用）")
    args = ap.parse_args()

    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = [l.strip() for l in f if l.strip()]
    else:
        questions = DEFAULT_QUESTIONS
    if not questions:
        print("没有任何问题可发，退出。")
        return

    base = args.url.rstrip("/")
    persist = not args.no_persist

    # 0) 健康检查
    try:
        async with httpx.AsyncClient(base_url=base) as hc:
            hr = await hc.get("/api/health", timeout=10.0)
            print(f"[health] {hr.status_code} {hr.text.strip()}")
            if hr.status_code != 200:
                print("服务未就绪，退出。")
                return
    except Exception as e:
        print(f"无法连接服务 {base}：{e}")
        return

    tasks_q = [q for _ in range((args.total + len(questions) - 1) // len(questions)) for q in questions][: args.total]

    # 认证准备（PRE-1）：登录/自动注册拿 token；无账号时不允许裸打（全部接口需登录）
    if not args.username and args.num_users <= 0:
        print("未提供压测账号：请用 --username/--password 或 --num-users N（全部业务接口需要登录）。")
        return
    tokens = await prepare_tokens(base, args.username or None, args.password, args.num_users)

    # 请求 → 账号轮换分配（独立账号场景：QA-2 的 5/20/40 用户同步起跑）
    pairs = [(q, tokens[i % len(tokens)]) for i, q in enumerate(tasks_q)]

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    stats = {}
    worker = hit_stream if args.endpoint == "stream" else hit_query

    print(f"\n=== 并发压测开始 ===")
    print(f"目标    : {base}{'/api/query' if args.endpoint=='query' else '/api/query/stream'}")
    print(f"总数    : {args.total}   并发: {args.concurrency}   persist: {persist}")
    print(f"账号    : {len(tokens)} 个（单账号={'是' if args.username else '否'}，随机注册={args.num_users}）")
    print(f"问题池  : {len(questions)} 条（轮流复用）\n")

    t_start = time.perf_counter()
    async with httpx.AsyncClient(base_url=base, headers={"Connection": "keep-alive"}) as client:
        tasks = []
        for q, tok in pairs:
            tasks.append(asyncio.create_task(worker(client, sem, q, tok, persist, results, stats)))
            if args.ramp > 0:
                await asyncio.sleep(args.ramp)
        # 实时进度
        done = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            done += 1
            if done % max(1, args.total // 10) == 0:
                print(f"  已完成 {done}/{args.total}")
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start

    # 汇总
    lat = [r[1] for r in results if isinstance(r[1], (int, float))]
    print("\n=== 结果汇总 ===")
    print(f"总请求数        : {len(results)}")
    print(f"总耗时(墙钟)    : {wall:.2f}s")
    print(f"吞吐(QPS)       : {len(results)/wall:.2f} req/s")
    print(f"状态码分布      : {dict(sorted((str(k), v) for k, v in stats.items()))}")
    if lat:
        print(f"单请求时延(ms)  : min={min(lat):.0f}  mean={statistics.mean(lat):.0f}  "
              f"p50={percentile(lat,50):.0f}  p90={percentile(lat,90):.0f}  "
              f"p95={percentile(lat,95):.0f}  p99={percentile(lat,99):.0f}  max={max(lat):.0f}")
        over_5s = sum(1 for x in lat if x > 5000)
        over_10s = sum(1 for x in lat if x > 10000)
        print(f"超时比例        : >5s={over_5s} ({over_5s/len(lat)*100:.1f}%)   >10s={over_10s} ({over_10s/len(lat)*100:.1f}%)")

    # 异常明细（最多打印 5 条）
    excs = [r for r in results if r[0] in ("EXC",)]
    if excs:
        print(f"\n异常样例（共 {len(excs)} 条，展示前 5）:")
        for r in excs[:5]:
            print(f"  - {r[2]}")

    # 落盘（可选）：每次请求的明细 + 汇总，便于多次压测对比
    if args.output:
        report = {
            "url": base,
            "endpoint": args.endpoint,
            "total": len(results),
            "concurrency": args.concurrency,
            "persist": persist,
            "wall_seconds": round(wall, 2),
            "qps": round(len(results) / wall, 2) if wall else 0,
            "status_distribution": {str(k): v for k, v in sorted(stats.items())},
            "latency_ms": {
                "min": round(min(lat)) if lat else 0,
                "mean": round(statistics.mean(lat), 1) if lat else 0,
                "p50": round(percentile(lat, 50), 1),
                "p90": round(percentile(lat, 90), 1),
                "p95": round(percentile(lat, 95), 1),
                "p99": round(percentile(lat, 99), 1),
                "max": round(max(lat)) if lat else 0,
            },
            "samples": [
                {"status": r[0], "latency_ms": round(r[1], 1)}
                for r in results if len(r) >= 2
            ],
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

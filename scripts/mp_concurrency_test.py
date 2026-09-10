#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""微信小程序通道 /api/mp/* 并发压测脚本。

与 scripts/concurrency_pressure_test.py（JWT Web 通道）的差异：

1. **无鉴权登录**：mp 通道身份来自请求体 userId（MP_TRUST_MODE=a），不需要
   register/login 换 token；MP_API_KEY 非空时统一带 X-API-Key 头。
2. **userId 池（关键）**：mp 通道「同一 userId 同时只能有 1 个在途请求」，
   重复的第二个请求直接 409 AI_REQUEST_IN_PROGRESS。因此并发度 N 必须由
   **N 个以上不同 userId** 承担，否则测到的是 409 而不是真实吞吐。
   本脚本默认建 max(concurrency*2, 50) 个 userId，并循环取用。
3. **覆盖 CRUD**：除 query / query/stream 外，还可压 register / update / profile。
4. **数据隔离校验**：每个 userId 的孩子昵称是唯一标记，任一回答里出现他人
   标记即记为串号（P0）。

子命令：steady（固定并发持续施压）/ boundary（同步起跑 N 个）/ spike（多轮尖峰）

用法示例见 tests/小程序接口并发测试流程.md。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

try:  # Windows 控制台中文输出兜底
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

DEFAULT_QUESTION = "我孩子最近写作业很磨蹭，一提作业就发脾气，我该怎么引导他？"
ROLES = ("爸爸", "妈妈", "其他")


# ---------------------------------------------------------------- 数据结构
@dataclass
class CallResult:
    endpoint: str
    user_id: str
    status: int
    ok: bool                 # 业务成功
    rejected: bool = False   # 正确拒绝（409/429/503 中的契约性拒绝）
    error_type: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    answer: str = ""
    sse_terminal: str = ""   # done / error / none
    detail: str = ""


@dataclass
class Metrics:
    results: list[CallResult] = field(default_factory=list)
    active_peak: int = 0
    queued_peak: int = 0
    max_active_cfg: int = 0
    max_queue_cfg: int = 0
    isolation_violations: list[dict] = field(default_factory=list)

    # ---- 聚合 ----
    def pct(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        vals = sorted(values)
        k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
        return round(vals[k], 1)

    def summary(self) -> dict:
        by_ep: dict[str, dict[str, Any]] = {}
        for r in self.results:
            e = by_ep.setdefault(
                r.endpoint,
                {"total": 0, "ok": 0, "fail": 0, "rejected": 0, "status": {}, "total_ms": [], "ttft_ms": []},
            )
            e["total"] += 1
            e["ok"] += int(r.ok)
            e["fail"] += int(not r.ok and not r.rejected)
            e["rejected"] += int(r.rejected)
            e["status"][str(r.status)] = e["status"].get(str(r.status), 0) + 1
            if r.total_ms:
                e["total_ms"].append(r.total_ms)
            if r.ttft_ms:
                e["ttft_ms"].append(r.ttft_ms)

        out: dict[str, Any] = {"endpoints": {}, "overall": {}}
        all_ok = all_fail = all_rej = 0
        all_total_ms: list[float] = []
        all_ttft: list[float] = []
        for name, e in by_ep.items():
            admitted = e["total"] - e["rejected"]
            out["endpoints"][name] = {
                "total": e["total"],
                "ok": e["ok"],
                "fail": e["fail"],
                "rejected": e["rejected"],
                "status_dist": e["status"],
                "success_rate": round(e["ok"] / admitted * 100, 2) if admitted else 0.0,
                "total_ms": {
                    "p50": self.pct(e["total_ms"], 50),
                    "p95": self.pct(e["total_ms"], 95),
                    "p99": self.pct(e["total_ms"], 99),
                    "max": round(max(e["total_ms"]), 1) if e["total_ms"] else 0.0,
                },
                "ttft_ms": {
                    "p50": self.pct(e["ttft_ms"], 50),
                    "p95": self.pct(e["ttft_ms"], 95),
                    "p99": self.pct(e["ttft_ms"], 99),
                    "max": round(max(e["ttft_ms"]), 1) if e["ttft_ms"] else 0.0,
                },
            }
            all_ok += e["ok"]
            all_fail += e["fail"]
            all_rej += e["rejected"]
            all_total_ms += e["total_ms"]
            all_ttft += e["ttft_ms"]

        admitted_all = len(self.results) - all_rej
        out["overall"] = {
            "requests": len(self.results),
            "ok": all_ok,
            "fail": all_fail,
            "rejected": all_rej,
            "success_rate": round(all_ok / admitted_all * 100, 2) if admitted_all else 0.0,
            "total_ms": {
                "p50": self.pct(all_total_ms, 50),
                "p95": self.pct(all_total_ms, 95),
                "p99": self.pct(all_total_ms, 99),
                "max": round(max(all_total_ms), 1) if all_total_ms else 0.0,
            },
            "ttft_ms": {
                "p50": self.pct(all_ttft, 50),
                "p95": self.pct(all_ttft, 95),
                "p99": self.pct(all_ttft, 99),
                "max": round(max(all_ttft), 1) if all_ttft else 0.0,
            },
            "admission": {
                "max_active_cfg": self.max_active_cfg,
                "max_queue_cfg": self.max_queue_cfg,
                "active_peak": self.active_peak,
                "queued_peak": self.queued_peak,
            },
            "isolation_violations": len(self.isolation_violations),
        }
        return out


# ---------------------------------------------------------------- 用户池
def build_users(prefix: str, count: int) -> list[dict]:
    """每个用户一份带唯一标记的档案（孩子昵称即串号探针）。"""
    users = []
    for i in range(count):
        uid = f"{prefix}_{i:05d}"
        users.append(
            {
                "userId": uid,
                "kid": f"童童{i:05d}",          # 唯一标记：出现在别人回答里即串号
                "profile": {
                    "userNickname": f"家长{i:05d}",
                    "birthday": "1990-01-01",
                    "parent_role": ROLES[i % len(ROLES)],
                    "children": [
                        {
                            "childId": f"c{i:05d}",
                            "childNickname": f"童童{i:05d}",
                            "childBirthDate": "2015-03-12",
                            "gender": "男" if i % 2 == 0 else "女",
                        }
                    ],
                },
            }
        )
    return users


def kid_marks(users: list[dict]) -> dict[str, str]:
    return {u["userId"]: u["kid"] for u in users}


# ---------------------------------------------------------------- 单次调用
def _headers(api_key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    return h


async def call_crud(
    client: httpx.AsyncClient, url: str, api_key: str, endpoint: str, user: dict
) -> CallResult:
    t0 = time.perf_counter()
    detail = ""
    try:
        if endpoint == "register":
            r = await client.post(
                f"{url}/api/mp/register",
                json={"userId": user["userId"], "profile": user["profile"]},
                headers=_headers(api_key),
            )
            ok = r.status_code in (200, 201)
        elif endpoint == "update":
            r = await client.post(
                f"{url}/api/mp/update",
                json={"userId": user["userId"], "profile": user["profile"]},
                headers=_headers(api_key),
            )
            ok = r.status_code == 200
        else:  # profile
            r = await client.get(
                f"{url}/api/mp/profile",
                params={"userId": user["userId"]},
                headers=_headers(api_key),
            )
            ok = r.status_code == 200
            if ok:
                try:
                    kids = (r.json() or {}).get("children") or []
                    names = {k.get("childNickname") for k in kids}
                    if names and user["kid"] not in names:
                        ok = False
                        detail = f"档案回读异常：{names}"
                except Exception:  # noqa: BLE001
                    pass
        return CallResult(
            endpoint=endpoint,
            user_id=user["userId"],
            status=r.status_code,
            ok=ok,
            total_ms=(time.perf_counter() - t0) * 1000,
            detail=detail if detail else ("" if ok else r.text[:160]),
        )
    except Exception as e:  # noqa: BLE001
        return CallResult(
            endpoint=endpoint, user_id=user["userId"], status=0, ok=False,
            error_type=type(e).__name__, total_ms=(time.perf_counter() - t0) * 1000,
            detail=str(e)[:160],
        )


async def call_query(
    client: httpx.AsyncClient, url: str, api_key: str, user: dict, question: str,
    timeout: float, tag: bool,
) -> CallResult:
    """非流式 /api/mp/query。"""
    q = f"{question}（标记 {uuid.uuid4().hex[:8]}）" if tag else question
    body = {
        "userId": user["userId"],
        "sessionId": f"sess_{user['userId']}",
        "query": q,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/api/mp/query", json=body, headers=_headers(api_key), timeout=timeout
        )
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            data = r.json() or {}
            return CallResult(
                endpoint="query", user_id=user["userId"], status=200, ok=bool(data.get("answer")),
                total_ms=ms, answer=data.get("answer") or "",
                detail="" if data.get("answer") else "空答案",
            )
        if r.status_code in (409, 429, 503):
            return CallResult(
                endpoint="query", user_id=user["userId"], status=r.status_code, ok=False,
                rejected=True, total_ms=ms, detail=r.text[:160],
            )
        return CallResult(
            endpoint="query", user_id=user["userId"], status=r.status_code, ok=False,
            total_ms=ms, detail=r.text[:160],
        )
    except Exception as e:  # noqa: BLE001
        return CallResult(
            endpoint="query", user_id=user["userId"], status=0, ok=False,
            error_type=type(e).__name__, total_ms=(time.perf_counter() - t0) * 1000,
            detail=str(e)[:160],
        )


async def call_stream(
    client: httpx.AsyncClient, url: str, api_key: str, user: dict, question: str,
    timeout: float, tag: bool,
) -> CallResult:
    """SSE /api/mp/query/stream：逐帧解析，区分 done / error。"""
    q = f"{question}（标记 {uuid.uuid4().hex[:8]}）" if tag else question
    body = {
        "userId": user["userId"],
        "sessionId": f"sess_{user['userId']}",
        "query": q,
    }
    t0 = time.perf_counter()
    ttft = 0.0
    answer_parts: list[str] = []
    terminal = "none"
    error_type = ""
    status = 0
    try:
        async with client.stream(
            "POST", f"{url}/api/mp/query/stream", json=body,
            headers=_headers(api_key), timeout=timeout,
        ) as resp:
            status = resp.status_code
            if status != 200:
                text = (await resp.aread()).decode("utf-8", "ignore")[:160]
                return CallResult(
                    endpoint="stream", user_id=user["userId"], status=status, ok=False,
                    rejected=status in (409, 429, 503), total_ms=(time.perf_counter() - t0) * 1000,
                    sse_terminal="none", detail=text,
                )
            event = ""
            async for raw in resp.aiter_lines():
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                try:
                    data = json.loads(payload)
                except Exception:  # noqa: BLE001
                    data = {}
                if event == "token":
                    if not ttft:
                        ttft = (time.perf_counter() - t0) * 1000
                    answer_parts.append(data.get("text") or "")
                elif event == "done":
                    terminal = "done"
                    answer_parts = [data.get("answer") or "".join(answer_parts)]
                elif event == "error":
                    terminal = "error"
                    error_type = str(data.get("error_type") or data.get("detail") or "")[:120]
    except Exception as e:  # noqa: BLE001
        return CallResult(
            endpoint="stream", user_id=user["userId"], status=status, ok=False,
            error_type=type(e).__name__, total_ms=(time.perf_counter() - t0) * 1000,
            sse_terminal="none", detail=str(e)[:160],
        )

    ms = (time.perf_counter() - t0) * 1000
    answer = "".join(answer_parts)
    ok = terminal == "done" and bool(answer)
    return CallResult(
        endpoint="stream", user_id=user["userId"], status=status, ok=ok,
        error_type=error_type, ttft_ms=ttft or ms, total_ms=ms, answer=answer,
        sse_terminal=terminal, detail="" if ok else f"terminal={terminal} {error_type}",
    )


# ---------------------------------------------------------------- 采样
async def sampler(client: httpx.AsyncClient, url: str, metrics: Metrics,
                  stop: asyncio.Event, poll_ms: float) -> None:
    while not stop.is_set():
        try:
            r = await client.get(f"{url}/api/concurrency/status", timeout=3.0)
            d = r.json() or {}
            metrics.active_peak = max(metrics.active_peak, int(d.get("active") or 0))
            metrics.queued_peak = max(metrics.queued_peak, int(d.get("queued") or 0))
            metrics.max_active_cfg = int(d.get("max_active") or metrics.max_active_cfg)
            metrics.max_queue_cfg = int(d.get("max_queue") or metrics.max_queue_cfg)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(max(poll_ms, 10) / 1000.0)


async def fetch_capacity(client: httpx.AsyncClient, url: str) -> tuple[int, int]:
    try:
        r = await client.get(f"{url}/api/concurrency/status", timeout=5.0)
        d = r.json() or {}
        return int(d.get("max_active") or 0), int(d.get("max_queue") or 0)
    except Exception:  # noqa: BLE001
        return 0, 0


# ---------------------------------------------------------------- 执行器
def pick_endpoint(mode: str, sse_ratio: float, crud_ratio: float) -> str:
    r = random.random()
    if mode == "query":
        return "query"
    if mode == "stream":
        return "stream"
    if mode == "crud":
        pool = ["register", "update", "profile"]
        return pool[int(r * len(pool)) % len(pool)]
    if mode == "mixed":
        return "stream" if r < sse_ratio else "query"
    # all：query 类占 (1-crud_ratio)，其余 CRUD
    if r < crud_ratio:
        pool = ["register", "update", "profile"]
        return pool[int(r * 1000) % len(pool)]
    return "stream" if random.random() < sse_ratio else "query"


async def run_once(args, client, endpoint, user, metrics) -> None:
    if endpoint in ("register", "update", "profile"):
        res = await call_crud(client, args.url, args.api_key, endpoint, user)
    elif endpoint == "stream":
        res = await call_stream(client, args.url, args.api_key, user, args.question,
                                args.timeout, not args.reuse_question)
    else:
        res = await call_query(client, args.url, args.api_key, user, args.question,
                               args.timeout, not args.reuse_question)
    metrics.results.append(res)


def check_isolation(metrics: Metrics, marks: dict[str, str]) -> None:
    for r in metrics.results:
        if not r.answer:
            continue
        for uid, kid in marks.items():
            if uid == r.user_id or not kid:
                continue
            if kid in r.answer:
                metrics.isolation_violations.append(
                    {"user_id": r.user_id, "endpoint": r.endpoint, "leaked_from": uid, "mark": kid}
                )
                break


async def wait_drain(client, url: str, timeout: float = 20.0) -> dict:
    """等待槽位与队列归零（资源泄漏断言）。"""
    end = time.perf_counter() + timeout
    last: dict = {}
    while time.perf_counter() < end:
        try:
            r = await client.get(f"{url}/api/concurrency/status", timeout=3.0)
            last = r.json() or {}
            if int(last.get("active") or 0) == 0 and int(last.get("queued") or 0) == 0:
                return last
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.5)
    return last


# ---------------------------------------------------------------- 模式实现
async def mode_steady(args) -> Metrics:
    metrics = Metrics()
    users = build_users(args.user_prefix, max(args.concurrency * 2, args.users, 8))
    marks = kid_marks(users)
    limits = httpx.Limits(max_connections=args.concurrency + 20, max_keepalive_connections=args.concurrency + 20)
    async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as client:
        metrics.max_active_cfg, metrics.max_queue_cfg = await fetch_capacity(client, args.url)
        stop = asyncio.Event()
        poll = asyncio.create_task(sampler(client, args.url, metrics, stop, args.poll_ms))

        slot = asyncio.Semaphore(args.concurrency)
        sent = 0
        start = time.perf_counter()

        async def worker(i: int) -> None:
            nonlocal sent
            while True:
                if args.total and sent >= args.total:
                    return
                if time.perf_counter() - start >= args.duration:
                    return
                async with slot:
                    if args.total and sent >= args.total:
                        return
                    sent += 1
                    # 循环取用户，保证同一 userId 不会被并发复用（否则测到的是 409）
                    user = users[(i + sent) % len(users)]
                    ep = pick_endpoint(args.endpoint, args.sse_ratio, args.crud_ratio)
                    await run_once(args, client, ep, user, metrics)

        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
        stop.set()
        await poll
        drain = await wait_drain(client, args.url)
        check_isolation(metrics, marks)
        metrics_dict = metrics.summary()
        metrics_dict["drain_final"] = drain
        metrics_dict["users"] = len(users)
        return _wrap(metrics, metrics_dict)


async def mode_boundary(args) -> Metrics:
    """同步起跑 N 个请求：验证槽位/队列/拒绝数量。"""
    metrics = Metrics()
    users = build_users(args.user_prefix, max(args.level, args.users, 8))
    marks = kid_marks(users)
    limits = httpx.Limits(max_connections=args.level + 20, max_keepalive_connections=args.level + 20)
    async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as client:
        metrics.max_active_cfg, metrics.max_queue_cfg = await fetch_capacity(client, args.url)
        stop = asyncio.Event()
        poll = asyncio.create_task(sampler(client, args.url, metrics, stop, args.poll_ms))
        barrier = asyncio.Barrier(args.level) if hasattr(asyncio, "Barrier") else None
        start_at = time.perf_counter() + 1.0

        async def one(i: int) -> None:
            user = users[i % len(users)]
            ep = pick_endpoint(args.endpoint, args.sse_ratio, args.crud_ratio)
            if barrier is not None:
                await barrier.wait()
            else:
                delay = start_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            await run_once(args, client, ep, user, metrics)

        await asyncio.gather(*(one(i) for i in range(args.level)))
        stop.set()
        await poll
        drain = await wait_drain(client, args.url)
        check_isolation(metrics, marks)
        d = metrics.summary()
        d["drain_final"] = drain
        d["users"] = len(users)
        return _wrap(metrics, d)


async def mode_spike(args) -> Metrics:
    metrics = Metrics()
    users = build_users(args.user_prefix, max(args.level * 2, args.users, 8))
    marks = kid_marks(users)
    limits = httpx.Limits(max_connections=args.level + 20, max_keepalive_connections=args.level + 20)
    async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as client:
        metrics.max_active_cfg, metrics.max_queue_cfg = await fetch_capacity(client, args.url)
        round_stats = []
        for rd in range(args.rounds):
            round_metrics = Metrics()
            stop = asyncio.Event()
            poll = asyncio.create_task(sampler(client, args.url, round_metrics, stop, args.poll_ms))
            barrier = asyncio.Barrier(args.level) if hasattr(asyncio, "Barrier") else None

            async def one(i: int) -> None:
                user = users[(rd * args.level + i) % len(users)]
                ep = pick_endpoint(args.endpoint, args.sse_ratio, args.crud_ratio)
                if barrier is not None:
                    await barrier.wait()
                await run_once(args, client, ep, user, round_metrics)

            await asyncio.gather(*(one(i) for i in range(args.level)))
            stop.set()
            await poll
            check_isolation(round_metrics, marks)
            metrics.results += round_metrics.results
            metrics.isolation_violations += round_metrics.isolation_violations
            metrics.active_peak = max(metrics.active_peak, round_metrics.active_peak)
            metrics.queued_peak = max(metrics.queued_peak, round_metrics.queued_peak)
            round_stats.append(round_metrics.summary()["overall"])
            print(f"  第 {rd + 1}/{args.rounds} 轮：{round_stats[-1]}")
            await asyncio.sleep(args.round_gap)

        drain = await wait_drain(client, args.url)
        d = metrics.summary()
        d["rounds"] = round_stats
        d["drain_final"] = drain
        d["users"] = len(users)
        return _wrap(metrics, d)


def _wrap(metrics: Metrics, d: dict) -> Metrics:
    metrics._summary = d  # type: ignore[attr-defined]
    return metrics


# ---------------------------------------------------------------- 判定
def judge(args, d: dict) -> tuple[bool, list[str]]:
    fails: list[str] = []
    ov = d["overall"]
    adm = ov["admission"]

    if adm["max_active_cfg"] and adm["active_peak"] > adm["max_active_cfg"]:
        fails.append(f"活跃槽位超限：peak={adm['active_peak']} > max_active={adm['max_active_cfg']}")
    if adm["max_queue_cfg"] and adm["queued_peak"] > adm["max_queue_cfg"]:
        fails.append(f"等待队列超限：peak={adm['queued_peak']} > max_queue={adm['max_queue_cfg']}")
    if ov["isolation_violations"] > 0:
        fails.append(f"跨用户数据串号 {ov['isolation_violations']} 次（P0）")

    # 5xx 一律视为服务异常（429/503 属背压拒绝，已在 rejected 中单独统计）
    server_err = 0
    for s in d["endpoints"].values():
        for code, cnt in s["status_dist"].items():
            if code.isdigit() and int(code) >= 500 and code != "503":
                server_err += cnt
    if server_err:
        fails.append(f"5xx 响应 {server_err} 次")

    # CRUD-only 模式要求零失败
    if args.endpoint == "crud":
        if ov["fail"] or ov["rejected"]:
            fails.append(f"CRUD 模式出现失败/拒绝：fail={ov['fail']} rejected={ov['rejected']}")
    else:
        if ov["success_rate"] < args.min_success:
            fails.append(f"成功率 {ov['success_rate']}% < 门槛 {args.min_success}%")
        if args.max_total_p95 and ov["total_ms"]["p95"] > args.max_total_p95:
            fails.append(f"完整响应 P95 {ov['total_ms']['p95']}ms > 门槛 {args.max_total_p95}ms")
        if args.max_ttft_p95 and ov["ttft_ms"]["p95"] and ov["ttft_ms"]["p95"] > args.max_ttft_p95:
            fails.append(f"TTFT P95 {ov['ttft_ms']['p95']}ms > 门槛 {args.max_ttft_p95}ms")

    drain = d.get("drain_final") or {}
    if int(drain.get("active") or 0) != 0 or int(drain.get("queued") or 0) != 0:
        fails.append(f"结束后未归零：active={drain.get('active')} queued={drain.get('queued')}（资源泄漏）")

    return (not fails), fails


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="微信小程序通道 /api/mp/* 并发压测")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default=os.getenv("MP_API_KEY", ""), help="与 .env 的 MP_API_KEY 一致；为空则不带 X-API-Key")
    p.add_argument("--endpoint", choices=("query", "stream", "mixed", "crud", "all"), default="mixed")
    p.add_argument("--sse-ratio", type=float, default=0.5, help="mixed/all 中流式占比")
    p.add_argument("--crud-ratio", type=float, default=0.2, help="all 模式中 CRUD 接口占比")
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--reuse-question", action="store_true", help="不追加唯一标记（会命中 embedding 缓存，仅缓存测试用）")
    p.add_argument("--user-prefix", default="mpc")
    p.add_argument("--users", type=int, default=0, help="userId 池大小；0=自动取 max(并发*2, 50)")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--poll-ms", type=float, default=50.0)
    p.add_argument("--output", default="results/mp-concurrency-latest.json")
    p.add_argument("--min-success", type=float, default=99.0)
    p.add_argument("--max-ttft-p95", type=float, default=0.0, help="0=不校验")
    p.add_argument("--max-total-p95", type=float, default=0.0, help="0=不校验")
    p.add_argument("--seed", type=int, default=20260909)

    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("steady", help="固定并发持续施压")
    s.add_argument("--concurrency", type=int, required=True)
    s.add_argument("--duration", type=float, default=600.0)
    s.add_argument("--total", type=int, default=0)

    b = sub.add_parser("boundary", help="同步起跑 N 个请求，验证槽位与拒绝")
    b.add_argument("--level", type=int, required=True)

    k = sub.add_parser("spike", help="多轮尖峰")
    k.add_argument("--level", type=int, required=True)
    k.add_argument("--rounds", type=int, default=10)
    k.add_argument("--round-gap", type=float, default=1.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    if args.users == 0:
        args.users = 50

    print(f"[mp-concurrency] 目标 {args.url}  endpoint={args.endpoint}  mode={args.mode}")
    print(f"[mp-concurrency] API-Key: {'已配置' if args.api_key else '未配置（服务端 MP_API_KEY 为空时才允许）'}")

    if args.mode == "steady":
        metrics = asyncio.run(mode_steady(args))
    elif args.mode == "boundary":
        metrics = asyncio.run(mode_boundary(args))
    else:
        metrics = asyncio.run(mode_spike(args))

    d = getattr(metrics, "_summary", metrics.summary())
    d["meta"] = {
        "url": args.url, "endpoint_mode": args.endpoint, "mode": args.mode,
        "sse_ratio": args.sse_ratio, "crud_ratio": args.crud_ratio,
        "question": args.question, "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
    }

    ok, fails = judge(args, d)
    d["verdict"] = {"pass": ok, "fails": fails}

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    ov = d["overall"]
    print("\n================ 汇总 ================")
    print(f"请求数 {ov['requests']}  成功 {ov['ok']}  失败 {ov['fail']}  拒绝 {ov['rejected']}")
    print(f"成功率 {ov['success_rate']}%   串号 {ov['isolation_violations']}")
    print(f"TTFT   P50/P95/P99 = {ov['ttft_ms']['p50']}/{ov['ttft_ms']['p95']}/{ov['ttft_ms']['p99']} ms")
    print(f"Total  P50/P95/P99 = {ov['total_ms']['p50']}/{ov['total_ms']['p95']}/{ov['total_ms']['p99']} ms")
    print(f"准入峰值 active={ov['admission']['active_peak']}/{ov['admission']['max_active_cfg']}  "
          f"queued={ov['admission']['queued_peak']}/{ov['admission']['max_queue_cfg']}")
    for ep, s in d["endpoints"].items():
        print(f"  - {ep}: total={s['total']} ok={s['ok']} fail={s['fail']} rej={s['rejected']} "
              f"rate={s['success_rate']}% status={s['status_dist']}")
    print("--------------------------------------")
    if ok:
        print("最终结论：通过")
    else:
        print("最终结论：不通过")
        for x in fails:
            print(f"  FAIL: {x}")
    print(f"原始结果：{out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""rag-psychology 并发压力测试工具。

目标服务版本：GitHub main fd976ec2（4.2并发初版）及兼容版本。

与旧脚本相比，本工具不会把 SSE 的 HTTP 200 直接视为业务成功，而是解析
queue/started/token/done/error 全部事件；同时轮询 /api/concurrency/status，验证
服务端 active/queued 没有突破 20/40 上限。

典型用法：

1. 精确边界测试（建议连接固定 5 秒响应的 Mock LLM）：
   python scripts/concurrency_pressure_test.py boundary --level 61 --strict-boundary

2. 真实模型 20 并发稳态测试：
   python scripts/concurrency_pressure_test.py steady --concurrency 20 --duration 600

3. 80 请求尖峰，连续 10 轮：
   python scripts/concurrency_pressure_test.py spike --level 80 --rounds 10 --strict-boundary

注意：
- 测试会通过公开注册接口创建唯一 load 用户，不读取任何 .env 文件；
- 正式边界测试必须使用响应时间可控的 Mock，否则请求过快完成会使 61 个请求
  无法同时占据 20 active + 40 queue，拒绝数自然不会稳定为 1；
- 输出不保存密码、JWT、问题正文或回答正文。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx


DEFAULT_QUESTION = "请给出三个帮助青少年缓解考试焦虑的具体步骤。"
QUEUE_FULL_CODE = "AI_QUEUE_FULL"


@dataclass
class RequestSample:
    """单次请求结果；不保存敏感正文与认证信息。"""

    index: int
    worker_id: int
    endpoint: str
    http_status: int | str
    terminal: str
    success: bool
    correct_rejection: bool
    error_code: str = ""
    request_id: str = ""
    queued: bool = False
    queue_wait_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    total_ms: float = 0.0
    answer_chars: int = 0
    detail: str = ""


@dataclass
class CapacityPoint:
    """服务端准入水位采样。"""

    elapsed_ms: float
    active: int
    queued: int
    accepting: bool


@dataclass
class HealthPoint:
    """满载期间健康检查采样。"""

    elapsed_ms: float
    status: int | str
    latency_ms: float


@dataclass
class MonitorResult:
    capacity: list[CapacityPoint] = field(default_factory=list)
    health: list[HealthPoint] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Thresholds:
    min_success_rate: float
    ttft_p95_ms: float
    total_p95_ms: float
    health_p95_ms: float


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    """线性插值分位数；空序列返回 None。"""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def rounded(value: Optional[float]) -> Optional[float]:
    return round(value, 1) if value is not None else None


def username_prefix(raw: str) -> str:
    """生成符合 3～32 字符限制的唯一测试账号前缀。"""

    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch == "_") or "load"
    return cleaned[:12]


def user_fingerprint(username: str) -> str:
    """报告只保留不可逆短指纹，不记录完整账号。"""

    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:10]


def choose_endpoint(configured: str, rng: random.Random, sse_ratio: float) -> str:
    if configured != "mixed":
        return configured
    return "stream" if rng.random() < sse_ratio else "query"


def automatic_thresholds(concurrency: int) -> Thresholds:
    """采用《并发压力测试验收方案》的候选门槛。"""

    if concurrency <= 5:
        return Thresholds(0.995, 1500.0, 7000.0, 200.0)
    if concurrency <= 20:
        return Thresholds(0.99, 2000.0, 8000.0, 200.0)
    if concurrency <= 40:
        return Thresholds(0.98, 7000.0, 13000.0, 200.0)
    return Thresholds(0.98, 12000.0, 20000.0, 200.0)


async def parse_json_safely(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


async def create_test_users(
    client: httpx.AsyncClient,
    count: int,
    prefix: str,
    password: str,
    auth_parallelism: int,
) -> tuple[list[str], list[str]]:
    """创建并登录独立测试账号，返回用户名与 token；日志不打印 token。"""

    run_id = uuid.uuid4().hex[:8]
    safe_prefix = username_prefix(prefix)
    usernames = [f"{safe_prefix}_{run_id}_{index:03d}"[:32] for index in range(count)]
    semaphore = asyncio.Semaphore(max(1, auth_parallelism))

    async def register_and_login(username: str) -> str:
        async with semaphore:
            register = await client.post(
                "/api/auth/register",
                json={"username": username, "password": password},
            )
            if register.status_code not in (201, 409):
                raise RuntimeError(
                    f"注册失败 user={user_fingerprint(username)} "
                    f"status={register.status_code} body={register.text[:120]!r}"
                )
            login = await client.post(
                "/api/auth/login",
                json={"username": username, "password": password},
            )
            if login.status_code != 200:
                raise RuntimeError(
                    f"登录失败 user={user_fingerprint(username)} "
                    f"status={login.status_code} body={login.text[:120]!r}"
                )
            payload = await parse_json_safely(login)
            token = str(payload.get("access_token") or "")
            if not token:
                raise RuntimeError(f"登录响应缺少 access_token user={user_fingerprint(username)}")
            return token

    tokens = await asyncio.gather(*(register_and_login(name) for name in usernames))
    return usernames, list(tokens)


async def call_query(
    client: httpx.AsyncClient,
    token: str,
    question: str,
    persist: bool,
    index: int,
    worker_id: int,
) -> RequestSample:
    """调用同步语义接口，并验证业务响应而非只看 HTTP 状态。"""

    started = time.perf_counter()
    try:
        response = await client.post(
            "/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": question, "persist": persist},
        )
        total_ms = (time.perf_counter() - started) * 1000
        payload = await parse_json_safely(response)
        error_code = str(payload.get("code") or "")
        if response.status_code == 200:
            answer = str(payload.get("answer") or "")
            valid = bool(answer.strip()) and bool(payload.get("request_id"))
            queue = payload.get("queue") if isinstance(payload.get("queue"), dict) else {}
            return RequestSample(
                index=index,
                worker_id=worker_id,
                endpoint="query",
                http_status=response.status_code,
                terminal="done" if valid else "invalid_success_payload",
                success=valid,
                correct_rejection=False,
                request_id=str(payload.get("request_id") or ""),
                queued=bool(queue.get("queued")),
                queue_wait_ms=float(queue.get("wait_ms") or 0.0),
                total_ms=total_ms,
                answer_chars=len(answer),
                detail="" if valid else "200 响应缺少答案或 request_id",
            )
        correct_rejection = response.status_code == 429 and error_code == QUEUE_FULL_CODE
        return RequestSample(
            index=index,
            worker_id=worker_id,
            endpoint="query",
            http_status=response.status_code,
            terminal="rejected" if correct_rejection else "http_error",
            success=False,
            correct_rejection=correct_rejection,
            error_code=error_code,
            total_ms=total_ms,
            detail=str(payload.get("detail") or response.text[:160]),
        )
    except Exception as exc:  # noqa: BLE001 - 压测必须把单请求异常落样本后继续
        return RequestSample(
            index=index,
            worker_id=worker_id,
            endpoint="query",
            http_status="EXCEPTION",
            terminal="exception",
            success=False,
            correct_rejection=False,
            total_ms=(time.perf_counter() - started) * 1000,
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )


async def call_stream(
    client: httpx.AsyncClient,
    token: str,
    question: str,
    persist: bool,
    index: int,
    worker_id: int,
) -> RequestSample:
    """完整消费 SSE，并以 done/error 业务终态判定成功或失败。"""

    started = time.perf_counter()
    request_id = ""
    queue_wait_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    queued = False
    answer_chars = 0
    terminal = "stream_closed"
    error_code = ""
    detail = ""

    try:
        async with client.stream(
            "POST",
            "/api/query/stream",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": question, "persist": persist},
        ) as response:
            if response.status_code != 200:
                raw = await response.aread()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = {}
                error_code = str(payload.get("code") or "") if isinstance(payload, dict) else ""
                correct_rejection = response.status_code == 429 and error_code == QUEUE_FULL_CODE
                return RequestSample(
                    index=index,
                    worker_id=worker_id,
                    endpoint="stream",
                    http_status=response.status_code,
                    terminal="rejected" if correct_rejection else "http_error",
                    success=False,
                    correct_rejection=correct_rejection,
                    error_code=error_code,
                    total_ms=(time.perf_counter() - started) * 1000,
                    detail=str(payload.get("detail") or "")[:160] if isinstance(payload, dict) else "",
                )

            current_event = "message"
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line.split(":", 1)[1].strip())
                except json.JSONDecodeError:
                    terminal = "malformed_sse"
                    detail = "SSE data 不是合法 JSON"
                    break
                if not isinstance(data, dict):
                    continue
                if current_event == "queue":
                    queued = True
                    request_id = str(data.get("request_id") or request_id)
                elif current_event == "started":
                    request_id = str(data.get("request_id") or request_id)
                    queue_wait_ms = float(data.get("queue_wait_ms") or 0.0)
                elif current_event == "token":
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    answer_chars += len(str(data.get("text") or ""))
                elif current_event == "done":
                    if ttft_ms is None:
                        # 高危拦截可能没有 token；done 即用户首次获得有效回答。
                        ttft_ms = (time.perf_counter() - started) * 1000
                    request_id = str(data.get("request_id") or request_id)
                    answer_chars = max(answer_chars, len(str(data.get("answer") or "")))
                    terminal = "done"
                    break
                elif current_event == "error":
                    terminal = "error"
                    error_code = str(data.get("code") or data.get("error_type") or "")
                    detail = str(data.get("detail") or "")[:160]
                    break

            total_ms = (time.perf_counter() - started) * 1000
            success = terminal == "done" and answer_chars > 0 and bool(request_id)
            return RequestSample(
                index=index,
                worker_id=worker_id,
                endpoint="stream",
                http_status=response.status_code,
                terminal=terminal if success or terminal != "done" else "invalid_done",
                success=success,
                correct_rejection=False,
                error_code=error_code,
                request_id=request_id,
                queued=queued,
                queue_wait_ms=queue_wait_ms,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                answer_chars=answer_chars,
                detail=detail if detail else ("" if success else "SSE 未得到合法 done"),
            )
    except Exception as exc:  # noqa: BLE001
        return RequestSample(
            index=index,
            worker_id=worker_id,
            endpoint="stream",
            http_status="EXCEPTION",
            terminal="exception",
            success=False,
            correct_rejection=False,
            request_id=request_id,
            queued=queued,
            queue_wait_ms=queue_wait_ms,
            ttft_ms=ttft_ms,
            total_ms=(time.perf_counter() - started) * 1000,
            answer_chars=answer_chars,
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )


async def execute_request(
    client: httpx.AsyncClient,
    endpoint: str,
    token: str,
    question: str,
    persist: bool,
    index: int,
    worker_id: int,
) -> RequestSample:
    if endpoint == "stream":
        return await call_stream(client, token, question, persist, index, worker_id)
    return await call_query(client, token, question, persist, index, worker_id)


async def monitor_service(
    client: httpx.AsyncClient,
    stop_event: asyncio.Event,
    poll_seconds: float,
) -> MonitorResult:
    """高频采集准入水位，低频采集健康检查时延。"""

    result = MonitorResult()
    started = time.perf_counter()
    next_health_at = 0.0
    while not stop_event.is_set():
        elapsed = time.perf_counter() - started
        try:
            status = await client.get("/api/concurrency/status", timeout=2.0)
            payload = await parse_json_safely(status)
            if status.status_code == 200:
                result.capacity.append(
                    CapacityPoint(
                        elapsed_ms=elapsed * 1000,
                        active=int(payload.get("active") or 0),
                        queued=int(payload.get("queued") or 0),
                        accepting=bool(payload.get("accepting")),
                    )
                )
            else:
                result.errors.append(f"status HTTP {status.status_code}")
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"status {type(exc).__name__}: {exc}"[:160])

        if elapsed >= next_health_at:
            health_started = time.perf_counter()
            try:
                health = await client.get("/api/health", timeout=2.0)
                result.health.append(
                    HealthPoint(
                        elapsed_ms=elapsed * 1000,
                        status=health.status_code,
                        latency_ms=(time.perf_counter() - health_started) * 1000,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                result.health.append(
                    HealthPoint(
                        elapsed_ms=elapsed * 1000,
                        status="EXCEPTION",
                        latency_ms=(time.perf_counter() - health_started) * 1000,
                    )
                )
                result.errors.append(f"health {type(exc).__name__}: {exc}"[:160])
            next_health_at = elapsed + 1.0

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
        except asyncio.TimeoutError:
            pass
    return result


async def run_synchronized_wave(
    client: httpx.AsyncClient,
    tokens: list[str],
    level: int,
    endpoint: str,
    sse_ratio: float,
    question: str,
    reuse_question: bool,
    persist: bool,
    index_offset: int,
    seed: int,
) -> list[RequestSample]:
    """所有请求先等待同一事件，再在同一事件循环轮次内起跑。"""

    start_event = asyncio.Event()
    rng = random.Random(seed)
    endpoints = [choose_endpoint(endpoint, rng, sse_ratio) for _ in range(level)]

    async def job(index: int) -> RequestSample:
        await start_event.wait()
        current_question = (
            question
            if reuse_question
            else f"{question}\n[pressure-test-id:{index_offset + index}]"
        )
        return await execute_request(
            client=client,
            endpoint=endpoints[index],
            token=tokens[index],
            question=current_question,
            persist=persist,
            index=index_offset + index,
            worker_id=index,
        )

    tasks = [asyncio.create_task(job(index)) for index in range(level)]
    await asyncio.sleep(0.1)  # 给所有任务一次进入 barrier 的调度机会
    start_event.set()
    return list(await asyncio.gather(*tasks))


async def run_steady_workers(
    client: httpx.AsyncClient,
    tokens: list[str],
    concurrency: int,
    duration: float,
    total: int,
    endpoint: str,
    sse_ratio: float,
    question: str,
    reuse_question: bool,
    persist: bool,
    seed: int,
) -> list[RequestSample]:
    """每个 worker 独占一个账号并串行发请求，避免单用户 in-flight 产生假 409。"""

    started = time.perf_counter()
    start_event = asyncio.Event()
    samples: list[RequestSample] = []
    state = {"next_index": 0}
    rng = random.Random(seed)

    def claim_index() -> Optional[int]:
        if total > 0 and state["next_index"] >= total:
            return None
        index = state["next_index"]
        state["next_index"] += 1
        return index

    async def worker(worker_id: int) -> None:
        await start_event.wait()
        while time.perf_counter() - started < duration:
            index = claim_index()
            if index is None:
                return
            selected = choose_endpoint(endpoint, rng, sse_ratio)
            current_question = (
                question if reuse_question else f"{question}\n[pressure-test-id:{index}]"
            )
            sample = await execute_request(
                client=client,
                endpoint=selected,
                token=tokens[worker_id],
                question=current_question,
                persist=persist,
                index=index,
                worker_id=worker_id,
            )
            samples.append(sample)

    tasks = [asyncio.create_task(worker(worker_id)) for worker_id in range(concurrency)]
    await asyncio.sleep(0.1)
    start_event.set()
    await asyncio.gather(*tasks)
    return samples


def summarize_samples(samples: list[RequestSample]) -> dict[str, Any]:
    successes = [sample for sample in samples if sample.success]
    rejections = [sample for sample in samples if sample.correct_rejection]
    failures = [sample for sample in samples if not sample.success and not sample.correct_rejection]
    admitted = len(samples) - len(rejections)
    ttft = [sample.ttft_ms for sample in successes if sample.ttft_ms is not None]
    total = [sample.total_ms for sample in successes]
    queue_wait = [sample.queue_wait_ms for sample in successes if sample.queue_wait_ms is not None]
    terminals: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for sample in samples:
        terminals[sample.terminal] = terminals.get(sample.terminal, 0) + 1
        key = str(sample.http_status)
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "requests": len(samples),
        "admitted": admitted,
        "business_success": len(successes),
        "correct_rejections": len(rejections),
        "unexpected_failures": len(failures),
        "success_rate_admitted": round(len(successes) / admitted, 6) if admitted else 1.0,
        "http_statuses": statuses,
        "terminals": terminals,
        "ttft_ms": distribution(ttft),
        "total_ms": distribution(total),
        "queue_wait_ms": distribution(queue_wait),
    }


def distribution(values: Iterable[float]) -> dict[str, Optional[float]]:
    values_list = [float(value) for value in values]
    if not values_list:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values_list),
        "mean": rounded(statistics.mean(values_list)),
        "p50": rounded(percentile(values_list, 50)),
        "p95": rounded(percentile(values_list, 95)),
        "p99": rounded(percentile(values_list, 99)),
        "max": rounded(max(values_list)),
    }


def summarize_monitor(monitor: MonitorResult) -> dict[str, Any]:
    active = [point.active for point in monitor.capacity]
    queued = [point.queued for point in monitor.capacity]
    health_latency = [point.latency_ms for point in monitor.health if point.status == 200]
    health_failures = sum(1 for point in monitor.health if point.status != 200)
    return {
        "capacity_samples": len(monitor.capacity),
        "active_max": max(active, default=0),
        "queued_max": max(queued, default=0),
        "health": distribution(health_latency),
        "health_failures": health_failures,
        "monitor_errors": monitor.errors,
    }


def evaluate(
    args: argparse.Namespace,
    samples: list[RequestSample],
    sample_summary: dict[str, Any],
    monitor_summary: dict[str, Any],
    max_active: int,
    max_queue: int,
    level_or_concurrency: int,
) -> dict[str, Any]:
    """计算机器可判定的发布门禁；每个检查项都给出证据。"""

    thresholds = automatic_thresholds(level_or_concurrency)
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, actual: Any, expected: str) -> None:
        checks[name] = {"passed": bool(passed), "actual": actual, "expected": expected}

    active_max = int(monitor_summary["active_max"])
    queued_max = int(monitor_summary["queued_max"])
    add("active_not_exceeded", active_max <= max_active, active_max, f"<= {max_active}")
    add("queue_not_exceeded", queued_max <= max_queue, queued_max, f"<= {max_queue}")
    add(
        "health_success",
        monitor_summary["health_failures"] == 0,
        monitor_summary["health_failures"],
        "0",
    )
    health_p95 = monitor_summary["health"]["p95"]
    add(
        "health_p95",
        health_p95 is not None and health_p95 <= thresholds.health_p95_ms,
        health_p95,
        f"<= {thresholds.health_p95_ms:.0f} ms",
    )

    if args.mode in ("boundary", "spike") and args.strict_boundary:
        expected_rejections_per_wave = max(0, level_or_concurrency - max_active - max_queue)
        waves = args.rounds if args.mode == "spike" else 1
        expected_rejections = expected_rejections_per_wave * waves
        add(
            "exact_queue_full_rejections",
            sample_summary["correct_rejections"] == expected_rejections,
            sample_summary["correct_rejections"],
            f"== {expected_rejections}",
        )
        expected_active_peak = min(level_or_concurrency, max_active)
        expected_queue_peak = min(max(0, level_or_concurrency - max_active), max_queue)
        add("active_peak_reached", active_max == expected_active_peak, active_max, f"== {expected_active_peak}")
        add("queue_peak_reached", queued_max == expected_queue_peak, queued_max, f"== {expected_queue_peak}")
        rejection_latencies = [
            sample.total_ms
            for sample in samples
            if sample.correct_rejection
        ]
        rejection_p95 = percentile(rejection_latencies, 95)
        if expected_rejections:
            add(
                "queue_full_rejection_p95",
                rejection_p95 is not None and rejection_p95 <= 1000.0,
                rounded(rejection_p95),
                "<= 1000 ms",
            )

    # 稳态模式以真实体验门槛验收；boundary/spike 主要验收精确状态与拒绝数。
    if args.mode == "steady":
        success_rate = float(sample_summary["success_rate_admitted"])
        add(
            "business_success_rate",
            success_rate >= thresholds.min_success_rate,
            success_rate,
            f">= {thresholds.min_success_rate}",
        )
        total_p95 = sample_summary["total_ms"]["p95"]
        add(
            "total_p95",
            total_p95 is not None and total_p95 <= thresholds.total_p95_ms,
            total_p95,
            f"<= {thresholds.total_p95_ms:.0f} ms",
        )
        if sample_summary["ttft_ms"]["count"]:
            ttft_p95 = sample_summary["ttft_ms"]["p95"]
            add(
                "ttft_p95",
                ttft_p95 is not None and ttft_p95 <= thresholds.ttft_p95_ms,
                ttft_p95,
                f"<= {thresholds.ttft_p95_ms:.0f} ms",
            )

    add(
        "no_unexpected_failures",
        sample_summary["unexpected_failures"] == 0,
        sample_summary["unexpected_failures"],
        "0",
    )
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "threshold_profile": asdict(thresholds),
    }


async def fetch_capacity_config(client: httpx.AsyncClient) -> tuple[int, int, dict[str, Any]]:
    response = await client.get("/api/concurrency/status", timeout=5.0)
    if response.status_code != 200:
        raise RuntimeError(f"无法读取并发状态：HTTP {response.status_code} {response.text[:160]}")
    payload = await parse_json_safely(response)
    max_active = int(payload.get("max_active") or 0)
    max_queue = int(payload.get("max_queue") or 0)
    if max_active <= 0 or max_queue < 0:
        raise RuntimeError(f"并发状态返回非法容量：{payload}")
    return max_active, max_queue, payload


def client_limits(concurrency: int) -> httpx.Limits:
    connections = max(20, concurrency + 20)
    return httpx.Limits(max_connections=connections, max_keepalive_connections=connections)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    samples = report["summary"]
    monitor = report["monitor_summary"]
    evaluation = report["evaluation"]
    print("\n=== 并发压力测试结果 ===")
    print(
        f"请求={samples['requests']} 成功={samples['business_success']} "
        f"正确拒绝={samples['correct_rejections']} 异常失败={samples['unexpected_failures']}"
    )
    print(
        f"active_max={monitor['active_max']} queued_max={monitor['queued_max']} "
        f"success_rate={samples['success_rate_admitted']:.4f}"
    )
    print(
        f"TTFT P95={samples['ttft_ms']['p95']}ms "
        f"Total P95={samples['total_ms']['p95']}ms "
        f"Health P95={monitor['health']['p95']}ms"
    )
    for name, check in evaluation["checks"].items():
        tag = "PASS" if check["passed"] else "FAIL"
        print(f"[{tag}] {name}: actual={check['actual']} expected={check['expected']}")
    print(f"最终结论：{'通过' if evaluation['passed'] else '不通过'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rag-psychology 最新版并发压力测试")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="目标服务地址")
    parser.add_argument("--endpoint", choices=("query", "stream", "mixed"), default="mixed")
    parser.add_argument("--sse-ratio", type=float, default=0.5, help="mixed 模式中 SSE 比例")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="固定测试问题")
    parser.add_argument(
        "--reuse-question",
        action="store_true",
        help="所有请求复用完全相同的问题；默认追加唯一标记以避免 embedding 缓存掩盖压力",
    )
    parser.add_argument("--persist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--user-prefix", default="load")
    parser.add_argument("--password", default="LoadTest@2026", help="仅用于本轮新建测试账号，不写入报告")
    parser.add_argument("--auth-parallelism", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求 HTTP 超时秒数")
    parser.add_argument("--poll-ms", type=float, default=50.0, help="并发状态采样间隔毫秒")
    parser.add_argument("--output", default="results/concurrency-pressure-latest.json")
    parser.add_argument("--seed", type=int, default=20260903)

    subparsers = parser.add_subparsers(dest="mode", required=True)
    boundary = subparsers.add_parser("boundary", help="单轮同步起跑，验证20/40边界")
    boundary.add_argument("--level", type=int, required=True, help="本轮同时请求数，例如20/21/60/61/80")
    boundary.add_argument("--strict-boundary", action="store_true", help="严格断言峰值与拒绝数量；仅限慢Mock")

    steady = subparsers.add_parser("steady", help="固定并发持续施压")
    steady.add_argument("--concurrency", type=int, required=True)
    steady.add_argument("--duration", type=float, default=600.0)
    steady.add_argument("--total", type=int, default=0, help="0表示只受duration限制")

    spike = subparsers.add_parser("spike", help="多轮同步尖峰")
    spike.add_argument("--level", type=int, required=True)
    spike.add_argument("--rounds", type=int, default=10)
    spike.add_argument("--round-gap", type=float, default=1.0)
    spike.add_argument("--strict-boundary", action="store_true", help="每轮严格断言峰值与拒绝数量")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if not 0.0 <= args.sse_ratio <= 1.0:
        raise ValueError("--sse-ratio 必须位于 0～1")
    if args.poll_ms < 10:
        raise ValueError("--poll-ms 不应低于10ms，避免监控本身制造压力")

    level = args.concurrency if args.mode == "steady" else args.level
    if level <= 0:
        raise ValueError("并发数必须大于0")

    timeout = httpx.Timeout(args.timeout, connect=10.0, pool=10.0)
    base_url = args.url.rstrip("/")
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        limits=client_limits(level),
        headers={"Connection": "keep-alive"},
    ) as client:
        max_active, max_queue, initial_capacity = await fetch_capacity_config(client)
        user_count = level
        print(
            f"目标={base_url} mode={args.mode} level={level} "
            f"server_capacity={max_active}+{max_queue} users={user_count}"
        )
        usernames, tokens = await create_test_users(
            client,
            count=user_count,
            prefix=args.user_prefix,
            password=args.password,
            auth_parallelism=args.auth_parallelism,
        )
        print(f"认证完成：{len(tokens)} 个独立账号")

        stop_monitor = asyncio.Event()
        monitor_task = asyncio.create_task(
            monitor_service(client, stop_monitor, args.poll_ms / 1000.0)
        )
        started = time.perf_counter()
        samples: list[RequestSample] = []
        try:
            if args.mode == "boundary":
                samples = await run_synchronized_wave(
                    client, tokens, args.level, args.endpoint, args.sse_ratio,
                    args.question, args.reuse_question, args.persist, 0, args.seed,
                )
            elif args.mode == "steady":
                samples = await run_steady_workers(
                    client, tokens, args.concurrency, args.duration, args.total,
                    args.endpoint, args.sse_ratio, args.question, args.reuse_question,
                    args.persist, args.seed,
                )
            else:
                for round_index in range(args.rounds):
                    wave = await run_synchronized_wave(
                        client, tokens, args.level, args.endpoint, args.sse_ratio,
                        args.question, args.reuse_question, args.persist,
                        len(samples), args.seed + round_index,
                    )
                    samples.extend(wave)
                    if round_index + 1 < args.rounds:
                        await asyncio.sleep(args.round_gap)
        finally:
            stop_monitor.set()
            monitor = await monitor_task
        wall_seconds = time.perf_counter() - started

    sample_summary = summarize_samples(samples)
    monitor_summary = summarize_monitor(monitor)
    evaluation = evaluate(
        args, samples, sample_summary, monitor_summary, max_active, max_queue, level
    )
    report = {
        "schema_version": 1,
        "target_commit": "fd976ec2",
        "generated_at_epoch": time.time(),
        "configuration": {
            "url": base_url,
            "mode": args.mode,
            "endpoint": args.endpoint,
            "sse_ratio": args.sse_ratio,
            "persist": args.persist,
            "reuse_question": args.reuse_question,
            "level": level,
            "duration": getattr(args, "duration", None),
            "rounds": getattr(args, "rounds", None),
            "strict_boundary": getattr(args, "strict_boundary", False),
            "poll_ms": args.poll_ms,
            "server_initial_capacity": initial_capacity,
            "test_user_count": len(usernames),
            "test_user_fingerprints": [user_fingerprint(name) for name in usernames],
            "seed": args.seed,
        },
        "wall_seconds": round(wall_seconds, 3),
        "summary": sample_summary,
        "monitor_summary": monitor_summary,
        "evaluation": evaluation,
        "samples": [asdict(sample) for sample in samples],
        "capacity_series": [asdict(point) for point in monitor.capacity],
        "health_series": [asdict(point) for point in monitor.health],
    }
    output = Path(args.output).resolve()
    write_report(output, report)
    print_summary(report)
    print(f"完整报告：{output}")
    return 0 if evaluation["passed"] else 2


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("测试被用户中断", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"测试启动或执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

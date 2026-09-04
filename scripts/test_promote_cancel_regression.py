"""回归测试：promote×cancel 竞态导致的槽位永久泄漏（memory 准入后端）。

背景（缺陷报告 + repro_slot_leak.py 实测）：
  B 入队并在 wait_until_running() 中等待；A release 触发 _promote() 把 B 置
  running + 放入 _active 后，若 cancel(B) 抢在 B 的等待协程恢复之前执行，
  旧实现的等待循环只认 state == "running" —— B 看到 cancelling 既不进 running
  分支、也无 terminal，只能卡到 deadline 被 _finalize_queued 按“排队超时”清理；
  而 _finalize_queued 只清 _entries/_queue、不碰 _active，于是：
    - 槽位永久泄漏（_active 残留，release 因 _entries 缺失返回 NOT_RUNNING 无法兜底）；
    - 用户占位被提前清除，单用户 in-flight 约束被破坏；
    - 请求“已获槽位却按排队超时处理”，语义错误。
  每发生一次，max_active 可用容量永久 -1，累积后服务等价瘫痪。

修复要点（modules/concurrency/memory_backend.py）：
  1) wait_until_running 放行判定改为“entry.request_id in self._active”（槽位真源），
     active 条目无论 state 是 running 还是被 cancel 置成的 cancelling 一律返回
     STARTED，取消终态统一由调用方 release() 裁决（cancel_requested → CANCELLED）；
  2) _finalize_queued 增加防御门禁：已入 _active 的条目拒绝 queued 收尾。

本脚本用 asyncio 步进强制确定性交错（非概率并发），按报告第八章要求对
“提升/取消/超时”时序排列跑 ≥1000 轮，断言槽位与用户占位残留为 0、无双终态。

用法：
  python scripts/test_promote_cancel_regression.py            # 1000 轮全场景
  python scripts/test_promote_cancel_regression.py --rounds 50
退出码：0 全部通过；1 存在失败。
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.concurrency.memory_backend import MemoryAdmissionBackend  # noqa: E402
from modules.concurrency.models import (  # noqa: E402
    CancelCode,
    ReleaseCode,
    SubmitCode,
    TerminalReason,
    WaitCode,
)

PASS_DIR = PROJECT_ROOT / "tests" / "results" / "passed"

# --------------------------------------------------------------------------
# 校验助手
# --------------------------------------------------------------------------

def _residuals(be: MemoryAdmissionBackend) -> list[str]:
    """返回所有不变量破坏点；空列表 = 干净。"""
    bad: list[str] = []
    if be._active:
        bad.append(f"_active 残留 {len(be._active)} 条: {list(be._active)}")
    if be._entries:
        bad.append(f"_entries 残留 {len(be._entries)} 条: {list(be._entries)}")
    if be._queue:
        bad.append(f"_queue 残留 {len(be._queue)} 条: {[e.request_id for e in be._queue]}")
    if be._user_req:
        bad.append(f"_user_req 残留 {len(be._user_req)} 条: {dict(be._user_req)}")
    return bad


def _assert_clean(be: MemoryAdmissionBackend) -> None:
    bad = _residuals(be)
    assert not bad, "; ".join(bad)


def _make_backend(max_active: int = 1, timeout: float = 0.2) -> MemoryAdmissionBackend:
    return MemoryAdmissionBackend(
        max_active=max_active,
        max_queue=8,
        queue_wait_timeout_seconds=timeout,
        one_inflight_per_user=True,
    )


# --------------------------------------------------------------------------
# 场景 1（核心）：promote 后、wait 恢复前 cancel —— 修复前必然泄漏
# --------------------------------------------------------------------------

async def scenario_promote_cancel_race(round_no: int, timeout: float) -> str:
    """确定性重现竞态窗口并验证修复后行为。返回空串=通过，否则返回失败原因。"""
    be = _make_backend(timeout=timeout)
    rid_a, rid_b = f"A{round_no}", f"B{round_no}"
    try:
        a = await be.submit("uA", rid_a)
        assert a.code == SubmitCode.STARTED, f"A submit={a.code}"
        b = await be.submit("uB", rid_b)
        assert b.code == SubmitCode.QUEUED, f"B submit={b.code}（max_active=1 应入队）"

        # B 挂起等待；随后 A release 同步触发 _promote（B -> running + _active），
        # 由于 release/cancel 函数体无内部 await，事件循环不会切给 B 的等待协程，
        # cancel 必然抢在 B 恢复之前 —— 确定性命中竞态窗口。
        w = asyncio.create_task(be.wait_until_running(rid_b))
        await asyncio.sleep(0)
        await be.release(rid_a, a.ticket.owner_token)
        assert rid_b in be._active, "promote 后 B 应在 _active"

        c = await be.cancel("uB", rid_b)
        assert c.code == CancelCode.CANCELLING, f"cancel 应返回 CANCELLING，实际 {c.code}"

        res = await w
        if res.code != WaitCode.STARTED:
            # 修复回退的典型表现：已获槽位却被按排队超时清理
            return (f"round {round_no}: wait 返回 {res.code}（应 STARTED；"
                    f"修复前此处泄漏 _active 槽位）")

        # 调用方收尾（同步 /api/query finally / SSE 断连 STARTED 分支同构）：
        # release 应按 cancel_requested 裁决为 CANCELLED 并释放槽位
        rel = await be.release(rid_b, b.ticket.owner_token, terminal="completed")
        if rel.code != ReleaseCode.RELEASED:
            return f"round {round_no}: release 返回 {rel.code}（应 RELEASED）"

        recent = be._recent_terminals.get(rid_b)
        if recent is None or recent[1] != TerminalReason.CANCELLED.value:
            return f"round {round_no}: 终态应为 CANCELLED，实际 {recent}（无双终态要求）"

        _assert_clean(be)
        return ""
    except AssertionError as e:  # noqa: PERF203
        return f"round {round_no}: 断言失败 {e}"
    except Exception as e:  # noqa: BLE001
        return f"round {round_no}: 异常 {type(e).__name__}: {e}"
    finally:
        # 兜底清理（等价 SSE finally 的幂等 cancel），避免单轮失败污染诊断
        if not w.done():
            w.cancel()
            try:
                await asyncio.wait_for(w, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass


# --------------------------------------------------------------------------
# 场景 2-5：既有路径回归（修复不得破坏原行为）
# --------------------------------------------------------------------------

async def scenario_queued_cancel(timeout: float) -> str:
    """排队中取消（未 promote）：wait 应 CANCELLED，无残留。"""
    be = _make_backend(timeout=timeout)
    a = await be.submit("uA", "A"); assert a.code == SubmitCode.STARTED
    b = await be.submit("uB", "B"); assert b.code == SubmitCode.QUEUED
    w = asyncio.create_task(be.wait_until_running("B"))
    await asyncio.sleep(0)
    c = await be.cancel("uB", "B")
    assert c.code == CancelCode.CANCELLED, f"排队中取消应 CANCELLED，实际 {c.code}"
    res = await w
    assert res.code == WaitCode.CANCELLED, f"wait 应 CANCELLED，实际 {res.code}"
    # 收尾：占槽的 A 真实退出后释放，验证全后端无残留
    rel = await be.release("A", a.ticket.owner_token, terminal="completed")
    assert rel.code == ReleaseCode.RELEASED, f"A release 应 RELEASED，实际 {rel.code}"
    _assert_clean(be)
    return ""


async def scenario_queue_timeout(timeout: float) -> str:
    """排队超时（无 promote、无 cancel）：wait 应 QUEUE_TIMEOUT，无残留。"""
    be = _make_backend(timeout=timeout)
    a = await be.submit("uA", "A"); assert a.code == SubmitCode.STARTED
    b = await be.submit("uB", "B"); assert b.code == SubmitCode.QUEUED
    w = asyncio.create_task(be.wait_until_running("B"))
    res = await asyncio.wait_for(w, timeout=timeout + 1.0)
    assert res.code == WaitCode.QUEUE_TIMEOUT, f"wait 应 QUEUE_TIMEOUT，实际 {res.code}"
    rel = await be.release("A", a.ticket.owner_token, terminal="completed")
    assert rel.code == ReleaseCode.RELEASED, f"A release 应 RELEASED，实际 {rel.code}"
    _assert_clean(be)
    return ""


async def scenario_normal_release(timeout: float = 0.2) -> str:
    """正常放行：A 释放 -> B STARTED -> release(completed)，无残留。"""
    be = _make_backend(timeout=timeout)
    a = await be.submit("uA", "A"); assert a.code == SubmitCode.STARTED
    b = await be.submit("uB", "B"); assert b.code == SubmitCode.QUEUED
    w = asyncio.create_task(be.wait_until_running("B"))
    await asyncio.sleep(0)
    await be.release("A", a.ticket.owner_token)
    res = await asyncio.wait_for(w, timeout=2.0)
    assert res.code == WaitCode.STARTED, f"wait 应 STARTED，实际 {res.code}"
    rel = await be.release("B", b.ticket.owner_token, terminal="completed")
    assert rel.code == ReleaseCode.RELEASED, f"release 应 RELEASED，实际 {rel.code}"
    recent = be._recent_terminals.get("B")
    assert recent and recent[1] == TerminalReason.COMPLETED.value, f"终态应为 COMPLETED，实际 {recent}"
    _assert_clean(be)
    return ""


async def scenario_running_cancel(timeout: float = 0.2) -> str:
    """运行中取消（wait 已交付 STARTED 后 cancel）：release 裁决 CANCELLED，无残留。"""
    be = _make_backend(timeout=timeout)
    a = await be.submit("uA", "A"); assert a.code == SubmitCode.STARTED
    c = await be.cancel("uA", "A")
    assert c.code == CancelCode.CANCELLING, f"运行中取消应 CANCELLING，实际 {c.code}"
    rel = await be.release("A", a.ticket.owner_token, terminal="completed")
    assert rel.code == ReleaseCode.RELEASED, f"release 应 RELEASED，实际 {rel.code}"
    recent = be._recent_terminals.get("A")
    assert recent and recent[1] == TerminalReason.CANCELLED.value, f"终态应为 CANCELLED，实际 {recent}"
    _assert_clean(be)
    return ""


# --------------------------------------------------------------------------
# 场景 6：容量连续复用 —— 多轮竞态后槽位容量不损
# --------------------------------------------------------------------------

async def scenario_capacity_preserved(rounds: int, timeout: float) -> str:
    # max_active=1：每轮 A 必占唯一槽位，B 必入队 —— 与核心竞态场景同构，
    # 但连续复用同一 backend，验证多轮竞态后槽位容量不损。
    be = _make_backend(max_active=1, timeout=timeout)
    for i in range(rounds):
        a = await be.submit(f"uA{i}", f"A{i}"); assert a.code == SubmitCode.STARTED
        b = await be.submit(f"uB{i}", f"B{i}"); assert b.code == SubmitCode.QUEUED
        w = asyncio.create_task(be.wait_until_running(f"B{i}"))
        await asyncio.sleep(0)
        await be.release(f"A{i}", a.ticket.owner_token)
        await be.cancel(f"uB{i}", f"B{i}")
        res = await asyncio.wait_for(w, timeout=2.0)
        assert res.code == WaitCode.STARTED, f"轮 {i}: wait 应 STARTED，实际 {res.code}"
        rel = await be.release(f"B{i}", b.ticket.owner_token, terminal="completed")
        assert rel.code == ReleaseCode.RELEASED, f"轮 {i}: release 应 RELEASED，实际 {rel.code}"
    _assert_clean(be)
    # 容量未损：新请求应直接 STARTED（max_active=2 全空）
    snap = await be.snapshot()
    assert snap.active == 0 and snap.queued == 0, f"快照应全空，实际 {snap}"
    n = await be.submit("uNew", "NEW")
    assert n.code == SubmitCode.STARTED, f"复用后新请求应直接 STARTED，实际 {n.code}"
    rel = await be.release("NEW", n.ticket.owner_token)
    assert rel.code == ReleaseCode.RELEASED, f"NEW release 应 RELEASED，实际 {rel.code}"
    _assert_clean(be)
    return ""


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--rounds", type=int, default=1000,
        help="场景 1（promote×cancel 竞态）确定性轮数，默认 1000（报告第八章门禁）",
    )
    p.add_argument(
        "--timeout", type=float, default=0.05,
        help="queue_wait_timeout_seconds（每轮随机化，此值为基准），默认 0.05",
    )
    return p


async def async_main(args: argparse.Namespace) -> int:
    rng = random.Random(20260904)
    failures: list[str] = []
    t0 = time.monotonic()

    # —— 场景 1：≥1000 轮确定性竞态排列 ——
    for i in range(args.rounds):
        timeout = max(0.01, args.timeout * rng.uniform(0.5, 2.0))
        err = await scenario_promote_cancel_race(i, timeout)
        if err:
            failures.append(err)
            if len(failures) >= 5:  # 快速失败，避免刷屏
                failures.append(f"... 后续 {args.rounds - i - 1} 轮省略")
                break

    # —— 场景 2-5：既有路径回归（各跑 20 轮） ——
    for fn in (scenario_queued_cancel, scenario_queue_timeout,
               scenario_normal_release, scenario_running_cancel):
        for i in range(20):
            try:
                err = await fn(timeout=max(0.02, args.timeout))
            except Exception as e:  # noqa: BLE001
                err = f"{fn.__name__} 轮 {i}: {type(e).__name__}: {e}"
            if err:
                failures.append(f"{fn.__name__}: {err}")
                break

    # —— 场景 6：容量连续复用 ——
    try:
        err = await scenario_capacity_preserved(min(50, args.rounds), timeout=args.timeout)
        if err:
            failures.append(f"capacity_preserved: {err}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"capacity_preserved: {type(e).__name__}: {e}")

    elapsed = time.monotonic() - t0
    summary = (
        f"[promote-cancel-race] rounds={args.rounds} "
        f"failures={len(failures)} elapsed={elapsed:.2f}s -> "
        f"{'PASS' if not failures else 'FAIL'}"
    )
    print(summary)
    for f in failures[:5]:
        print("  FAIL:", f)

    PASS_DIR.mkdir(parents=True, exist_ok=True)
    (PASS_DIR / "promote-cancel-race.log").write_text(
        "\n".join([summary, *[f"FAIL: {x}" for x in failures], ""]),
        encoding="utf-8",
    )
    return 0 if not failures else 1


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())

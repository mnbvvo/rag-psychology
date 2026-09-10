"""AI 问答的公共编排工具与持久化（/api/query 系列与小程序 /api/mp/* 共用）。

2026-09-07 从 api/main.py 平移：Web 端 query 与小程序 mp 通道都需要
「会话落库 + 危机审计 + 长期记忆后台化」以及「SSE / 准入响应的格式化」，
收拢到本模块避免两份实现漂移（修 bug 只改一处）。行为与平移前完全一致，
main.py 改为 import 本模块同名函数。

约定：本模块只放「纯编排工具」，不持有 admission 状态机（modules.concurrency）
与业务查询（db/crud_async）——那些仍在各自归属模块。
"""
import json
import logging
import time
from typing import Optional

logger = logging.getLogger("rag.gateway")

from config.settings import settings


# 后台持久化可靠性指标（/api/health 暴露；落库失败必须对监控可见）
_persist_total = 0                 # 已投递的持久化任务数（含同步回退执行）
_persist_failures = 0              # 会话落库最终失败累计（重试后仍失败）
_persist_critical_failures = 0     # 危机审计/高危落库最终失败累计（最敏感数据，单独计数）


def get_persist_metrics() -> dict:
    """持久化可靠性指标快照（供 /api/health 聚合）。"""
    return {
        "total": _persist_total,
        "failures": _persist_failures,
        "critical_failures": _persist_critical_failures,
    }


# AI 问答并发准入的 429 Retry-After：建议与排队超时同量级（上限 60s）
_ADMISSION_RETRY_AFTER = str(max(1, min(60, int(settings.AI_QUEUE_WAIT_TIMEOUT_SECONDS))))


# ---------------- 上游 LLM 网关异常分类（背压 vs 服务缺陷） ----------------
# 2026-09-10 实测背景：上游百炼限流（type/code = limit_requests）原先被端点
# `except Exception → 500` 一律吞成「内部处理失败」，导致
#   ① 压测把它判成「非 503 的 5xx」一票否决 → 把「模型侧容量不足」误记成服务缺陷；
#   ② 非流式路径连日志都不打，现场无法归因；
#   ③ 流式与非流式降级行为不一致（SSE error vs 500）。
# 现在统一：上游容量类错误 → 503 + Retry-After（503 是本项目唯一允许的非 2xx 5xx，
# 也是压测口径里的「正确拒绝」）；4xx 类（密钥/参数错）仍按 500 处理，
# 因为那属于必须修掉的配置缺陷，不该被隐藏成背压。
AI_UPSTREAM_LIMITED = "AI_UPSTREAM_LIMITED"          # 上游 429：请求速率超限
AI_UPSTREAM_UNAVAILABLE = "AI_UPSTREAM_UNAVAILABLE"  # 上游超时/连接失败/5xx
_UPSTREAM_RETRY_AFTER = "5"                          # 秒；上游限流窗口通常秒级

_UPSTREAM_DETAIL = {
    AI_UPSTREAM_LIMITED: "上游模型请求频率超限，请稍后重试",
    AI_UPSTREAM_UNAVAILABLE: "上游模型暂不可用，请稍后重试",
}


def classify_upstream_error(exc: BaseException) -> Optional[str]:
    """上游容量类错误 → 背压错误码；不是上游容量问题 → None（调用方按 500 处理）。

    openai 的异常层级：RateLimitError / APITimeoutError / APIConnectionError 都是
    APIError 子类，且 RateLimitError ⊂ APIStatusError，所以判定顺序不能反。
    """
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )
    except Exception:  # noqa: BLE001 —— 极端环境下 openai 不可用时退化为「非背压」
        return None
    if isinstance(exc, RateLimitError):
        return AI_UPSTREAM_LIMITED
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return AI_UPSTREAM_UNAVAILABLE
    if isinstance(exc, APIStatusError) and int(getattr(exc, "status_code", 0) or 0) >= 500:
        return AI_UPSTREAM_UNAVAILABLE
    return None


def upstream_json(code: str) -> "JSONResponse":
    """上游背压的统一响应体：503 + Retry-After + {detail, code}。"""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=503,
        content={"detail": _UPSTREAM_DETAIL.get(code, "上游模型暂不可用，请稍后重试"), "code": code},
        headers={"Retry-After": _UPSTREAM_RETRY_AFTER},
    )


def upstream_sse_payload(exc: BaseException, code: str) -> dict:
    """上游背压的 SSE error 事件体。

    带 code + retry_after，让客户端（与压测脚本）能把「背压」和「真故障」分开：
    背压应退避重试，真故障才该报障。
    """
    return {
        "detail": _UPSTREAM_DETAIL.get(code, "上游模型暂不可用，请稍后重试"),
        "error_type": type(exc).__name__,
        "code": code,
        "retry_after": _UPSTREAM_RETRY_AFTER,
    }


def admission_json(status: int, detail: str, code: str, headers: dict = None) -> "JSONResponse":
    """准入类错误的统一响应体：{detail, code} + 可选头（如 Retry-After）。

    用 JSONResponse 而非 HTTPException，是为了让前端能按 code 精确分支，
    同时保持 detail 中文提示向后兼容。
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status,
        content={"detail": detail, "code": code},
        headers=headers or {},
    )


def sse(event: str, data: dict) -> str:
    """格式化一个 SSE 事件：`event:` + `data:` 两行 + 空行结尾。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_queue_event(request_id: str, position: int, queued: int, active: int) -> str:
    return sse("queue", {
        "request_id": request_id,
        "position": position,
        "queued": queued,
        "active": active,
        "wait_timeout_seconds": settings.AI_QUEUE_WAIT_TIMEOUT_SECONDS,
    })


def flush_db_turn_sync(
    session_id: str,
    question: str,
    answer: str,
    title: Optional[str],
    user_id: str,
    safety_check: Optional[dict] = None,
    is_crisis_response: bool = False,
    safety_note: Optional[str] = None,
    answer_safety_check: Optional[dict] = None,
) -> None:
    """会话落库 + 危机审计（纯 DB 写，毫秒级）。

    由请求完成路径**同步**执行（query/stream 返回前）：服务端短期窗口改为
    从 messages 表按会话组装，本轮若不在返回前落库，下一条请求的窗口就会
    永远缺这一轮。失败重试一次后告警，不影响已生成的回答。
    危机/高危审计属关键数据（合规留痕），同步直写、不依赖内存队列。
    """
    global _persist_total, _persist_failures, _persist_critical_failures
    from db import crud

    current_question = (question or "").strip()
    # 高危/危机审计属于关键数据：失败必须在计数上单独体现，不允许与普通
    # 会话混在一起被"尽力而为"掩盖。
    critical = bool(is_crisis_response) or bool(
        safety_check and safety_check.get("is_crisis")
    ) or bool(answer_safety_check and answer_safety_check.get("is_crisis"))

    def _flush_once() -> None:
        with crud.get_db() as db:
            crud.append_turn(
                db,
                session_id,
                current_question,
                answer or "",
                # 自动命名提示：优先用前端首次提问传入的标题，否则回退到当前问题
                title=(title or current_question or None),
                user_id=user_id,
            )
            sc = safety_check
            if sc and sc.get("is_crisis"):
                crud.log_crisis(
                    db,
                    session_id,
                    level=sc.get("level", "unknown"),
                    keywords_found=sc.get("keywords_found"),
                    question=current_question,
                    response=answer if is_crisis_response else safety_note,
                    is_crisis_response=bool(is_crisis_response),
                    detect_method=sc.get("detect_method") if isinstance(sc, dict) else None,
                    confidence=sc.get("confidence") if isinstance(sc, dict) else None,
                    user_id=user_id,
                )
            # 回答侧命中高危：另记一条审计（detect_method=answer_check）
            ans_sc = answer_safety_check
            if ans_sc and ans_sc.get("is_crisis"):
                crud.log_crisis(
                    db,
                    session_id,
                    level=ans_sc.get("level", "high"),
                    keywords_found=ans_sc.get("keywords_found"),
                    question=current_question,
                    response=answer or "",
                    is_crisis_response=False,
                    detect_method="answer_check",
                    user_id=user_id,
                )

    _persist_total += 1
    try:
        _flush_once()
    except Exception as e:
        # 瞬时故障（连接抖动/锁等待/网络闪断）重试一次，降低偶发静默丢失
        time.sleep(0.3)
        try:
            _flush_once()
            logger.warning("[persist] 首次落库失败后重试成功: %s", type(e).__name__)
        except Exception as e2:
            _persist_failures += 1
            if critical:
                _persist_critical_failures += 1
                logger.error(
                    "[persist][CRITICAL] 危机审计/高危落库最终失败（累计 %s）: %s: %s",
                    _persist_critical_failures, type(e2).__name__, e2,
                )
            else:
                logger.error(
                    "[persist][ERROR] 会话持久化最终失败（累计 %s，回答已正常返回）: %s: %s",
                    _persist_failures, type(e2).__name__, e2,
                )


def flush_memory_sync(user_id: str, question: str, answer: str) -> None:
    """长期记忆落库（慢路径：2 次 embedding API 调用，数百 ms），后台队列执行。

    与会话落库拆分：会话同步（毫秒级，窗口一致性），embedding 后台（不阻塞
    SSE 返回）。失败不影响回答本身，只打告警。
    """
    if not settings.MEMORY_ENABLED or not (question or "").strip():
        return
    try:
        from modules.memory import memory_service

        memory_service.save_turn(user_id, (question or "").strip(), answer or "")
    except Exception as e:
        logger.warning("[memory] 长期记忆落库失败: %s", e)


async def enqueue_persist(
    session_id: str,
    question: str,
    answer: str,
    title: Optional[str],
    user_id: str,
    safety_check: Optional[dict] = None,
    is_crisis_response: bool = False,
    safety_note: Optional[str] = None,
    answer_safety_check: Optional[dict] = None,
) -> None:
    """持久化编排（2026-09-04 起会话同步落库）：

    1. 会话 + 审计（纯 DB，毫秒级）：**请求内同步执行**——服务端短期窗口已改为
       从 messages 表按会话组装（前端只发 session_id+本轮问题），本轮若不在
       返回前落库，下一条请求的窗口会永远缺这一轮。危机审计同属关键数据，
       绝不进内存队列（进程崩溃也不丢）。
    2. 长期记忆 embedding（2 次 API，数百 ms）：后台队列执行，队列不可用/已满
       时回退请求内线程池同步执行（可靠性兜底）。
    失败不影响回答本身（会话落库失败仅告警并计数）。
    """
    from fastapi.concurrency import run_in_threadpool

    from modules.bg_queue import bg_queue

    # 1) 会话落库 + 危机审计（同步）
    await run_in_threadpool(
        flush_db_turn_sync,
        session_id, question, answer, title, user_id,
        safety_check, is_crisis_response, safety_note, answer_safety_check,
    )
    # 2) 长期记忆 embedding（慢）入队；队列不可用回退请求内同步
    if settings.MEMORY_ENABLED and (question or "").strip():
        payload = (user_id, (question or "").strip(), answer or "")
        ok = await bg_queue.enqueue(flush_memory_sync, *payload)
        if not ok:
            await run_in_threadpool(flush_memory_sync, *payload)

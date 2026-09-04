"""
FastAPI服务接口
提供RESTful API供前端调用
"""
import sys
import os
import re
import atexit
import socket
import subprocess
import tempfile
from pathlib import Path

# 确保无论从哪个工作目录启动（如 `python api/main.py`），
# 项目根都在 sys.path 上，使 `from config.settings` / `from modules` 稳定可用。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict
import json
import asyncio
from modules import rag_system
from modules.rag_core import build_sources
from modules.bg_queue import bg_queue
from modules.concurrency.service import admission
from modules.concurrency.models import (
    AI_QUEUE_FULL,
    AI_QUEUE_TIMEOUT,
    AI_REQUEST_ALREADY_FINISHED,
    AI_REQUEST_CANCELLED,
    AI_REQUEST_IN_PROGRESS,
    CancelCode,
    SubmitCode,
    TerminalReason,
    WaitCode,
)
from config.settings import settings
from db import init_db, crud, crud_async
from api.auth import router as auth_router
from api.deps import get_current_user, get_db_session, require_admin
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

app = FastAPI(
    title="青少年心理RAG系统API",
    description="基于RAG的6-18岁青少年心理咨询系统（含登录鉴权与危机干预）",
    version="1.1.0",
)

# 认证路由（/api/auth/register、/api/auth/login、/api/auth/me）
app.include_router(auth_router)

# 跨域：默认仅放行本服务同源 + localhost（前端由本服务托管时同源本不需要跨域；
# 若前端独立部署 / 用开发服务器，请在 .env 用 CORS_ORIGINS 显式放行，切勿用 "*"）。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# 简单内存限流：仅针对 POST /api/query 与 /api/query/stream，防止单客户端刷接口
_rate_limit_store: dict[str, deque] = defaultdict(deque)
_rate_limit_last_sweep = 0.0  # 上次空桶清扫时间（限流桶键回收用）

# 后台持久化可靠性指标（/api/health 暴露；落库失败必须对监控可见，见验收方案 §8）
_persist_total = 0                 # 已投递的持久化任务数（含同步回退执行）
_persist_failures = 0              # 会话落库最终失败累计（重试后仍失败）
_persist_critical_failures = 0     # 危机审计/高危落库最终失败累计（最敏感数据，单独计数）


# ---------------- 越权预校验（在调用 LLM 之前快速失败） ----------------
def _assert_no_impersonation(body_user_id, real_user_id: str) -> None:
    """请求体篡改 user_id → 403：身份永远以 token 为准，客户端传的 user_id 一律不信任。"""
    if body_user_id and body_user_id != real_user_id:
        raise HTTPException(status_code=403, detail="无权以其他用户身份操作")


async def _assert_session_ownership(session_id, user_id: str, db: AsyncSession) -> None:
    """水平越权：session_id 已存在但不属于当前用户 → 403（未创建的新 id 放行）。

    使用请求级 AsyncSession（异步数据层，见 db/crud_async.py）。
    """
    if not session_id:
        return
    if not await crud_async.session_belongs_to(db, session_id, user_id):
        raise HTTPException(status_code=403, detail="无权访问该会话")


def _persist_payload_sync(
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
    """一轮问答的落库 + 长期记忆（同步实现，由后台 Worker 线程池调用）。

    对应架构图「持久化/记忆写入 → Queue → Worker → DB」虚线路径的任务体：
    - 含同步 DB 写（get_db/crud）与同步 embedding（长期记忆 save_turn 两次 embed）；
    - 由 modules/bg_queue 的 Worker 经 asyncio.to_thread 执行，每个任务独立 DB Session；
    - 失败不影响回答本身，只打告警。
    """
    global _persist_total, _persist_failures, _persist_critical_failures
    current_question = (question or "").strip()
    # 高危/危机审计属于关键数据：若本函数被请求内同步执行（见 _enqueue_persist），
    # 其失败必须在计数上单独体现，不允许与普通会话混在一起被"尽力而为"掩盖。
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
            print(f"[persist][WARN] 首次落库失败后重试成功: {type(e).__name__}", flush=True)
        except Exception as e2:
            _persist_failures += 1
            if critical:
                _persist_critical_failures += 1
                print(
                    f"[persist][CRITICAL] 危机审计/高危落库最终失败（累计 {_persist_critical_failures}）: "
                    f"{type(e2).__name__}: {e2}",
                    flush=True,
                )
            else:
                print(
                    f"[persist][ERROR] 会话持久化最终失败（累计 {_persist_failures}，回答已正常返回）: "
                    f"{type(e2).__name__}: {e2}",
                    flush=True,
                )

    # 长期记忆落库（向量检索式）：本轮问答 + embedding 写入 user_chat_history
    if settings.MEMORY_ENABLED and current_question:
        try:
            from modules.memory import memory_service

            memory_service.save_turn(user_id, current_question, answer or "")
        except Exception as e:
            print(f"[memory][WARN] 长期记忆落库失败: {e}", flush=True)


async def _enqueue_persist(
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
    """投递持久化任务到后台队列；队列未启动/已满时回退请求内同步执行（可靠性兜底）。

    高危/危机审计属于关键数据（验收方案 §8：不得仅依靠内存队列"尽力而为"保存），
    一律**请求内同步写入**，不进入进程内内存队列——即使进程随后崩溃/关闭，
    危机审计也已落库，不会被内存队列丢弃。
    """
    from modules.bg_queue import bg_queue

    payload = (
        session_id, question, answer, title, user_id,
        safety_check, is_crisis_response, safety_note, answer_safety_check,
    )
    critical = bool(is_crisis_response) or bool(
        safety_check and safety_check.get("is_crisis")
    ) or bool(answer_safety_check and answer_safety_check.get("is_crisis"))
    if critical:
        # 关键数据：同步直写（高危低频，毫秒级 DB insert，请求内可接受）
        await run_in_threadpool(_persist_payload_sync, *payload)
        return
    ok = await bg_queue.enqueue(_persist_payload_sync, *payload)
    if not ok:
        # 回退：请求内线程池执行（与旧行为一致）
        await run_in_threadpool(_persist_payload_sync, *payload)


# ---------------- AI 问答并发准入（总稿 §4，Phase 1 memory） ----------------
# 429 Retry-After：建议与排队超时同量级（上限 60s）
_ADMISSION_RETRY_AFTER = str(max(1, min(60, int(settings.AI_QUEUE_WAIT_TIMEOUT_SECONDS))))


def _admission_json(status: int, detail: str, code: str, headers: dict = None) -> JSONResponse:
    """准入类错误的统一响应体：{detail, code} + 可选头（如 Retry-After）。

    用 JSONResponse 而非 HTTPException，是为了让前端能按 code 精确分支，
    同时保持 detail 中文提示向后兼容。
    """
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "code": code},
        headers=headers or {},
    )


def _sse_queue_event(request_id: str, position: int, queued: int, active: int) -> str:
    return _sse("queue", {
        "request_id": request_id,
        "position": position,
        "queued": queued,
        "active": active,
        "wait_timeout_seconds": settings.AI_QUEUE_WAIT_TIMEOUT_SECONDS,
    })


# ---------------- Phase 1 memory 准入后端：单实例守卫（验收方案 G-08 / 一票否决项 8） ----------------
def _pid_alive(pid: int) -> bool:
    """探测 PID 是否存活。POSIX 用 kill(pid, 0)；Windows 用 tasklist（os.kill 在
    Windows 对非 CTRL 信号会 TerminateProcess，不能用于探测）。"""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _release_single_instance_lock(lock_path: Path, pid: int) -> None:
    """atexit 释放单实例锁：仅当锁仍由本进程持有才删除，避免误删后来者。"""
    try:
        if lock_path.exists():
            cur = lock_path.read_text(encoding="utf-8").strip().split()
            if cur and int(cur[0]) == pid:
                lock_path.unlink()
    except Exception:
        pass


def _acquire_single_instance_lock() -> None:
    """memory 准入后端只允许一个 app 实例承载 20 active + 40 queue。

    关键事实：uvicorn 的 `--workers` 不会设置任何环境变量，且与 `--reload` 的
    socket 拓扑完全相同（master 预绑定后子进程共享），因此无法靠环境变量或端口
    探测区分；但 `--reload`/直跑都只有一个 app 实例，`--workers N` 有 N 个。
    所以守卫维度 = **app 实例级原子锁**：每个进程在 startup 时用 O_EXCL 原子
    创建锁文件，第二个实例创建失败且持有者存活 → 拒绝启动（uvicorn 多 worker
    的第二个 worker 启动失败退出，最终只剩一个 20 槽位实例，满足 G-08b）。
    """
    if (settings.AI_ADMISSION_BACKEND or "memory").strip().lower() != "memory":
        return  # redis（Phase 2 多实例）才允许多进程；memory 必须单实例
    lock_path = Path(tempfile.gettempdir()) / f"rag_psychology_memory_{settings.PORT}.lock"

    for attempt in (0, 1):  # 第二轮仅用于：stale 锁（持有者已死）清理后重试
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()} {time.time()}\n")
            atexit.register(_release_single_instance_lock, lock_path, os.getpid())
            return
        except FileExistsError:
            holder_pid = None
            try:
                holder_pid = int(lock_path.read_text(encoding="utf-8").strip().split()[0])
            except Exception:
                pass
            if holder_pid is not None and _pid_alive(holder_pid):
                raise RuntimeError(
                    f"检测到已有 rag-psychology 实例在运行（PID {holder_pid}，锁 {lock_path}）。"
                    "Phase 1 memory 准入后端只允许单实例/单 worker 承载 20 active + 40 queue，"
                    "请勿使用 uvicorn --workers 或重复启动；多实例需 Phase 2 redis 后端。"
                )
            # 持有者已退出（如 kill -9 残留）：清掉 stale 锁后进入下一轮重试
            try:
                lock_path.unlink()
            except OSError:
                pass
        except OSError as e:
            print(f"[warn] 单实例锁不可用（继续启动）: {e}", flush=True)
            return
    raise RuntimeError(
        f"无法获取单实例锁（{lock_path}），疑似存在并发启动的实例。"
        "Phase 1 memory 准入后端只允许单实例/单 worker。"
    )


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/api/query", "/api/query/stream") and request.method == "POST":
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = settings.RATE_LIMIT_SECONDS
        limit = settings.RATE_LIMIT_TIMES
        bucket = _rate_limit_store[client]
        # 丢弃时间窗之外的旧请求记录
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试。"},
            )
        bucket.append(now)
        # 桶键回收：defaultdict 的键只 prune 不删除，公网暴露后键数会随来源 IP 无限
        # 增长（每个 IP 残留一个空 deque）→ 每 10 分钟清扫一次空桶，防止内存缓慢泄漏
        global _rate_limit_last_sweep
        if len(_rate_limit_store) > 1000 and now - _rate_limit_last_sweep > 600:
            _rate_limit_last_sweep = now
            for key in [k for k, v in _rate_limit_store.items() if not v]:
                del _rate_limit_store[key]
    return await call_next(request)


class QueryRequest(BaseModel):
    """查询请求：支持单轮 question 或多轮 messages。"""
    question: Optional[str] = Field(None, description="用户当前问题（单轮模式，与 messages 二选一）", min_length=1, max_length=2000)
    messages: Optional[List[Dict[str, str]]] = Field(
        None,
        description="多轮对话历史，每条含 role 与 content；role 可为 human/ai/user/assistant。最后一条须为用户问题。",
    )
    session_id: Optional[str] = Field(
        None,
        description="会话 id（前端本地生成，如 session-<timestamp>）；不传则由服务端生成并在响应中返回。用于把多轮对话与危机审计持久化到关系库。",
    )
    user_id: Optional[str] = Field(
        None,
        description="（忽略项）兼容外部测试传入的伪造字段；服务端身份一律以 token 为准，传入值不等于当前用户时返回 403。",
    )
    title: Optional[str] = Field(
        None,
        description="可选：会话标题提示（前端在首次提问时传入问题前若干字）。服务端仅在会话标题仍是占位名（如“新的对话”）时用它自动命名，已命名的会话不受影响。",
    )
    persist: Optional[bool] = Field(
        True,
        description="是否将本轮对话持久化到关系库（sessions/messages）。对话联调页应保留默认 true；"
        "提示词对比页请传 false，避免对比用的临时问答被写入会话表、泄漏到历史对话列表。",
    )
    rag_enabled: Optional[bool] = Field(
        None,
        description="是否启用检索增强生成（RAG）。None=用全局配置 settings.RAG_ENABLED；"
        "false=跳过检索，纯 LLM 对话；true=完整 RAG。",
    )
    safety_enabled: Optional[bool] = Field(
        None,
        description="是否启用安全检测（L0 关键词 + L1 语义 + 回答侧复查）。"
        "None=用全局配置 settings.SAFETY_ENABLED；false=跳过整条安全链路（仅限联调/实验，生产保持 true）。",
    )

    @model_validator(mode="after")
    def check_question_or_messages(self):
        if not self.question and not self.messages:
            raise ValueError("必须提供 question 或 messages 之一")
        if self.messages:
            if not isinstance(self.messages, list) or len(self.messages) == 0:
                raise ValueError("messages 不能为空数组")
            for i, m in enumerate(self.messages):
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise ValueError(f"messages[{i}] 必须包含 role 和 content")
                if m["role"] not in ("human", "ai", "user", "assistant"):
                    raise ValueError(f"messages[{i}].role 必须是 human/ai/user/assistant 之一")
                if not isinstance(m["content"], str) or not m["content"].strip():
                    raise ValueError(f"messages[{i}].content 不能为空")
                if len(m["content"]) > 4000:
                    raise ValueError(f"messages[{i}].content 超长（最多 4000 字符）")
        return self


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    sources: List[dict] = []
    safety_note: Optional[str] = None
    is_crisis_response: bool = False
    safety_check: Optional[dict] = None
    timings: Optional[dict] = Field(
        None,
        description="各阶段耗时（毫秒）：safety/embed/retrieve/llm/total",
    )
    session_id: Optional[str] = Field(
        None,
        description="本次对话归属的会话 id（与请求中的 session_id 对应；未传时由服务端生成）。",
    )
    request_id: Optional[str] = Field(
        None,
        description="本次请求的准入凭证 id（同步接口新增，便于取消/对账）。",
    )
    queue: Optional[dict] = Field(
        None,
        description="准入信息：{wait_ms, queued}。queued=true 表示曾排队等待放行。",
    )


class SessionCreate(BaseModel):
    """新建会话"""
    name: Optional[str] = Field(None, description="会话名称，留空则默认“新的对话”")


class SessionRename(BaseModel):
    """重命名会话"""
    name: str = Field(..., description="新名称")


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    """
    查询接口（同步语义，内部为 async 流收集完整答案）

    昂贵链路（安全检测/RAG/LLM/持久化）前先过统一准入（总稿 §4）：
    空闲槽位立即执行；无槽位 FIFO 排队等待；重复提交/队满/超时返回确定错误码。
    """
    # —— 越权预校验：在调用 LLM 之前快速失败，杜绝资源泄露与算力浪费 ——
    _assert_no_impersonation(request.user_id, current_user.id)
    await _assert_session_ownership(request.session_id, current_user.id, db)
    user_id = current_user.id
    request_id = uuid.uuid4().hex

    # —— 准入：submit（去重 + 槽位检查 + 入队原子完成） ——
    sub = await admission.submit(user_id, request_id)
    if sub.code == SubmitCode.REJECTED_DUPLICATE:
        return _admission_json(409, "你已有问题正在处理", AI_REQUEST_IN_PROGRESS)
    if sub.code == SubmitCode.REJECTED_FULL:
        return _admission_json(
            429, "当前排队已满，请稍后重试", AI_QUEUE_FULL,
            headers={"Retry-After": _ADMISSION_RETRY_AFTER},
        )
    ticket = sub.ticket
    wait_ms = 0.0
    if sub.code == SubmitCode.QUEUED:
        # 排队等待放行（同步接口无 SSE，仅静默等待）
        wait_res = await admission.wait_until_running(request_id)
        wait_ms = wait_res.wait_ms
        if wait_res.code == WaitCode.QUEUE_TIMEOUT:
            admission.record_dropped(ticket, wait_ms, TerminalReason.QUEUE_TIMEOUT.value)
            return _admission_json(503, "排队等待超时，请重新发起", AI_QUEUE_TIMEOUT)
        if wait_res.code == WaitCode.CANCELLED:
            admission.record_dropped(ticket, wait_ms, TerminalReason.CANCELLED.value)
            return _admission_json(409, "请求已取消", AI_REQUEST_CANCELLED)
    admission.note_started(ticket, wait_ms)

    terminal = TerminalReason.COMPLETED.value
    try:
        # 异步完整流程：prepare（同步检索/embedding）在线程池，生成走 llm_stream
        # 原生 async —— LLM 秒级等待不占线程，摆脱 run_in_threadpool 线程闸限制
        result = await rag_system.aquery(
            question=request.question,
            messages=request.messages,
            check_safety=request.safety_enabled,
            user_id=user_id,
            rag_enabled=request.rag_enabled,
            # 同步接口取消语义：生成过程中逐 chunk 检查取消标记，命中即终止上游生成，
            # 不再白烧 token（验收方案 §2.2：不能只设置 cancelling 标志后仍生成完整答案）
            cancel_check=lambda: admission.is_cancelling(request_id),
        )
        if result.get("cancelled"):
            # 已被用户取消：不返回半截答案，按取消契约给 409（finally 以 cancelled 释放槽位）
            terminal = TerminalReason.CANCELLED.value
            return _admission_json(409, "请求已取消", AI_REQUEST_CANCELLED)

        # —— 持久化到关系库（会话 + 危机审计，归属当前用户） ——
        # 仅对话联调页（persist=true）需要落库。落库已后台化（Queue→Worker→DB，
        # 见 _enqueue_persist / modules/bg_queue）：请求只负责生成并拿到 session_id，
        # 不再阻塞等待同步 DB 写与 embedding；队列不可用/已满时回退请求内同步执行。
        # 运行中被取消的请求不写会话（总稿 FR-BE-06）。
        session_id = None
        if request.persist and not admission.is_cancelling(request_id):
            session_id = request.session_id or uuid.uuid4().hex
            await _enqueue_persist(
                session_id,
                result.get("question", ""),
                result.get("answer", ""),
                request.title,
                user_id,
                safety_check=result.get("safety_check"),
                is_crisis_response=bool(result.get("is_crisis_response", False)),
                safety_note=result.get("safety_note"),
                answer_safety_check=result.get("answer_safety_check"),
            )
        if admission.is_cancelling(request_id):
            terminal = TerminalReason.CANCELLED.value

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources") or [],
            safety_note=result.get("safety_note"),
            is_crisis_response=bool(result.get("is_crisis_response", False)),
            safety_check=result.get("safety_check"),
            timings=result.get("timings"),
            session_id=session_id,
            request_id=request_id,
            queue={"wait_ms": round(wait_ms, 1), "queued": sub.code == SubmitCode.QUEUED},
        )
    except ValueError as e:
        # 参数校验类错误返回 400，便于前端定位
        terminal = TerminalReason.FAILED.value
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # 不把内部异常细节（路径/堆栈）回传给客户端
        terminal = TerminalReason.FAILED.value
        raise HTTPException(status_code=500, detail="内部处理失败，请稍后重试。")
    finally:
        # 无论成功/失败/取消，真实退出后释放槽位（幂等；防超卖）
        await admission.release(ticket, terminal=terminal)


def _sse(event: str, data: dict) -> str:
    """格式化一个 SSE 事件：`event:` + `data:` 两行 + 空行结尾。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/query/stream")
async def query_stream(
    stream_request: Request,
    request: QueryRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    """
    SSE 流式问答（对话联调页使用）。需要登录（Bearer JWT）。

    事件流（总稿 §4.3.4）：queue（0..N 次，仅排队时）→ started（1 次）→
    sources（1 次）→ token×N（逐块文本）→ done（完整回答 + timings +
    session_id + request_id，已落库）；高危危机直接发 done；异常发 error。
    旧前端忽略未知事件后仍可消费原有 sources/token/done/error（向后兼容）。
    """
    # 越权预校验（与 /api/query 一致，在调用 LLM 之前快速失败）
    _assert_no_impersonation(request.user_id, current_user.id)
    await _assert_session_ownership(request.session_id, current_user.id, db)
    user_id = current_user.id
    request_id = uuid.uuid4().hex

    # —— 准入：submit（去重 + 槽位检查 + 入队原子完成）；409/429 在建立流前返回 ——
    sub = await admission.submit(user_id, request_id)
    if sub.code == SubmitCode.REJECTED_DUPLICATE:
        return _admission_json(409, "你已有问题正在处理", AI_REQUEST_IN_PROGRESS)
    if sub.code == SubmitCode.REJECTED_FULL:
        return _admission_json(
            429, "当前排队已满，请稍后重试", AI_QUEUE_FULL,
            headers={"Retry-After": _ADMISSION_RETRY_AFTER},
        )
    ticket = sub.ticket

    async def event_stream():
        # 全流程墙钟计时（用于 done 事件里的 total，与 /api/query 语义一致）
        t_total = time.perf_counter()
        terminal = TerminalReason.COMPLETED.value
        wait_ms = 0.0
        try:
            # ============ 1) 排队阶段：发 queue 事件直到放行/超时/取消/断连 ============
            if sub.code == SubmitCode.QUEUED:
                upd0 = admission.queue_update(request_id)
                yield _sse_queue_event(request_id, upd0.position, upd0.queued, upd0.active)
                outcome_fut = asyncio.ensure_future(
                    admission.wait_until_running(request_id)
                )
                queued_disconnected = False
                while not outcome_fut.done():
                    try:
                        # 0.5s 粒度轮询断连/取消：排队中客户端断开应在 ~0.5s 内被移除
                        # （验收方案 R-10 要求 1s 内；1.0s 轮询贴线，收紧到 0.5s）
                        await asyncio.wait_for(asyncio.shield(outcome_fut), timeout=0.5)
                    except asyncio.TimeoutError:
                        # 客户端在排队中断开 → 尽力取消排队
                        if await stream_request.is_disconnected():
                            queued_disconnected = True
                            break
                        upd = admission.queue_update(request_id)
                        if upd.position >= 0:
                            yield _sse_queue_event(request_id, upd.position, upd.queued, upd.active)
                        continue
                if queued_disconnected:
                    await admission.cancel(user_id, request_id)
                    outcome = await asyncio.shield(outcome_fut)
                    if outcome.code == WaitCode.STARTED:
                        # 取消前已被提升：本协程持有槽位，释放防止泄漏
                        admission.note_started(ticket, outcome.wait_ms)
                        await admission.release(
                            ticket, terminal=TerminalReason.DISCONNECTED.value
                        )
                    else:
                        reason = (
                            TerminalReason.QUEUE_TIMEOUT.value
                            if outcome.code == WaitCode.QUEUE_TIMEOUT
                            else TerminalReason.CANCELLED.value
                        )
                        admission.record_dropped(ticket, outcome.wait_ms, reason)
                    return
                outcome = await asyncio.shield(outcome_fut)
                wait_ms = outcome.wait_ms
                if outcome.code == WaitCode.QUEUE_TIMEOUT:
                    admission.record_dropped(ticket, wait_ms, TerminalReason.QUEUE_TIMEOUT.value)
                    yield _sse("error", {
                        "detail": "排队等待超时，请重新发起",
                        "code": AI_QUEUE_TIMEOUT,
                        "error_type": "queue_timeout",
                    })
                    return
                if outcome.code == WaitCode.CANCELLED:
                    admission.record_dropped(ticket, wait_ms, TerminalReason.CANCELLED.value)
                    yield _sse("error", {
                        "detail": "请求已取消",
                        "code": AI_REQUEST_CANCELLED,
                        "error_type": "cancelled",
                    })
                    return
                admission.note_started(ticket, wait_ms)
                snap = await admission.snapshot()
                yield _sse("started", {
                    "request_id": request_id,
                    "queue_wait_ms": round(wait_ms, 1),
                    "active": snap.active,
                })
            else:
                # 立即获得槽位：无需 queue 事件，直接 started
                admission.note_started(ticket, 0.0)
                snap = await admission.snapshot()
                yield _sse("started", {
                    "request_id": request_id,
                    "queue_wait_ms": 0,
                    "active": snap.active,
                })

            # ============ 2) 放行后：安全检测 + 混合检索 + 重排（同步放线程池） ============
            prep = await run_in_threadpool(
                rag_system.prepare,
                question=request.question,
                messages=request.messages,
                check_safety=request.safety_enabled,
                user_id=user_id,
                rag_enabled=request.rag_enabled,
            )
            if prep.get("is_crisis_response"):
                # 高危拦截同样写会话与危机审计（与 /api/query 高危路径对齐；落库后台化）
                session_id = None
                if request.persist and not admission.is_cancelling(request_id):
                    session_id = request.session_id or uuid.uuid4().hex
                    await _enqueue_persist(
                        session_id,
                        prep.get("question", ""),
                        prep.get("answer", ""),
                        request.title,
                        user_id,
                        safety_check=prep.get("safety_check"),
                        is_crisis_response=True,
                        safety_note=prep.get("answer", ""),
                    )
                yield _sse("done", {
                    "answer": prep.get("answer", ""),
                    "is_crisis_response": True,
                    "safety_check": prep.get("safety_check"),
                    "session_id": session_id,
                    "request_id": request_id,
                })
                return

            # 3) 来源与检索耗时（生成尚未开始，前端可先展示引用）
            yield _sse("sources", {
                "sources": prep.get("sources") or [],
                "timings": prep.get("timings") or {},
            })

            # 4) 流式生成：逐 token 下发；断连 / 被取消时及时终止
            # prompt_messages 预构建（含同步长期记忆检索/embedding）放线程池，
            # 避免首次生成前在事件循环上跑同步 embed，阻塞所有并发流
            prompt_messages = await asyncio.to_thread(
                rag_system.rag._build_messages,
                prep["question"],
                prep.get("context") or [],
                prep.get("norm_messages"),
                user_id,
                (bool(prep.get("rag_enabled")) and not prep.get("context")),
            )
            timings = prep.get("timings") or {}
            full: list[str] = []
            t_gen = time.perf_counter()
            stopped = False
            async for chunk in rag_system.rag.stream_generate(
                prep["question"],
                prep.get("context") or [],
                messages=prep.get("norm_messages"),
                user_id=user_id,
                low_relevance=(bool(prep.get("rag_enabled")) and not prep.get("context")),
                prompt_messages=prompt_messages,
            ):
                full.append(chunk)
                yield _sse("token", {"text": chunk})
                # 客户端断开（关页面/刷新）或用户取消 → 终止生成，避免浪费上游 token
                if await stream_request.is_disconnected():
                    print("[query/stream] 客户端断开，终止生成", flush=True)
                    terminal = TerminalReason.DISCONNECTED.value
                    stopped = True
                    break
                if admission.is_cancelling(request_id):
                    print("[query/stream] 用户取消，终止生成", flush=True)
                    terminal = TerminalReason.CANCELLED.value
                    stopped = True
                    break
            answer = "".join(full)
            # 回答侧安全复查：命中高危关键词时追加安全提醒（纯 L0，CPU 级）
            ans_check = None
            if request.safety_enabled is not False and settings.SAFETY_ENABLED:
                answer, ans_check = rag_system.safety_checker.review_answer(answer)
            timings["llm"] = (time.perf_counter() - t_gen) * 1000
            timings["total"] = (time.perf_counter() - t_total) * 1000

            if stopped:
                # 断连直接结束；取消发 error 让前端复位按钮与 loading
                if terminal == TerminalReason.CANCELLED.value:
                    yield _sse("error", {
                        "detail": "回答已停止",
                        "code": AI_REQUEST_CANCELLED,
                        "error_type": "cancelled",
                    })
                return

            # 5) 持久化后台化（Queue→Worker→DB，见 modules/bg_queue）；
            #    失败不影响已生成的回答；session_id 在请求内生成并随 done 下发
            session_id = None
            if request.persist:
                session_id = request.session_id or uuid.uuid4().hex
                await _enqueue_persist(
                    session_id,
                    prep.get("question", ""),
                    answer,
                    request.title,
                    user_id,
                    safety_check=prep.get("safety_check"),
                    is_crisis_response=False,
                    safety_note=prep.get("safety_note"),
                    answer_safety_check=ans_check,
                )

            yield _sse("done", {
                "answer": answer,
                "safety_note": prep.get("safety_note"),
                "safety_check": prep.get("safety_check"),
                "timings": timings,
                "session_id": session_id,
                "request_id": request_id,
            })
        except Exception as e:
            # 记录完整异常到服务端日志；SSE error 事件只带异常类型（不暴露堆栈/路径）
            terminal = TerminalReason.FAILED.value
            print(f"[query/stream][ERROR] {type(e).__name__}: {e}", flush=True)
            yield _sse("error", {
                "detail": f"生成失败（{type(e).__name__}），请稍后重试。",
                "error_type": type(e).__name__,
            })
        finally:
            # 工作真实退出后才释放活跃槽位（幂等；断连/取消/异常均覆盖）
            await admission.release(ticket, terminal=terminal)
            # 兜底防泄漏（R-10 实测暴露）：排队阶段因客户端断连/异常提前退出时，
            # 条目仍在 _queue（release 对 queued 条目返回 NOT_RUNNING、无清理效果），
            # 会被后续 _promote 提升为“无消费者协程”的 running 请求 → 槽位永久泄漏。
            # 此处幂等补 cancel：已在活跃/已终态/已释放时返回 CANCELLING/ALREADY_FINISHED，无副作用。
            try:
                await admission.cancel(user_id, request_id)
            except Exception as _e:  # noqa: BLE001 - 清理兜底失败仅告警，不影响主流程
                print(f"[query/stream][WARN] 排队残留清理失败: {type(_e).__name__}: {_e}", flush=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防止反代缓冲（本地直连无影响）
        },
    )


# ---------------- 准入：取消与状态（总稿 FR-BE-06 / FR-BE-10） ----------------
@app.delete("/api/query/requests/{request_id}")
async def cancel_request(request_id: str, current_user=Depends(get_current_user)):
    """
    取消自己的问答请求（Phase 1 memory）。
    QUEUED → 原子移除并释放用户占位；RUNNING → 标记取消，等待工作协程真实退出。
    操作他人请求 → 403；已终态/不存在 → 409。
    """
    owner = admission.resolve_user_of(request_id)
    if owner is None:
        return _admission_json(409, "请求不存在或已结束", AI_REQUEST_ALREADY_FINISHED)
    if owner != current_user.id:
        raise HTTPException(status_code=403, detail="无权操作该请求")
    res = await admission.cancel(current_user.id, request_id)
    if res.code == CancelCode.CANCELLED:
        return {"ok": True, "status": "cancelled", "request_id": request_id}
    if res.code == CancelCode.CANCELLING:
        return {"ok": True, "status": "cancelling", "request_id": request_id}
    return _admission_json(409, "请求已结束", AI_REQUEST_ALREADY_FINISHED)


@app.get("/api/concurrency/status")
async def concurrency_status():
    """准入状态聚合（只返回聚合信息，不涉及个人数据，无需登录）。"""
    snap = await admission.snapshot()
    return {
        "backend": snap.backend,
        "max_active": snap.max_active,
        "active": snap.active,
        "max_queue": snap.max_queue,
        "queued": snap.queued,
        "accepting": snap.accepting,
    }


@app.get("/api/health")
async def health_check():
    """
    健康检查接口：同时暴露后台持久化可靠性指标（落库失败必须可观测，验收方案 §8）。
    """
    from modules.bg_queue import bg_queue

    return {
        "status": "healthy",
        "version": "1.0.0",
        "persist": {
            "queue_depth": bg_queue.queue_depth(),
            "total": _persist_total,
            "completed": bg_queue.completed_count,
            "failures": _persist_failures,
            "critical_failures": _persist_critical_failures,
        },
    }


@app.get("/api/sessions")
async def list_sessions(
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    """列出当前用户的最近会话（含消息数）；他人会话不可见（数据隔离）。

    请求级 AsyncSession + selectinload(messages)（异步数据层，防 MissingGreenlet）。
    """
    rows = await crud_async.list_sessions(db, current_user.id, limit)
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "message_count": len(s.messages),
        }
        for s in rows
    ]


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    """获取某会话的全部消息（按时间顺序）；非本人会话 → 403。"""
    sess = await crud_async.get_session_with_messages(db, session_id)
    if sess is None:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    if sess.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    return [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in sess.messages
    ]


@app.post("/api/sessions")
async def create_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    """新建一个空会话（归属当前用户），返回服务端生成的 id。"""
    sess = await crud_async.create_session(
        db, uuid.uuid4().hex, payload.name or "新的对话", current_user.id
    )
    return {
        "id": sess.id,
        "name": sess.title,
        "created_at": sess.created_at.isoformat() if sess.created_at else None,
        "updated_at": sess.updated_at.isoformat() if sess.updated_at else None,
        "message_count": 0,
    }


@app.patch("/api/sessions/{session_id}")
async def rename_session(
    session_id: str,
    payload: SessionRename,
    db: AsyncSession = Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    """重命名会话；非本人会话 → 403。"""
    sess = await crud_async.get_session_with_messages(db, session_id)
    if sess is None or sess.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权操作该会话")
    sess.title = payload.name[:255]
    return {"id": sess.id, "name": sess.title}


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    """删除会话（级联删除其消息）；非本人会话 → 403。"""
    sess = await crud_async.get_session_with_messages(db, session_id)
    if sess is None or sess.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权操作该会话")
    await db.delete(sess)
    return {"ok": True, "id": session_id}


# ---------------- 管理员接口（垂直越权测试目标：普通用户访问 → 403） ----------------
@app.get("/api/admin/users")
async def admin_list_users(
    db: AsyncSession = Depends(get_db_session),
    admin=Depends(require_admin),
):
    """管理员：查看用户列表。"""
    users = await crud_async.list_users(db)
    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]


@app.get("/api/admin/crisis-audit")
async def admin_list_crisis_audits(
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session),
    admin=Depends(require_admin),
):
    """管理员：查看危机审计记录（合规留痕，仅管理员可见）。"""
    rows = await crud_async.list_crisis_audits(db, limit=limit)
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "session_id": r.session_id,
            "crisis_level": r.crisis_level,
            "keywords_found": json.loads(r.keywords_found) if r.keywords_found else None,
            "question": r.question,
            "is_crisis_response": r.is_crisis_response,
            "detect_method": r.detect_method,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.on_event("startup")
async def startup_event():
    """启动时验证配置、建表、引导账号"""
    settings.validate()
    # Phase 1 memory 准入：单实例守卫（阻断 uvicorn --workers / 重复启动的第二个实例）
    _acquire_single_instance_lock()
    init_db()
    # 历史遗留的占位标题会话（“新的对话”等）按最早一条用户消息自动命名（幂等）
    try:
        with crud.get_db() as db:
            renamed = crud.rename_unnamed_sessions(db)
        if renamed:
            print(f"[startup] 已为 {renamed} 个未命名会话自动生成标题")
    except Exception as e:
        print(f"[startup][WARN] 自动命名未执行（不影响启动）: {e}")
    # 预热本地重排模型：后台线程加载（约 5-10 秒），不阻塞启动；
    # 预热失败静默（问答时自动回退到原排序），保证首次问答不卡顿。
    # 仅 RAG 检索开启时才需要重排/BM25：RAG_ENABLED=False（纯对话模式）时跳过加载。
    if settings.RERANK_ENABLED and settings.RAG_ENABLED:
        import threading

        def _warm_reranker():
            try:
                from modules.reranker import get_reranker

                get_reranker()._load()
                print("[startup] 本地重排模型已加载（bge-reranker-v2-m3）")
            except Exception as e:
                print(f"[startup][WARN] 重排模型预热失败，问答时将回退原排序: {e}")
            # 混合检索 BM25 索引预热（失败静默，首次问答时懒构建兜底）
            try:
                from modules.hybrid_search import warm_up_index

                warm_up_index()
            except Exception:
                pass

        threading.Thread(target=_warm_reranker, daemon=True).start()
    # 预热语义危机检测锚点（后台批量 embed 约几秒，不阻塞启动；
    # 未就绪时问答回退关键词，避免首次请求被同步构建卡住）。
    # 仅安全链路总开关开启时才需要：SAFETY_ENABLED=False 时整条安全检测被跳过，锚点不会被使用。
    if settings.SEMANTIC_CHECK_ENABLED and settings.SAFETY_ENABLED:
        import threading as _th

        def _warm_crisis_detector():
            try:
                from modules.crisis_detector import get_crisis_detector

                get_crisis_detector().warm_up()
            except Exception as e:
                print(f"[startup][WARN] 语义危机检测预热失败（首次问答将回退关键词）: {e}", flush=True)

        _th.Thread(target=_warm_crisis_detector, daemon=True).start()
    print("=" * 50)
    print("青少年心理RAG系统已启动")
    print(f"模型: {settings.CHAT_MODEL}")
    print(f"向量数据库: {settings.VECTOR_BACKEND}")
    # 脱敏打印：DB URL 含密码，不得直接写入终端/日志（统一日志聚合时会泄露）
    print(f"关系数据库: {re.sub(r'(://[^:/]+:)[^@/]+(@)', lambda m: m.group(1) + '******' + m.group(2), settings.DB_URL)}")
    print(f"RAG 检索: {'启用' if settings.RAG_ENABLED else '禁用（纯对话模式，不加载重排/BM25）'}")
    print(f"安全检查: {'启用' if settings.SAFETY_ENABLED else '禁用（不加载语义锚点）'}")
    print(f"本地重排: {'启用（' + settings.RERANK_MODEL + '）' if settings.RERANK_ENABLED and settings.RAG_ENABLED else '未启用'}")
    print(f"并发准入: {settings.AI_ADMISSION_BACKEND}（活跃 {settings.AI_MAX_ACTIVE_REQUESTS} / 队列 {settings.AI_MAX_QUEUED_REQUESTS} / 等待超时 {settings.AI_QUEUE_WAIT_TIMEOUT_SECONDS}s）")
    # 启动后台持久化 Worker（Queue→Worker→DB；Redis 属 Phase 2，本期不引入）
    bg_queue.start()
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时优雅停止后台 Worker。"""
    try:
        await bg_queue.shutdown()
    except Exception as e:
        print(f"[shutdown][WARN] 后台 Worker 停止异常: {e}", flush=True)


# 前端静态资源托管：在 API 路由之后挂载，/api、/docs 等显式路由优先匹配。
# 注意：必须位于 __main__ 块之前 —— `python api/main.py` 直接运行时若挂在 __main__
# 之后，_serve() 阻塞执行时挂载代码尚未运行，前端会全部 404（import 方式无此问题）。
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIR), html=True),
        name="frontend",
    )


if __name__ == "__main__":
    # Windows 兼容：psycopg(async) 需要 SelectorEventLoop，而本机 uvicorn 版本在
    # Windows 上无论 --loop 参数都强制 Proactor（uvicorn.run 内部建循环时无视已设策略）。
    # 解法：自己持有循环 —— 先切 Selector 策略，再由 asyncio.run 创建循环，
    # 在循环内以 uvicorn.Server 对象方式 serve()（不再走 uvicorn.run 的循环创建）。
    import asyncio
    import sys

    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass

    import uvicorn

    async def _serve() -> None:
        config = uvicorn.Config(
            app,
            host=settings.HOST,
            port=settings.PORT,
            log_level="info",
            reload=False,  # 调试热重载请用 DEBUG=False + 手动重启；本入口不启子进程（保 Selector）
        )
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(_serve())

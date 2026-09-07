"""微信小程序通道 /api/mp/*（与 JWT Web 端并存的第二身份通道）。

身份模型（MP_TRUST_MODE=a · 内网直传，见 config/settings.py）：
请求体 userId 即身份来源，后端**不自建账号**——首次出现时自动建 users 镜像行
（不可登录，username=mp_<sha1>），此后会话/长期记忆/危机审计/家庭档案均挂该 id，
与外部既有小程序（它自己在别处负责注册登录）直接对齐，无需映射表。

⚠️ 信任边界（务必遵守）：模式 a 下任何能访问本服务的人都可伪造任意 userId，
越权读取他人会话/危机审计/档案。因此本通道**只应部署在受信边界内**：
- 本机/内网联调（HOST=127.0.0.1 默认）；
- 或经网关 IP 白名单 / API-Key 保护；
切勿不加保护直接暴露公网。升级到公网直连请先实现 MP_TRUST_MODE=b（code2session）。

纵深防御（2026-09-07）：即使在内网，也建议在 .env 配置 MP_API_KEY（非空后
/api/mp/* 全部端点要求 X-API-Key 头，router 级强制，防同网段其它进程/误暴露时
直接调用与枚举档案）。空 = 不启用（本地联调免 key，保持现状）。

接口（字段名与外部小程序约定一致，注册/资料编辑/对话均由小程序发起）：
- POST /api/mp/register  {userId, profile}             注册时写入家庭档案（含建镜像号，幂等）
- POST /api/mp/update    {userId, profile}             资料更新（同 upsert，幂等）
- GET  /api/mp/profile   ?userId=xxx                   回读档案（资料页回显）
- POST /api/mp/query     {userId, sessionId, query}    非流式对话
- POST /api/mp/query/stream {userId, sessionId, query} SSE 流式对话（小程序基础库
  ≥2.20.2 用 wx.request enableChunked:true + onChunkReceived 分块解析，事件与
  /api/query/stream 一致：queue→started→sources→token×N→done；高危直达 done/error）
"""
import asyncio
import hmac
import logging
import time
import uuid
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from db import crud_async
from api.deps import get_db_session
from modules.concurrency.models import (
    AI_QUEUE_FULL,
    AI_QUEUE_TIMEOUT,
    AI_REQUEST_CANCELLED,
    AI_REQUEST_IN_PROGRESS,
    SubmitCode,
    TerminalReason,
    WaitCode,
)
from modules.concurrency.service import admission
from modules.family_profile import format_profile_for_prompt
from modules.gateway import (
    _ADMISSION_RETRY_AFTER,
    admission_json,
    enqueue_persist,
    sse,
    sse_queue_event,
)
from modules import rag_system

# ---- 共享 API-Key（纵深防御）：settings.MP_API_KEY 非空时全 router 强制 ----
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_mp_key(cred: Optional[str] = Depends(_api_key_header)) -> None:
    """router 级依赖：MP_API_KEY 配置后，/api/mp/* 全部端点须携带正确 X-API-Key。

    空配置 = 不启用（本地联调免 key）。compare_digest 防时序侧信道。
    目的：即使信任边界（内网/白名单）失守，同网段其它进程也不能直接调用
    mp 通道或枚举 /api/mp/profile 读取家庭档案（含未成年人出生日期/性别）。
    """
    if not settings.MP_API_KEY:
        return
    if not cred or not hmac.compare_digest(cred.encode("utf-8"), settings.MP_API_KEY.encode("utf-8")):
        raise HTTPException(status_code=401, detail="缺少或无效的 API-Key")


router = APIRouter(
    prefix="/api/mp",
    tags=["mp-miniprogram"],
    dependencies=[Depends(require_mp_key)],
)

logger = logging.getLogger("rag.api.mp")


# ---------------- 请求体模型（字段名对齐外部小程序约定） ----------------
class MpChild(BaseModel):
    childId: str = Field(..., min_length=1, max_length=64, description="孩子 id（小程序侧生成）")
    childNickname: str = Field("", max_length=64)
    childBirthDate: Optional[str] = Field(None, description="孩子出生日期 YYYY-MM-DD")
    gender: str = Field("", max_length=8, description="性别（前端原值，如 男/女）")


class MpProfile(BaseModel):
    userNickname: str = Field("", max_length=64, description="家长昵称")
    birthday: Optional[str] = Field(None, description="家长生日 YYYY-MM-DD")
    parent_role: str = Field("", max_length=20, description="家长角色（爸爸/妈妈/其他）")
    children: List[MpChild] = Field(
        default_factory=list,
        max_length=10,  # 防超大 payload：单家庭孩子数上限（正常 ≤ 4）
        description="孩子列表（整份提交，全量替换，最多 10 个）",
    )


class MpRegisterBody(BaseModel):
    userId: str = Field(..., min_length=1, max_length=64, description="外部用户 id（小程序生态已有，后端不自建）")
    profile: MpProfile = Field(default_factory=MpProfile)


class MpUpdateBody(MpRegisterBody):
    """资料更新：与 register 同 body（upsert 幂等）。"""


class MpQueryBody(BaseModel):
    userId: str = Field(..., min_length=1, max_length=64)
    # 与 sessions.id 列宽一致（String(36)）：超长会在落库时 DataError 500，
    # 这里直接 422 提前拒绝（LLM 前快速失败）。小程序生成 id 建议 sess_<ts>_<rand> ≤36
    sessionId: Optional[str] = Field(None, max_length=36, description="会话 id（小程序生成 ≤36 字符，如 sess_<ts>_<rand>；缺省服务端生成）")
    query: str = Field(..., min_length=1, max_length=2000, description="用户本轮问题")


# ---------------- 工具 ----------------
def _clean_user_id(user_id: str) -> str:
    """userId 清洗：非空 + 长度上限（users.id 列宽 64）。"""
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="userId 不能为空")
    if len(uid) > settings.MP_USER_ID_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"userId 超长（最多 {settings.MP_USER_ID_MAX_LEN} 字符）",
        )
    return uid


def _to_date(value: Optional[str]) -> Optional[date]:
    """'YYYY-MM-DD' → date；空 → None；非法 → 400（避免脏数据进 DATE 列）。"""
    if value is None or str(value).strip() == "":
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        raise HTTPException(status_code=400, detail=f"日期格式须为 YYYY-MM-DD，收到: {value}")


def _profile_to_crud(profile: MpProfile):
    """把 camelCase 档案模型拆成 crud_async.upsert_family_profile 需要的主表/子表字段。"""
    main = {
        "user_nickname": profile.userNickname,
        "birthday": _to_date(profile.birthday),
        "parent_role": profile.parent_role,
    }
    children = [
        {
            "child_id": c.childId,
            "child_nickname": c.childNickname,
            "child_birth_date": _to_date(c.childBirthDate),
            "gender": c.gender,
        }
        for c in profile.children
    ]
    return main, children


def _profile_to_camel(profile: dict) -> dict:
    """回读档案（snake dict）→ camel JSON（小程序资料页直接渲染）。"""
    return {
        "userNickname": profile.get("user_nickname") or "",
        "birthday": profile.get("birthday"),
        "parent_role": profile.get("parent_role") or "",
        "children": [
            {
                "childId": k["child_id"],
                "childNickname": k["child_nickname"] or "",
                "childBirthDate": k.get("child_birth_date"),
                "gender": k.get("gender") or "",
            }
            for k in profile.get("children") or []
        ],
    }


async def _load_profile_text(db: AsyncSession, uid: str) -> str:
    """读取档案并加工为注入文本；未注册档案 / 注入开关关闭 → ""（不注入）。"""
    if not settings.PROFILE_INJECT_ENABLED:
        return ""
    try:
        profile = await crud_async.get_family_profile(db, uid)
    except Exception:
        return ""
    if not profile:
        return ""
    try:
        return format_profile_for_prompt(profile)
    except Exception:
        return ""


async def _assert_session_ownership(session_id, user_id: str, db: AsyncSession) -> None:
    """水平越权：session_id 已存在但不属于当前 userId → 403（未创建的新 id 放行）。

    与 Web 端语义一致；差异仅在身份来源：模式 a 下请求体 userId 即当前用户。
    """
    if not session_id:
        return
    if not await crud_async.session_belongs_to(db, session_id, user_id):
        raise HTTPException(status_code=403, detail="无权访问该会话")


async def _admission_enter(user_id: str, request_id: str):
    """准入（与 /api/query 语义一致）：submit → 去重/队满/排队/超时 → 占槽。

    返回 (ticket, wait_ms, error)；error 非 None 时调用方直接返回该响应。
    """
    sub = await admission.submit(user_id, request_id)
    if sub.code == SubmitCode.REJECTED_DUPLICATE:
        return None, 0.0, admission_json(409, "你已有问题正在处理", AI_REQUEST_IN_PROGRESS)
    if sub.code == SubmitCode.REJECTED_FULL:
        return None, 0.0, admission_json(
            429, "当前排队已满，请稍后重试", AI_QUEUE_FULL,
            headers={"Retry-After": _ADMISSION_RETRY_AFTER},
        )
    ticket = sub.ticket
    wait_ms = 0.0
    try:
        if sub.code == SubmitCode.QUEUED:
            wait_res = await admission.wait_until_running(request_id)
            wait_ms = wait_res.wait_ms
            if wait_res.code == WaitCode.QUEUE_TIMEOUT:
                admission.record_dropped(ticket, wait_ms, TerminalReason.QUEUE_TIMEOUT.value)
                return None, 0.0, admission_json(503, "排队等待超时，请重新发起", AI_QUEUE_TIMEOUT)
            if wait_res.code == WaitCode.CANCELLED:
                admission.record_dropped(ticket, wait_ms, TerminalReason.CANCELLED.value)
                return None, 0.0, admission_json(409, "请求已取消", AI_REQUEST_CANCELLED)
        admission.note_started(ticket, wait_ms)
    except BaseException:
        # 与 Web /api/query 同步路径同款兜底（2026-09-07）：排队等待期间协程被取消
        # （客户端断开）/异常时，条目与 _user_req 占位残留 → 该用户永久 409 +
        # 队列孤儿被 _promote 提升为无消费者 running → 槽位泄漏。release 对非
        # _active 条目无效 → 先 cancel（queued→清占位；running→置 cancelling）
        # 再 release 释放活跃槽位（幂等，两分支覆盖）。
        try:
            await admission.cancel(user_id, request_id)
        finally:
            await admission.release(ticket, terminal=TerminalReason.CANCELLED.value)
        raise
    return ticket, wait_ms, None


async def _ensure_mp_user(db: AsyncSession, uid: str, nickname: str = "") -> None:
    """宽容建号 + 镜像守卫：命中内部账号（username 非 mp_ 前缀）→ 400。

    外部 userId 不得接管系统内部账号（JWT 注册的 uuid / legacy / admin）的数据
    上下文；正常小程序生态 id 首次出现时自动建镜像行（幂等、并发安全）。
    """
    try:
        await crud_async.get_or_create_mp_user(db, uid, nickname=nickname)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------- 档案：register / update / profile 回读 ----------------
async def _upsert_profile(body: MpRegisterBody, db: AsyncSession, create_user: bool) -> dict:
    """register/update 共用：清洗 userId → （可选）宽容建镜像号 → upsert 档案（幂等）。"""
    uid = _clean_user_id(body.userId)
    if create_user:
        await _ensure_mp_user(db, uid, nickname=body.profile.userNickname)
    main, children = _profile_to_crud(body.profile)
    await crud_async.upsert_family_profile(db, uid, **main, children=children)
    return {"ok": True, "userId": uid}


@router.post("/register", status_code=201)
async def register(body: MpRegisterBody, db: AsyncSession = Depends(get_db_session)):
    """注册时写入家庭档案：users 镜像行不存在则创建 + 档案 upsert（幂等，可重放）。"""
    return await _upsert_profile(body, db, create_user=True)


@router.post("/update")
async def update_profile(body: MpUpdateBody, db: AsyncSession = Depends(get_db_session)):
    """资料更新：同 upsert 幂等语义（宽容建号，与 register 行为对齐，重复提交无害）。"""
    return await _upsert_profile(body, db, create_user=True)


@router.get("/profile")
async def get_profile(userId: str, db: AsyncSession = Depends(get_db_session)):
    """回读整份家庭档案（camelCase，供小程序资料页回显）；未注册返回 {}。"""
    uid = _clean_user_id(userId)
    profile = await crud_async.get_family_profile(db, uid)
    return _profile_to_camel(profile) if profile else {}


# ---------------- 对话：非流式 ----------------
@router.post("/query")
async def mp_query(
    body: MpQueryBody,
    db: AsyncSession = Depends(get_db_session),
):
    """非流式对话：{userId, sessionId, query} → JSON。

    复用 Web 端完整链路（安全检测/RAG/生成/持久化/危机审计/长期记忆）的
    rag_system.aquery；差异：身份来自请求体 userId（模式 a）、档案加工后注入
    prompt。RAG/安全开关沿用 settings 全局配置。
    """
    uid = _clean_user_id(body.userId)
    await _ensure_mp_user(db, uid)  # 宽容建号：未调 register 也能对话
    await _assert_session_ownership(body.sessionId, uid, db)
    profile_text = await _load_profile_text(db, uid)

    request_id = uuid.uuid4().hex
    ticket, wait_ms, err = await _admission_enter(uid, request_id)
    if err is not None:
        return err

    terminal = TerminalReason.COMPLETED.value
    try:
        result = await rag_system.aquery(
            question=body.query,
            messages=None,
            check_safety=None,
            user_id=uid,
            rag_enabled=None,
            session_id=body.sessionId,
            profile_text=profile_text,
        )
        if result.get("cancelled"):
            terminal = TerminalReason.CANCELLED.value
            return admission_json(409, "请求已取消", AI_REQUEST_CANCELLED)

        session_id = body.sessionId or uuid.uuid4().hex
        if not admission.is_cancelling(request_id):
            await enqueue_persist(
                session_id,
                result.get("question", ""),
                result.get("answer", ""),
                title=None,
                user_id=uid,
                safety_check=result.get("safety_check"),
                is_crisis_response=bool(result.get("is_crisis_response", False)),
                safety_note=result.get("safety_note"),
                answer_safety_check=result.get("answer_safety_check"),
            )
        if admission.is_cancelling(request_id):
            terminal = TerminalReason.CANCELLED.value

        return {
            "answer": result.get("answer", ""),
            "sessionId": session_id,
            "sources": result.get("sources") or [],
            "safetyNote": result.get("safety_note"),
            "isCrisisResponse": bool(result.get("is_crisis_response", False)),
            "timings": result.get("timings"),
        }
    except ValueError as e:
        terminal = TerminalReason.FAILED.value
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        terminal = TerminalReason.FAILED.value
        raise HTTPException(status_code=500, detail="内部处理失败，请稍后重试。")
    finally:
        await admission.release(ticket, terminal=terminal)


# ---------------- 对话：流式（SSE，事件协议与 /api/query/stream 一致） ----------------
@router.post("/query/stream")
async def mp_query_stream(
    stream_request: Request,
    body: MpQueryBody,
    db: AsyncSession = Depends(get_db_session),
):
    """SSE 流式对话：{userId, sessionId, query} → 事件流。

    事件序列（对齐 Web 端）：queue（仅排队时 0..N）→ started → sources →
    token×N → done（含完整回答/sessionId/timings，已落库）；高危拦截直接 done；
    异常发 error。小程序端解析：基础库 ≥2.20.2 的 wx.request({enableChunked:true})
    + requestTask.onChunkReceived() 按 event:/data: 帧拆包，done 后 abort()。
    """
    uid = _clean_user_id(body.userId)
    await _ensure_mp_user(db, uid)
    await _assert_session_ownership(body.sessionId, uid, db)
    profile_text = await _load_profile_text(db, uid)

    request_id = uuid.uuid4().hex
    sub = await admission.submit(uid, request_id)
    if sub.code == SubmitCode.REJECTED_DUPLICATE:
        return admission_json(409, "你已有问题正在处理", AI_REQUEST_IN_PROGRESS)
    if sub.code == SubmitCode.REJECTED_FULL:
        return admission_json(
            429, "当前排队已满，请稍后重试", AI_QUEUE_FULL,
            headers={"Retry-After": _ADMISSION_RETRY_AFTER},
        )
    ticket = sub.ticket

    async def event_stream():
        t_total = time.perf_counter()
        terminal = TerminalReason.COMPLETED.value
        wait_ms = 0.0
        try:
            # 1) 排队阶段
            if sub.code == SubmitCode.QUEUED:
                upd0 = admission.queue_update(request_id)
                yield sse_queue_event(request_id, upd0.position, upd0.queued, upd0.active)
                outcome_fut = asyncio.ensure_future(
                    admission.wait_until_running(request_id)
                )
                while not outcome_fut.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(outcome_fut), timeout=0.5)
                    except asyncio.TimeoutError:
                        if await stream_request.is_disconnected():
                            await admission.cancel(uid, request_id)
                            outcome = await asyncio.shield(outcome_fut)
                            if outcome.code == WaitCode.STARTED:
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
                        upd = admission.queue_update(request_id)
                        if upd.position >= 0:
                            yield sse_queue_event(request_id, upd.position, upd.queued, upd.active)
                        continue
                outcome = await asyncio.shield(outcome_fut)
                wait_ms = outcome.wait_ms
                if outcome.code == WaitCode.QUEUE_TIMEOUT:
                    admission.record_dropped(ticket, wait_ms, TerminalReason.QUEUE_TIMEOUT.value)
                    yield sse("error", {"detail": "排队等待超时，请重新发起", "code": AI_QUEUE_TIMEOUT, "error_type": "queue_timeout"})
                    return
                if outcome.code == WaitCode.CANCELLED:
                    admission.record_dropped(ticket, wait_ms, TerminalReason.CANCELLED.value)
                    yield sse("error", {"detail": "请求已取消", "code": AI_REQUEST_CANCELLED, "error_type": "cancelled"})
                    return
                admission.note_started(ticket, wait_ms)
                yield sse("started", {"request_id": request_id, "queue_wait_ms": round(wait_ms, 1)})
            else:
                admission.note_started(ticket, 0.0)
                yield sse("started", {"request_id": request_id, "queue_wait_ms": 0})

            # 2) 安全检测 + 检索（同步部分放线程池）
            from fastapi.concurrency import run_in_threadpool

            prep = await run_in_threadpool(
                rag_system.prepare,
                question=body.query,
                messages=None,
                check_safety=None,
                user_id=uid,
                rag_enabled=None,
                session_id=body.sessionId,
            )
            if prep.get("is_crisis_response"):
                session_id = body.sessionId or uuid.uuid4().hex
                if not admission.is_cancelling(request_id):
                    await enqueue_persist(
                        session_id,
                        prep.get("question", ""),
                        prep.get("answer", ""),
                        title=None,
                        user_id=uid,
                        safety_check=prep.get("safety_check"),
                        is_crisis_response=True,
                        safety_note=prep.get("answer", ""),
                    )
                yield sse("done", {
                    "answer": prep.get("answer", ""),
                    "is_crisis_response": True,
                    "safety_check": prep.get("safety_check"),
                    "sessionId": session_id,
                    "request_id": request_id,
                })
                return

            yield sse("sources", {
                "sources": prep.get("sources") or [],
                "timings": prep.get("timings") or {},
            })

            # 3) prompt 预构建（含记忆检索与家庭档案）放线程池，避免阻塞事件循环
            prompt_messages = await asyncio.to_thread(
                rag_system.rag._build_messages,
                prep["question"],
                prep.get("context") or [],
                prep.get("norm_messages"),
                uid,
                (bool(prep.get("rag_enabled")) and not prep.get("context")),
                profile_text,
            )
            timings = prep.get("timings") or {}
            full: list[str] = []
            t_gen = time.perf_counter()
            stopped = False
            async for chunk in rag_system.rag.stream_generate(
                prep["question"],
                prep.get("context") or [],
                messages=prep.get("norm_messages"),
                user_id=uid,
                low_relevance=(bool(prep.get("rag_enabled")) and not prep.get("context")),
                prompt_messages=prompt_messages,
            ):
                full.append(chunk)
                yield sse("token", {"text": chunk})
                if await stream_request.is_disconnected():
                    terminal = TerminalReason.DISCONNECTED.value
                    stopped = True
                    break
            answer = "".join(full)
            # 回答侧安全复查（与 Web 端一致：safety 总开关开启时生效）
            ans_check = None
            if settings.SAFETY_ENABLED:
                answer, ans_check = rag_system.safety_checker.review_answer(answer)
            timings["llm"] = (time.perf_counter() - t_gen) * 1000
            timings["total"] = (time.perf_counter() - t_total) * 1000

            if stopped:
                return

            session_id = body.sessionId or uuid.uuid4().hex
            if not admission.is_cancelling(request_id):
                await enqueue_persist(
                    session_id,
                    prep.get("question", ""),
                    answer,
                    title=None,
                    user_id=uid,
                    safety_check=prep.get("safety_check"),
                    is_crisis_response=False,
                    safety_note=prep.get("safety_note"),
                    answer_safety_check=ans_check,
                )
            yield sse("done", {
                "answer": answer,
                "safety_note": prep.get("safety_note"),
                "safety_check": prep.get("safety_check"),
                "timings": timings,
                "sessionId": session_id,
                "request_id": request_id,
            })
        except Exception as e:
            terminal = TerminalReason.FAILED.value
            logger.exception("[api/mp][query/stream] 流式生成异常 [rid=%s]: %s",
                             getattr(stream_request.state, "request_id", "-"), e)
            yield sse("error", {
                "detail": f"生成失败（{type(e).__name__}），请稍后重试。",
                "error_type": type(e).__name__,
            })
        finally:
            await admission.release(ticket, terminal=terminal)
            try:
                await admission.cancel(uid, request_id)
            except Exception as _e:  # noqa: BLE001 清理兜底失败仅告警
                logger.warning("[api/mp][query/stream] 排队残留清理失败 [rid=%s]: %s",
                               getattr(stream_request.state, "request_id", "-"), _e)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

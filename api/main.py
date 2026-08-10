"""
FastAPI服务接口
提供RESTful API供前端调用
"""
import sys
from pathlib import Path

# 确保无论从哪个工作目录启动（如 `python api/main.py`），
# 项目根都在 sys.path 上，使 `from config.settings` / `from modules` 稳定可用。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict
import json
from modules import rag_system
from modules.rag_core import build_sources
from modules.prompt_store import (
    get_prompt_config,
    update_prompt_config,
    reset_prompt_config,
    ensure_prompts_seeded,
)
from config.settings import settings
from db import init_db, crud
from db.models import Session as ConvSession, CompareHistory
from sqlalchemy import select, desc, func
import uuid

app = FastAPI(
    title="青少年心理RAG系统API",
    description="基于RAG的6-18岁青少年心理咨询系统",
    version="1.0.0",
)

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
    return await call_next(request)


class QueryRequest(BaseModel):
    """查询请求：支持单轮 question 或多轮 messages。"""
    question: Optional[str] = Field(None, description="用户当前问题（单轮模式，与 messages 二选一）", min_length=1, max_length=2000)
    messages: Optional[List[Dict[str, str]]] = Field(
        None,
        description="多轮对话历史，每条含 role 与 content；role 可为 human/ai/user/assistant。最后一条须为用户问题。",
    )
    system_prompt_override: Optional[str] = Field(
        None,
        description="可选：覆盖使用的系统提示词（不落盘），用于前端在不保存的情况下预览/对比提示词效果。",
    )
    prompt_id: Optional[str] = Field(
        None,
        description="可选：使用提示词库中指定 id 的提示词（与 override 互斥，override 优先）。",
    )
    session_id: Optional[str] = Field(
        None,
        description="会话 id（前端本地生成，如 session-<timestamp>）；不传则由服务端生成并在响应中返回。用于把多轮对话与危机审计持久化到关系库。",
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


class SessionCreate(BaseModel):
    """新建会话"""
    name: Optional[str] = Field(None, description="会话名称，留空则默认“新的对话”")


class SessionRename(BaseModel):
    """重命名会话"""
    name: str = Field(..., description="新名称")


class CompareHistoryItem(BaseModel):
    """新增一条对比历史记录"""
    input: str = Field(..., description="对比用的测试问题")
    a: Dict = Field(..., description="A 侧完整结果（answer/sources/timings 等）")
    b: Dict = Field(..., description="B 侧完整结果")


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    查询接口
    接收用户问题，返回RAG生成的回答，并把本轮对话与（若命中）危机事件
    持久化到关系库（与 Chroma 向量检索互补）。
    """
    try:
        result = await run_in_threadpool(
            rag_system.query,
            question=request.question,
            messages=request.messages,
            check_safety=True,
            system_prompt_override=request.system_prompt_override,
            prompt_id=request.prompt_id,
        )

        # —— 持久化到关系库（会话 + 危机审计） ——
        # 与 Chroma 向量库分工：Chroma 管检索，这里管结构化留痕。
        # 仅对话联调页（persist=true）需要落库；提示词对比页传 persist=false，
        # 其临时问答不写入会话表，避免泄漏到历史对话列表。
        # 持久化失败不影响已经生成的回答，只打告警日志。
        current_question = result.get("question", "")
        answer = result.get("answer", "")
        session_id = None
        if request.persist:
            session_id = request.session_id or uuid.uuid4().hex
            try:
                with crud.get_db() as db:
                    crud.append_turn(
                        db,
                        session_id,
                        current_question,
                        answer,
                        # 自动命名提示：优先用前端首次提问传入的标题，否则回退到当前问题；
                        # 服务端只会在会话标题仍为占位名时采纳（见 crud._auto_title）
                        title=(request.title or current_question or None),
                    )
                    sc = result.get("safety_check")
                    if sc and sc.get("is_crisis"):
                        crud.log_crisis(
                            db,
                            session_id,
                            level=sc.get("level", "unknown"),
                            keywords_found=sc.get("keywords_found"),
                            question=current_question,
                            response=answer if result.get("is_crisis_response") else result.get("safety_note"),
                            is_crisis_response=bool(result.get("is_crisis_response")),
                            detect_method=sc.get("detect_method") if isinstance(sc, dict) else None,
                            confidence=sc.get("confidence") if isinstance(sc, dict) else None,
                        )
                    # 回答侧命中高危：另记一条审计（detect_method=answer_check）
                    ans_sc = result.get("answer_safety_check")
                    if ans_sc and ans_sc.get("is_crisis"):
                        crud.log_crisis(
                            db,
                            session_id,
                            level=ans_sc.get("level", "high"),
                            keywords_found=ans_sc.get("keywords_found"),
                            question=current_question,
                            response=answer,
                            is_crisis_response=False,
                            detect_method="answer_check",
                        )
            except Exception as e:
                print(f"[persist][WARN] 会话持久化失败（回答已正常返回）: {e}", flush=True)

        return QueryResponse(
            answer=answer,
            sources=result.get("sources") or [],
            safety_note=result.get("safety_note"),
            is_crisis_response=bool(result.get("is_crisis_response", False)),
            safety_check=result.get("safety_check"),
            timings=result.get("timings"),
            session_id=session_id,
        )
    except ValueError as e:
        # 参数校验类错误返回 400，便于前端定位
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # 不把内部异常细节（路径/堆栈）回传给客户端
        raise HTTPException(status_code=500, detail="内部处理失败，请稍后重试。")


def _sse(event: str, data: dict) -> str:
    """格式化一个 SSE 事件：`event:` + `data:` 两行 + 空行结尾。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/query/stream")
async def query_stream(stream_request: Request, request: QueryRequest):
    """
    SSE 流式问答（对话联调页使用）。

    事件流：sources（来源 + 检索耗时）→ token×N（逐块文本）→ done（完整回答 +
    safety_note + timings + session_id，已落库）；高危危机直接发 done；异常发 error。
    """
    async def event_stream():
        try:
            # 全流程墙钟计时（用于 done 事件里的 total，与 /api/query 语义一致）
            t_total = time.perf_counter()
            # 1) 安全检测 + 混合检索 + 重排（同步阻塞放线程池，避免卡事件循环）
            prep = await run_in_threadpool(
                rag_system.prepare,
                question=request.question,
                messages=request.messages,
                check_safety=True,
                system_prompt_override=request.system_prompt_override,
                prompt_id=request.prompt_id,
            )
            if prep.get("is_crisis_response"):
                # 高危拦截同样写会话与危机审计（此前缺失：前端对话页走的就是 stream，
                # 高危问答既不进历史也不留审计；现与 /api/query 高危路径对齐）
                if request.persist:
                    session_id = request.session_id or uuid.uuid4().hex
                    try:
                        with crud.get_db() as db:
                            crud.append_turn(
                                db, session_id, prep.get("question", ""), prep.get("answer", ""),
                                title=(request.title or prep.get("question") or None),
                            )
                            sc = prep.get("safety_check") or {}
                            crud.log_crisis(
                                db, session_id,
                                level=sc.get("level", "unknown"),
                                keywords_found=sc.get("keywords_found"),
                                question=prep.get("question", ""),
                                response=prep.get("answer", ""),
                                is_crisis_response=True,
                                detect_method=sc.get("detect_method") if isinstance(sc, dict) else None,
                                confidence=sc.get("confidence") if isinstance(sc, dict) else None,
                            )
                    except Exception as e:
                        print(f"[persist][WARN] 高危危机审计写入失败: {e}", flush=True)
                yield _sse("done", {
                    "answer": prep.get("answer", ""),
                    "is_crisis_response": True,
                    "safety_check": prep.get("safety_check"),
                })
                return

            # 2) 先发来源与检索耗时（生成尚未开始，前端可先展示引用）
            yield _sse("sources", {
                "sources": prep.get("sources") or [],
                "timings": prep.get("timings") or {},
            })

            # 3) 流式生成：逐 token 下发
            timings = prep.get("timings") or {}
            full: list[str] = []
            t_gen = time.perf_counter()
            async for chunk in rag_system.rag.stream_generate(
                prep["question"],
                prep.get("context") or [],
                system_prompt_override=request.system_prompt_override,
                prompt_id=request.prompt_id,
                # 必须传归一化后的消息（role=human/ai），否则前端 user/assistant
                # 角色在 _build_messages 里不匹配被静默丢弃，多轮历史全部丢失
                messages=prep.get("norm_messages"),
            ):
                full.append(chunk)
                yield _sse("token", {"text": chunk})
                # 客户端断开（关页面/刷新）时及时终止生成，避免浪费上游 token 与算力
                if await stream_request.is_disconnected():
                    print("[query/stream] 客户端断开，终止生成", flush=True)
                    break
            answer = "".join(full)
            # 回答侧安全复查：命中高危关键词时追加安全提醒（token 已发，追加部分随 done 的 answer 下发）
            ans_check = None
            answer, ans_check = rag_system.safety_checker.review_answer(answer)
            # 生成耗时（从流式调用开始到结束）与全流程总耗时，供前端耗时栏展示
            timings["llm"] = (time.perf_counter() - t_gen) * 1000
            timings["total"] = (time.perf_counter() - t_total) * 1000

            # 4) 持久化（与 /api/query 相同的落库语义；失败不影响已生成的回答）
            session_id = None
            if request.persist:
                session_id = request.session_id or uuid.uuid4().hex
                try:
                    with crud.get_db() as db:
                        crud.append_turn(
                            db, session_id, prep.get("question", ""), answer,
                            title=(request.title or prep.get("question") or None),
                        )
                        sc = prep.get("safety_check")
                        if sc and sc.get("is_crisis"):
                            crud.log_crisis(
                                db, session_id,
                                level=sc.get("level", "unknown"),
                                keywords_found=sc.get("keywords_found"),
                                question=prep.get("question", ""),
                                response=prep.get("safety_note"),
                                is_crisis_response=False,
                                detect_method=sc.get("detect_method") if isinstance(sc, dict) else None,
                                confidence=sc.get("confidence") if isinstance(sc, dict) else None,
                            )
                        # 回答侧命中高危：另记一条审计（detect_method=answer_check）
                        if ans_check and ans_check.get("is_crisis"):
                            crud.log_crisis(
                                db, session_id,
                                level=ans_check.get("level", "high"),
                                keywords_found=ans_check.get("keywords_found"),
                                question=prep.get("question", ""),
                                response=answer,
                                is_crisis_response=False,
                                detect_method="answer_check",
                            )
                except Exception as e:
                    print(f"[persist][WARN] 会话持久化失败（回答已正常返回）: {e}", flush=True)

            yield _sse("done", {
                "answer": answer,
                "safety_note": prep.get("safety_note"),
                "safety_check": prep.get("safety_check"),
                "timings": timings,
                "session_id": session_id,
            })
        except Exception as e:
            # 记录完整异常到服务端日志；SSE error 事件只带异常类型（不暴露堆栈/路径），
            # 便于前端展示具体原因、下次复现时直接定位
            print(f"[query/stream][ERROR] {type(e).__name__}: {e}", flush=True)
            yield _sse("error", {
                "detail": f"生成失败（{type(e).__name__}），请稍后重试。",
                "error_type": type(e).__name__,
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防止反代缓冲（本地直连无影响）
        },
    )


@app.get("/api/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
    }


@app.get("/api/sessions")
async def list_sessions(limit: int = 50):
    """列出最近会话（含消息数），用于审计/排查。"""
    with crud.get_db() as db:
        rows = (
            db.execute(select(ConvSession).order_by(desc(ConvSession.updated_at)).limit(limit))
            .scalars()
            .all()
        )
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
async def get_session_messages(session_id: str):
    """获取某会话的全部消息（按时间顺序）。"""
    with crud.get_db() as db:
        sess = db.get(ConvSession, session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in sess.messages
        ]


@app.post("/api/sessions")
async def create_session(payload: SessionCreate):
    """新建一个空会话，返回服务端生成的 id（前端以此为后续对话归属）。"""
    with crud.get_db() as db:
        sess = ConvSession(id=uuid.uuid4().hex, title=(payload.name or "新的对话")[:255])
        db.add(sess)
        db.flush()
        return {
            "id": sess.id,
            "name": sess.title,
            "created_at": sess.created_at.isoformat() if sess.created_at else None,
            "updated_at": sess.updated_at.isoformat() if sess.updated_at else None,
            "message_count": 0,
        }


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, payload: SessionRename):
    """重命名会话。"""
    with crud.get_db() as db:
        sess = db.get(ConvSession, session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        sess.title = payload.name[:255]
        return {"id": sess.id, "name": sess.title}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话（级联删除其消息）。"""
    with crud.get_db() as db:
        sess = db.get(ConvSession, session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        db.delete(sess)
    return {"ok": True, "id": session_id}


@app.get("/api/compare-history")
async def list_compare_history(limit: int = 50):
    """列出对比历史记录（含 A/B 完整结果）。"""
    with crud.get_db() as db:
        rows = crud.list_compare_history(db, limit=limit)
        return [
            {
                "id": r.id,
                "input": r.input,
                "a": json.loads(r.result_a) if r.result_a else None,
                "b": json.loads(r.result_b) if r.result_b else None,
                "createdAt": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


@app.post("/api/compare-history")
async def add_compare_history(item: CompareHistoryItem):
    """新增一条对比历史记录。"""
    with crud.get_db() as db:
        r = crud.add_compare_history(
            db,
            item.input,
            json.dumps(item.a, ensure_ascii=False),
            json.dumps(item.b, ensure_ascii=False),
        )
        return {
            "id": r.id,
            "input": r.input,
            "a": item.a,
            "b": item.b,
            "createdAt": r.created_at.isoformat() if r.created_at else None,
        }


@app.delete("/api/compare-history/{item_id}")
async def delete_compare_history(item_id: int):
    """删除一条对比历史记录。"""
    with crud.get_db() as db:
        r = crud.get_compare_history(db, item_id)
        if r is None:
            raise HTTPException(status_code=404, detail="记录不存在")
        db.delete(r)
    return {"ok": True, "id": item_id}


class PromptItem(BaseModel):
    """提示词库条目"""
    id: str
    name: str
    content: str


class PromptAdd(BaseModel):
    """新增提示词"""
    name: str
    content: str


class PromptUpdateSingle(BaseModel):
    """更新单条提示词"""
    id: str
    name: Optional[str] = None
    content: Optional[str] = None


class PromptUpdate(BaseModel):
    """系统提示词库更新请求（局部更新，未提供的字段保持不变）"""
    prompts: Optional[List[PromptItem]] = Field(None, description="完整替换提示词库")
    activeId: Optional[str] = Field(None, description="设置当前激活的提示词 id")
    add: Optional[PromptAdd] = Field(None, description="新增一条提示词")
    update: Optional[PromptUpdateSingle] = Field(None, description="更新某条提示词")
    deleteId: Optional[str] = Field(None, description="删除指定 id 的提示词")


@app.get("/api/system-prompt")
async def get_system_prompt():
    """
    获取当前与默认的系统提示词配置，供前端双栏对比展示。
    返回 { current: {...}, default: {...} }。
    """
    return get_prompt_config()


@app.put("/api/system-prompt")
async def put_system_prompt(payload: PromptUpdate):
    """
    更新系统提示词库（持久化到 SQLite 的 prompts 表）。
    支持增删改、完整替换、设置激活提示词。
    """
    try:
        update_payload = {}
        if payload.prompts is not None:
            update_payload["prompts"] = [p.model_dump() for p in payload.prompts]
        if payload.activeId is not None:
            update_payload["activeId"] = payload.activeId
        if payload.add is not None:
            update_payload["add"] = payload.add.model_dump()
        if payload.update is not None:
            update_payload["update"] = payload.update.model_dump()
        if payload.deleteId is not None:
            update_payload["deleteId"] = payload.deleteId

        if not update_payload:
            raise ValueError("未提供任何更新字段")

        saved = update_prompt_config(update_payload)
        return {"ok": True, "config": saved}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/system-prompt/reset")
async def reset_system_prompt():
    """
    还原系统提示词为出厂默认（清空 prompts 表并重新从出厂文件 seed）。
    """
    saved = reset_prompt_config()
    return {"ok": True, "config": saved}


@app.on_event("startup")
async def startup_event():
    """启动时验证配置、建表、seed 提示词库"""
    settings.validate()
    init_db()
    ensure_prompts_seeded()
    # 历史遗留的占位标题会话（“新的对话”等）按最早一条用户消息自动命名（幂等）
    try:
        with crud.get_db() as db:
            renamed = crud.rename_unnamed_sessions(db)
        if renamed:
            print(f"[startup] 已为 {renamed} 个未命名会话自动生成标题")
    except Exception as e:
        print(f"[startup][WARN] 自动命名未执行（不影响启动）: {e}")
    # 预热本地重排模型：后台线程加载（约 5-10 秒），不阻塞启动；
    # 预热失败静默（问答时自动回退到原排序），保证首次问答不卡顿
    if settings.RERANK_ENABLED:
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
    # 未就绪时问答回退关键词，避免首次请求被同步构建卡住）
    if settings.SEMANTIC_CHECK_ENABLED:
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
    print(f"向量数据库: Chroma")
    print(f"关系数据库: {settings.DB_URL}")
    print(f"安全检查: {'启用' if settings.SAFETY_CHECK_ENABLED else '禁用'}")
    print(f"本地重排: {'启用（' + settings.RERANK_MODEL + '）' if settings.RERANK_ENABLED else '禁用'}")
    print("=" * 50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


# 前端静态资源托管：在 API 路由之后挂载，/api、/docs 等显式路由优先匹配。
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIR), html=True),
        name="frontend",
    )

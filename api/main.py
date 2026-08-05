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
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict
from modules import rag_system
from modules.prompt_store import (
    get_prompt_config,
    update_prompt_config,
    reset_prompt_config,
)
from config.settings import settings

app = FastAPI(
    title="青少年心理RAG系统API",
    description="基于RAG的6-18岁青少年心理咨询系统",
    version="1.0.0",
)

# 允许本地前端跨域访问（前端既可由本服务托管，也可直接以 file:// 打开）。
# 服务本身仅监听 127.0.0.1，不存在暴露到局域网的风险。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 简单内存限流：仅针对 POST /api/query，防止单客户端刷接口
_rate_limit_store: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/api/query" and request.method == "POST":
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
    age_group: Optional[str] = Field(
        None,
        description="年龄段分桶：child / early_teen / teen / late_teen。留空表示不限年龄，语气默认按 teen。",
        pattern="^(child|early_teen|teen|late_teen)$",
    )
    system_prompt_override: Optional[str] = Field(
        None,
        description="可选：覆盖使用的系统提示词（不落盘），用于前端在不保存的情况下预览/对比提示词效果。",
    )
    prompt_id: Optional[str] = Field(
        None,
        description="可选：使用提示词库中指定 id 的提示词（与 override 互斥，override 优先）。",
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


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    查询接口
    接收用户问题，返回RAG生成的回答
    """
    try:
        result = await run_in_threadpool(
            rag_system.query,
            question=request.question,
            messages=request.messages,
            age_group=request.age_group,
            check_safety=True,
            system_prompt_override=request.system_prompt_override,
            prompt_id=request.prompt_id,
        )

        return QueryResponse(**result)
    except ValueError as e:
        # 参数校验类错误返回 400，便于前端定位
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # 不把内部异常细节（路径/堆栈）回传给客户端
        raise HTTPException(status_code=500, detail="内部处理失败，请稍后重试。")

@app.get("/api/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
    }


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
    更新系统提示词库并同步写入后端文件 config/system_prompt.json。
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
    还原系统提示词为出厂默认（覆盖 config/system_prompt.json）。
    """
    saved = reset_prompt_config()
    return {"ok": True, "config": saved}


@app.on_event("startup")
async def startup_event():
    """启动时验证配置"""
    settings.validate()
    print("=" * 50)
    print("青少年心理RAG系统已启动")
    print(f"模型: {settings.CHAT_MODEL}")
    print(f"向量数据库: Chroma")
    print(f"安全检查: {'启用' if settings.SAFETY_CHECK_ENABLED else '禁用'}")
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

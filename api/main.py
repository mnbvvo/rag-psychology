"""
FastAPI服务接口
提供RESTful API供前端调用
"""
import sys
from pathlib import Path

# 确保无论从哪个工作目录启动（如 `python api/main.py`），
# 项目根都在 sys.path 上，使 `from config.settings` / `from modules` 稳定可用。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from modules import rag_system
from config.settings import settings
from typing import Optional
app = FastAPI(
    title="青少年心理RAG系统API",
    description="基于RAG的6-18岁青少年心理咨询系统",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    age_group: Optional[str] = Field(
        None,
        description="年龄段分桶：child / early_teen / teen / late_teen。留空表示不限年龄，语气默认按 teen。",
        pattern="^(child|early_teen|teen|late_teen)$",
    )


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    sources: List[dict] = []
    safety_note: Optional[str] = None
    is_crisis_response: bool = False
    safety_check: Optional[dict] = None


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    查询接口
    接收用户问题，返回RAG生成的回答
    """
    try:
        result = rag_system.query(
            question=request.question,
            age_group=request.age_group,
            check_safety=True,
        )

        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
    }


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

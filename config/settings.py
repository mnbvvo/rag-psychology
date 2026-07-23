"""
系统配置管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings:
    """系统配置类"""

    # DashScope（通义千问兼容模式）配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # 阿里云兼容接口地址
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 模型配置
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5-35b-a3b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

    # RAG 配置
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "psychology_knowledge")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

    # 检索策略
    # 召回候选数量（先多召回，再重排/截断）
    FETCH_K = int(os.getenv("FETCH_K", "10"))
    # similarity: 按相关性分数重排；mmr: 最大边际相关（兼顾多样性）
    SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")
    MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))
    # 相关性分数下限（0 表示不启用）。text-embedding-v3 余弦相似度，建议 0.2~0.35
    MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))
    # 年龄段分桶（与导入脚本写入的 age_group metadata 对应）
    AGE_GROUPS = ["child", "early_teen", "teen", "late_teen"]

    # 生成参数
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.3"))

    # 安全配置
    CRISIS_KEYWORDS_FILE = os.getenv(
        "CRISIS_KEYWORDS_FILE", "./config/crisis_keywords.json"
    )
    SAFETY_CHECK_ENABLED = os.getenv("SAFETY_CHECK_ENABLED", "true").lower() == "true"

    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent

    @classmethod
    def validate(cls):
        """验证必要配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未设置，请在 .env 文件中配置")
        return True


settings = Settings()
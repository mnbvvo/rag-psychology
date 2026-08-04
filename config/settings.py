"""
系统配置管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 显式加载项目根目录的 .env（与 settings.py 同级的上级目录），
# 避免依赖启动时的 cwd，否则从非项目根目录启动时拿不到 OPENAI_API_KEY。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 路径锚定基目录（settings.py 位于 config/ 下）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str, base: Path) -> str:
    """将相对路径锚定到 base 目录，绝对路径原样返回。

    这样无论从哪个工作目录启动应用，配置文件与数据目录都能正确定位，
    避免 FileNotFoundError: ./config/... 这类因 cwd 不同导致的问题。
    """
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve())


class Settings:
    """系统配置类"""

    # DashScope（通义千问兼容模式）配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 模型配置
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5-35b-a3b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

    # RAG 配置（路径锚定到项目根目录，避免 cwd 不同导致找不到文件/库）
    CHROMA_PERSIST_DIR = _resolve_path(
        os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
        _PROJECT_ROOT,
    )
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "psychology_knowledge")
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

    # 检索策略
    FETCH_K = int(os.getenv("FETCH_K", "10"))
    SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")
    MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))
    MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))
    AGE_GROUPS = ["child", "early_teen", "teen", "late_teen"]

    # 生成参数
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.3"))

    # 安全配置（路径锚定到项目根目录，避免 cwd 不同导致找不到文件）
    CRISIS_KEYWORDS_FILE = _resolve_path(
        os.getenv("CRISIS_KEYWORDS_FILE", "config/crisis_keywords.json"),
        _PROJECT_ROOT,
    )
    SAFETY_CHECK_ENABLED = os.getenv("SAFETY_CHECK_ENABLED", "true").lower() == "true"

    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # 项目根目录
    PROJECT_ROOT = _PROJECT_ROOT

    @classmethod
    def validate(cls):
        """验证必要配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未设置，请在 .env 文件中配置")
        return True


settings = Settings()

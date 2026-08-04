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

    # 接口与密钥（密钥为必填，写在 .env；base 地址随部署环境覆盖）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 必填：OpenAI 兼容接口密钥
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")  # 兼容模式基地址

    # 模型配置（更换模型/部署环境时在 .env 覆盖；不填则用下列默认值）
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5-35b-a3b")  # 对话生成模型
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")  # 向量化模型（与 chroma_db 绑定，换模型须 --reset 重导）

    # RAG 配置（路径锚定到项目根目录，避免 cwd 不同导致找不到文件/库）
    CHROMA_PERSIST_DIR = _resolve_path(
        os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
        _PROJECT_ROOT,
    )  # 本地向量库持久化目录
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "psychology_knowledge")  # Chroma 集合名
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))  # 最终喂给模型的最相关文档数
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))  # 相似度召回后重排截断保留的条数

    # 检索策略
    FETCH_K = int(os.getenv("FETCH_K", "10"))  # 相似度检索初召回候选数（应 >= RETRIEVAL_TOP_K）
    SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")  # similarity 或 mmr（最大边际相关，兼顾多样性）
    MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # mmr 模式下多样性权重：0=最多样，1=最相关
    MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))  # 相关性下限，0=不启用（建议 0.2~0.35）
    AGE_GROUPS = ["child", "early_teen", "teen", "late_teen"]  # 年龄分桶（检索过滤 + 回答语气适配）

    # 生成参数
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.3"))  # 生成温度，事实/建议类问答偏低以减少幻觉

    # 安全配置（路径锚定到项目根目录，避免 cwd 不同导致找不到文件）
    CRISIS_KEYWORDS_FILE = _resolve_path(
        os.getenv("CRISIS_KEYWORDS_FILE", "config/crisis_keywords.json"),
        _PROJECT_ROOT,
    )  # 危机关键词 + 等级 + 热线定义文件
    SAFETY_CHECK_ENABLED = os.getenv("SAFETY_CHECK_ENABLED", "true").lower() == "true"  # 是否启用关键词级危机检测

    # 限流（仅 POST /api/query，内存级；多进程部署需改用共享存储）
    RATE_LIMIT_TIMES = int(os.getenv("RATE_LIMIT_TIMES", "20"))  # 时间窗内单客户端最大请求数
    RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "60"))  # 限流时间窗长度（秒）

    # 服务配置（心理应用含危机内容，默认只绑本机，避免暴露到局域网）
    HOST = os.getenv("HOST", "127.0.0.1")  # 监听地址；切勿改为 0.0.0.0 以免暴露到局域网
    PORT = int(os.getenv("PORT", "8000"))  # 监听端口
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"  # 调试模式（开启时 uvicorn --reload 且单进程）

    # 项目根目录
    PROJECT_ROOT = _PROJECT_ROOT

    @classmethod
    def validate(cls):
        """验证必要配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未设置，请在 .env 文件中配置")
        return True


settings = Settings()

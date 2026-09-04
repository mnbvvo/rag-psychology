"""
系统配置管理

配置分层原则：
- .env 只放「部署环境相关」的三项：OPENAI_API_KEY / OPENAI_API_BASE / CHAT_MODEL。
- 其余所有调参（检索 / 重排 / 安全 / 限流 / 服务等）都是这里的静态常量，
  改配置直接改本文件（含中文注释），不依赖环境变量。

路径类配置统一经 _resolve_path 锚定到项目根目录，避免 cwd 不同导致找不到文件。
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

    # ============ 接口与模型（唯一从 .env 读取的部分） ============
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 必填：OpenAI 兼容接口密钥（LLM 对话用）
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")  # 兼容模式基地址（LLM）
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.6-flash")  # 对话生成模型

    # ============ 向量化模型（可独立于 LLM 配置 API 端点/密钥） ============
    # 默认回退到 LLM 同一套（EMBEDDING_API_* 未设置时沿用 OPENAI_*），
    # 需要 embedding 与回答走不同服务商时，在 .env 里单独覆盖即可。
    # 注意：embedding 模型与向量库绑定，换模型后旧向量静默失效，必须重导。
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")  # 独立密钥；未设置回退 OPENAI_API_KEY
    EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE") or os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")  # 独立基地址；未设置回退 OPENAI_API_BASE
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    # 思考/推理模式（仅部分 OpenAI 兼容模型支持，如 Qwen3 / DeepSeek；端点不支持时请把下面的值改为 False，否则可能报 400）
    ENABLE_THINKING = False  # 是否在请求体注入 enable_thinking 控制思考模式

    # ============ RAG 检索增强生成开关 ============
    # True：走完整 RAG（安全检测 + 向量/混合检索 + 重排 + 生成）；
    # False：跳过检索，context 为空，直接让 LLM 基于提示词回答（纯对话模式）。
    # 可通过请求参数 rag_enabled 按次覆盖（None 时用此全局值）。
    RAG_ENABLED = False

    # ============ RAG 检索（路径锚定到项目根目录，避免 cwd 不同导致找不到文件/库） ============
    CHROMA_PERSIST_DIR = _resolve_path("data/chroma", _PROJECT_ROOT)  # 本地向量库持久化目录（统一收在 data/ 下）
    COLLECTION_NAME = "psychology_knowledge"  # Chroma 集合名
    RERANK_TOP_K = 3  # 最终喂给模型/展示的来源条数（similarity 与 mmr 两模式统一使用此值）
    FETCH_K = 10  # 相似度检索初召回候选数（应 >= RERANK_TOP_K，即最终条数）
    SEARCH_TYPE = "similarity"  # similarity 或 mmr（最大边际相关，兼顾多样性）
    MMR_LAMBDA = 0.5  # mmr 模式下多样性权重：0=最多样，1=最相关
    MIN_RELEVANCE_SCORE = 0.2  # 相关性下限，0=不启用（建议 0.2~0.35；仅未开重排时生效，重排开启时跳过该硬阈值）

    # ============ 混合检索（向量召回 ∪ BM25 关键词召回 → 重排精排） ============
    # 心理领域高频症状词（失眠/厌学/霸凌等）词面命中比语义匹配更可靠；
    # 与重排配合：多路召回取并集去重，由重排器统一精排，无需分数融合。
    HYBRID_ENABLED = True  # 是否启用混合检索（仅重排开启时生效）
    HYBRID_KEYWORD_K = 5  # 关键词（BM25）召回条数，并入向量候选后交给重排器

    # ============ 本地重排序（Cross-Encoder，bge-reranker-v2-m3） ============
    # 召回候选后按「问题 × 文档」逐对打分精排，替代仅按向量分数截断的假重排。
    # 模型失败/异常时自动回退到原排序逻辑，不影响检索可用性。
    RERANK_ENABLED = True  # 是否启用本地重排
    RERANK_MODEL = _resolve_path("data/rerank_models/bge-reranker-v2-m3", _PROJECT_ROOT)  # 本地模型目录（含 config.json/model.safetensors），也支持 HF 模型名
    RERANK_DEVICE = ""  # 留空=自动（优先 cuda），可显式 cuda/cpu
    RERANK_BATCH_SIZE = 8  # 每批重排文档数
    RERANK_MAX_LENGTH = 512  # 单条文本截断长度（token）
    RERANK_MIN_SCORE = 0  # 重排分数下限（bge 分数 0~1；本项目实测相关文档约 0.4+、无关 <0.15，建议从 0.2~0.3 起试，勿设太高否则全部过滤）；0=不启用。低于阈值的候选被丢弃，全被丢弃时回答会提示"没有足够信息"防编造

    # ============ 生成参数 ============
    CHAT_TEMPERATURE = 0.3  # 生成温度，事实/建议类问答偏低以减少幻觉
    # LLM 客户端超时/重试（修复 P1：旧值 30s 与排队超时 30s 同界——慢上游首 token 超过
    # 30s 会被当成故障直接 500、且与排队者 30s 超时形成竞争，Q-03/Q-04 无法按 30±1s 验收）。
    # 约束：必须严格大于 AI_QUEUE_WAIT_TIMEOUT_SECONDS（validate() 强制），
    # 使「慢 LLM 仍在正常等待」与「排队者 30s 超时」两个语义解耦。
    LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    # embedding 客户端超时（安全 L1/记忆/检索共用；默认 30s 保持原行为，可 env 覆盖）
    EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))

    # ============ 长期记忆（向量检索式） ============
    # 方案：每轮问答落库 user_chat_history 并打双向量（query 向量 + qa 向量，
    # qa = query+answer 拼接，检索主用）；提问时用当前问题向量检索该用户
    # 相似历史（fn_search_chat_history，余弦相似度 top_k）注入 system prompt。
    # 替代旧的「全量拼接历史」——检索成本恒定，不随历史总量增长，token 开销也固定。
    MEMORY_ENABLED = True  # 是否启用长期记忆（检索相似历史注入上下文）
    MEMORY_TOP_K = 5  # 每次注入的相似历史条数（与 fn_search_chat_history 的 LIMIT 一致）
    MEMORY_MIN_SIMILARITY = 0.3  # 相似度下限（0~1），低于阈值的历史不注入；0=不启用。建议 0.25~0.35 防无关历史
    MEMORY_EMBEDDING_MODEL = ""  # 记忆用 embedding 模型；留空复用 EMBEDDING_MODEL
    # 双通道：当前会话最近 N 轮原始对话（human/ai 交替）直接拼入消息列表，
    # 与跨会话向量记忆互补——解决指代消解（"那个方法""刚才说的"向量检索不到，
    # 只能靠最近几轮原文）。0=只走向量记忆通道。
    MEMORY_RECENT_ROUNDS = 6
    # 短期窗口字符预算（双约束：轮数上限 MEMORY_RECENT_ROUNDS 之外，窗口内历史
    # 原文合计字符数不超过此值，从最新轮次向前累计，超预算即截断，至少保底 1 轮）。
    # 中文约 1 字 ≈ 1 token 上下，6000 字符 ≈ 6k~9k token；请按所用模型 context
    # 与 system/RAG/输出占比调优，别让窗口占满上下文。0=仅按轮数不设预算。
    MEMORY_RECENT_MAX_CHARS = int(os.getenv("MEMORY_RECENT_MAX_CHARS", "6000"))

    # 遗留参数（旧「全量拼接历史」模式使用，已由向量检索式长期记忆取代，保留以兼容引用）
    MAX_HISTORY_TURNS = 5

    # ============ 安全配置（路径锚定到项目根目录，避免 cwd 不同导致找不到文件） ============
    # 安全总开关：False 时整条安全链路（L0 关键词 + L1 语义 + 回答侧复查）全部跳过，
    # 不调 embedding、不调安全检测。可通过请求参数 safety_enabled 按次覆盖（None 用此全局值）。
    # ⚠️ 仅用于联调/对比实验；生产环境应保持 True。
    SAFETY_ENABLED = False
    CRISIS_KEYWORDS_FILE = _resolve_path("config/crisis_keywords.json", _PROJECT_ROOT)  # 危机关键词 + 等级 + 热线定义文件
    SAFETY_CHECK_ENABLED = True  # 是否启用关键词级危机检测（SAFETY_ENABLED 之下的 L0 细分开关）

    # 语义危机检测（L1：高危意图原型距离）
    # 隐喻表达无限但危险意图有限：种子集（config/high_risk_intents.json）embed 后按意图簇
    # 聚合成原型向量，用户问题与原型算余弦距离 —— 复用检索阶段的 embedding（带缓存），
    # 不引入额外 API 成本。距离 ≤ 拦截半径 → 高危；≤ 灰区半径 → 疑似（附关怀，不拦截）。
    SEMANTIC_CHECK_ENABLED = True
    CRISIS_SEED_FILE = _resolve_path("config/high_risk_intents.json", _PROJECT_ROOT)  # 高危意图标注种子集（标准句 + 隐喻变体）
    CRISIS_PROTOTYPE_CACHE = _resolve_path("data/crisis_prototypes.json", _PROJECT_ROOT)  # 原型向量缓存文件（种子文件未变更时直接复用，避免启动重建）
    CRISIS_INTERCEPT_DIST = 0.25  # 到最近锚点距离 ≤ 此值 → 高危拦截（锚点集合方案实测：高危隐喻 0.20~0.24、负例最近 0.285，留缓冲防边界波动）
    CRISIS_GRAY_DIST = 0.36  # 距离介于拦截值与灰区值之间 → 疑似（附关怀 + 转介，不拦截）；0.36 可放行"孩子发脾气/说谎"类远邻误报

    # embedding 进程内缓存（同一问题的向量在检测器与检索间复用，减少 API 调用）
    EMBED_CACHE_SIZE = 2048

    # ============ 认证与授权（JWT Bearer + bcrypt，RBAC 双角色 user/admin） ============
    # 安全基线：JWT_SECRET 必须由 .env 提供强随机密钥（≥32 字节），
    # 无配置 / 仍为旧公开默认值时 validate() 拒绝启动（fail-closed），
    # 防止退回公开密钥导致任意身份可伪造（含 admin）。
    # 生成方式：python -c "import secrets;print(secrets.token_urlsafe(48))" 或 openssl rand -hex 32
    JWT_SECRET = os.getenv("JWT_SECRET", "")  # JWT 签名密钥（必填，.env 覆盖）
    JWT_ALGORITHM = "HS256"  # 签名算法
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))  # access token 有效期（分钟）

    # 初始管理员（首次启动 users 表为空时自动创建，仅供本地原型/测试；生产应改为人工置数）
    INIT_ADMIN_USERNAME = os.getenv("INIT_ADMIN_USERNAME", "admin")
    INIT_ADMIN_PASSWORD = os.getenv("INIT_ADMIN_PASSWORD", "admin123456")

    # 登录失败限流（内存级；多进程部署需改用共享存储）
    LOGIN_MAX_FAILS = int(os.getenv("LOGIN_MAX_FAILS", "5"))  # 时间窗内最大失败次数（账号级锁定）
    LOGIN_LOCK_SECONDS = int(os.getenv("LOGIN_LOCK_SECONDS", "900"))  # 失败锁定时间窗（秒）= 15 分钟
    LOGIN_IP_MAX_REQUESTS = int(os.getenv("LOGIN_IP_MAX_REQUESTS", "200"))  # 单 IP 60 秒内最大登录请求数（防止单 IP 爆破；并发压测需注册多账号时可调高）
    REGISTER_IP_MAX_REQUESTS = int(os.getenv("REGISTER_IP_MAX_REQUESTS", "100"))  # 单 IP 60 秒内最大注册请求数（默认 100，压测建号可注入调高）

    # ============ 限流（仅 POST /api/query，内存级；多进程部署需改用共享存储） ============
    RATE_LIMIT_TIMES = int(os.getenv("RATE_LIMIT_TIMES", "2000"))  # 时间窗内单客户端最大请求数
    RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "60"))  # 限流时间窗长度（秒）

    # ============ AI 问答并发准入（AdmissionController，总稿 Phase 1 memory） ============
    # 昂贵链路（安全检测/RAG/LLM/持久化）前的统一准入：有界活跃槽位 + 有界 FIFO 队列 +
    # 单用户单 in-flight。同步与 SSE 共用同一组容量。说明见 docs/并发能力方案总稿.md §4。
    AI_ADMISSION_BACKEND = os.getenv("AI_ADMISSION_BACKEND", "memory")  # memory（Phase 1）| redis（Phase 2，未实现）
    AI_MAX_ACTIVE_REQUESTS = int(os.getenv("AI_MAX_ACTIVE_REQUESTS", "20"))  # 同时活跃上限
    AI_MAX_QUEUED_REQUESTS = int(os.getenv("AI_MAX_QUEUED_REQUESTS", "40"))  # 等待队列上限（不含活跃）
    AI_QUEUE_WAIT_TIMEOUT_SECONDS = float(os.getenv("AI_QUEUE_WAIT_TIMEOUT_SECONDS", "30"))  # 最大排队时长
    AI_ACTIVE_LEASE_SECONDS = float(os.getenv("AI_ACTIVE_LEASE_SECONDS", "45"))  # Phase 2 活跃租约（memory 模式预留）
    AI_ONE_INFLIGHT_PER_USER = os.getenv("AI_ONE_INFLIGHT_PER_USER", "true").lower() not in ("0", "false", "no")

    # ============ 后台持久化 Worker（进程内队列；对应架构图 Queue→Worker 虚线路径） ============
    # Phase 1 用 asyncio.Queue + 线程池 worker 落库（Redis 属 Phase 2，本期不引入）；
    # Worker 每次用独立 DB Session，符合「Session 不跨任务共享」。
    AI_BG_WORKERS = int(os.getenv("AI_BG_WORKERS", "1"))        # 后台落库 worker 数
    AI_BG_QUEUE_SIZE = int(os.getenv("AI_BG_QUEUE_SIZE", "512"))  # 队列上限（队满时回退请求内同步落库）

    # ============ 服务配置（心理应用含危机内容，默认只绑本机，避免暴露到局域网） ============
    HOST = "127.0.0.1"  # 监听地址；切勿改为 0.0.0.0 以免暴露到局域网
    PORT = int(os.getenv("PORT", "8000"))  # 监听端口（env 可覆盖，便于多服务并存测试）
    DEBUG = False  # 调试模式（开启时 uvicorn --reload 且单进程）

    # 跨域白名单（前端若独立部署 / 用 Vite 等开发服务器时需在此放行；
    # 默认由本服务同源托管前端，无需跨域；留空则仅允许本服务自身 origin，切勿用 "*"）
    CORS_ORIGINS = [f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"]

    # ============ 关系型数据库（结构化持久化：会话 / 消息 / 危机审计） ============
    # 双后端：默认 SQLite（单文件、零部署，本地原型）；配置 PG_* 后自动切换 PostgreSQL。
    # 生产/多 worker 推荐 PostgreSQL：DB_BACKEND=postgres 时使用 postgresql+psycopg 驱动。
    DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")  # sqlite | postgres
    PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_USER = os.getenv("PG_USER", "postgres")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "")
    PG_DB = os.getenv("PG_DB", "rag_psychology")
    # 关系库连接池（总稿 §3.4/§6）：数值起步后按压测调；SQLite 忽略以下参数
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))          # async 主池常驻连接（请求路径）
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))    # async 主池高峰临时连接
    DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))    # 无连接可借时的等待秒数
    # sync Worker 小池（bg Worker 后台落库 / 记忆检索 / sync def 端点，在子线程跑 AsyncSession 不可用）
    DB_WORKER_POOL_SIZE = int(os.getenv("DB_WORKER_POOL_SIZE", "2"))
    DB_WORKER_MAX_OVERFLOW = int(os.getenv("DB_WORKER_MAX_OVERFLOW", "4"))

    _DB_PATH = _resolve_path("data/rag_psychology.sqlite3", _PROJECT_ROOT)
    if DB_BACKEND == "postgres":
        DB_URL = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    else:
        DB_URL = f"sqlite:///{_DB_PATH.replace('\\', '/')}"

    # ============ 向量库（语义检索） ============
    # 双后端：pgvector（生产，向量存 PostgreSQL）| chroma（本地原型，data/chroma/）。
    # PGVECTOR_URL 独立连接串；留空则复用 DB_URL（DB_BACKEND=postgres 时）。
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pgvector")  # pgvector | chroma
    PGVECTOR_URL = os.getenv("PGVECTOR_URL", "")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))  # 向量维度（text-embedding-v3 = 1024）

    # 项目根目录
    PROJECT_ROOT = _PROJECT_ROOT

    @classmethod
    def validate(cls):
        """验证必要配置（fail-closed：缺失/弱配置直接拒绝启动）"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未设置，请在 .env 文件中配置")
        if cls.VECTOR_BACKEND == "pgvector" and not cls.PGVECTOR_URL and cls.DB_BACKEND != "postgres":
            raise ValueError("VECTOR_BACKEND=pgvector 需要 PostgreSQL：请配置 DB_BACKEND=postgres 或 PGVECTOR_URL")
        if not cls.JWT_SECRET or len(cls.JWT_SECRET) < 32:
            raise ValueError(
                "JWT_SECRET 未配置或强度不足（<32 字节）：拒绝启动以防止公开密钥伪造身份。"
                "请在 .env 设置强随机密钥：python -c \"import secrets;print(secrets.token_urlsafe(48))\""
            )
        # LLM 超时与排队超时解耦（P1 修复）：LLM 客户端超时必须严格大于排队超时，
        # 否则慢上游（首 token 30~60s）会被 LLM 超时误杀成 500，排队者也无法按
        # 30±1s 稳定进入 QUEUE_TIMEOUT（Q-03/Q-04 验收前提）。
        if cls.LLM_TIMEOUT_SECONDS <= cls.AI_QUEUE_WAIT_TIMEOUT_SECONDS:
            raise ValueError(
                f"LLM_TIMEOUT_SECONDS（{cls.LLM_TIMEOUT_SECONDS:.0f}s）必须大于 "
                f"AI_QUEUE_WAIT_TIMEOUT_SECONDS（{cls.AI_QUEUE_WAIT_TIMEOUT_SECONDS:.0f}s），"
                "否则慢上游会被 LLM 超时误判为故障，排队超时语义无法独立成立。"
            )
        # 准入后端守卫（fail-closed）：
        # memory 后端（Phase 1）只允许单 worker——多 worker 各自计数会放大实际并发；
        # redis 后端（Phase 2）尚未实现，禁止静默错配成每进程独立调度。
        if cls.AI_ADMISSION_BACKEND == "memory":
            wc = (os.getenv("WEB_CONCURRENCY") or "1").strip()
            if wc not in ("", "1"):
                raise ValueError(
                    f"AI_ADMISSION_BACKEND=memory（Phase 1）只允许单 worker，"
                    f"检测到 WEB_CONCURRENCY={wc}。多 worker 需 Phase 2 的 redis 后端。"
                )
        elif cls.AI_ADMISSION_BACKEND == "redis":
            raise ValueError(
                "AI_ADMISSION_BACKEND=redis 属于 Phase 2（多实例共享调度），尚未实现。"
                "Phase 1 请使用 AI_ADMISSION_BACKEND=memory。"
            )
        else:
            raise ValueError(f"未知 AI_ADMISSION_BACKEND: {cls.AI_ADMISSION_BACKEND}（可选 memory/redis）")
        return True


settings = Settings()

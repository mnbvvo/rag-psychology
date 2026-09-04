"""
向量存储和检索模块（双后端：pgvector / chroma）
基于 PostgreSQL + pgvector（生产）或 Chroma（本地原型）实现心理学知识的存储和检索。
"""
import time
import threading
from collections import OrderedDict
from typing import List, Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.settings import settings

# 线程隔离的 embedding 真实耗时累计区，供拆分 embed / retrieve / safety 耗时。
# 只累计「真实调用 embedding API」的耗时（缓存命中≈0ms 不计），语义是：
# 同一问题在安全阶段首次向量化是真实成本，检索阶段命中 LRU 缓存就是 0ms。
_embed_tls = threading.local()

# embedding 进程内 LRU 缓存：同一文本（如用户问题）在语义检测器与向量检索之间
# 只真实调用一次 embedding API，第二次起直接命中缓存（省 300ms+ 与费用）。
_embed_cache: "OrderedDict[tuple, list]" = OrderedDict()
_embed_cache_lock = threading.Lock()


def _record_embed_ms(ms: float) -> None:
    """把一次真实 embedding API 调用的耗时累加到当前线程（缓存命中不调用此函数）。"""
    _embed_tls.real_ms = getattr(_embed_tls, "real_ms", 0.0) + ms


def reset_embed_timer() -> None:
    """清零当前线程的 embedding 真实耗时累计（每次请求开始时调用一次）。"""
    _embed_tls.real_ms = 0.0


def get_embed_ms() -> float:
    """读取当前线程自 reset 以来真实 embedding 调用的总耗时（毫秒）。"""
    return getattr(_embed_tls, "real_ms", 0.0) or 0.0


def _cached_embed(key: tuple, do_embed, record_ms):
    """按 (model, text) 缓存单条文本的向量；缓存满时淘汰最久未用项。

    record_ms：缓存未命中（真实 API 调用）时回调，把耗时计入线程累计；
    命中缓存时不计时 —— 这是"嵌入 0ms"与"嵌入几百 ms"语义正确分界的关键。
    """
    with _embed_cache_lock:
        if key in _embed_cache:
            _embed_cache.move_to_end(key)
            return list(_embed_cache[key])
    t = time.perf_counter()
    vec = do_embed()
    if vec is not None:
        record_ms((time.perf_counter() - t) * 1000)
        with _embed_cache_lock:
            _embed_cache[key] = list(vec)
            if len(_embed_cache) > settings.EMBED_CACHE_SIZE:
                _embed_cache.popitem(last=False)
    return vec


class TimedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings 的子类，仅用于给 embed_query/embed_documents 包一层耗时统计。

    只累计真实调用 embedding API 的耗时（缓存命中不计入），按线程隔离存放在
    _embed_tls.real_ms；配合 reset_embed_timer / get_embed_ms 把耗时正确
    拆分到 safety / embed / retrieve 各栏，并发请求互不串扰。
    """

    def embed_query(self, text: str, *args, **kwargs):
        base_embed = super().embed_query  # 零参数 super() 只能在方法体内直接使用
        key = (self.model, text)
        return _cached_embed(key, lambda: base_embed(text, *args, **kwargs), _record_embed_ms)

    def embed_documents(self, texts, *args, **kwargs):
        t = time.perf_counter()
        try:
            return super().embed_documents(texts, *args, **kwargs)
        finally:
            _record_embed_ms((time.perf_counter() - t) * 1000)


class PsychologyVectorStore:
    """心理学向量存储（pgvector / chroma 双后端，接口一致）"""

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = None,
    ):
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.backend = settings.VECTOR_BACKEND

        # 初始化OpenAI Embeddings（子类化以实现 embedding 耗时统计）
        # embedding 走独立配置（EMBEDDING_API_*），未单独配置时回退 LLM 同一套
        self.embeddings = TimedOpenAIEmbeddings(
            model=embedding_model or settings.EMBEDDING_MODEL,
            openai_api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_API_BASE,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            chunk_size=10,
            timeout=settings.EMBEDDING_TIMEOUT_SECONDS,  # 上游挂起时及时失败，避免导入/检索永久阻塞
            max_retries=2,    # 瞬时网络/限流错误自动重试
        )

        if self.backend == "pgvector":
            self._init_pgvector()
        else:
            self._init_chroma()

    # ---------------- 后端初始化 ----------------
    def _init_chroma(self):
        """Chroma 后端（本地原型 / 迁移前兜底）。"""
        from langchain_chroma import Chroma

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def _pg_connection(self) -> str:
        url = settings.PGVECTOR_URL or settings.DB_URL
        if not url.startswith("postgresql"):
            raise RuntimeError(
                f"VECTOR_BACKEND=pgvector 需要 PostgreSQL 连接串，当前为 {url}。"
                "请配置 DB_BACKEND=postgres 或 PGVECTOR_URL"
            )
        return url

    def _init_pgvector(self):
        """PostgreSQL + pgvector 后端（生产）。"""
        try:
            from langchain_postgres import PGVector

            self.vectorstore = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self._pg_connection(),
                embedding_length=settings.VECTOR_DIMENSION,
                use_jsonb=True,
                create_extension=True,  # 自动 CREATE EXTENSION IF NOT EXISTS vector
            )
        except Exception as e:
            raise RuntimeError(
                f"PGVector 初始化失败（请确认 pgvector 扩展已安装、数据库可连接）: {e}"
            ) from e

    # ---------------- 通用接口 ----------------
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加或更新文档到向量存储。"""
        if not documents:
            return []

        stored_ids = self.vectorstore.add_documents(documents, ids=ids)
        print(f"成功写入 {len(stored_ids)} 个文档到向量存储（{self.backend}）")
        return stored_ids

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[tuple]:
        """带相关性分数的语义搜索（分数越高越相关）"""
        k = k or settings.RERANK_TOP_K
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """最大边际相关性搜索（平衡相关性和多样性）"""
        k = k or settings.RERANK_TOP_K
        return self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter_dict,
        )

    def delete_collection(self):
        """清空并重新创建当前集合。"""
        if self.backend == "pgvector":
            self._delete_pg_collection()
        else:
            self.vectorstore.reset_collection()
        print(f"已重建集合（{self.backend}）: {self.collection_name}")

    def _delete_pg_collection(self):
        """删除 pgvector 中当前 collection 及其全部 embedding（级联）。"""
        from sqlalchemy import create_engine, text as sql_text

        engine = create_engine(self._pg_connection(), future=True)
        with engine.begin() as conn:
            conn.execute(
                sql_text(
                    "DELETE FROM langchain_pg_collection WHERE name = :n"
                ),
                {"n": self.collection_name},
            )
        engine.dispose()
        # 重新初始化（重建 collection 记录）
        self._init_pgvector()


def _all_documents_from_store() -> List[Document]:
    """从当前向量库读取全量文档（混合检索 BM25 索引构建用）。

    后端无关：pgvector 直接查 embedding 表；chroma 走 _collection.get。
    """
    store = PsychologyVectorStore()
    if store.backend == "pgvector":
        from sqlalchemy import create_engine, text as sql_text

        engine = create_engine(store._pg_connection(), future=True)
        docs: List[Document] = []
        with engine.connect() as conn:
            rows = conn.execute(
                sql_text(
                    "SELECT e.document, e.cmetadata FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                    "WHERE c.name = :n"
                ),
                {"n": store.collection_name},
            ).fetchall()
            for doc_text, meta in rows:
                docs.append(
                    Document(page_content=doc_text or "", metadata=meta or {})
                )
        engine.dispose()
        return docs

    data = store.vectorstore._collection.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    return [
        Document(page_content=text or "", metadata=meta or {})
        for text, meta in zip(docs, metas)
    ]

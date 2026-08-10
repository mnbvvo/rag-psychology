"""
向量存储和检索模块
基于Chroma实现心理学知识的存储和检索
"""
import time
import threading
from collections import OrderedDict
from typing import List, Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings

# 线程隔离的 embedding 耗时存放区，供检索阶段拆分 embed/retrieve 耗时。
_embed_tls = threading.local()

# embedding 进程内 LRU 缓存：同一文本（如用户问题）在语义检测器与向量检索之间
# 只真实调用一次 embedding API，第二次起直接命中缓存（省 300ms+ 与费用）。
_embed_cache: "OrderedDict[tuple, list]" = OrderedDict()
_embed_cache_lock = threading.Lock()


def _cached_embed(key: tuple, do_embed):
    """按 (model, text) 缓存单条文本的向量；缓存满时淘汰最久未用项。"""
    with _embed_cache_lock:
        if key in _embed_cache:
            _embed_cache.move_to_end(key)
            return list(_embed_cache[key])
    vec = do_embed()
    if vec is not None:
        with _embed_cache_lock:
            _embed_cache[key] = list(vec)
            if len(_embed_cache) > settings.EMBED_CACHE_SIZE:
                _embed_cache.popitem(last=False)
    return vec


class TimedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings 的子类，仅用于给 embed_query/embed_documents 包一层耗时统计。

    耗时按线程隔离存放在 _embed_tls.embed_ms（每次调用覆盖）。
    即使是并发请求共用同一实例，各线程读到的也是自己那次调用的耗时。
    """

    def embed_query(self, text: str, *args, **kwargs):
        t = time.perf_counter()
        try:
            base_embed = super().embed_query  # 零参数 super() 只能在方法体内直接使用
            key = (self.model, text)
            return _cached_embed(key, lambda: base_embed(text, *args, **kwargs))
        finally:
            _embed_tls.embed_ms = (time.perf_counter() - t) * 1000

    def embed_documents(self, texts, *args, **kwargs):
        t = time.perf_counter()
        try:
            return super().embed_documents(texts, *args, **kwargs)
        finally:
            _embed_tls.embed_ms = (time.perf_counter() - t) * 1000


def get_last_embed_ms() -> float:
    """读取当前线程最近一次 embedding 调用的耗时（毫秒）。"""
    return getattr(_embed_tls, "embed_ms", 0.0) or 0.0


class PsychologyVectorStore:
    """心理学向量存储"""

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = None,
    ):
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR

        # 初始化OpenAI Embeddings（子类化以实现 embedding 耗时统计）
        self.embeddings = TimedOpenAIEmbeddings(
            model=embedding_model or settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            chunk_size=10,
            timeout=30,       # 上游挂起时及时失败，避免导入/检索永久阻塞
            max_retries=2,    # 瞬时网络/限流错误自动重试
        )

        # 初始化Chroma向量存储
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加或更新文档到向量存储。"""
        if not documents:
            return []

        stored_ids = self.vectorstore.add_documents(documents, ids=ids)
        print(f"成功写入 {len(stored_ids)} 个文档到向量存储")
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
        self.vectorstore.reset_collection()
        print(f"已重建集合: {self.collection_name}")

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        count = self.vectorstore._collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
        }

    def as_retriever(self, k: int = None, search_type: str = "similarity"):
        """返回LangChain retriever对象"""
        k = k or settings.RERANK_TOP_K

        if search_type == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": 20,
                    "lambda_mult": 0.5,
                }
            )
        else:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

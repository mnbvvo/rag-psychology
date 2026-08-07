"""
混合检索模块：向量召回 ∪ 关键词（BM25）召回 → 去重 → 交给重排器精排。

设计要点：
- 因为最终排序由重排器（Cross-Encoder）统一完成，多路召回只需取并集，
  无需做分数归一化/加权融合 —— 相比传统混合检索大幅简化且更鲁棒。
- BM25 索引首次使用时懒构建（从向量库全量文档 + jieba 分词；本地数百条
  毫秒级构建），构建结果常驻内存；构建/检索任何异常都向上抛，由
  rag_core 回退到纯向量检索，不影响检索链路可用性。
"""
import threading
from typing import List, Optional

from config.settings import settings


class BM25HybridSearcher:
    """基于 rank_bm25 的中文关键词检索器（jieba 分词）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List = []
        self._index = None
        self._built = False
        self._lock = threading.Lock()

    def build(self, documents: List) -> None:
        """从候选文档构建 BM25 索引（线程安全，重复调用只建一次）。"""
        if self._built:
            return
        with self._lock:
            if self._built:
                return
            import jieba
            from rank_bm25 import BM25Okapi

            tokenized = [list(jieba.cut(d.page_content or "")) for d in documents]
            self._docs = documents
            self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)
            self._built = True

    def search(self, query: str, k: int = 5) -> List:
        """BM25 关键词召回 top-k 文档（未构建或分词为空时返回空列表）。"""
        if not self._built or self._index is None:
            return []
        import jieba

        tokens = [t for t in jieba.cut(query) if t.strip()]
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(zip(self._docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]


_searcher: Optional[BM25HybridSearcher] = None
_searcher_lock = threading.Lock()


def get_hybrid_searcher() -> BM25HybridSearcher:
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                _searcher = BM25HybridSearcher()
    return _searcher


def _docs_from_vectorstore() -> List:
    """从 Chroma 拉取全量文档用于构建索引。"""
    from langchain_core.documents import Document

    from modules.vector_store import PsychologyVectorStore

    store = PsychologyVectorStore()
    data = store.vectorstore._collection.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    return [
        Document(page_content=text or "", metadata=meta or {})
        for text, meta in zip(docs, metas)
    ]


def keyword_search(query: str, k: Optional[int] = None) -> List:
    """关键词召回：首次调用自动从向量库全量构建 BM25 索引。

    异常直接上抛，由 rag_core 回退到纯向量检索。
    """
    searcher = get_hybrid_searcher()
    if not searcher._built:
        searcher.build(_docs_from_vectorstore())
    return searcher.search(query, k=k or settings.HYBRID_KEYWORD_K)


def warm_up_index() -> None:
    """启动预热：后台构建 BM25 索引（失败静默，首次问答时懒构建兜底）。"""
    try:
        keyword_search("预热", k=1)
        print("[startup] 混合检索 BM25 索引已构建")
    except Exception as e:
        print(f"[startup][WARN] BM25 索引预热失败（首次问答时将懒构建）: {e}")

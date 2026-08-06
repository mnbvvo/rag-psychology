"""
本地重排序模块（Cross-Encoder）

基于 sentence-transformers 的 CrossEncoder 加载 bge-reranker-v2-m3，
对「问题 × 候选文档」逐对打分，按分数降序返回 top_k —— 替代仅按
向量相似度分数截断的"假重排"，提升 top 命中率。

设计要点：
- 懒加载全局单例：首次调用时才加载模型（启动不拖慢，加载后常驻内存）
- 模型路径 / 设备可配置（settings.RERANK_MODEL / RERANK_DEVICE）
- 任何异常（模型缺失、OOM、设备问题等）都向上抛，由 rag_core 回退到
  原排序逻辑，保证检索链路永不因重排而不可用
"""
import os
import threading
from typing import List, Optional

from config.settings import settings


class LocalReranker:
    """本地 Cross-Encoder 重排器。"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or settings.RERANK_MODEL
        self.device = device or settings.RERANK_DEVICE or None
        self._model = None
        self._lock = threading.Lock()
        self._load_failed = False  # 加载失败后置位，避免每次请求都重试拖慢链路

    def _load(self):
        """懒加载模型（线程安全，进程内只加载一次）。

        - 若模型已在加载中（如启动预热线程持锁），请求线程不等待，
          直接抛错由调用方快速回退，避免首次问答被加载阻塞 10 秒。
        - 加载失败置位 _load_failed，后续请求立即抛错走回退，不再重试。
        """
        if self._model is not None:
            return self._model
        if self._load_failed:
            raise RuntimeError("重排模型加载失败，本次回退到原排序")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("重排模型正在加载中，本次回退到原排序")
        try:
            if self._model is not None:
                return self._model
            # 本地路径快速探测：路径配置错误时立即失败，不交给 transformers
            # 当 HF 模型名尝试联网下载（可能卡几十秒）；纯 HF 名（如 BAAI/xxx）才走下载
            p = self.model_path
            looks_local = ":" in p or "\\" in p or p.startswith(("/", "./", "../"))
            if os.path.isdir(p):
                if not os.path.exists(os.path.join(p, "config.json")):
                    raise FileNotFoundError(f"模型目录缺少 config.json: {p}")
            elif looks_local:
                raise FileNotFoundError(f"本地模型路径不存在: {p}")

            from sentence_transformers import CrossEncoder

            kwargs = {"max_length": settings.RERANK_MAX_LENGTH}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(p, **kwargs)
            return self._model
        except Exception:
            self._load_failed = True
            raise
        finally:
            self._lock.release()

    def rerank(self, query: str, documents: List, top_k: int = 3) -> List:
        """对候选文档重排，返回按分数降序的 top_k 条（保持原对象引用）。

        documents：任意带 ``page_content`` 属性的对象（LangChain Document）。
        启用 RERANK_MIN_SCORE 时，低于阈值的候选被丢弃；全部被丢弃则返回
        空列表，由上游触发"没有足够信息"的低相关兜底（防幻觉）。
        """
        if not documents:
            return []
        model = self._load()
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(
            pairs,
            batch_size=settings.RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        min_score = settings.RERANK_MIN_SCORE
        if min_score > 0:
            ranked = [(doc, score) for doc, score in ranked if score >= min_score]
        return [doc for doc, _ in ranked[:top_k]]


# 全局单例（懒加载）
_reranker: Optional[LocalReranker] = None
_reranker_lock = threading.Lock()


def get_reranker() -> LocalReranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = LocalReranker()
    return _reranker


def rerank_documents(query: str, documents: List, top_k: int = 3) -> List:
    """便捷入口：返回重排后的 top_k 文档；异常直接上抛，由调用方回退。"""
    return get_reranker().rerank(query, documents, top_k=top_k)

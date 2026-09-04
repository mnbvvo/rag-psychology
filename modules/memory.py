"""长期记忆服务（向量检索式）。

三件事：
- save_turn：每轮问答落库 user_chat_history，打双向量——query 向量 +
  query+answer 拼接的 qa 向量（检索主用，语义从「问题↔问题」升级为
  「问题↔问答内容」，用户换措辞也能靠历史 answer 召回）；
- search：当前问题 embedding → fn_search_chat_history 检索该用户相似历史
  （SQL 内 COALESCE(qa_embedding, embedding) 优先 qa 向量）；
- build_context：把相似历史拼成注入 system prompt 的上下文文本。

与旧「全量拼接历史」的区别：检索成本恒定（一次 embedding + 数据库 top_k），
不随历史总量线性增长；相似历史以「参考上下文」形式注入，不会把旧问题
重复成新问题。

embedding 复用 modules.vector_store.TimedOpenAIEmbeddings（带进程内 LRU
缓存：同一文本在安全检测与记忆检索之间只真实调用一次 embedding API）。
"""
from typing import Dict, List, Optional

from db import SessionLocal
from db.crud import add_chat_history, search_chat_history
from config.settings import settings


class MemoryService:
    """向量检索式长期记忆：落库 + 检索 + 上下文拼装。"""

    def __init__(self):
        from modules.vector_store import TimedOpenAIEmbeddings

        model = settings.MEMORY_EMBEDDING_MODEL or settings.EMBEDDING_MODEL
        self.embeddings = TimedOpenAIEmbeddings(
            model=model,
            openai_api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_API_BASE,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            chunk_size=10,
            timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
            max_retries=2,
        )

    # ---------------- embedding ----------------
    def embed(self, text: str) -> List[float]:
        """文本 → embedding 向量（复用 LRU 缓存）。"""
        return list(self.embeddings.embed_query(text))

    # ---------------- 落库 ----------------
    def save_turn(self, user_id: str, query: str, answer: Optional[str]) -> None:
        """问答落库（query 向量 + qa 向量，双向量）。

        双向量说明：
        - embedding：仅 query 的向量（兼容存量数据，保留回退用）；
        - qa_embedding：query+answer 拼接后的向量，检索主用（SQL 函数优先）。
          匹配语义从「问题↔问题」升级为「问题↔问答内容」——用户换措辞
          （"睡眠不好"→"老是失眠"）时，靠历史 answer 的语义仍能召回。
        answer 为空时 qa_embedding 回退为 query 向量；answer 超过 500 字符
        时截断再拼接（embedding 只需要语义，超长尾部增益低且费 token）。

        落库失败不影响主流程，只打告警。
        """
        try:
            if not query or not query.strip():
                return
            q_text = query.strip()
            q_vec = self.embed(q_text)
            ans_text = (answer or "").strip()
            if ans_text:
                qa_vec = self.embed(f"{q_text}\n{ans_text[:500]}")
            else:
                qa_vec = q_vec  # 空 answer 时 qa 向量退化为 query 向量
            with SessionLocal() as db:
                add_chat_history(db, user_id, q_text, ans_text, q_vec, qa_vec)
                db.commit()
        except Exception as e:
            print(f"[memory][WARN] 长期记忆落库失败: {e}", flush=True)

    # ---------------- 检索 ----------------
    def search(self, user_id: str, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """检索该用户与 query 最相似的 top_k 条历史（余弦相似度降序）。"""
        top_k = top_k or settings.MEMORY_TOP_K
        q_vec = self.embed(query)
        with SessionLocal() as db:
            rows = search_chat_history(db, user_id, q_vec, limit=top_k)
        min_sim = settings.MEMORY_MIN_SIMILARITY
        if min_sim > 0:
            rows = [r for r in rows if (r.get("cosine_similarity") or 0) >= min_sim]
        return rows

    # ---------------- 上下文拼装 ----------------
    def build_context(self, query: str, user_id: str) -> str:
        """检索相似历史并拼成注入 system prompt 的上下文文本；无相关记忆返回 ""。"""
        try:
            rows = self.search(user_id, query)
        except Exception as e:
            print(f"[memory][WARN] 记忆检索失败，本次不带记忆: {e}", flush=True)
            return ""
        if not rows:
            return ""
        parts = []
        for i, r in enumerate(rows, 1):
            q = (r.get("query") or "").strip()
            a = (r.get("answer") or "").strip()
            sim = r.get("cosine_similarity")
            sim_txt = f"{sim:.3f}" if isinstance(sim, float) else "-"
            parts.append(f"[历史{i}](相似度 {sim_txt})\n用户：{q}\n助手：{a}")
        return (
            "以下是该用户与当前问题相关的历史对话记录（仅作参考背景，"
            "注意区分时间先后，不要把它们当作本次的新问题）：\n\n"
            + "\n\n".join(parts)
        )


# 全局单例（与 rag_system 同级；embedding 实例惰性初始化，import 不发起网络调用）
memory_service = MemoryService()

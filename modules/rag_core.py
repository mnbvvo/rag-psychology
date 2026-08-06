"""
RAG核心流程
实现青少年心理领域的检索增强生成
"""
import time
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from modules.vector_store import PsychologyVectorStore
from modules.prompt_store import build_system_prompt
from config.settings import settings


class PsychologyRAG:
    """青少年心理RAG系统"""

    def __init__(self, vectorstore: PsychologyVectorStore):
        self.vectorstore = vectorstore

        # 初始化LLM（事实/建议类问答，温度偏低以减少幻觉）
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            temperature=settings.CHAT_TEMPERATURE,
            max_tokens=4096,
            timeout=30,       # 上游挂起时及时失败，避免请求永久阻塞
            max_retries=2,    # 瞬时网络/限流错误自动重试
            extra_body={"enable_thinking": settings.ENABLE_THINKING},  # 思考/推理模式开关
        )

    def _build_age_filter(self, age_group: str | None) -> dict | None:
        """仅当 age_group 命中已知分桶时才构建元数据过滤条件。"""
        if age_group in settings.AGE_GROUPS:
            return {"age_group": age_group}
        return None

    def retrieve(
        self,
        question: str,
        age_group: str = "teen",
        timings: Optional[Dict] = None,
    ) -> List[Document]:
        """检索相关文档：多召回 -> （可选重排）-> 截断。

        启用本地重排（RERANK_ENABLED）时：召回候选集 → Cross-Encoder 精排取 top3；
        重排失败自动回退到「按向量分数排序截断」的原逻辑，保证可用性。
        """
        from modules.vector_store import get_last_embed_ms

        filter_dict = self._build_age_filter(age_group)
        rerank_enabled = settings.RERANK_ENABLED

        t0 = time.perf_counter()
        if settings.SEARCH_TYPE == "mmr":
            # 重排开启时用 MMR 产出更多候选（多样性），交给重排精排；否则保持原 top_k 行为
            mmr_k = settings.FETCH_K if rerank_enabled else settings.RERANK_TOP_K
            docs = self.vectorstore.max_marginal_relevance_search(
                question,
                k=mmr_k,
                fetch_k=settings.FETCH_K * 2,
                lambda_mult=settings.MMR_LAMBDA,
                filter_dict=filter_dict,
            )
            scored = None
        else:
            # 带分数的语义检索，便于阈值过滤与回退排序
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=settings.FETCH_K, filter_dict=filter_dict
            )
            if rerank_enabled:
                # 重排前不做硬阈值过滤（避免误杀），把全部候选交给重排器精排
                docs = [doc for doc, _ in scored]
            else:
                if settings.MIN_RELEVANCE_SCORE > 0:
                    scored = [
                        (doc, score)
                        for doc, score in scored
                        if score >= settings.MIN_RELEVANCE_SCORE
                    ]
                # 按相关性降序重排，仅取最相关的 RERANK_TOP_K 条喂给模型
                scored.sort(key=lambda item: item[1], reverse=True)
                top = scored[: settings.RERANK_TOP_K]
                docs = [doc for doc, _ in top]

        # 混合检索：向量候选 ∪ BM25 关键词候选（去重），扩大召回面交给重排器精排。
        # 仅重排开启时生效（最终排序依赖重排器统一精排）；关键词检索异常自动回退纯向量。
        if rerank_enabled and settings.HYBRID_ENABLED and docs:
            try:
                from modules.hybrid_search import keyword_search

                t_h = time.perf_counter()
                kw_docs = keyword_search(question, k=settings.HYBRID_KEYWORD_K)
                seen = {d.metadata.get("card_id") for d in docs}
                for d in kw_docs:
                    cid = d.metadata.get("card_id")
                    if cid and cid not in seen:
                        docs.append(d)
                        seen.add(cid)
                if timings is not None:
                    timings["hybrid"] = (time.perf_counter() - t_h) * 1000
            except Exception as e:
                print(f"[hybrid][WARN] 关键词检索失败，仅用向量召回: {e}", flush=True)

        # 本地重排：对候选按「问题 × 文档」逐对打分，取 top_k
        if rerank_enabled and docs:
            try:
                from modules.reranker import rerank_documents

                t_r = time.perf_counter()
                docs = rerank_documents(question, docs, top_k=settings.RERANK_TOP_K)
                if timings is not None:
                    timings["rerank"] = (time.perf_counter() - t_r) * 1000
            except Exception as e:
                # 回退：similarity 按原始分数截断；mmr 直接取前 k 条
                print(f"[rerank][WARN] 重排失败，回退到原排序: {e}", flush=True)
                if scored:
                    scored.sort(key=lambda item: item[1], reverse=True)
                    docs = [doc for doc, _ in scored[: settings.RERANK_TOP_K]]
                else:
                    docs = docs[: settings.RERANK_TOP_K]

        # 拆分 embed / retrieve / rerank：检索总耗时 - embedding - 重排
        search_total_ms = (time.perf_counter() - t0) * 1000
        embed_ms = get_last_embed_ms()
        rerank_ms = (timings.get("rerank", 0) if timings else 0) or 0
        retrieve_ms = max(0.0, search_total_ms - embed_ms - rerank_ms)
        if timings is not None:
            timings["embed"] = embed_ms
            timings["retrieve"] = retrieve_ms

        return docs

    def compress_messages(self, messages: List[Dict]) -> List[Dict]:
        """对话历史压缩：保留最近 MAX_HISTORY_TURNS 轮，对更早历史做摘要。

        每轮约 2 条消息（human + ai）。摘要失败时直接截断，保证可用性。
        """
        max_turns = settings.MAX_HISTORY_TURNS
        if max_turns <= 0 or len(messages) <= max_turns * 2:
            return messages

        keep_count = max_turns * 2
        recent = messages[-keep_count:]
        older = messages[:-keep_count]

        try:
            conversation_text = "\n\n".join([
                f"{'用户' if m['role'] == 'human' else '助手'}：{m['content']}"
                for m in older
            ])
            summary_prompt = ChatPromptTemplate.from_messages([
                ("human", "请用一段话简要总结以下心理咨询对话的要点，保留用户核心诉求和关键建议，控制在 200 字以内：\n\n{conversation}"),
            ])
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            summary = summary_chain.invoke({"conversation": conversation_text})
            if summary and summary.strip():
                return [{"role": "human", "content": f"前文摘要：{summary.strip()}"}] + recent
        except Exception:
            # 摘要失败时回退到截断
            pass

        return recent

    def generate(
        self,
        question: str,
        context: List[Document],
        age_group: str = "teen",
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
        timings: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict:
        """基于检索到的内容生成回答

        system_prompt_override：非 None 时，使用传入文本作为系统提示词基础，
        便于前端在不落盘的情况下预览/对比不同提示词的效果。
        prompt_id：指定使用提示词库中的某条提示词（与 override 互斥，override 优先）。
        messages：多轮对话历史；提供时会把完整历史拼入 prompt，question 仅用于检索与日志。
        """
        # 构建上下文文本
        context_text = "\n\n".join(
            [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(context)]
        )

        # 低相关提示：检索结果偏弱时，要求模型如实说明而非编造
        low_relevance = not context

        # 构建系统提示词（base + 年龄段片段 + 参考资料），
        system_prompt = build_system_prompt(
            age_group=age_group,
            context_text=context_text,
            low_relevance=low_relevance,
            system_prompt_override=system_prompt_override,
            prompt_id=prompt_id,
        )

        # 多轮模式：system + 压缩后的历史消息（最后一条为当前问题）
        if messages:
            compressed = self.compress_messages(messages)
            prompt_messages = [("system", system_prompt)]
            for m in compressed:
                if m["role"] == "human":
                    prompt_messages.append(("human", m["content"]))
                elif m["role"] == "ai":
                    prompt_messages.append(("ai", m["content"]))
            rag_prompt = ChatPromptTemplate.from_messages(prompt_messages)
            chain = rag_prompt | self.llm | StrOutputParser()
            invoke_input = {}
        else:
            # 单轮模式：system + 当前问题
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}"),
            ])
            chain = rag_prompt | self.llm | StrOutputParser()
            invoke_input = {"question": question}

        # 生成回答（LLM 调用是主要耗时来源，单独计时）
        t0 = time.perf_counter()
        answer = chain.invoke(invoke_input)
        if timings is not None:
            timings["llm"] = (time.perf_counter() - t0) * 1000

        # 构建来源信息
        sources = []
        for i, doc in enumerate(context):
            sources.append({
                "index": i + 1,
                "card_id": doc.metadata.get("card_id"),
                "title": doc.metadata.get("title", "未知"),
                "source_id": doc.metadata.get("source_id"),
                "risk_level": doc.metadata.get("risk_level"),
            })

        return {
            "answer": answer,
            "sources": sources,
        }


    def run(
        self,
        question: str,
        age_group: str = "teen",
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
        timings: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict:
        """同步执行完整的RAG流程"""
        # 检索
        context = self.retrieve(question, age_group, timings=timings)

        # 生成
        result = self.generate(question, context, age_group, system_prompt_override, prompt_id, timings=timings, messages=messages)

        return {
            "question": question,
            "age_group": age_group,
            "context": context,
            "answer": result["answer"],
            "sources": result["sources"],
        }

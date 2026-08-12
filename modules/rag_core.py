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


def build_sources(context: List[Document]) -> List[Dict]:
    """从检索文档构建来源信息（供 generate 结果与 SSE sources 事件复用）。"""
    sources = []
    for i, doc in enumerate(context):
        sources.append({
            "index": i + 1,
            "card_id": doc.metadata.get("card_id"),
            "title": doc.metadata.get("title", "未知"),
            "source_id": doc.metadata.get("source_id"),
            "risk_level": doc.metadata.get("risk_level"),
        })
    return sources


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

    def retrieve(
        self,
        question: str,
        timings: Optional[Dict] = None,
    ) -> List[Document]:
        """检索相关文档：多召回 -> （可选重排）-> 截断。

        启用本地重排（RERANK_ENABLED）时：召回候选集 → Cross-Encoder 精排取 top3；
        重排失败自动回退到「按向量分数排序截断」的原逻辑，保证可用性。
        """
        from modules.vector_store import get_embed_ms

        filter_dict = None  # 年龄分桶过滤已移除（age_group 元数据保留在库中，检索不再按年龄过滤）
        rerank_enabled = settings.RERANK_ENABLED

        t0 = time.perf_counter()
        embed_before = get_embed_ms()  # retrieve 窗口起点：用于拆分本窗口内真实 embedding 耗时
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

        # 拆分 embed / retrieve / hybrid / rerank：
        # - embed：本 retrieve 窗口内「真实调用 embedding API」的耗时增量
        #   （结束时累计 - 开始时累计）。正常流程下问题向量已在安全阶段 embed
        #   并被缓存，检索阶段命中缓存不计时 → 显示 0ms 是真实语义；若未走
        #   prepare（直接调 retrieve），本窗口的真实 embed 会被这里兜底记上。
        # - retrieve：检索总耗时 - 本窗口 embed 增量 - 混合 - 重排，四栏相加
        #   与总墙钟一致，便于按耗时定位瓶颈。
        search_total_ms = (time.perf_counter() - t0) * 1000
        embed_in_retrieve = max(0.0, get_embed_ms() - embed_before)
        rerank_ms = (timings.get("rerank", 0) if timings else 0) or 0
        hybrid_ms = (timings.get("hybrid", 0) if timings else 0) or 0
        retrieve_ms = max(0.0, search_total_ms - embed_in_retrieve - rerank_ms - hybrid_ms)
        if timings is not None:
            # prepare() 已把安全阶段的真实 embed 写入 timings["embed"]，保留之；
            # 仅在未被写入（直接调 retrieve 的路径）时用本窗口增量兜底。
            if "embed" not in timings:
                timings["embed"] = embed_in_retrieve
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
                f"{'用户' if m['role'] in ('human', 'user') else '助手'}：{m['content']}"
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

    def _build_messages(
        self,
        question: str,
        context: List[Document],
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> List:
        """组装直接传给 LLM 的消息列表（generate 与 stream_generate 共用）。

        直接返回消息元组列表交给 llm.invoke/astream，**绕过 ChatPromptTemplate
        的 {var} 模板变量解析**：即使提示词/空上下文里出现 {context} 字面量，
        也不会被当成模板变量报 KeyError（参考此前 RERANK_MIN_SCORE 过滤导致
        检索为空时 {context} 占位触发 MissingInput 的 bug）。
        """
        context_text = "\n\n".join(
            [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(context)]
        )
        # 低相关提示：检索结果偏弱时，要求模型如实说明而非编造
        system_prompt = build_system_prompt(
            context_text=context_text,
            low_relevance=not context,
            system_prompt_override=system_prompt_override,
            prompt_id=prompt_id,
        )

        if messages:
            compressed = self.compress_messages(messages)
            prompt_messages = [("system", system_prompt)]
            for m in compressed:
                role = m.get("role", "")
                # 兼容 human/ai 与 user/assistant 两种角色命名，避免历史被静默丢弃
                if role in ("human", "user"):
                    prompt_messages.append(("human", m["content"]))
                elif role in ("ai", "assistant"):
                    prompt_messages.append(("ai", m["content"]))
            return prompt_messages
        return [("system", system_prompt), ("human", question)]

    def generate(
        self,
        question: str,
        context: List[Document],
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
        prompt_messages = self._build_messages(
            question, context, system_prompt_override, prompt_id, messages
        )

        # 生成回答（LLM 调用是主要耗时来源，单独计时）
        t0 = time.perf_counter()
        msg = self.llm.invoke(prompt_messages)
        answer = msg.content if hasattr(msg, "content") else str(msg)
        if timings is not None:
            timings["llm"] = (time.perf_counter() - t0) * 1000

        return {
            "answer": answer,
            "sources": build_sources(context),
        }

    async def stream_generate(
        self,
        question: str,
        context: List[Document],
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ):
        """流式生成：与 generate 相同的提示词组装，逐 token 产出文本块（async generator）。

        供 /api/query/stream（SSE）使用：检索已由 prepare 完成，这里只做生成。
        """
        prompt_messages = self._build_messages(
            question, context, system_prompt_override, prompt_id, messages
        )
        async for chunk in self.llm.astream(prompt_messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content


    def run(
        self,
        question: str,
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
        timings: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict:
        """同步执行完整的RAG流程"""
        # 检索
        context = self.retrieve(question, timings=timings)

        # 生成
        result = self.generate(question, context, system_prompt_override, prompt_id, timings=timings, messages=messages)

        return {
            "question": question,
            "context": context,
            "answer": result["answer"],
            "sources": result["sources"],
        }

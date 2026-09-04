"""
RAG核心流程
实现青少年心理领域的检索增强生成
"""
import asyncio
import time
from typing import Callable, List, Dict, Optional
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
        # enable_thinking 仅流式调用携带：部分兼容接口（如 deepseek-v3.1 非流式）
        # 对 enable_thinking 报 400（"only support stream call"），非流式用不带该参数的实例。
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            temperature=settings.CHAT_TEMPERATURE,
            max_tokens=4096,
            timeout=settings.LLM_TIMEOUT_SECONDS,  # 参数化：须 > 排队超时（settings.validate 强制）
            max_retries=settings.LLM_MAX_RETRIES,  # 瞬时网络/限流错误自动重试
        )
        # 流式实例：保留 enable_thinking（思考/推理模式开关，仅流式接口支持）
        self.llm_stream = ChatOpenAI(
            model=settings.CHAT_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            temperature=settings.CHAT_TEMPERATURE,
            max_tokens=4096,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            extra_body={"enable_thinking": settings.ENABLE_THINKING},
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


    def _build_messages(
        self,
        question: str,
        context: List[Document],
        messages: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        low_relevance: Optional[bool] = None,
    ) -> List:
        """组装直接传给 LLM 的消息列表（generate 与 stream_generate 共用）。

        user_id：当前用户（透传至提示词解析，见 prompt_store.build_system_prompt）。
        messages：归一化后的多轮历史（含当前问题，见 rag_system._normalize）。
        low_relevance：是否追加「未检索到足够资料」说明；None 时按 context 为空判定
        （兼容未显式传参的调用）。RAG 关闭（纯对话）时调用方应显式传 False——
        没有检索动作，不应向模型声明"本次未检索到资料"。
        本方法同时承载两条记忆通道：
        - 跨会话长期记忆（向量检索，注入 system prompt）；
        - 本会话最近 N 轮原文（human/ai 交替插入，解决指代消解）。
        直接返回消息元组列表交给 llm.invoke/astream，**绕过 ChatPromptTemplate
        的 {var} 模板变量解析**：即使提示词/空上下文里出现 {context} 字面量，
        也不会被当成模板变量报 KeyError（参考此前 RERANK_MIN_SCORE 过滤导致
        检索为空时 {context} 占位触发 MissingInput 的 bug）。
        """
        context_text = "\n\n".join(
            [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(context)]
        )
        # 低相关提示：仅「RAG 开启且检索为空」时追加，要求模型如实说明而非编造
        if low_relevance is None:
            low_relevance = not context
        system_prompt = build_system_prompt(
            context_text=context_text,
            low_relevance=low_relevance,
        )

        # 长期记忆（双通道）：
        # 通道一（跨会话·向量检索）：相似历史注入 system prompt，召回成本恒定。
        # 通道二（本会话·最近 N 轮）：messages 中除当前问题外的最近
        #   MEMORY_RECENT_ROUNDS 轮原文，按时间顺序插入 system 之后、当前问题
        #   之前——解决指代消解（"那个方法""刚才说的"等指代词向量检索不到，
        #   只能靠最近几轮原文）；轮数由 settings.MEMORY_RECENT_ROUNDS 控制。
        if settings.MEMORY_ENABLED and user_id:
            try:
                from modules.memory import memory_service

                mem_text = memory_service.build_context(question, user_id)
                if mem_text:
                    system_prompt = f"{system_prompt}\n\n{mem_text}"
            except Exception as e:
                print(f"[memory][WARN] 记忆上下文注入失败，忽略: {e}", flush=True)

        prompt_messages = [("system", system_prompt)]
        if settings.MEMORY_RECENT_ROUNDS > 0 and messages:
            # messages 的最后一条是当前问题（_normalize 保证），跳过它取历史；
            # 1 轮 = 一问一答 2 条消息，先按轮数上限（MEMORY_RECENT_ROUNDS*2）
            # 截出候选，再按字符预算（MEMORY_RECENT_MAX_CHARS）从最新向旧累计，
            # 超预算即停——双约束先到先停，至少保底最近 1 条完整消息（不截断
            # 单条内部）。预算只统计窗口内原文，system/RAG/长期记忆另计。
            history = messages[:-1][-settings.MEMORY_RECENT_ROUNDS * 2:]
            budget = settings.MEMORY_RECENT_MAX_CHARS
            if budget > 0 and len(history) > 1:
                used = 0
                kept: list = []
                for m in reversed(history):
                    content = (m.get("content") or "").strip()
                    if not content:
                        continue
                    used += len(content)
                    if used > budget and kept:
                        break  # 超预算即停；kept 为空时保底放行最近 1 条
                    kept.append(m)
                history = list(reversed(kept))
            for m in history:
                role = (m.get("role") or "").lower()
                content = (m.get("content") or "").strip()
                if not content:
                    continue  # 防御：跳过空内容占位消息
                if role in ("user", "human"):
                    prompt_messages.append(("human", content))
                elif role in ("assistant", "ai"):
                    prompt_messages.append(("assistant", content))
                # 其他 role 忽略（防御：不注入未知角色）
        prompt_messages.append(("human", question))
        return prompt_messages

    def generate(
        self,
        question: str,
        context: List[Document],
        timings: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        low_relevance: Optional[bool] = None,
    ) -> Dict:
        """基于检索到的内容生成回答。

        messages：多轮对话历史；提供时会把完整历史拼入 prompt，question 仅用于检索与日志。
        user_id：当前用户（提示词全局激活项解析，透传保留）。
        low_relevance：是否追加「未检索到足够资料」说明（见 _build_messages）。
        """
        prompt_messages = self._build_messages(
            question, context, messages, user_id, low_relevance
        )

        # 生成回答（LLM 调用是主要耗时来源，单独计时）
        t0 = time.perf_counter()
        if settings.ENABLE_THINKING:
            # 所有模型统一开启思考模式：deepseek 兼容接口非流式不支持 enable_thinking
            # （报 400 "only support stream call"），故同步接口内部走「同步流式」收集
            # 完整内容——与流式接口（llm_stream）同一实例、同一思考参数，行为一致。
            answer = ""
            for chunk in self.llm_stream.stream(prompt_messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    answer += content
        else:
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
        messages: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        low_relevance: Optional[bool] = None,
        prompt_messages: Optional[List] = None,
    ):
        """流式生成：与 generate 相同的提示词组装，逐 token 产出文本块（async generator）。

        供 /api/query/stream（SSE）使用：检索已由 prepare 完成，这里只做生成。
        prompt_messages：可选。已组装好的消息列表（调用方若已在线程池完成
        _build_messages——含同步记忆检索/embedding——可传入跳过重复构建，避免
        这些同步调用直接跑在事件循环上阻塞所有并发请求）。
        """
        if prompt_messages is None:
            prompt_messages = self._build_messages(
                question, context, messages, user_id, low_relevance
            )
        async for chunk in self.llm_stream.astream(prompt_messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content

    async def agenerate(
        self,
        question: str,
        context: List[Document],
        timings: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        low_relevance: Optional[bool] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """异步生成（非流式路径用）：与 generate 相同语义，但 LLM 调用走原生 async。

        - 消息组装（_build_messages，含同步长期记忆检索/embedding）放线程池，
          不占用事件循环 —— 避免单次缓存未命中的 embedding API 调用卡住所有并发请求；
        - LLM 调用用 llm_stream.astream 逐块收集 —— 秒级生成期间事件循环空闲，
          可同时托管大量 in-flight 请求，不再受 run_in_threadpool 线程闸（≈40）限制。
        - cancel_check：可选无参回调；每收到一个 chunk 前调用，返回 True 时立即终止
          收集（async for break → astream 生成器关闭 → 上游连接被真正终止），
          避免"取消后仍生成完整答案白烧 token"。返回 dict 带 cancelled=True 标记。
        - 返回 {answer, sources[, cancelled]}，与 generate 一致；llm 耗时写入 timings。
        """
        # 记忆检索等同步调用移出事件循环（线程池内完成）
        prompt_messages = await asyncio.to_thread(
            self._build_messages, question, context, messages, user_id, low_relevance
        )
        # 首 token 前若已被取消：不发起任何 LLM 调用，直接返回 cancelled
        if cancel_check is not None and cancel_check():
            return {"answer": "", "sources": build_sources(context), "cancelled": True}
        t0 = time.perf_counter()
        full: List[str] = []
        cancelled = False
        async for chunk in self.stream_generate(
            question,
            context,
            messages=messages,
            user_id=user_id,
            low_relevance=low_relevance,
            prompt_messages=prompt_messages,
        ):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            full.append(chunk)
        if timings is not None:
            timings["llm"] = (time.perf_counter() - t0) * 1000

        result: Dict = {
            "answer": "".join(full),
            "sources": build_sources(context),
        }
        if cancelled:
            result["cancelled"] = True
        return result


    def run(
        self,
        question: str,
        timings: Optional[Dict] = None,
        messages: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
    ) -> Dict:
        """同步执行完整的RAG流程"""
        # 检索
        context = self.retrieve(question, timings=timings)

        # 生成
        result = self.generate(question, context, timings=timings, messages=messages, user_id=user_id)

        return {
            "question": question,
            "context": context,
            "answer": result["answer"],
            "sources": result["sources"],
        }

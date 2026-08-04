"""
RAG核心流程
实现青少年心理领域的检索增强生成
"""
from typing import List, Dict, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from modules.vector_store import PsychologyVectorStore
from modules.prompt_store import build_system_prompt
from config.settings import settings


class RAGState(TypedDict):
    """RAG状态"""
    question: str
    age_group: str  # 年龄段: child(6-9), early_teen(10-12), teen(13-15), late_teen(16-18)
    context: List[Document]
    answer: str
    sources: List[Dict]


class PsychologyRAG:
    """青少年心理RAG系统"""

    def __init__(self, vectorstore: PsychologyVectorStore):
        self.vectorstore = vectorstore

        # 初始化LLM（事实/建议类问答，温度偏低以减少幻觉）
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=settings.CHAT_TEMPERATURE,
            max_tokens=4096,
        )

        # 检索相关性分数（供低相关判定使用）
        self._last_scores: list[float] | None = None

    def _build_age_filter(self, age_group: str | None) -> dict | None:
        """仅当 age_group 命中已知分桶时才构建元数据过滤条件。"""
        if age_group in settings.AGE_GROUPS:
            return {"age_group": age_group}
        return None

    def retrieve(self, question: str, age_group: str = "teen") -> List[Document]:
        """检索相关文档：多召回 -> 阈值 -> 重排序截断。"""
        filter_dict = self._build_age_filter(age_group)

        if settings.SEARCH_TYPE == "mmr":
            # 最大边际相关：在相关性基础上提升多样性
            docs = self.vectorstore.max_marginal_relevance_search(
                question,
                k=min(settings.RETRIEVAL_TOP_K, settings.FETCH_K),
                fetch_k=settings.FETCH_K * 2,
                lambda_mult=settings.MMR_LAMBDA,
                filter_dict=filter_dict,
            )
            self._last_scores = None
        else:
            # 带分数的语义检索，便于阈值过滤与重排序
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=settings.FETCH_K, filter_dict=filter_dict
            )
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
            self._last_scores = [score for _, score in top]

        return docs

    def generate(
        self,
        question: str,
        context: List[Document],
        age_group: str = "teen",
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> Dict:
        """基于检索到的内容生成回答

        system_prompt_override：非 None 时，使用传入文本作为系统提示词基础，
        便于前端在不落盘的情况下预览/对比不同提示词的效果。
        prompt_id：指定使用提示词库中的某条提示词（与 override 互斥，override 优先）。
        """
        # 构建上下文文本
        context_text = "\n\n".join(
            [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(context)]
        )

        # 低相关提示：检索结果偏弱时，要求模型如实说明而非编造
        low_relevance = not context

        # 构建系统提示词（base + 年龄段片段 + 参考资料），
        # 提示词内容来自 config/system_prompt.json，可由前端实时修改并同步。
        system_prompt = build_system_prompt(
            age_group=age_group,
            context_text=context_text,
            low_relevance=low_relevance,
            system_prompt_override=system_prompt_override,
            prompt_id=prompt_id,
        )

        # 构建RAG prompt
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # 创建链
        chain = rag_prompt | self.llm | StrOutputParser()

        # 生成回答
        answer = chain.invoke({"question": question})

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

    async def arun(
        self,
        question: str,
        age_group: str = "teen",
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> RAGState:
        """异步执行完整的RAG流程"""
        # 检索
        context = self.retrieve(question, age_group)

        # 生成
        result = self.generate(question, context, age_group, system_prompt_override, prompt_id)

        return RAGState(
            question=question,
            age_group=age_group,
            context=context,
            answer=result["answer"],
            sources=result["sources"],
        )

    def run(
        self,
        question: str,
        age_group: str = "teen",
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> RAGState:
        """同步执行完整的RAG流程"""
        # 检索
        context = self.retrieve(question, age_group)

        # 生成
        result = self.generate(question, context, age_group, system_prompt_override, prompt_id)

        return RAGState(
            question=question,
            age_group=age_group,
            context=context,
            answer=result["answer"],
            sources=result["sources"],
        )

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
            max_tokens=2000,
        )

        # 检索相关性分数（供低相关判定使用）
        self._last_scores: list[float] | None = None

    def _get_age_specific_prompt(self, age_group: str) -> str:
        """根据年龄段获取合适的prompt风格"""
        prompts = {
            "child": (
                "你是一位温柔、有耐心的儿童心理辅导老师。"
                "请用简单、温暖、容易理解的语言回答，避免使用复杂术语。"
                "多使用比喻和故事来解释，让孩子感到被理解和关爱。"
            ),
            "early_teen": (
                "你是一位理解青少年的心理辅导老师。"
                "请用友善、尊重的语气回答，避免说教。"
                "认可他们的感受，给予实用建议，让他们知道寻求帮助是正常的。"
            ),
            "teen": (
                "你是一位专业的青少年心理咨询师。"
                "请用平等、理解的态度回答，尊重他们的想法和感受。"
                "提供专业但易懂的解释，引导他们积极思考。"
            ),
            "late_teen": (
                "你是一位资深心理咨询师。"
                "请用专业但亲切的方式回答，把他们当作成年人对待。"
                "提供深入的心理学知识，帮助他们自我成长和问题解决。"
            ),
        }
        return prompts.get(age_group, prompts["teen"])

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
    ) -> Dict:
        """基于检索到的内容生成回答"""
        # 构建上下文文本
        context_text = "\n\n".join(
            [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(context)]
        )

        # 获取年龄段特定的系统提示
        age_prompt = self._get_age_specific_prompt(age_group)

        # 低相关提示：检索结果偏弱时，要求模型如实说明而非编造
        low_relevance_note = ""
        if not context:
            low_relevance_note = (
                "\n\n【重要】本次未检索到足够相关的参考资料。"
                "若无法直接、有据地回答，请明确说明“我目前没有足够的信息来回答这个问题”，"
                "不要编造内容。\n"
            )

        # 构建RAG prompt
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是专业的青少年心理咨询师。{age_prompt}

【重要原则】
1. 仅基于提供的参考资料回答，不要编造信息
2. 如果资料不足以回答问题，请诚实地说"我目前没有足够的信息来回答这个问题"
3. 始终提供温暖、专业的态度
4. 如果涉及安全问题，优先提供危机干预资源
5. 在回答末尾标注信息来源编号（如[1][2]）
{low_relevance_note}
参考资料:
{context_text}
"""),
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
    ) -> RAGState:
        """异步执行完整的RAG流程"""
        # 检索
        context = self.retrieve(question, age_group)

        # 生成
        result = self.generate(question, context, age_group)

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
    ) -> RAGState:
        """同步执行完整的RAG流程"""
        # 检索
        context = self.retrieve(question, age_group)

        # 生成
        result = self.generate(question, context, age_group)

        return RAGState(
            question=question,
            age_group=age_group,
            context=context,
            answer=result["answer"],
            sources=result["sources"],
        )

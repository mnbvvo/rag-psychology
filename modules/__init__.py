"""
心理学RAG系统主入口
整合所有模块提供统一的接口
"""
import time
import threading
from modules.vector_store import PsychologyVectorStore
from modules.rag_core import PsychologyRAG, build_sources
from modules.safety_checker import SafetyChecker
from config.settings import settings
from typing import Dict, List, Optional, Union

# 避免超长问题刷屏，日志里只截取前 48 个字符
_Q_PREVIEW = 48

# 危机等级排序（多轮检测取最高等级用）
_LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _log_query_timings(question: str, timings: Dict, source_count: int):
    """统一打印一次 /api/query 的分阶段耗时（后台 print，便于定位瓶颈）。"""
    def ms(key):
        v = timings.get(key)
        return f"{v:.0f}ms" if v is not None else "-"

    q = (question or "").replace("\n", " ").strip()
    q_preview = (q[:_Q_PREVIEW] + "…") if len(q) > _Q_PREVIEW else q
    tid = threading.get_ident() % 100000  # 短线程号，便于并发时区分 A/B
    safety = ms("safety")
    embed = ms("embed")
    retrieve = ms("retrieve")
    llm = ms("llm")
    total = ms("total")
    print(
        f"[query][t:{tid}] q=\"{q_preview}\" "
        f"safety: {safety} | embed: {embed} | retrieve: {retrieve} | "
        f"llm: {llm} | total: {total} | src: {source_count}",
        flush=True,
    )


class PsychologyRAGSystem:
    """青少年心理RAG系统"""

    def __init__(self):
        # 初始化各模块
        self.vectorstore = PsychologyVectorStore()
        self.rag = PsychologyRAG(self.vectorstore)
        self.safety_checker = SafetyChecker()

    @staticmethod
    def _normalize(messages, question):
        """统一归一化 messages 并提取当前问题；返回 (norm_messages, current_question)。"""
        if messages:
            norm_messages = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if role in ("user", "human"):
                    norm_messages.append({"role": "human", "content": content})
                elif role in ("assistant", "ai"):
                    norm_messages.append({"role": "ai", "content": content})
                else:
                    norm_messages.append({"role": role, "content": content})
            # 当前问题取最后一条 human 消息
            current_question = ""
            for m in reversed(norm_messages):
                if m["role"] == "human":
                    current_question = m["content"]
                    break
            if not current_question:
                raise ValueError("messages 中未找到有效的用户问题")
            return norm_messages, current_question
        if question:
            return [{"role": "human", "content": question}], question
        raise ValueError("必须提供 question 或 messages")

    def prepare(
        self,
        question: str = None,
        check_safety: bool = True,
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict:
        """安全检测 + 检索（不含生成）。

        供 SSE 端点复用：先同步完成 safety + 混合检索 + 重排，返回
        context/sources/safety/timings；高危时 is_crisis_response=True 且
        answer 为危机响应，此时不应再进入生成阶段。
        """
        norm_messages, current_question = self._normalize(messages, question)
        result = {
            "question": current_question,
            "context": [],
            "answer": "",
            "sources": [],
            "safety_check": None,
            "is_crisis_response": False,
            "norm_messages": norm_messages,
        }
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # 安全检查（基于当前问题）：L0 关键词 + L1 语义（高危意图锚点距离）。
        # 语义层复用本向量库的 embedding（带缓存），与检索共用同一次 API 调用。
        #
        # 多轮兜底：不只看最后一条 human —— 用户可能在某一轮暴露危机信号、
        # 下一轮用指代句（"那该怎么办"）继续，只查最后一条会漏。
        # 对全部 human 轮次逐一检测，取最高等级作为本轮判定（历史高危同样拦截）。
        if check_safety:
            ts = time.perf_counter()
            emb_query_fn = self.vectorstore.embeddings.embed_query
            emb_docs_fn = self.vectorstore.embeddings.embed_documents
            safety_result = self.safety_checker.check_full(current_question, emb_query_fn, emb_docs_fn)
            for m in norm_messages:
                if m["role"] != "human" or m["content"] == current_question:
                    continue
                try:
                    r = self.safety_checker.check_full(m["content"], emb_query_fn, emb_docs_fn)
                except Exception:
                    continue  # 单轮检测异常不影响主流程
                if _LEVEL_RANK.get(r.get("level"), 0) > _LEVEL_RANK.get(safety_result.get("level"), 0):
                    safety_result = r
            timings["safety"] = (time.perf_counter() - ts) * 1000
            result["safety_check"] = safety_result
            # 高危：直接返回危机响应，不走检索与生成
            if safety_result.get("is_crisis") and safety_result["level"] == "high":
                result["answer"] = safety_result["safety_response"]["message"]
                result["is_crisis_response"] = True
                timings["total"] = (time.perf_counter() - t0) * 1000
                result["timings"] = timings
                _log_query_timings(current_question, timings, 0)
                return result

        # 检索：混合召回 + 重排
        context = self.rag.retrieve(current_question, timings=timings)
        result["context"] = context
        result["sources"] = build_sources(context)
        # 中/低危：附带关怀提示
        if check_safety and safety_result.get("is_crisis"):
            result["safety_note"] = safety_result["safety_response"]["message"]
        result["timings"] = timings
        return result

    def query(
        self,
        question: str = None,
        check_safety: bool = True,
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict:
        """查询系统（同步完整流程：prepare + generate）。

        支持单轮 question 或多轮 messages。多轮模式下，从最后一条 human/user
        消息提取当前问题用于检索与安全检测，完整历史传给 LLM 作为上下文。
        """
        # 全流程墙钟：从 prepare（安全+检索+重排）到生成结束，与 SSE 端点 total 语义一致
        t_query = time.perf_counter()
        prep = self.prepare(question, check_safety, system_prompt_override, prompt_id, messages)
        timings = prep.get("timings") or {}

        # 高危：直接返回危机响应（prepare 内已记录全程 total）
        if prep.get("is_crisis_response"):
            return prep

        # 生成
        gen = self.rag.generate(
            prep["question"], prep.get("context") or [],
            system_prompt_override, prompt_id, timings=timings, messages=prep.get("norm_messages"),
        )
        prep["answer"] = gen["answer"]
        prep["sources"] = gen["sources"]
        # 回答侧安全复查：LLM 输出命中高危关键词时追加安全提醒（并落审计）
        if check_safety:
            answer, ans_check = self.safety_checker.review_answer(gen["answer"])
            prep["answer"] = answer
            if ans_check:
                prep["answer_safety_check"] = ans_check
        timings["total"] = (time.perf_counter() - t_query) * 1000
        prep["timings"] = timings
        _log_query_timings(prep["question"], timings, len(prep["sources"]))
        return prep

# 创建全局实例
rag_system = PsychologyRAGSystem()

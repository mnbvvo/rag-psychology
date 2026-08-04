"""
心理学RAG系统主入口
整合所有模块提供统一的接口
"""
import time
import threading
from modules.vector_store import PsychologyVectorStore
from modules.rag_core import PsychologyRAG
from modules.safety_checker import SafetyChecker
from config.settings import settings
from typing import Dict, List, Optional, Union

# 避免超长问题刷屏，日志里只截取前 24 个字符
_Q_PREVIEW = 24


def _log_query_timings(question: str, age_group, timings: Dict, source_count: int):
    """统一打印一次 /api/query 的分阶段耗时（后台 print，便于定位瓶颈）。"""
    def ms(key):
        v = timings.get(key)
        return f"{v:.0f}ms" if v is not None else "-"

    q = (question or "").replace("\n", " ").strip()
    q_preview = (q[:_Q_PREVIEW] + "…") if len(q) > _Q_PREVIEW else q
    tid = threading.get_ident() % 100000  # 短线程号，便于并发时区分 A/B
    ag = age_group or "-"
    safety = ms("safety")
    embed = ms("embed")
    retrieve = ms("retrieve")
    llm = ms("llm")
    total = ms("total")
    print(
        f"[query][t:{tid}][{ag}] q=\"{q_preview}\" "
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

    def query(
        self,
        question: str,
        age_group: str = None,
        check_safety: bool = True,
        system_prompt_override: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> Dict:
        """查询系统"""
        result = {
            "question": question,
            "age_group": age_group,
            "answer": "",
            "sources": [],
            "safety_check": None,
        }

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # 安全检查
        if check_safety:
            ts = time.perf_counter()
            safety_result = self.safety_checker.check_and_respond(question)
            timings["safety"] = (time.perf_counter() - ts) * 1000
            result["safety_check"] = safety_result

            # 如果是高危情况，优先返回安全提示
            if safety_result.get("is_crisis") and safety_result["level"] == "high":
                result["answer"] = safety_result["safety_response"]["message"]
                result["is_crisis_response"] = True
                timings["total"] = (time.perf_counter() - t0) * 1000
                _log_query_timings(question, age_group, timings, 0)
                return result

        # 执行RAG查询
        rag_result = self.rag.run(
            question, age_group, system_prompt_override, prompt_id, timings=timings
        )
        result["answer"] = rag_result["answer"]
        result["sources"] = rag_result["sources"]
        result["is_crisis_response"] = False

        # 如果有中危信号，附加安全提示
        if check_safety and safety_result.get("is_crisis"):
            result["safety_note"] = safety_result["safety_response"]["message"]

        timings["total"] = (time.perf_counter() - t0) * 1000
        _log_query_timings(question, age_group, timings, len(result["sources"]))
        return result

# 创建全局实例
rag_system = PsychologyRAGSystem()

"""
心理学RAG系统主入口
整合所有模块提供统一的接口
"""
from modules.vector_store import PsychologyVectorStore
from modules.rag_core import PsychologyRAG
from modules.safety_checker import SafetyChecker
from config.settings import settings
from typing import Dict, List, Optional, Union


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
    ) -> Dict:
        """查询系统"""
        result = {
            "question": question,
            "age_group": age_group,
            "answer": "",
            "sources": [],
            "safety_check": None,
        }

        # 安全检查
        if check_safety:
            safety_result = self.safety_checker.check_and_respond(question)
            result["safety_check"] = safety_result

            # 如果是高危情况，优先返回安全提示
            if safety_result.get("is_crisis") and safety_result["level"] == "high":
                result["answer"] = safety_result["safety_response"]["message"]
                result["is_crisis_response"] = True
                return result

        # 执行RAG查询
        rag_result = self.rag.run(question, age_group)
        result["answer"] = rag_result["answer"]
        result["sources"] = rag_result["sources"]
        result["is_crisis_response"] = False

        # 如果有中危信号，附加安全提示
        if check_safety and safety_result.get("is_crisis"):
            result["safety_note"] = safety_result["safety_response"]["message"]

        return result

# 创建全局实例
rag_system = PsychologyRAGSystem()

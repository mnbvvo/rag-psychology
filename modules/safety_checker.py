"""
青少年心理安全检测模块
检测和预警潜在的心理危机情况
"""
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from config.settings import settings


class SafetyChecker:
    """心理安全检测器"""

    def __init__(self, keywords_file: str = None):
        self.keywords_file = keywords_file or settings.CRISIS_KEYWORDS_FILE
        self.crisis_data = self._load_keywords()
        self.enabled = settings.SAFETY_CHECK_ENABLED

    def _load_keywords(self) -> Dict:
        """加载危机关键词配置"""
        path = Path(self.keywords_file)
        if not path.exists():
            raise FileNotFoundError(f"危机关键词文件不存在: {self.keywords_file}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_text(self, text: str) -> Dict:
        """检测文本中的危机信号。

        每个关键词都在 config 中显式归属 high/medium/low 之一，
        直接遍历 crisis_levels 判断，避免「列了关键词却没配等级」导致被兜底判为 low。
        """
        if not self.enabled:
            return {"is_crisis": False, "level": "none", "keywords_found": []}

        text_lower = text.lower()
        keywords_found = []
        level_scores = {"high": 0, "medium": 0, "low": 0}

        # 按等级遍历所有关键词，命中即记录其所属等级
        for level in ("high", "medium", "low"):
            for keyword in self.crisis_data.get("crisis_levels", {}).get(level, []):
                if keyword and keyword in text_lower:
                    level_scores[level] += 1
                    keywords_found.append({"keyword": keyword, "level": level})

        # 确定最高危机等级
        crisis_level = self._determine_crisis_level(level_scores)
        is_crisis = crisis_level != "none"

        return {
            "is_crisis": is_crisis,
            "level": crisis_level,
            "keywords_found": keywords_found,
            "level_scores": level_scores,
            "hotlines": self.crisis_data.get("hotlines", {}),
            "response_strategy": self.crisis_data["emergency_response"].get(
                crisis_level, ""
            ),
        }

    def _determine_crisis_level(self, level_scores: Dict) -> str:
        """根据得分确定最终危机等级"""
        if level_scores["high"] > 0:
            return "high"
        elif level_scores["medium"] > 0:
            return "medium"
        elif level_scores["low"] > 0:
            return "low"
        return "none"

    def get_crisis_response(self, level: str) -> Dict:
        """获取危机应对建议"""
        hotlines = self.crisis_data.get("hotlines", {})

        response_templates = {
            "high": (
                "⚠️ **紧急安全警告**\n\n"
                "检测到可能存在严重安全风险的情况。\n"
                "请立即联系专业心理危机干预机构：\n\n"
                + "\n".join([f"• {name}: {number}" for name, number in hotlines.items()])
                + "\n\n**如果您或他人处于立即危险中，请立即拨打110或120。**"
            ),
            "medium": (
                "⚠️ **安全提示**\n\n"
                "我们注意到您可能正在经历困难时期。\n"
                "请记住，寻求帮助是勇敢的表现。\n\n"
                "专业支持资源：\n"
                + "\n".join([f"• {name}: {number}" for name, number in hotlines.items()])
            ),
            "low": (
                "💚 **关怀提示**\n\n"
                "如果您感到困扰，以下是一些可能有帮助的资源：\n\n"
                + "\n".join([f"• {name}: {number}" for name, number in hotlines.items()])
                + "\n\n记住，寻求帮助是坚强而非软弱的表现。"
            ),
        }

        return {
            "level": level,
            "message": response_templates.get(level, ""),
            "hotlines": hotlines,
            "should_intervene": level in ["high", "medium"],
        }

    def check_and_respond(self, text: str) -> Dict:
        """检测并返回安全建议"""
        check_result = self.check_text(text)

        if check_result["is_crisis"]:
            response = self.get_crisis_response(check_result["level"])
            check_result["safety_response"] = response

        return check_result

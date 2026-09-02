"""
青少年心理安全检测模块
检测和预警潜在的心理危机情况

分层检测：
- L0 关键词快检（check_text）：显式高危表达，毫秒级兜底；
  命中 high 但属「求助型提问」（孩子/朋友等称谓 + 求助动作词）时降级为
  medium，避免把家长/老师的求助误当成危机实施者直接拦截。
- L1 语义比对（semantic_check）：高危意图原型距离，识别隐喻/隐晦表达
  （如"想和房梁比赛"= 上吊），复用检索 embedding，无额外 API 成本。
- 综合判定（check_full）：取 L0/L1 较高等级，附带 detect_method 与
  confidence（语义距离），供 crisis_audit 审计追溯检测来源。
"""
import json
from typing import Callable, Dict, List, Optional
from pathlib import Path
from config.settings import settings


# 求助型提问标记：他人称谓 + 求助动作词同时出现，才视为"询问如何帮助他人"。
# 注意必须成对出现——"我想死"没有称谓不会降级，"孩子说想死怎么办"才会。
_HELP_SEEKING_PEOPLE = (
    "孩子", "女儿", "儿子", "朋友", "同学", "学生", "家长", "老师",
    "侄", "外甥", "表弟", "表妹", "闺蜜", "室友", "邻居",
)
_HELP_SEEKING_VERBS = (
    "怎么办", "如何", "怎么帮", "帮助", "帮帮", "求助", "预防",
    "避免", "识别", "应对", "处理", "引导", "劝", "开导", "安慰",
)

# 回答侧复查追加的安全提醒（追加而非替换：正常科普回答也常含"自伤/抑郁"等词，
# 整段替换会误伤家长/老师的正常求助；追加保证干预信息一定送达）
_ANSWER_SAFETY_NOTE = (
    "\n\n——\n安全提醒：如果以上话题与你或他人的安全有关，"
    "请不要独自承受，及时联系专业帮助："
    "全国心理援助热线 400-161-9995，或青少年心理咨询热线 12355。"
)


def _is_help_seeking(text: str) -> bool:
    """判断是否为求助型提问（他人口吻 + 求助动作），用于高危降级防误伤。"""
    if not text:
        return False
    has_people = any(p in text for p in _HELP_SEEKING_PEOPLE)
    has_verb = any(v in text for v in _HELP_SEEKING_VERBS)
    return has_people and has_verb


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

        # 求助型降级：命中 high 但明显是"询问如何帮助他人"（如"孩子有自伤倾向怎么办"），
        # 降为 medium —— 附关怀 + 转介即可，不拦截（避免误伤家长/老师这类核心用户）。
        downgraded = False
        if crisis_level == "high" and _is_help_seeking(text):
            crisis_level = "medium"
            downgraded = True

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
            "downgraded": downgraded,
        }

    def semantic_check(
        self,
        text: str,
        embed_query_fn: Optional[Callable] = None,
        embed_documents_fn: Optional[Callable] = None,
    ) -> Optional[Dict]:
        """L1 语义检测：高危意图原型距离（识别隐喻/隐晦表达）。

        需要传入 embedding 调用（带缓存），复用检索阶段的那一次 API 调用。
        返回 None 表示语义层未启用/不可用；命中返回含 distance 的结构化结果。
        """
        if not settings.SEMANTIC_CHECK_ENABLED or embed_query_fn is None:
            return None
        from modules.crisis_detector import get_crisis_detector

        return get_crisis_detector().detect(
            text,
            embed_query_fn=embed_query_fn,
            embed_documents_fn=embed_documents_fn,
        )

    def check_full(
        self,
        text: str,
        embed_query_fn: Optional[Callable] = None,
        embed_documents_fn: Optional[Callable] = None,
    ) -> Dict:
        """综合检测（L0 关键词 + L1 语义），供问答入口调用。

        等级取两者较高者；返回含 semantic 明细与 detect_method / confidence，
        便于 crisis_audit 审计追溯"这条是靠哪层拦下来的"。
        """
        l0 = self.check_text(text)
        l1 = self.semantic_check(text, embed_query_fn, embed_documents_fn)

        # 合并等级（高者优先）：semantic 命中 high/medium 时并入
        level = l0["level"]
        methods = ["keyword"] if l0.get("keywords_found") else []
        confidence = None
        if l1 and l1.get("is_crisis") and l1.get("level") != "none":
            l1_level = l1["level"]
            if l1_level in ("high", "medium") and l1_level == "high":
                level = "high"
            elif l1_level == "medium" and level == "none":
                level = "medium"
            elif l1_level == "medium" and level == "low":
                level = "medium"
            methods.append("semantic")
            confidence = l1.get("distance")

        result = dict(l0)
        # 求助型降级（无论 L0 还是 L1 判出的 high）："孩子说想死怎么办"是家长在求助，
        # 不是危机实施者 —— 附关怀 + 转介即可，不拦截。
        if level == "high" and _is_help_seeking(text):
            level = "medium"
        result["level"] = level
        result["is_crisis"] = level != "none"
        result["semantic"] = l1
        result["detect_method"] = "+".join(dict.fromkeys(methods)) or "none"
        result["confidence"] = confidence
        # response_strategy 跟随最终等级
        result["response_strategy"] = self.crisis_data["emergency_response"].get(level, "")
        # prepare() 的高危拦截/中低危关怀都读 safety_response.message
        if level != "none":
            result["safety_response"] = self.get_crisis_response(level)
        return result

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
        """获取危机应对建议（先共情、后指引：接住情绪再给热线，避免命令式冰冷感）"""
        hotlines = self.crisis_data.get("hotlines", {})
        hotline_lines = "\n".join([f"- {name}：{number}" for name, number in hotlines.items()])

        response_templates = {
            "high": (
                "我知道你现在一定非常难受，谢谢你愿意把这些话告诉我。\n\n"
                "你的感受是真实且重要的，你不需要一个人扛着这一切。"
                "请先停一停，身边有很多人愿意听你说，也有专业的心理工作者可以陪着你一起度过这段艰难的时间。\n\n"
                "现在就可以联系他们，好吗？\n\n"
                + hotline_lines
                + "\n\n如果你或身边的人正处在立即的危险之中，请马上拨打 110 或 120。"
                "你的安全，是这个世界上最重要的事。"
            ),
            "medium": (
                "看得出来，你最近一定很不好受。辛苦你了。\n\n"
                "能够把这些说出来，本身就是一件很勇敢的事。"
                "请记得，你不需要独自面对这一切——家人、朋友，还有专业的心理支持，都可以成为你的依靠。\n\n"
                "需要的时候，随时可以打给他们，会有人愿意听你说：\n\n"
                + hotline_lines
            ),
            "low": (
                "谢谢你愿意向我倾诉。每个人都会有感到低落和困惑的时候，这很正常，你并不孤单。\n\n"
                "如果你想找人说说话，下面这些渠道随时可以找到愿意倾听、支持你的人：\n\n"
                + hotline_lines
                + "\n\n请记得照顾好自己，你值得被温柔以待。"
            ),
        }

        return {
            "level": level,
            "message": response_templates.get(level, ""),
            "hotlines": hotlines,
            "should_intervene": level in ["high", "medium"],
        }

    def review_answer(self, answer: str):
        """回答侧安全复查（L0）：LLM 输出命中高危关键词时追加安全提醒。

        返回 (处理后的回答, 复查结果)；未命中返回 (原样, None)。
        注意：追加而非替换 —— LLM 正常科普回答也常含"自伤/抑郁"等敏感词，
        整段替换会误伤正常内容；追加提醒保证干预信息一定送达。
        """
        if not answer:
            return answer, None
        try:
            check = self.check_text(answer)
            if check.get("is_crisis") and check["level"] == "high":
                return answer + _ANSWER_SAFETY_NOTE, check
        except Exception:
            pass
        return answer, None

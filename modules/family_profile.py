"""家庭档案 → prompt 注入文本（隐私加工层）。

输入：db/crud_async.get_family_profile 返回的档案 dict（{user_nickname, birthday,
parent_role, children:[{child_id, child_nickname, child_birth_date, gender}]}，
日期为 YYYY-MM-DD 字符串或 None）。

输出：一段注入 system prompt 的中文「用户家庭档案」文本，或 ""（无档案/全空时
调用方跳过注入）。

隐私与提示工程规则（定稿于 2026-09-07，微信小程序通道）：
1. 原始生日日期（家长/孩子）一律不落 prompt——换算成「约 X 岁 / X 个月」等
   相对表述；生日字段只在服务端用于年龄换算。
2. 保留昵称/家长角色/孩子性别词，用于让陪伴语气自然贴近用户家庭情境。
3. 附克制性说明：不逐条复述、不主动质询档案，档案可能有出入、以用户当下叙述
   为准——避免模型每轮机械朗读档案或拿档案"教育"用户。
4. 输出长度受 settings.PROFILE_INJECT_MAX_CHARS 约束，超长截断（家庭多子女场景
   也不会挤占上下文预算）。

纯函数模块：不依赖 DB，便于单测。
"""
from datetime import date

from config.settings import settings


def _parse_date(value) -> date | None:
    """'YYYY-MM-DD' 字符串或 date → date；非法/空 → None。"""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _age_text(birth_value, ref: date | None = None) -> str:
    """把出生日期换算为自然的中文年龄表述；解析失败/未来日期返回 ""。

    <24 个月 → "约 N 个月大"（≥1 月）；≥24 个月 → "约 N 岁"。
    """
    birth = _parse_date(birth_value)
    if birth is None:
        return ""
    ref = ref or date.today()
    if birth > ref:
        return ""
    months = (ref.year - birth.year) * 12 + (ref.month - birth.month)
    if ref.day < birth.day:
        months -= 1
    if months < 0:
        months = 0
    if months < 1:
        return "约 1 个月大"
    if months < 24:
        return f"约 {months} 个月大"
    return f"约 {months // 12} 岁"


def _parent_line(profile: dict, ref: date | None) -> str:
    """用户（家长本人）行：昵称（角色）。昵称缺失时用角色；两者皆缺则返回 ""。

    家长自身生日只用于年龄换算；角色已足够表达身份时不再附年龄（对陪伴价值低）。
    行首固定「用户：」锚定档案主体是**对话对象**，防止模型把用户背景当成自己身份。
    """
    nickname = (profile.get("user_nickname") or "").strip()
    role = (profile.get("parent_role") or "").strip()
    if nickname and role:
        return f"- 用户：{nickname}（{role}）"
    if nickname:
        age = _age_text(profile.get("birthday"), ref)
        return f"- 用户：{nickname}（{age}）" if age else f"- 用户：{nickname}"
    if role:
        return f"- 用户（{role}）"
    return ""


def _child_line(child: dict, ref: date | None) -> str:
    """用户的孩子行：昵称（性别，年龄）；性别与年龄都缺失时只写昵称。"""
    nickname = (child.get("child_nickname") or "").strip() or "孩子"
    gender = (child.get("gender") or "").strip()
    age = _age_text(child.get("child_birth_date"), ref)
    detail = "，".join(x for x in (gender, age) if x)
    return f"- 用户的孩子：{nickname}（{detail}）" if detail else f"- 用户的孩子：{nickname}"


def format_profile_for_prompt(profile: dict | None, ref: date | None = None) -> str:
    """加工档案 → 注入文本；无档案 / 无可写字段 → ""（调用方据此跳过注入）。"""
    if not profile:
        return ""
    ref = ref or date.today()
    lines = [_parent_line(profile, ref)]
    lines += [_child_line(c, ref) for c in (profile.get("children") or [])]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    text = (
        "以下是当前对话用户（孩子家长本人）的家庭背景，仅作参考。"
        "注意：这是用户的情况，不是你的身份——你仍是心桥，陪伴这位家长"
        "的家庭心理健康助手。不要逐条复述档案，也不要主动质询档案内容；"
        "档案可能与现实有出入，以用户当下的叙述为准：\n"
        + "\n".join(lines)
    )
    cap = settings.PROFILE_INJECT_MAX_CHARS
    if cap > 0 and len(text) > cap:
        text = text[:cap].rstrip() + "……"
    return text

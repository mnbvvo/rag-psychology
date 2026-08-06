"""会话 / 消息 / 危机审计的持久化操作。

与 Chroma 向量库互补：Chroma 负责语义检索，这里负责结构化留痕
（多轮对话可被服务端审计、危机事件可追溯——心理类产品的合规硬伤）。
"""
import json
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import select, desc, func

from . import SessionLocal
from .models import Session, Message, CrisisAudit, Prompt, CompareHistory


@contextmanager
def get_db():
    """DB Session 上下文管理器：退出时自动提交；异常时回滚；始终关闭。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# 未命名会话的占位标题集合：命中即视为"还没取名"，首次提问时自动命名。
# 注意历史遗留：后端旧默认名是"新会话"，前端创建会话传的默认名是"新的对话"，
# 两个都要认，否则自动命名条件永远不成立（曾导致标题一直停留在占位名）。
_UNNAMED_TITLES = ("", "新会话", "新的对话")


def _auto_title(db, session_id: str, fallback: str) -> str:
    """为未命名会话生成标题：优先取该会话最早的一条用户消息，否则用 fallback。

    注意 SessionLocal 是 autoflush=False，新追加的 human 消息不会混入查询，
    因此这里拿到的始终是会话里最早那条用户问题（对首次提问的会话则回退到 fallback）。
    """
    first = (
        db.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.role == "human")
            .order_by(Message.id.asc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    text = (first.content if first and first.content else fallback) or "新对话"
    return " ".join(text.split())[:30]


def ensure_session(db, session_id: str, title: str | None = None) -> Session:
    """确保会话行存在（不存在则按 id 创建）。"""
    sess = db.get(Session, session_id)
    if sess is None:
        sess = Session(id=session_id, title=(title or "新会话")[:255])
        db.add(sess)
        db.flush()
    return sess


def _recent_messages(db, session_id: str, n: int = 2) -> list[Message]:
    return (
        db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.id))
            .limit(n)
        )
        .scalars()
        .all()
    )


def append_turn(
    db,
    session_id: str,
    user_text: str,
    ai_text: str,
    title: str | None = None,
) -> None:
    """追加一轮对话（用户提问 + AI 回答）。

    轻量幂等：若最近两条消息恰好等于本次内容（通常是重复提交/重试），
    则跳过，避免同一轮在 DB 里出现重复。
    """
    ensure_session(db, session_id, title=title)
    recent = _recent_messages(db, session_id, n=2)
    if len(recent) == 2:
        last, prev = recent[0], recent[1]  # 倒序：last 为最新
        if (
            last.role == "ai"
            and prev.role == "human"
            and last.content == ai_text
            and prev.content == user_text
        ):
            return
    db.add(Message(session_id=session_id, role="human", content=user_text))
    db.add(Message(session_id=session_id, role="ai", content=ai_text))
    sess = db.get(Session, session_id)
    if sess is not None:
        sess.updated_at = datetime.now(timezone.utc)
        # 未命名会话：首次提问时用最早一条用户消息自动命名（不再停留在"新的对话"）
        if title and sess.title in _UNNAMED_TITLES:
            sess.title = _auto_title(db, session_id, title)


def rename_unnamed_sessions(db) -> int:
    """把历史遗留的占位标题会话（"新的对话"/"新会话"/空）按最早一条用户消息自动命名。

    服务启动时调用一次，幂等：只处理仍未命名的会话，已命名的保持不变。
    返回本次改名的会话数。
    """
    renamed = 0
    rows = db.execute(select(Session)).scalars().all()
    for sess in rows:
        if sess.title in _UNNAMED_TITLES:
            new_title = _auto_title(db, sess.id, sess.title)
            if new_title and new_title != sess.title:
                sess.title = new_title
                renamed += 1
    return renamed


def log_crisis(
    db,
    session_id: str | None,
    level: str,
    keywords_found,
    question: str,
    response: str | None,
    is_crisis_response: bool = False,
) -> None:
    """记录一次危机命中（合规审计，可追溯）。"""
    try:
        kw_text = json.dumps(keywords_found, ensure_ascii=False) if keywords_found else None
    except (TypeError, ValueError):
        kw_text = None
    db.add(
        CrisisAudit(
            session_id=session_id,
            crisis_level=level,
            keywords_found=kw_text,
            question=question,
            response=response,
            is_crisis_response=bool(is_crisis_response),
        )
    )


# ---------------- 提示词库（SQLite 持久化，替代原 JSON 文件） ----------------
def count_prompts(db) -> int:
    return db.execute(select(func.count()).select_from(Prompt)).scalar() or 0


def list_prompts(db) -> list[Prompt]:
    return db.execute(select(Prompt).order_by(Prompt.created_at)).scalars().all()


def get_active_prompt_row(db) -> Prompt | None:
    p = db.execute(select(Prompt).where(Prompt.is_active == True)).scalars().first()
    if p is None:
        p = db.execute(select(Prompt).order_by(Prompt.created_at)).scalars().first()
    return p


# ---------------- 对比历史（用户生成的对比记录，持久化到 SQLite） ----------------
def add_compare_history(db, input_text: str, result_a: str | None, result_b: str | None) -> CompareHistory:
    r = CompareHistory(input=input_text, result_a=result_a, result_b=result_b)
    db.add(r)
    db.flush()
    return r


def list_compare_history(db, limit: int = 50) -> list[CompareHistory]:
    return (
        db.execute(select(CompareHistory).order_by(desc(CompareHistory.created_at)).limit(limit))
        .scalars()
        .all()
    )


def get_compare_history(db, item_id: int) -> CompareHistory | None:
    return db.get(CompareHistory, item_id)

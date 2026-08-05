"""
系统提示词可编辑存储（提示词库）

把原本硬编码在 ``modules/rag_core.py`` 里的系统提示词外置，并持久化到
SQLite（``prompts`` 表），支持前端读取 / 修改 / 还原，并与 RAG 实时联动。

与 Chroma 向量库分工：Chroma 管语义检索，SQLite 管结构化数据
（提示词库 / 会话 / 消息 / 危机审计 / 对比历史），成为前端唯一数据源。

结构（与前端约定一致）：
{
  "version": 1,
  "updated_at": "...",
  "activeId": "default",
  "prompts": [
    {"id": "default", "name": "默认青少年心理", "content": "..."}
  ]
}

组装规则：
    最终系统提示词 = active prompt content + 参考资料({context})
"""
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import select

from db import crud
from db.models import Prompt

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_FILE = _CONFIG_DIR / "system_prompt.default.json"
_CURRENT_FILE = _CONFIG_DIR / "system_prompt.json"  # 旧文件：首次启动用于迁移，之后不再是数据源

# 出厂默认提示词库（首次运行会写入 default 文件，并作为 reset 的基线）
DEFAULT_CONTENT = (
    "你是专业的青少年心理咨询师。\n\n"
    "【重要原则】\n"
    "1. 仅基于下方提供的参考资料回答，不要编造信息。\n"
    "2. 如果资料不足以回答问题，请诚实地说“我目前没有足够的信息来回答这个问题”。\n"
    "3. 始终以温暖、专业、非评判的态度回应。\n"
    "4. 如果涉及安全或危机信号，优先提供危机干预资源与求助渠道。\n"
    "5. 在回答末尾标注信息来源编号（如 [1][2]）。"
)

DEFAULT_CONFIG: Dict = {
    "version": 1,
    "updated_at": None,
    "activeId": "default",
    "prompts": [
        {
            "id": "default",
            "name": "默认青少年心理",
            "content": DEFAULT_CONTENT,
        }
    ],
}

# 低相关时的补充说明（由代码动态插入，不进入可编辑文本）
LOW_RELEVANCE_NOTE = (
    "\n【重要】本次未检索到足够相关的参考资料。"
    "若无法直接、有据地回答，请明确说明“我目前没有足够的信息来回答这个问题”，不要编造内容。"
)

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _ensure_default_file() -> None:
    """确保出厂默认文件存在（首次运行从常量写入），作为 reset / 首次 seed 的基线。"""
    if not _DEFAULT_FILE.exists():
        _DEFAULT_FILE.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _migrate_single_prompt(data: Optional[Dict]) -> Dict:
    """兼容旧版单条 system_prompt 结构，自动迁移为 prompts[] 结构。"""
    data = data or {}
    if "system_prompt" in data and isinstance(data["system_prompt"], str):
        old_text = data["system_prompt"]
        prompts = DEFAULT_CONFIG["prompts"].copy()
        if old_text != DEFAULT_CONFIG["prompts"][0]["content"]:
            prompts = [
                {
                    "id": "migrated-" + uuid.uuid4().hex[:8],
                    "name": "已迁移提示词",
                    "content": old_text,
                },
                *prompts,
            ]
        return {
            "version": DEFAULT_CONFIG["version"],
            "updated_at": data.get("updated_at"),
            "activeId": prompts[0]["id"],
            "prompts": prompts,
        }
    return _merge_with_default(data)


def _merge_with_default(data: Optional[Dict]) -> Dict:
    """用默认值兜底缺失字段，保证结构完整。"""
    data = data or {}
    if "prompts" not in data or not isinstance(data.get("prompts"), list):
        return _migrate_single_prompt(data)

    prompts = []
    for p in data["prompts"]:
        if not isinstance(p, dict):
            continue
        pid = p.get("id") or uuid.uuid4().hex[:8]
        prompts.append({
            "id": pid,
            "name": p.get("name", "未命名提示词") or "未命名提示词",
            "content": p.get("content", "") or "",
        })

    if not prompts:
        prompts = [{
            "id": "default",
            "name": DEFAULT_CONFIG["prompts"][0]["name"],
            "content": DEFAULT_CONFIG["prompts"][0]["content"],
        }]

    active_id = data.get("activeId")
    if not active_id or not any(p["id"] == active_id for p in prompts):
        active_id = prompts[0]["id"]

    return {
        "version": DEFAULT_CONFIG["version"],
        "updated_at": data.get("updated_at"),
        "activeId": active_id,
        "prompts": prompts,
    }


# ---------------- DB 读写辅助 ----------------
def _row_to_prompt(p: Prompt) -> Dict:
    return {"id": p.id, "name": p.name, "content": p.content}


def _config_from_rows(rows: List[Prompt], active_id: str) -> Dict:
    return {
        "version": 1,
        "updated_at": _now_iso(),
        "activeId": active_id,
        "prompts": [_row_to_prompt(p) for p in rows],
    }


def ensure_prompts_seeded() -> None:
    """首次启动：若 prompts 表为空，则从已有 system_prompt.json 迁移，否则用出厂默认 seed。"""
    _ensure_default_file()
    with _lock:
        with crud.get_db() as db:
            if crud.count_prompts(db) > 0:
                return
            src = _CURRENT_FILE if _CURRENT_FILE.exists() else _DEFAULT_FILE
            try:
                data = _merge_with_default(json.loads(src.read_text(encoding="utf-8")))
            except Exception:
                data = _merge_with_default(json.loads(_DEFAULT_FILE.read_text(encoding="utf-8")))
            active_id = data.get("activeId")
            for i, p in enumerate(data["prompts"]):
                is_active = (p.get("id") == active_id) or (i == 0 and not active_id)
                db.add(Prompt(
                    id=p["id"],
                    name=p.get("name", "未命名提示词"),
                    content=p.get("content", ""),
                    is_active=bool(is_active),
                ))


def get_prompt_config() -> Dict:
    """返回 {current, default}，供前端展示完整提示词库。current 来自 SQLite，default 来自出厂文件。"""
    with _lock:
        with crud.get_db() as db:
            rows = crud.list_prompts(db)
            active = crud.get_active_prompt_row(db)
            active_id = active.id if active else (rows[0].id if rows else "")
            current = _config_from_rows(rows, active_id)
    _ensure_default_file()
    default = json.loads(_DEFAULT_FILE.read_text(encoding="utf-8"))
    return {"current": current, "default": default}


def update_prompt_config(partial: Dict) -> Dict:
    """更新提示词库（SQLite 持久化）。

    支持：
    - partial.prompts: 完整替换 prompts[]
    - partial.activeId: 设置激活提示词
    - partial.add: {name, content} 新增一条提示词（自动设为激活）
    - partial.update: {id, name, content} 更新某条
    - partial.deleteId: 删除指定 id
    """
    with _lock:
        with crud.get_db() as db:
            rows = crud.list_prompts(db)
            by_id = {p.id: p for p in rows}

            if "prompts" in partial and isinstance(partial["prompts"], list):
                new_ids: set[str] = set()
                for item in partial["prompts"]:
                    pid = item.get("id") or uuid.uuid4().hex[:8]
                    new_ids.add(pid)
                    p = by_id.get(pid)
                    if p is None:
                        p = Prompt(id=pid, name=item.get("name", "未命名提示词"), content=item.get("content", ""))
                        db.add(p)
                        by_id[pid] = p
                    else:
                        if item.get("name") is not None:
                            p.name = item["name"]
                        if item.get("content") is not None:
                            p.content = item["content"]
                # 删除不在新列表中的旧提示词
                for pid in list(by_id.keys()):
                    if pid not in new_ids:
                        db.delete(by_id[pid])
                        del by_id[pid]

            if partial.get("add") and isinstance(partial["add"], dict):
                new_id = uuid.uuid4().hex[:8]
                for other in crud.list_prompts(db):
                    other.is_active = False
                p = Prompt(id=new_id, name=partial["add"].get("name", "新提示词"),
                           content=partial["add"].get("content", ""), is_active=True)
                db.add(p)
                by_id[new_id] = p

            if partial.get("update") and isinstance(partial["update"], dict):
                upd = partial["update"]
                pid = upd.get("id")
                p = by_id.get(pid) or db.get(Prompt, pid)
                if p:
                    if upd.get("name") is not None:
                        p.name = upd["name"]
                    if upd.get("content") is not None:
                        p.content = upd["content"]

            if partial.get("deleteId"):
                pid = partial["deleteId"]
                p = by_id.get(pid) or db.get(Prompt, pid)
                if p:
                    was_active = p.is_active
                    db.delete(p)
                    if pid in by_id:
                        del by_id[pid]
                    if was_active:
                        nxt = crud.list_prompts(db)
                        if nxt:
                            nxt[0].is_active = True

            if partial.get("activeId"):
                aid = partial["activeId"]
                for p in crud.list_prompts(db):
                    p.is_active = (p.id == aid)

            db.flush()
            rows = crud.list_prompts(db)
            active = crud.get_active_prompt_row(db)
            active_id = active.id if active else (rows[0].id if rows else "")
            return _config_from_rows(rows, active_id)


def reset_prompt_config() -> Dict:
    """还原为出厂默认（清空 prompts 表，重新从出厂文件 seed）。"""
    with _lock:
        with crud.get_db() as db:
            for p in crud.list_prompts(db):
                db.delete(p)
            _ensure_default_file()
            data = _merge_with_default(json.loads(_DEFAULT_FILE.read_text(encoding="utf-8")))
            active_id = data.get("activeId")
            for i, p in enumerate(data["prompts"]):
                is_active = (p.get("id") == active_id) or (i == 0 and not active_id)
                db.add(Prompt(
                    id=p["id"],
                    name=p.get("name", "未命名提示词"),
                    content=p.get("content", ""),
                    is_active=bool(is_active),
                ))
            db.flush()
            rows = crud.list_prompts(db)
            active = crud.get_active_prompt_row(db)
            active_id = active.id if active else (rows[0].id if rows else "")
            return _config_from_rows(rows, active_id)


def get_active_prompt() -> Dict:
    """返回当前激活的提示词对象。"""
    with _lock:
        with crud.get_db() as db:
            p = crud.get_active_prompt_row(db)
            return _row_to_prompt(p) if p else {"id": "", "name": "", "content": ""}


def get_prompt_by_id(prompt_id: str) -> Optional[Dict]:
    """按 id 查找提示词。"""
    with _lock:
        with crud.get_db() as db:
            p = db.get(Prompt, prompt_id)
            return _row_to_prompt(p) if p else None


def build_system_prompt(
    age_group: str = "teen",
    context_text: str = "",
    low_relevance: bool = False,
    system_prompt_override: Optional[str] = None,
    prompt_id: Optional[str] = None,
) -> str:
    """组装最终系统提示词：active prompt content + 参考资料({context})。

    - system_prompt_override: 若传入，直接用该字符串作为基础文本（前端不保存测试）。
    - prompt_id: 指定使用库中某条提示词；若未指定且未传 override，则使用 activeId。
    """
    if system_prompt_override is not None:
        base = system_prompt_override
    elif prompt_id:
        p = get_prompt_by_id(prompt_id)
        base = p["content"] if p else ""
    else:
        base = get_active_prompt()["content"]

    ref = "参考资料:\n" + (context_text or "{context}")
    if low_relevance:
        ref = LOW_RELEVANCE_NOTE + "\n\n" + ref
    return (base + "\n\n" + ref) if base else ref

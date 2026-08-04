"""
系统提示词可编辑存储（提示词库）

把原本硬编码在 ``modules/rag_core.py`` 里的系统提示词外置为 JSON 文件，
支持前端读取 / 修改 / 还原，并与后端文件实时同步。

文件约定：
- ``config/system_prompt.default.json``：出厂默认提示词库（不可变参考）。
- ``config/system_prompt.json``：当前可编辑提示词库（运行时生成，可被前端修改）。

结构：
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

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_CURRENT_FILE = _CONFIG_DIR / "system_prompt.json"
_DEFAULT_FILE = _CONFIG_DIR / "system_prompt.default.json"

# 出厂默认提示词库（首次运行会写入 default 文件）
DEFAULT_CONFIG: Dict = {
    "version": 1,
    "updated_at": None,
    "activeId": "default",
    "prompts": [
        {
            "id": "default",
            "name": "默认青少年心理",
            "content": (
                "你是专业的青少年心理咨询师。\n\n"
                "【重要原则】\n"
                "1. 仅基于下方提供的参考资料回答，不要编造信息。\n"
                "2. 如果资料不足以回答问题，请诚实地说“我目前没有足够的信息来回答这个问题”。\n"
                "3. 始终以温暖、专业、非评判的态度回应。\n"
                "4. 如果涉及安全或危机信号，优先提供危机干预资源与求助渠道。\n"
                "5. 在回答末尾标注信息来源编号（如 [1][2]）。"
            ),
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
    """确保出厂默认文件存在（首次运行从常量写入）。"""
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
        # 把旧提示词作为第一条可编辑提示词
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


def _load_current() -> Dict:
    """读取当前可编辑配置；若不存在则用默认值初始化。"""
    _ensure_default_file()
    if not _CURRENT_FILE.exists():
        _save_current(DEFAULT_CONFIG, stamp=False)
    text = _CURRENT_FILE.read_text(encoding="utf-8")
    return _merge_with_default(json.loads(text))


def _save_current(data: Dict, stamp: bool = True) -> Dict:
    payload = _merge_with_default(data)
    if stamp:
        payload["updated_at"] = _now_iso()
    _CURRENT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _get_current() -> Dict:
    with _lock:
        return _load_current()


def get_prompt_config() -> Dict:
    """返回 {current, default}，供前端展示完整提示词库。"""
    with _lock:
        current = _load_current()
        default_text = _DEFAULT_FILE.read_text(encoding="utf-8")
        default = json.loads(default_text)
        return {"current": current, "default": default}


def update_prompt_config(partial: Dict) -> Dict:
    """更新提示词库。

    支持：
    - partial.prompts: 完整替换 prompts[]
    - partial.activeId: 设置激活提示词
    - partial.add: {name, content} 新增一条提示词
    - partial.update: {id, name, content} 更新某条
    - partial.deleteId: 删除指定 id
    """
    with _lock:
        current = _load_current()

        if "prompts" in partial and isinstance(partial["prompts"], list):
            current["prompts"] = partial["prompts"]

        if "activeId" in partial and isinstance(partial["activeId"], str):
            current["activeId"] = partial["activeId"]

        if partial.get("add") and isinstance(partial["add"], dict):
            new_prompt = {
                "id": uuid.uuid4().hex[:8],
                "name": partial["add"].get("name", "新提示词"),
                "content": partial["add"].get("content", ""),
            }
            current["prompts"].append(new_prompt)
            current["activeId"] = new_prompt["id"]

        if partial.get("update") and isinstance(partial["update"], dict):
            upd = partial["update"]
            for p in current["prompts"]:
                if p["id"] == upd.get("id"):
                    if "name" in upd:
                        p["name"] = upd["name"]
                    if "content" in upd:
                        p["content"] = upd["content"]
                    break

        if partial.get("deleteId"):
            before = len(current["prompts"])
            current["prompts"] = [p for p in current["prompts"] if p["id"] != partial["deleteId"]]
            if len(current["prompts"]) < before:
                if current["activeId"] == partial["deleteId"]:
                    current["activeId"] = current["prompts"][0]["id"] if current["prompts"] else ""

        return _save_current(current, stamp=True)


def reset_prompt_config() -> Dict:
    """还原为出厂默认。"""
    with _lock:
        return _save_current(DEFAULT_CONFIG, stamp=True)


def get_active_prompt() -> Dict:
    """返回当前激活的提示词对象。"""
    cfg = _get_current()
    active_id = cfg.get("activeId")
    for p in cfg["prompts"]:
        if p["id"] == active_id:
            return p
    return cfg["prompts"][0] if cfg["prompts"] else {"id": "", "name": "", "content": ""}


def get_prompt_by_id(prompt_id: str) -> Optional[Dict]:
    """按 id 查找提示词。"""
    cfg = _get_current()
    for p in cfg["prompts"]:
        if p["id"] == prompt_id:
            return p
    return None


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

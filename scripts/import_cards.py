"""将知识卡片 JSONL 离线写入本地 Chroma 向量库。"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from modules.vector_store import PsychologyVectorStore


def join_values(values: Any) -> str:
    if not values:
        return ""
    return "、".join(str(value) for value in values)


# 年龄分桶与导入写入的 age_group 元数据对应；rag_core 据此做过滤。
# 顺序即优先级：必须先把更长的标签（“青少年”）放在含其子串的（“少年”）之前。
_AGE_BUCKET_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("child", ("婴儿", "幼儿", "儿童", "0-2", "3-6")),
    ("teen", ("青少年",)),
    ("early_teen", ("少年", "中小学", "小学生", "初中生", "中学生")),
    ("late_teen", ("青年", "高中生", "职高", "大学")),
]


def normalize_age_group(age_stages: Any) -> str:
    """把卡片里的 age_stages（中文标签列表）映射到分桶，无法识别返回空串。"""
    text = join_values(age_stages)
    if not text:
        return ""
    for bucket, keywords in _AGE_BUCKET_RULES:
        if any(keyword in text for keyword in keywords):
            return bucket
    return ""


def format_actions(actions: list[dict[str, Any]]) -> str:
    lines = []
    for index, action in enumerate(actions or [], start=1):
        details = [
            f"步骤：{action.get('step', '')}",
            f"频率：{action.get('frequency', '')}",
            f"时长：{action.get('duration', '')}",
            f"观察：{action.get('observe', '')}",
        ]
        if action.get("safety_note"):
            details.append(f"安全提示：{action['safety_note']}")
        lines.append(f"{index}. " + "；".join(item for item in details if item.split("：", 1)[1]))
    return "\n".join(lines)


def to_document(record: dict[str, Any], source_file: Path) -> Document:
    card = record["card_json"]
    content = "\n".join([
        f"标题：{card.get('title', '')}",
        f"适用对象：{join_values(card.get('audiences'))}",
        f"年龄阶段：{join_values(card.get('age_stages'))}",
        f"领域：{join_values(card.get('domains'))}",
        f"场景：{card.get('scenario', '')}",
        f"适用条件：{card.get('applicable_conditions', '')}",
        f"澄清问题：{join_values(card.get('clarifying_questions'))}",
        f"可能解释：{join_values(card.get('possible_explanations'))}",
        f"建议行动：\n{format_actions(card.get('actions', []))}",
        f"不适用情况：{join_values(card.get('do_not_use_when'))}",
        f"转介条件：{join_values(card.get('referral_conditions'))}",
        f"风险等级：{card.get('risk_level', '')}",
        f"证据等级：{card.get('evidence_level', '')}",
    ])
    metadata = {
        "card_id": record["card_id"],
        "source_id": record.get("source_id", ""),
        "chunk_id": record.get("chunk_id", ""),
        "title": card.get("title", ""),
        "domains": join_values(card.get("domains")),
        "audiences": join_values(card.get("audiences")),
        "age_stages": join_values(card.get("age_stages")),
        "risk_level": card.get("risk_level", ""),
        "evidence_level": card.get("evidence_level", ""),
        "review_status": record.get("review_status", ""),
        "age_group": normalize_age_group(card.get("age_stages")),
        "source": str(source_file),
        "filename": source_file.name,
    }
    return Document(page_content=content, metadata=metadata)


def load_documents(path: Path, review_status: str | None) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if review_status and record.get("review_status") != review_status:
                continue
            card_id = record.get("card_id")
            if not card_id:
                raise ValueError(f"第 {line_number} 行缺少 card_id")
            if card_id in seen_ids:
                raise ValueError(f"发现重复 card_id: {card_id}")
            seen_ids.add(card_id)
            documents.append(to_document(record, path))
            ids.append(card_id)
    return documents, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="离线导入知识卡片到本地 Chroma")
    parser.add_argument("cards_path", type=Path, help="output_cards.jsonl 的绝对路径")
    parser.add_argument("--review-status", help="只导入指定审核状态，例如 approved")
    parser.add_argument("--reset", action="store_true", help="先删除当前 Chroma 集合")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    if not args.cards_path.is_file():
        raise FileNotFoundError(f"找不到知识卡片文件: {args.cards_path}")
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")

    documents, ids = load_documents(args.cards_path, args.review_status)
    if not documents:
        raise ValueError("没有符合条件的知识卡片可导入")

    store = PsychologyVectorStore()
    if args.reset:
        store.delete_collection()

    for offset in range(0, len(documents), args.batch_size):
        batch_documents = documents[offset:offset + args.batch_size]
        batch_ids = ids[offset:offset + args.batch_size]
        store.add_documents(batch_documents, ids=batch_ids)
        print(f"已写入 {min(offset + len(batch_documents), len(documents))}/{len(documents)} 张卡片")

    print(f"导入完成：{len(documents)} 张卡片，向量库位于 {store.persist_directory}")


if __name__ == "__main__":
    #python scripts/import_cards.py "C:\Users\Thunderobot\Documents\paper\knowledge_base_automation\out\output_cards.jsonl" --reset
    main()

"""检索质量诊断：测试「问题 × 知识库」的相关性分数。

原理：把知识库全量文档直接交给重排器打分（不经过 top10/top5 召回截断），
统计每个查询的最高分与分数分布 —— 用于定位"检索相似度低"的根源：

- 该主题最高分高（>0.5）：重排模型正常，问题可能在召回/阈值环节
  （调 FETCH_K / HYBRID_KEYWORD_K / RERANK_MIN_SCORE）
- 该主题最高分低（<0.3）：知识库缺该主题内容，补知识库卡片才是正解

用法：
    python scripts/diagnose_retrieval.py                        # 交互模式：直接在控制台输入问题（空行/exit 退出）
    python scripts/diagnose_retrieval.py --builtin              # 跑内置默认查询集
    python scripts/diagnose_retrieval.py --queries "失眠|考试焦虑|抑郁"
    python scripts/diagnose_retrieval.py --file queries.txt     # 每行一个查询

统计阈值（≥ 阈值卡数）与明细显示条数在文件顶部 THRESHOLD / TOP_N 直接调整。
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.hybrid_search import _docs_from_vectorstore
from modules.reranker import get_reranker

THRESHOLD = 0.2   # 统计"≥ 阈值的卡片数"用的分数下限
TOP_N = 5         # 每个查询显示的最高分卡片条数（0=不显示明细）

DEFAULT_QUERIES = [
    "如何培养孩子的专注力？",            # 对照：库内强主题（预期高分）
    "家长怎么和孩子好好沟通？",          # 对照：库内强主题（预期高分）
    "孩子厌学不想上学怎么办？",
    "孩子最近总是睡不着，作为家长该怎么帮他？",
    "孩子考试前焦虑紧张怎么缓解？",
    "如何识别孩子是否有抑郁倾向？",
    "孩子在学校被同学霸凌怎么办？",
    "孩子沉迷手机游戏怎么办？",
    "孩子总是发脾气，家长怎么办？",
    "怎么帮孩子建立自信？",
]


def _judge(top_score: float) -> str:
    """根据最高分给出覆盖判断（分数阈值随 bge-reranker 实测标定）。"""
    if top_score >= 0.7:
        return "库里强相关 → 覆盖充足"
    if top_score >= 0.5:
        return "有相关卡"
    if top_score >= 0.3:
        return "间接相关（覆盖偏弱）"
    return "库里基本无相关内容（需补库）"


def _diagnose_one(model, docs, query: str, index: int, total: int):
    """诊断单个查询：全库重排打分，返回汇总行并打印明细。"""
    print(f"[{index}/{total}] 诊断中：{query[:24]}…", flush=True)
    scores = model.predict(
        [[query, d.page_content] for d in docs],
        batch_size=32,
        show_progress_bar=False,
    )
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_score = ranked[0][1] if ranked else 0.0
    over = sum(1 for _, s in ranked if s >= THRESHOLD)

    if TOP_N > 0:
        print(f"    top{TOP_N}:")
        for d, s in ranked[: TOP_N]:
            print(f"      {s:.3f} [{d.metadata.get('card_id')}] {d.metadata.get('title', '')[:30]}")

    return (query, top_score, over, _judge(top_score))


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量诊断：问题 × 知识库相关性分数")
    parser.add_argument("--builtin", action="store_true", help="跑内置默认查询集（默认无参数为交互模式）")
    parser.add_argument("--queries", type=str, default="", help="用 | 分隔的自定义查询列表")
    parser.add_argument("--file", type=str, default="", help="查询文件路径，每行一个查询")
    args = parser.parse_args()

    interactive = not args.builtin and not args.queries and not args.file
    queries: list[str] = []
    if args.queries:
        queries = [q.strip() for q in args.queries.split("|") if q.strip()]
    elif args.file:
        p = Path(args.file)
        if not p.is_file():
            raise FileNotFoundError(f"查询文件不存在: {args.file}")
        queries = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.builtin:
        queries = DEFAULT_QUERIES

    print("加载知识库全量文档…", flush=True)
    t0 = time.perf_counter()
    docs = _docs_from_vectorstore()
    print(f"共 {len(docs)} 张卡片（{time.perf_counter() - t0:.0f}ms），加载重排模型（首次较慢）…", flush=True)
    model = get_reranker()._load()

    rows = []
    if interactive:
        print("交互模式：直接输入问题开始诊断，空行或输入 exit 退出", flush=True)
        total_marker = "?"
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() in ("exit", "quit", "q"):
                break
            rows.append(_diagnose_one(model, docs, line, 1, total_marker))
    else:
        for i, q in enumerate(queries, 1):
            rows.append(_diagnose_one(model, docs, q, i, len(queries)))

    # 汇总表（Markdown 格式，便于复制）
    print("\n" + "=" * 92)
    print(f"诊断汇总（阈值 {THRESHOLD}）")
    print("| 查询 | 最高分 | ≥阈值卡数 | 说明 |")
    print("|---|---|---|---|")
    for q, top_score, over, judge in rows:
        print(f"| {q[:26]} | {top_score:.3f} | {over} | {judge} |")
    print("=" * 92)
    print("判读：最高分高(>0.5)→模型正常，问题在召回/阈值参数；最高分低(<0.3)→知识库缺该主题，优先补卡。")


if __name__ == "__main__":
    main()

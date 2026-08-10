"""
语义危机检测器（L1：高危意图原型距离）

原理：隐喻表达无限，但危险意图有限。把种子集（标准意图句 + 隐喻变体，
见 config/high_risk_intents.json）embed 后按意图簇聚合成「原型向量」；
用户问题与各原型算余弦距离 ——
  距离 ≤ 拦截半径（CRISIS_INTERCEPT_DIST）→ 高危，拦截；
  距离 ≤ 灰区半径（CRISIS_GRAY_DIST）→ 疑似，附关怀 + 转介（不拦截）；
  远离所有簇 → 放行。

泛化来自 embedding 的稠密语义表示：没收录过的隐喻（如"想变成星星"）
在语义空间里天然落在"自杀意图"原型附近，不需要逐个枚举。

工程要点：
- 复用检索阶段的 embedding（TimedOpenAIEmbeddings 已带 LRU 缓存），
  同一问题的向量在检测器与检索之间只调用一次 API，零额外成本。
- 原型向量缓存到 data/crisis_prototypes.json；种子文件 mtime 变更自动重建。
- 任何异常（embedding 失败 / 种子缺失）都返回 None，由上游回退到纯关键词，
  不影响问答链路可用性。
"""
import json
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from config.settings import settings


def _cosine_dist(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return 1.0 - float(np.dot(a, b) / denom)


class SemanticCrisisDetector:
    """基于高危意图原型向量的语义危机检测器。"""

    def __init__(self):
        self._prototypes: Dict[str, Dict] = {}  # cluster -> {vectors, level, label}
        self._loaded = False
        self._build_failed = False  # 构建失败置位，避免每次请求都重试批量 embed
        self._lock = threading.Lock()
        self._seed_file = Path(settings.CRISIS_SEED_FILE)
        self._cache_file = Path(settings.CRISIS_PROTOTYPE_CACHE)

    # ---------------- 原型构建 ----------------
    def _seed_mtime(self) -> float:
        try:
            return self._seed_file.stat().st_mtime
        except OSError:
            return 0.0

    def _load_cache(self) -> bool:
        """从缓存文件加载锚点集合（种子文件未变更时直接复用）。"""
        try:
            if not self._cache_file.is_file():
                return False
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            if data.get("format") != 2 or data.get("seed_mtime") != self._seed_mtime():
                return False
            self._prototypes = {
                cid: {
                    "vectors": [np.asarray(v, dtype=np.float32) for v in p["vectors"]],
                    "level": p["level"],
                    "label": p["label"],
                }
                for cid, p in data.get("prototypes", {}).items()
                if p.get("vectors")
            }
            return bool(self._prototypes)
        except Exception as e:
            print(f"[crisis_detector][WARN] 原型缓存加载失败，将重建: {e}", flush=True)
            return False

    def _build(self, embed_documents_fn: Callable) -> None:
        """从种子集构建锚点集合（首次或种子变更时），并写缓存文件。

        每个意图簇保留全部种子句子的向量（而非均值原型）：簇内比喻跨度大时，
        均值点会偏离个别成员（如"庄稼长大的小药水"离"服药自杀"均值较远），
        而"到最近锚点的距离"能保证已收录表达必然命中、未收录表达靠近最近锚点。
        """
        seed = json.loads(self._seed_file.read_text(encoding="utf-8"))
        clusters_cfg = seed.get("clusters", {})
        by_cluster: Dict[str, Dict] = {}
        for item in seed.get("intents", []):
            cid = item.get("cluster", "")
            if not cid:
                continue
            bucket = by_cluster.setdefault(cid, {"texts": []})
            bucket["texts"].append(item.get("intent", ""))
            bucket["texts"].extend(v for v in item.get("variants", []) if v)

        # 批量 embed 全部句子（去重），按文本映射向量
        texts: List[str] = []
        for bucket in by_cluster.values():
            for t in bucket["texts"]:
                if t and t not in texts:
                    texts.append(t)
        if not texts:
            raise ValueError("种子集为空，无法构建高危意图锚点")
        vectors = embed_documents_fn(texts) or []
        vec_map = {t: v for t, v in zip(texts, vectors) if v is not None}

        prototypes: Dict[str, Dict] = {}
        for cid, bucket in by_cluster.items():
            vecs = [vec_map[t] for t in bucket["texts"] if t in vec_map]
            if not vecs:
                continue
            cfg = clusters_cfg.get(cid, {})
            prototypes[cid] = {
                "vectors": vecs,
                "level": cfg.get("level", "high"),
                "label": cfg.get("label", cid),
            }
        if not prototypes:
            raise ValueError("种子集无法聚合成任何原型")

        self._prototypes = prototypes
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(
                json.dumps({
                    "format": 2,
                    "seed_mtime": self._seed_mtime(),
                    "prototypes": {
                        cid: {
                            "vectors": [list(v) for v in p["vectors"]],
                            "level": p["level"],
                            "label": p["label"],
                        }
                        for cid, p in prototypes.items()
                    },
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[crisis_detector][WARN] 锚点缓存写入失败（不影响本次检测）: {e}", flush=True)

    def _ensure_loaded(self, embed_documents_fn: Callable) -> bool:
        """确保锚点已加载；未就绪/构建中/构建失败时返回 False，由调用方回退。

        非阻塞设计：批量 embed（首次构建约 3-10 秒）可能耗时较长，
        若在构建中（锁被占用）或上次构建失败，立即返回 False 回退到关键词，
        避免请求排队等锁把问答链路卡死（与 reranker 的加载策略一致）。
        """
        if self._loaded:
            return True
        if self._build_failed:
            return False
        if not self._lock.acquire(blocking=False):
            return False  # 正在构建中，本次回退
        try:
            if self._loaded:
                return True
            try:
                if not self._load_cache():
                    self._build(embed_documents_fn)
                self._loaded = True
                return True
            except Exception as e:
                self._build_failed = True
                print(f"[crisis_detector][WARN] 锚点构建失败，本次回退关键词: {e}", flush=True)
                return False
        finally:
            self._lock.release()

    def warm_up(self) -> None:
        """启动预热：后台构建锚点（失败静默，首次问答懒构建兜底）。"""
        try:
            from modules.vector_store import PsychologyVectorStore

            store = PsychologyVectorStore()
            self._ensure_loaded(store.embeddings.embed_documents)
            if self._loaded:
                print(f"[startup] 语义危机检测锚点已就绪（{len(self._prototypes)} 个意图簇）")
            else:
                print("[startup][WARN] 语义危机检测锚点未就绪，首次问答将回退关键词")
        except Exception as e:
            print(f"[startup][WARN] 语义危机检测锚点预热失败（首次问答将回退关键词）: {e}")

    # ---------------- 检测 ----------------
    def detect(self, text: str, embed_query_fn: Callable, embed_documents_fn: Callable) -> Optional[Dict]:
        """语义危机检测。

        embed_query_fn：单条文本 → 向量（带缓存，复用检索的调用）。
        embed_documents_fn：批量文本 → 向量列表（仅首次构建原型时用）。
        返回 None 表示语义层不可用/未命中；命中返回结构化结果。
        """
        if not settings.SEMANTIC_CHECK_ENABLED or not text:
            return None
        try:
            if not self._ensure_loaded(embed_documents_fn):
                return None  # 未就绪/构建中/失败：回退关键词
            if not self._prototypes:
                return None
            vec = embed_query_fn(text)
            if vec is None:
                return None

            best_cluster, best_dist = None, float("inf")
            for cid, proto in self._prototypes.items():
                for v in proto["vectors"]:
                    d = _cosine_dist(vec, v)
                    if d < best_dist:
                        best_dist, best_cluster = d, cid
            if best_cluster is None:
                return None

            inter = settings.CRISIS_INTERCEPT_DIST
            gray = settings.CRISIS_GRAY_DIST
            proto = self._prototypes[best_cluster]
            if best_dist <= inter:
                level = proto["level"]  # high 或 medium（如被伤害求助）
            elif best_dist <= gray:
                level = "medium"  # 灰区 → 疑似
            else:
                return {
                    "is_crisis": False,
                    "level": "none",
                    "cluster": best_cluster,
                    "label": proto["label"],
                    "distance": round(best_dist, 4),
                    "gray_zone": False,
                    "detect_method": "semantic",
                }

            return {
                "is_crisis": level in ("high", "medium"),
                "level": level,
                "cluster": best_cluster,
                "label": proto["label"],
                "distance": round(best_dist, 4),
                "gray_zone": best_dist > inter,
                "detect_method": "semantic",
            }
        except Exception as e:
            # 语义层任何异常都不阻断问答：回退到纯关键词
            print(f"[crisis_detector][WARN] 语义检测失败，回退关键词: {e}", flush=True)
            return None


# 全局单例（懒加载）
_detector: Optional[SemanticCrisisDetector] = None
_detector_lock = threading.Lock()


def get_crisis_detector() -> SemanticCrisisDetector:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = SemanticCrisisDetector()
    return _detector

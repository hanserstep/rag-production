"""
多路检索融合
============

为什么需要融合？
    向量检索返回的是余弦相似度（越大越相似，范围 -1~1）
    BM25 返回的是词频打分（越大越相关，范围 0~几十，随语料规模漂移）

两路分数的**量纲完全不同**，直接相加等于拿米和斤相加。
更麻烦的是：BM25 的绝对分值会随语料库大小变化，同一个 query 在 100 篇
和 10000 篇文档上拿到的分数不在一个量级，任何写死的权重都会失效。

三种做法：

1. 按顺序截断合并（本项目旧版本的做法）
   向量取 top_k，BM25 取 top_k，去重后拼接。
   问题：两路在结果里的相对顺序没有意义，重排序模型看到的是一个
   完全由"谁先被遍历到"决定的列表。

2. 分数归一化后加权（score fusion）
   各自 min-max 归一化到 [0,1] 再加权求和。
   问题：min-max 依赖当前批次的最大最小值。一批结果里如果有一条
   特别高分，其余全被压到接近 0，归一化反而破坏了两路各自的
   相对关系。而且权重需要按数据分布调，换语料就得重调。

3. RRF（Reciprocal Rank Fusion）
   只用排名，不用分数：  score(d) = Σ weight_i / (k + rank_i(d))
   k 取 60（论文经验值），作用是压低头部差异——排第 1 和排第 2 的
   贡献差不太多，避免某一路的 top1 独断。
   好处：对量纲免疫，对分数分布免疫，不需要调参。

本项目默认用 RRF，同时保留 score fusion 作为对照组，
在 eval/ 里用评测集量化两者的差距。
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[str]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    top_n: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    RRF 融合。

    Args:
        ranked_lists: {"vector": [id1, id2, ...], "bm25": [id3, id1, ...]}
                      每个列表必须已按相关性从高到低排好序。
        k: 平滑常数，默认 60。调小 → 更看重头部结果；调大 → 各路更平均。
        weights: 各路的权重，{"vector": 1.0, "bm25": 0.7}。不传则等权。
        top_n: 只返回前 n 个。

    Returns:
        [(doc_id, rrf_score), ...]  按分数从高到低排序。

    注意：rank 从 1 开始，不是 0。用 0 会让 1/(k+0) 和 1/(k+1)
    的差距被放大，k 的平滑作用就失效了。
    """
    if not ranked_lists:
        return []

    weights = weights or {}
    scores: Dict[str, float] = defaultdict(float)
    hit_count: Dict[str, int] = defaultdict(int)

    for source, ids in ranked_lists.items():
        w = weights.get(source, 1.0)
        if w == 0:
            continue
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] += w / (k + rank)
            hit_count[doc_id] += 1

    # 顺序：先按分数降序；分数相同（只在等权且名次相同的情况下出现）
    # 时，被多路同时召回的优先——两路都命中的文档通常更可信
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], -hit_count[kv[0]]),
    )

    return ranked[:top_n] if top_n else ranked


def min_max_normalize(scores: List[float]) -> List[float]:
    """把分数线性映射到 [0,1]。全相同时返回全 0（避免除零）。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_score_fusion(
    scored_lists: Dict[str, List[Tuple[str, float]]],
    weights: Optional[Dict[str, float]] = None,
    top_n: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    分数归一化后加权求和。保留这个实现是为了在评测里和 RRF 做对照，
    证明 RRF 更好——面试时"我对比了两种方案"比"我用了 RRF"有说服力得多。

    Args:
        scored_lists: {"vector": [(id, score), ...], "bm25": [...]}
                      注意分数方向要统一：BM25 是越大越好，余弦距离是越小越好，
                      调用方必须先统一成"越大越好"再传进来。
    """
    if not scored_lists:
        return []

    weights = weights or {}
    fused: Dict[str, float] = defaultdict(float)

    for source, items in scored_lists.items():
        w = weights.get(source, 1.0)
        if not items:
            continue
        scores = [s for _, s in items]
        normed = min_max_normalize(scores)
        for (doc_id, _), n in zip(items, normed):
            fused[doc_id] += w * n

    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    return ranked[:top_n] if top_n else ranked

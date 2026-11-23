"""RRF 融合的单元测试。

这些用例的价值不只是"测通了"，而是每个用例都对应一个面试问题：
- test_single_list_preserves_order   → "RRF 会破坏单路的排序吗"
- test_both_hit_ranks_higher         → "两路都召回的文档为什么更可信"
- test_k_controls_head_weight        → "k 调大调小有什么区别"
- test_weight_zero_disables_source   → "某一路效果差怎么办"
"""

import pytest

from app.retrieval.fusion import (
    reciprocal_rank_fusion,
    min_max_normalize,
    weighted_score_fusion,
)


class TestRRF:

    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"vector": []}) == []

    def test_single_list_preserves_order(self):
        """只有一路时，RRF 必须保持原顺序，否则等于把召回结果打乱。"""
        ids = ["a", "b", "c", "d"]
        result = reciprocal_rank_fusion({"vector": ids})
        assert [doc_id for doc_id, _ in result] == ids

    def test_score_matches_formula(self):
        """分数必须严格等于 1/(k+rank)，rank 从 1 开始。"""
        result = reciprocal_rank_fusion({"vector": ["a"]}, k=60)
        assert result[0][1] == pytest.approx(1 / 61)

    def test_both_hit_ranks_higher(self):
        """被两路同时召回的文档，应排在只被一路召回的文档前面。

        这是融合的核心价值：两路独立判断都认为相关 → 置信度更高。
        """
        result = reciprocal_rank_fusion({
            "vector": ["a", "b"],
            "bm25": ["c", "b"],
        })
        order = [doc_id for doc_id, _ in result]
        # b 在 vector 排第 2、bm25 排第 2，累计 2/(60+2)
        # a 在 vector 排第 1，只有 1/61
        # c 在 bm25 排第 1，只有 1/61
        assert order[0] == "b"
        assert set(order[1:]) == {"a", "c"}

    def test_k_controls_head_weight(self):
        """k 越小，头部结果的优势越明显。

        k=1 时 rank1 得 1/2，rank2 得 1/3，差距 1.5 倍。
        k=100 时 rank1 得 1/101，rank2 得 1/102，几乎相等。
        """
        small_k = reciprocal_rank_fusion({"v": ["a", "b"]}, k=1)
        large_k = reciprocal_rank_fusion({"v": ["a", "b"]}, k=100)
        assert small_k[0][1] > large_k[0][1]
        # k 小时的相对差距应更大
        assert (small_k[0][1] - small_k[1][1]) / small_k[1][1] > \
               (large_k[0][1] - large_k[1][1]) / large_k[1][1]

    def test_weight_zero_disables_source(self):
        """某一路权重设 0，等于关掉它——用于线上按效果动态降级。"""
        result = reciprocal_rank_fusion(
            {"vector": ["a"], "bm25": ["b"]},
            weights={"bm25": 0},
        )
        assert [d for d, _ in result] == ["a"]

    def test_weight_boosts_source(self):
        result = reciprocal_rank_fusion(
            {"vector": ["a"], "bm25": ["b"]},
            weights={"bm25": 2.0},
        )
        assert result[0][0] == "b"

    def test_top_n_truncates(self):
        result = reciprocal_rank_fusion({"v": ["a", "b", "c"]}, top_n=2)
        assert len(result) == 2

    def test_no_duplicate_output(self):
        """同一文档在两路都出现，输出里只能有一次。"""
        result = reciprocal_rank_fusion({
            "vector": ["a", "a"],
            "bm25": ["a"],
        })
        ids = [doc_id for doc_id, _ in result]
        assert ids.count("a") == 1


class TestScoreFusion:

    def test_min_max_normalize(self):
        assert min_max_normalize([0, 5, 10]) == [0.0, 0.5, 1.0]

    def test_normalize_flat_returns_zeros(self):
        """全相同不能除零。"""
        assert min_max_normalize([3, 3, 3]) == [0.0, 0.0, 0.0]

    def test_scale_invariance_is_why_rrf_wins(self):
        """RRF 对量纲免疫，加权融合不免疫——这是选 RRF 的核心理由。

        两路的分数绝对值差 100 倍时：
        加权融合的结果被大量纲那一路主导；RRF 只看排名，不受影响。
        """
        vector = [("a", 0.9), ("b", 0.8)]
        bm25 = [("c", 90.0), ("a", 80.0)]

        fused = dict(weighted_score_fusion({"vector": vector, "bm25": bm25}))
        # bm25 的量纲把 a 顶到了接近 1.0，向量路的贡献被稀释
        assert fused["a"] > fused["b"]

        rrf = dict(reciprocal_rank_fusion({
            "vector": ["a", "b"],
            "bm25": ["c", "a"],
        }))
        # 而 RRF 只看名次，a 两路都在前列，稳定胜出
        assert rrf["a"] > rrf["b"] and rrf["a"] > rrf["c"]

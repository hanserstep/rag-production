# 检索评测报告

- 知识库：6 篇文档 / 35 个 chunk
- 评测集：50 条问题（人工标注相关 chunk）
- 向量后端：milvus_rest　Embedding：api
- 生成时间：2026-11-24 00:48

## 消融结果

| 配置 | 5 条召回 Hit Rate | MRR | Recall | P50 延迟 | 说明 |
|---|---|---|---|---|---|
| `vector_only` | 0.880 | 0.637 | 0.823 | 95ms | 只有向量检索，基线 |
| `vector_bm25_concat` | 0.880 | 0.637 | 0.823 | 98ms | 双路召回 + 旧版的截断拼接 |
| `vector_bm25_score` | 0.940 | 0.682 | 0.883 | 94ms | 双路召回 + 分数归一化加权 |
| `rrf_no_rerank` | 0.940 | 0.669 | 0.903 | 91ms | 双路召回 + RRF 融合 |
| `rrf_rerank` | 0.940 | 0.704 | 0.903 | 224ms | RRF + Cross-Encoder 重排（默认线上配置） |

## 相对基线的增益

| 配置 | Hit Rate 变化 | MRR 变化 |
|---|---|---|
| `vector_bm25_concat` | +0.000 | +0.000 |
| `vector_bm25_score` | +0.060 | +0.046 |
| `rrf_no_rerank` | +0.060 | +0.032 |
| `rrf_rerank` | +0.060 | +0.067 |

## 结论

最优配置 `rrf_rerank` 相对纯向量基线，Hit Rate@5 +0.060，MRR +0.067。

注意 MRR 的提升幅度通常大于 Hit Rate——因为融合和重排序主要改变的是**排序位置**，而不是"能不能召回"。这也说明只报 Hit Rate 会低估这两层的价值。

> 表中数字由 `eval/run_eval.py` 实测生成，可复现：
> `python -m eval.run_eval --backend milvus`

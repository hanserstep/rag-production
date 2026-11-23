"""检索层消融评测。

跑法：
    python -m eval.run_eval                          # 用默认配置，输出到 eval/EVAL_REPORT.md
    python -m eval.run_eval --backend memory         # 不依赖 Milvus，先用内存版验证流程
    python -m eval.run_eval --configs vector_only,rff   # 只跑指定配置

指标定义（面试必被问，要能讲清区别）：

    Hit Rate@k   前 k 条里是否命中至少一个相关片段。只看"有没有"，
                 不看"排第几"。用户真正在意的是这个——只要答案在
                 前 k 条里，模型就能答出来。

    MRR@k        Mean Reciprocal Rank，命中片段排名的倒数取均值。
                 排第 1 得 1.0，排第 3 得 0.333。反映"排得有多靠前"，
                 对重排序的效果最敏感。

    Recall@k     前 k 条覆盖了多少比例的相关片段。当一个问题对应
                 多个片段时才有区分度。Hit Rate 会掩盖"只召回一半"
                 的问题，Recall 不会。

三个一起看的原因：只报 Hit Rate 容易自欺——把 k 调大到 20，
Hit Rate 一定好看，但 context 塞满噪声，生成质量反而下降。
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings
from app.ingest.chunker import DocumentChunker
from app.ingest.embedder import build_embedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import NoOpReranker, build_reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import build_store


@dataclass
class EvalConfig:
    name: str
    use_bm25: bool = True
    fusion_mode: str = "rrf"
    use_rerank: bool = True
    use_hyde: bool = False
    description: str = ""


DEFAULT_CONFIGS: List[EvalConfig] = [
    EvalConfig("vector_only", use_bm25=False, fusion_mode="rrf", use_rerank=False,
               description="只有向量检索，基线"),
    EvalConfig("vector_bm25_concat", use_bm25=True, fusion_mode="concat", use_rerank=False,
               description="双路召回 + 旧版的截断拼接"),
    EvalConfig("vector_bm25_score", use_bm25=True, fusion_mode="score", use_rerank=False,
               description="双路召回 + 分数归一化加权"),
    EvalConfig("rrf_no_rerank", use_bm25=True, fusion_mode="rrf", use_rerank=False,
               description="双路召回 + RRF 融合"),
    EvalConfig("rrf_rerank", use_bm25=True, fusion_mode="rrf", use_rerank=True,
               description="RRF + Cross-Encoder 重排（默认线上配置）"),
]


def load_dataset(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(docs_dir: str) -> Dict[str, str]:
    """读取 docs/ 下的 markdown 作为知识库。doc_id 用文件名（不含扩展名）。"""
    corpus = {}
    for fn in sorted(os.listdir(docs_dir)):
        if not fn.endswith((".md", ".txt")):
            continue
        path = os.path.join(docs_dir, fn)
        with open(path, encoding="utf-8") as f:
            corpus[os.path.splitext(fn)[0]] = f.read()
    return corpus


def build_index(cfg, corpus: Dict[str, str], backend: str, embedder):
    store = build_store(
        backend,
        uri=cfg.milvus_uri,
        collection=cfg.collection_name + "_eval",
        hnsw_m=cfg.hnsw_m,
        ef_construction=cfg.hnsw_ef_construction,
        ef_search=cfg.hnsw_ef_search,
    )
    bm25 = BM25Index(persist_path=os.path.join("data", "bm25_eval.pkl"))
    retriever = HybridRetriever(
        store=store, bm25=bm25, embedder=embedder,
        reranker=NoOpReranker(), rrf_k=cfg.rrf_k, dim=embedder.dim,
    )

    chunker = DocumentChunker()
    total = 0
    for doc_id, text in corpus.items():
        chunks = chunker.split(text, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        total += retriever.index_chunks(doc_id, chunks)
    return retriever, total


def compute_metrics(rows: List[Dict], retriever: HybridRetriever,
                    config: EvalConfig, k_values: List[int],
                    reranker, hyde_fn=None) -> Dict:
    hits = {k: 0 for k in k_values}
    rr_sum = {k: 0.0 for k in k_values}
    recall_sum = {k: 0.0 for k in k_values}
    latencies: List[float] = []
    max_k = max(k_values)

    for row in rows:
        question = row["question"]
        relevant = set(row.get("relevant_chunk_ids") or row.get("relevant", []))

        search_query = None
        if config.use_hyde and hyde_fn:
            search_query = hyde_fn(question)

        res = retriever.retrieve(
            query=question,
            top_k=max_k,
            rerank_top_n=max_k,
            fusion_mode=config.fusion_mode,
            use_bm25=config.use_bm25,
            use_rerank=config.use_rerank,
            search_query=search_query,
        )
        latencies.append(res.trace.total_ms)

        ranked_ids = [h.id for h in res.hits]
        for k in k_values:
            topk = ranked_ids[:k]
            hit_positions = [i + 1 for i, cid in enumerate(topk) if cid in relevant]
            if hit_positions:
                hits[k] += 1
                rr_sum[k] += 1.0 / hit_positions[0]
            if relevant:
                recall_sum[k] += len(set(topk) & relevant) / len(relevant)

    n = max(1, len(rows))
    lat_sorted = sorted(latencies)
    return {
        "config": config.name,
        "description": config.description,
        "n_queries": len(rows),
        **{f"hit_rate@{k}": round(hits[k] / n, 4) for k in k_values},
        **{f"mrr@{k}": round(rr_sum[k] / n, 4) for k in k_values},
        **{f"recall@{k}": round(recall_sum[k] / n, 4) for k in k_values},
        "p50_ms": round(lat_sorted[int(len(lat_sorted) * 0.5)], 1) if lat_sorted else 0,
        "p95_ms": round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))], 1) if lat_sorted else 0,
    }


def render_markdown(results: List[Dict], k_values: List[int], meta: Dict) -> str:
    k_main = k_values[-1]
    lines = [
        "# 检索评测报告",
        "",
        f"- 知识库：{meta['n_docs']} 篇文档 / {meta['n_chunks']} 个 chunk",
        f"- 评测集：{meta['n_queries']} 条问题（人工标注相关 chunk）",
        f"- 向量后端：{meta['backend']}　Embedding：{meta['embedder']}",
        f"- 生成时间：{meta['timestamp']}",
        "",
        "## 消融结果",
        "",
        f"| 配置 | {k_main} 条召回 Hit Rate | MRR | Recall | P50 延迟 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['config']}` | {r[f'hit_rate@{k_main}']:.3f} | "
            f"{r[f'mrr@{k_main}']:.3f} | {r[f'recall@{k_main}']:.3f} | "
            f"{r['p50_ms']:.0f}ms | {r['description']} |"
        )

    # 相对基线的增益，这是面试里最有说服力的一组数字
    base = results[0]
    lines += ["", "## 相对基线的增益", "", f"| 配置 | Hit Rate 变化 | MRR 变化 |", "|---|---|---|"]
    for r in results[1:]:
        dh = r[f"hit_rate@{k_main}"] - base[f"hit_rate@{k_main}"]
        dm = r[f"mrr@{k_main}"] - base[f"mrr@{k_main}"]
        lines.append(f"| `{r['config']}` | {dh:+.3f} | {dm:+.3f} |")

    lines += [
        "",
        "## 结论",
        "",
        _conclusion(results, k_main),
        "",
        "> 表中数字由 `eval/run_eval.py` 实测生成，可复现：",
        "> `python -m eval.run_eval --backend milvus`",
        "",
    ]
    return "\n".join(lines)


def _conclusion(results: List[Dict], k: int) -> str:
    """从结果自动生成结论，避免手抄数字抄错。"""
    if len(results) < 2:
        return "样本配置过少，未生成结论。"
    base = results[0]
    best = max(results, key=lambda r: r[f"mrr@{k}"])
    dh = best[f"hit_rate@{k}"] - base[f"hit_rate@{k}"]
    dm = best[f"mrr@{k}"] - base[f"mrr@{k}"]
    return (
        f"最优配置 `{best['config']}` 相对纯向量基线，"
        f"Hit Rate@{k} {dh:+.3f}，MRR {dm:+.3f}。\n\n"
        "注意 MRR 的提升幅度通常大于 Hit Rate——因为融合和重排序主要改变的是"
        "**排序位置**，而不是\"能不能召回\"。这也说明只报 Hit Rate 会低估这两层的价值。"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/dataset.jsonl")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--backend", default="memory", choices=["memory", "milvus", "milvus_rest"])
    ap.add_argument("--embedding", default=None, choices=["api", "local", "ollama", "hashing"])
    ap.add_argument("--report", default="eval/EVAL_REPORT.md")
    ap.add_argument("--k", default="1,3,5,10")
    ap.add_argument("--configs", default="")
    args = ap.parse_args()

    cfg = get_settings()
    k_values = [int(x) for x in args.k.split(",")]

    rows = load_dataset(args.dataset)
    corpus = load_corpus(args.docs)
    if not rows or not corpus:
        print("评测集或知识库为空，先跑 eval/build_dataset.py")
        return 1

    emb_backend = args.embedding or cfg.embedding_backend
    if emb_backend == "hashing":
        from app.ingest.embedder import HashingEmbedder
        embedder = HashingEmbedder(dim=256)
    else:
        embedder = build_embedder(
            emb_backend,
            api_key=cfg.siliconflow_api_key,
            model=cfg.embedding_model,
            dim=cfg.embedding_dim,
            model_path=cfg.embedding_model,
        )

    print(f"[1/3] 建索引：{len(corpus)} 篇文档，backend={args.backend}, embedder={emb_backend}")
    t0 = time.time()
    retriever, n_chunks = build_index(cfg, corpus, args.backend, embedder)
    print(f"      完成：{n_chunks} 个 chunk，耗时 {time.time() - t0:.1f}s")

    names = [x.strip() for x in args.configs.split(",") if x.strip()]
    configs = [c for c in DEFAULT_CONFIGS if not names or c.name in names] or DEFAULT_CONFIGS

    reranker = NoOpReranker()
    if any(c.use_rerank for c in configs):
        if emb_backend == "hashing":
            # 哈希向量只是为了让流程可跑，没有真实语义能力，此时调真实
            # Cross-Encoder 没有意义（分数和上下文对不上）。明确跳过而不是
            # 静默降级，避免报告里出现"重排无效"的错误结论。
            print("      [警告] hashing 模式不启用重排序，rff_rerank 与 rff_no_rerank 结果会相同")
            reranker = NoOpReranker()
        elif emb_backend == "api":
            reranker = build_reranker("api", api_key=cfg.siliconflow_api_key)
        else:
            reranker = build_reranker("local", model_path=cfg.rerank_model)
    retriever.reranker = reranker

    print(f"[2/3] 跑 {len(configs)} 组配置 × {len(rows)} 条 query")
    results = []
    for c in configs:
        r = compute_metrics(rows, retriever, c, k_values, reranker)
        results.append(r)
        print(f"      {c.name:22} hit@{k_values[-1]}={r[f'hit_rate@{k_values[-1]}']:.3f} "
              f"mrr={r[f'mrr@{k_values[-1]}']:.3f}")

    print("[3/3] 生成报告")
    meta = {
        "n_docs": len(corpus),
        "n_chunks": n_chunks,
        "n_queries": len(rows),
        "backend": args.backend,
        "embedder": emb_backend,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    md = render_markdown(results, k_values, meta)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(md)

    with open(args.report.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"      报告：{args.report}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""并发压测，产出 P50 / P95 / P99 延迟和 QPS。

用法：
    python scripts/bench.py --n 100 --concurrency 8 --top-k 20
    python scripts/bench.py --question-file eval/dataset.jsonl

为什么要自己写而不是用 wrk / ab：
    RAG 的延迟包含 embedding 调用、两次检索、重排序、LLM 生成，
    每一段的耗时分布差异很大。通用压测工具只能给出端到端总延迟，
    这个脚本复用项目里的 retriever 直接测检索段——这一段的延迟是
    系统可控的部分，LLM 生成延迟取决于上游 API，不该混在一起看。
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings
from app.ingest.embedder import build_embedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import NoOpReranker, build_reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import build_store


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="总请求数")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--backend", default="memory", choices=["memory", "milvus"])
    ap.add_argument("--embedding", default="ollama", choices=["api", "local", "ollama", "hashing"])
    ap.add_argument("--question-file", default="eval/dataset.jsonl")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    cfg = get_settings()

    embedder = build_embedder(
        args.embedding,
        api_key=cfg.siliconflow_api_key,
        model=cfg.embedding_model,
        dim=cfg.embedding_dim,
        model_path=cfg.embedding_model,
    )
    store = build_store(args.backend, uri=cfg.milvus_uri,
                        collection=cfg.collection_name + "_bench")
    bm25 = BM25Index(persist_path="data/bm25_bench.pkl")

    reranker = NoOpReranker()
    if args.embedding != "hashing":
        try:
            reranker = build_reranker("api", api_key=cfg.siliconflow_api_key) \
                if args.embedding == "api" else NoOpReranker()
        except Exception:
            reranker = NoOpReranker()

    retriever = HybridRetriever(store=store, bm25=bm25, embedder=embedder,
                                reranker=reranker, rrf_k=cfg.rrf_k, dim=embedder.dim)

    # 建索引
    from app.ingest.chunker import DocumentChunker
    chunker = DocumentChunker()
    n_chunks = 0
    for fn in sorted(os.listdir(args.docs)):
        if not fn.endswith((".md", ".txt")):
            continue
        text = open(os.path.join(args.docs, fn), encoding="utf-8").read()
        chunks = chunker.split(text, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        n_chunks += retriever.index_chunks(os.path.splitext(fn)[0], chunks)
    print(f"索引就绪：{n_chunks} chunks，backend={args.backend}，embedding={args.embedding}")

    questions = []
    with open(args.question_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
    if not questions:
        print("评测集为空")
        return 1

    # 预热：第一次调用包含模型加载、连接建立，不能算进延迟统计
    retriever.retrieve(questions[0], top_k=args.top_k, use_rerank=False)

    def one(i):
        q = questions[i % len(questions)]
        t0 = time.perf_counter()
        res = retriever.retrieve(q, top_k=args.top_k, rerank_top_n=cfg.rerank_top_n)
        return (time.perf_counter() - t0) * 1000, res.trace

    print(f"开始压测：{args.n} 请求，并发 {args.concurrency}")
    latencies, traces = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, i) for i in range(args.n)]
        for fu in as_completed(futures):
            lat, tr = fu.result()
            latencies.append(lat)
            traces.append(tr)
    wall = time.perf_counter() - started

    qps = args.n / wall if wall else 0
    report = {
        "requests": args.n,
        "concurrency": args.concurrency,
        "wall_time_s": round(wall, 2),
        "qps": round(qps, 1),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "p99": round(percentile(latencies, 0.99), 1),
            "max": round(max(latencies), 1),
        },
        "stage_breakdown_p50_ms": {
            "vector": round(percentile([t.vector_ms for t in traces], 0.5), 1),
            "bm25": round(percentile([t.bm25_ms for t in traces], 0.5), 1),
            "fusion": round(percentile([t.fusion_ms for t in traces], 0.5), 2),
            "rerank": round(percentile([t.rerank_ms for t in traces], 0.5), 1),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    os.makedirs("eval", exist_ok=True)
    out = f"eval/bench_{args.backend}_{args.embedding}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

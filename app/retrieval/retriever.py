"""检索编排层。

一次检索的完整链路：

    query ──┬─▶ 向量检索 (Bi-Encoder + ANN)  ─┬─▶ 融合(RRF) ─▶ 重排(Cross-Encoder) ─▶ top_n
            └─▶ BM25 关键词检索              ─┘

每个环节都可以单独关掉，这是刻意设计的——评测脚本要跑消融实验
（ablation），量化每一层到底贡献了多少：

    vector_only   → 只有向量，基线
    +bm25         → 双路召回，看关键词路带来多少提升
    +rrf          → 换融合策略，看排序质量的提升
    +rerank       → 加重排，看精排的增益
    +hyde         → 加查询改写

面试时"我做了消融，RRF 贡献 +0.07，重排贡献 +0.03"的说服力，
远高于"我用了混合检索和重排序"。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bm25 import BM25Index
from .fusion import min_max_normalize, reciprocal_rank_fusion, weighted_score_fusion
from .reranker import BaseReranker, NoOpReranker
from .vector_store import BaseVectorStore, Chunk, SearchHit


@dataclass
class Trace:
    """一次检索的可观测记录。

    线上排障时最想知道的是"慢在哪、哪一路没召回"，
    没有这个结构就只能靠加 print，查一次改一次代码。
    """
    query: str = ""
    fusion_mode: str = "rrf"
    vector_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    vector_candidates: int = 0
    bm25_candidates: int = 0
    fused_candidates: int = 0
    final_count: int = 0
    rerank_skipped: int = 0  # 被相似度阈值过滤掉的数量

    def as_dict(self) -> Dict:
        return {
            "query": self.query,
            "fusion_mode": self.fusion_mode,
            "latency_ms": {
                "vector": round(self.vector_ms, 1),
                "bm25": round(self.bm25_ms, 1),
                "fusion": round(self.fusion_ms, 2),
                "rerank": round(self.rerank_ms, 1),
                "total": round(self.total_ms, 1),
            },
            "candidates": {
                "vector": self.vector_candidates,
                "bm25": self.bm25_candidates,
                "fused": self.fused_candidates,
                "final": self.final_count,
            },
            "rerank_filtered": self.rerank_skipped,
        }


@dataclass
class RetrievalResult:
    hits: List[SearchHit] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)


class HybridRetriever:

    def __init__(
        self,
        store: BaseVectorStore,
        bm25: BM25Index,
        embedder,
        reranker: Optional[BaseReranker] = None,
        rrf_k: int = 60,
        dim: int = 1024,
    ):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker or NoOpReranker()
        self.rrf_k = rrf_k
        self.dim = dim

    # ---- 写入 ----
    def index_chunks(self, doc_id: str, texts: List[str]) -> int:
        """把一批 chunk 同时写进向量库和 BM25 索引。

        两条链路写同一批 id，融合时才能按 id 对齐。
        """
        if not texts:
            return 0

        self.store.ensure_collection(self.dim)
        embeddings = self.embedder.encode_documents(texts)
        chunks = [
            Chunk(
                id=f"{doc_id}#{i}",
                content=t,
                embedding=emb,
                doc_id=doc_id,
                chunk_index=i,
            )
            for i, (t, emb) in enumerate(zip(texts, embeddings))
        ]
        n_vec = self.store.upsert(chunks)
        n_bm25 = self.bm25.add(chunks)
        if self.bm25.persist_path:
            self.bm25.save()
        return min(n_vec, n_bm25)

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_by_doc(doc_id)
        self.bm25.delete_by_doc(doc_id)
        if self.bm25.persist_path:
            self.bm25.save()

    # ---- 检索 ----
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_n: int = 3,
        fusion_mode: str = "rrf",
        use_bm25: bool = True,
        use_rerank: bool = True,
        search_query: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        rrf_weights: Optional[Dict[str, float]] = None,
    ) -> RetrievalResult:
        """
        Args:
            query: 原始问题，用于 BM25 和重排序（必须是用户的原始问法）
            search_query: 用于向量检索的查询。开了 HyDE 时传假设答案，
                          没开就传 None 表示用 query 本身。
        """
        started = time.perf_counter()
        trace = Trace(query=query, fusion_mode=fusion_mode)

        # ---- 路 1：向量检索 ----
        t0 = time.perf_counter()
        query_vec = self.embedder.encode_query(search_query or query)
        vec_hits = self.store.search(query_vec, top_k=top_k, doc_ids=doc_ids)
        trace.vector_ms = (time.perf_counter() - t0) * 1000
        trace.vector_candidates = len(vec_hits)

        # ---- 路 2：BM25 ----
        bm25_hits: List[SearchHit] = []
        if use_bm25:
            t0 = time.perf_counter()
            bm25_hits = self.bm25.search(query, top_k=top_k, doc_ids=doc_ids)
            trace.bm25_ms = (time.perf_counter() - t0) * 1000
            trace.bm25_candidates = len(bm25_hits)

        # ---- 融合 ----
        t0 = time.perf_counter()
        by_id: Dict[str, SearchHit] = {h.id: h for h in bm25_hits}
        by_id.update({h.id: h for h in vec_hits})  # 向量结果优先保留（带 embedding 侧元数据）

        if not bm25_hits:
            fused_ids = [(h.id, h.score) for h in vec_hits]
        elif fusion_mode == "concat":
            # 旧版做法：向量截断 + BM25 截断，去重拼接。留作对照组
            seen, fused_ids = set(), []
            for h in vec_hits + bm25_hits:
                if h.id not in seen:
                    seen.add(h.id)
                    fused_ids.append((h.id, 0.0))
        elif fusion_mode == "score":
            # 对照组：归一化加权。向量是 IP 越大越好，BM25 也是越大越好，方向一致
            fused_ids = weighted_score_fusion(
                {"vector": [(h.id, h.score) for h in vec_hits],
                 "bm25": [(h.id, h.score) for h in bm25_hits]},
                weights={"vector": 1.0, "bm25": 0.5},
                top_n=top_k,
            )
        else:
            fused_ids = reciprocal_rank_fusion(
                {"vector": [h.id for h in vec_hits],
                 "bm25": [h.id for h in bm25_hits]},
                k=self.rrf_k,
                weights=rrf_weights,
                top_n=top_k,
            )

        fused = []
        for doc_id_, fscore in fused_ids:
            hit = by_id.get(doc_id_)
            if hit is None:
                continue
            hit.meta["fusion_score"] = float(fscore)
            hit.meta["sources"] = [
                name for name, hits in (("vector", vec_hits), ("bm25", bm25_hits))
                if any(h.id == doc_id_ for h in hits)
            ]
            fused.append(hit)

        trace.fusion_ms = (time.perf_counter() - t0) * 1000
        trace.fused_candidates = len(fused)

        # ---- 重排 ----
        if use_rerank and fused:
            t0 = time.perf_counter()
            before = len(fused)
            final = self.reranker.rerank(query, fused, top_n=rerank_top_n)
            trace.rerank_ms = (time.perf_counter() - t0) * 1000
            trace.rerank_skipped = before - len(final)
        else:
            final = fused[:rerank_top_n]

        trace.final_count = len(final)
        trace.total_ms = (time.perf_counter() - started) * 1000

        return RetrievalResult(hits=final, trace=trace)

    def stats(self) -> Dict:
        return {
            "vector_store": type(self.store).__name__,
            "chunks": self.store.count(),
            "bm25": self.bm25.stats(),
        }

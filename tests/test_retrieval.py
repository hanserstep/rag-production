"""检索链路端到端测试。

用 HashingEmbedder + InMemoryStore，不依赖模型、不依赖 Milvus、不依赖网络。
目的是验证：混入两路结果后，相关文档确实能排到前面，且消融开关生效。
"""

import pytest

from app.ingest.embedder import HashingEmbedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import InMemoryStore, build_store

DOCS = {
    "milvus": [
        "Milvus 是一个开源的向量数据库，支持 HNSW 和 IVF_FLAT 索引。",
        "HNSW 索引的 M 参数控制每个节点的出边数量，M 越大召回率越高但内存占用也越大。",
        "Milvus 的 efConstruction 参数影响建索引时的候选队列长度，调大会让图质量更好但建索引更慢。",
    ],
    "rag": [
        "RAG 是检索增强生成，先从知识库检索相关文档再交给大模型生成回答。",
        "混合检索结合了 BM25 关键词检索和向量语义检索，两路结果融合后召回率更高。",
        "RRF 即倒数排名融合，只用名次不用分数，因此不受两路分数量纲不同的影响。",
    ],
    "docker": [
        "Docker Compose 可以用一个配置文件编排多个容器，一条命令启动全部服务。",
        "容器的数据卷 volumes 用于持久化，删除容器后数据不会丢失。",
        "Docker 镜像分层构建，把不常变的层放在前面能利用缓存加速。",
    ],
}


@pytest.fixture
def retriever(tmp_path):
    embedder = HashingEmbedder(dim=256)
    store = InMemoryStore()
    bm25 = BM25Index(persist_path=str(tmp_path / "bm25.pkl"))
    r = HybridRetriever(store=store, bm25=bm25, embedder=embedder, dim=256)
    for doc_id, texts in DOCS.items():
        r.index_chunks(doc_id, texts)
    return r


class TestIndexing:

    def test_index_counts(self, retriever):
        stats = retriever.stats()
        assert stats["chunks"] == 9
        assert stats["bm25"]["docs"] == 3

    def test_bm25_persisted_and_reloadable(self, retriever, tmp_path):
        """索引必须能落盘重载——这是旧版最大的缺陷。"""
        pkl = tmp_path / "bm25.pkl"
        assert pkl.exists()
        reloaded = BM25Index(persist_path=str(pkl))
        assert reloaded.size == 9
        assert len(reloaded.search("向量数据库", top_k=3)) > 0

    def test_delete_removes_from_both(self, retriever):
        retriever.delete_document("docker")
        assert retriever.store.count() == 6
        assert retriever.bm25.size == 6


class TestRetrieval:

    def test_finds_relevant_doc(self, retriever):
        """问 BM25 融合的问题，应召回 rag 文档。"""
        result = retriever.retrieve("RRF 倒数排名融合怎么用", top_k=5, rerank_top_n=3)
        assert result.hits
        assert result.hits[0].doc_id == "rag"

    def test_hnsw_param_question_hits_milvus(self, retriever):
        result = retriever.retrieve("HNSW 的 M 参数是什么", top_k=5, rerank_top_n=3)
        assert result.hits[0].doc_id == "milvus"

    def test_trace_populated(self, retriever):
        result = retriever.retrieve("向量数据库", top_k=5)
        t = result.trace
        assert t.vector_candidates > 0
        assert t.bm25_candidates > 0
        assert t.total_ms >= 0
        d = t.as_dict()
        assert "latency_ms" in d and "candidates" in d

    def test_bm25_catches_exact_term_vector_misses(self, retriever):
        """查询里带精确的技术名词时，BM25 应该能命中。

        这正是混合检索存在的理由：向量对 ERR_XXX 这类字面量无能为力。
        """
        result = retriever.retrieve("efConstruction 调大有什么影响", top_k=5, use_rerank=False)
        docs = [h.doc_id for h in result.hits]
        assert "milvus" in docs

    def test_ablation_switches(self, retriever):
        """关掉 BM25 只剩向量路，结果仍应可用（不能崩）。"""
        result = retriever.retrieve("向量数据库", top_k=5, use_bm25=False)
        assert result.trace.bm25_candidates == 0
        assert result.hits

    def test_fusion_modes_all_work(self, retriever):
        for mode in ("rrf", "score", "concat"):
            result = retriever.retrieve("混合检索", top_k=5, fusion_mode=mode)
            assert result.hits, f"{mode} 模式返回空"
            assert result.trace.fusion_mode == mode

    def test_doc_id_filter(self, retriever):
        result = retriever.retrieve("索引", top_k=5, doc_ids=["docker"])
        assert all(h.doc_id == "docker" for h in result.hits)

    def test_empty_query_is_safe(self, retriever):
        result = retriever.retrieve("", top_k=5)
        assert isinstance(result.hits, list)  # 不崩即可


class TestStoreFactory:

    def test_memory_backend(self):
        s = build_store("memory")
        assert isinstance(s, InMemoryStore)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            build_store("redis")

    def test_inmemory_upsert_overwrites(self):
        from app.retrieval.vector_store import Chunk
        s = InMemoryStore()
        s.ensure_collection(4)
        s.upsert([Chunk(id="a#0", content="旧", embedding=[1, 0, 0, 0], doc_id="a", chunk_index=0)])
        s.upsert([Chunk(id="a#0", content="新", embedding=[0, 1, 0, 0], doc_id="a", chunk_index=0)])
        assert s.count() == 1
        hits = s.search([0, 1, 0, 0], top_k=1)
        assert hits[0].content == "新"

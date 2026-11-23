"""向量存储层。

设计取舍：
    用一个抽象基类 + 多实现，而不是直接在业务代码里调 pymilvus。
    原因是评测和单测需要在不启动 Milvus 的情况下跑通全链路——
    评测要跑几十组配置，每次都起容器不现实。

    另一个原因是旧版本的设计缺陷：它把每个文档建成一个 Chroma
    collection，检索时要遍历所有 collection 逐个查询。文档一多，
    查询次数线性增长，而且跨 collection 的分数没法统一排序。
    现在改成全局单 collection + doc_id 字段过滤，检索永远是 O(1) 次
    ANN 查询。

三个实现：
    MilvusStore   生产用，HNSW 索引
    QdrantStore   轻量部署备选（单容器，不需要 etcd/minio）
    InMemoryStore 单测和评测用，numpy 暴力检索
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Chunk:
    """入库的最小单元。"""
    id: str
    content: str
    embedding: List[float]
    doc_id: str
    chunk_index: int


@dataclass
class SearchHit:
    """检索返回的一条结果。

    score 的语义由具体实现决定（Milvus 的 IP 是越大越相似），
    所以上层一律不直接用 score 排序，而是交给 fusion 层按名次融合。
    """
    id: str
    content: str
    score: float
    doc_id: str = ""
    chunk_index: int = -1
    meta: Dict = field(default_factory=dict)


class BaseVectorStore(ABC):

    @abstractmethod
    def ensure_collection(self, dim: int, drop: bool = False) -> None: ...

    @abstractmethod
    def upsert(self, chunks: List[Chunk]) -> int: ...

    @abstractmethod
    def search(self, embedding: List[float], top_k: int,
               doc_ids: Optional[List[str]] = None) -> List[SearchHit]: ...

    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> None: ...

    @abstractmethod
    def list_documents(self) -> List[Dict]: ...

    @abstractmethod
    def count(self) -> int: ...

    def close(self) -> None:
        pass


class InMemoryStore(BaseVectorStore):
    """numpy 暴力检索。

    几千条 chunk 的规模下，一次矩阵乘法比 HNSW 的图遍历还快，
    而且没有部署依赖。评测脚本默认用它。
    """

    def __init__(self, normalize: bool = True):
        self._chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None
        self._normalize = normalize
        self._dim = 0

    def ensure_collection(self, dim: int, drop: bool = False) -> None:
        self._dim = dim
        if drop:
            self._chunks = []
            self._matrix = None

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        # 先移除同 id 的旧记录，再整体插入——语义对齐 Milvus 的 upsert。
        # 注意不能只插入"原本不存在"的那些，否则同 id 更新会变成删除。
        incoming = {c.id for c in chunks}
        if incoming & {c.id for c in self._chunks}:
            self._chunks = [c for c in self._chunks if c.id not in incoming]
        self._chunks.extend(chunks)
        self._rebuild()
        return len(chunks)

    def _rebuild(self) -> None:
        if not self._chunks:
            self._matrix = None
            return
        mat = np.asarray([c.embedding for c in self._chunks], dtype=np.float32)
        if self._normalize:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        self._matrix = mat

    def search(self, embedding: List[float], top_k: int,
               doc_ids: Optional[List[str]] = None) -> List[SearchHit]:
        if self._matrix is None or not self._chunks:
            return []

        q = np.asarray(embedding, dtype=np.float32)
        if self._normalize:
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n

        scores = self._matrix @ q

        allowed = set(doc_ids) if doc_ids else None
        if allowed is not None:
            mask = np.array([c.doc_id in allowed for c in self._chunks])
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)

        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            SearchHit(
                id=self._chunks[i].id,
                content=self._chunks[i].content,
                score=float(scores[i]),
                doc_id=self._chunks[i].doc_id,
                chunk_index=self._chunks[i].chunk_index,
            )
            for i in top_idx
            if np.isfinite(scores[i])
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        self._rebuild()

    def list_documents(self) -> List[Dict]:
        agg: Dict[str, int] = {}
        for c in self._chunks:
            agg[c.doc_id] = agg.get(c.doc_id, 0) + 1
        return [{"doc_id": k, "chunks": v} for k, v in agg.items()]

    def count(self) -> int:
        return len(self._chunks)


class MilvusStore(BaseVectorStore):
    """Milvus 单 collection + HNSW。

    索引参数取舍（面试高频）：
        M                 每个节点的出边数。调大 → 召回率高、内存占用大、
                          建索引慢。16 是 1024 维向量的常见起点。
        efConstruction    建索引时搜索的候选队列长度。调大 → 图质量好、
                          建索引慢。200 对应 10 万级 chunk 比较稳。
        ef（search 时）   查询时搜索的候选数，必须 >= top_k。调大 → 召回高、
                          延迟高。线上一般取 top_k 的 2-4 倍。

    为什么用 IP 而不是 COSINE：
        向量入库前已经 L2 归一化，此时内积恒等于余弦相似度，
        而 IP 的计算比 COSINE（要先算模长）省一步。代价是检索侧
        必须保证 query 也归一化——这个约束写在 embedder 里。
    """

    def __init__(self, uri: str, collection: str = "rag_chunks",
                 hnsw_m: int = 16, ef_construction: int = 200, ef_search: int = 64):
        from pymilvus import MilvusClient  # 懒加载：只用 InMemory 时不依赖 pymilvus

        self.client = MilvusClient(uri=uri)
        self.collection = collection
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def ensure_collection(self, dim: int, drop: bool = False) -> None:
        from pymilvus import DataType

        if drop and self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        if self.client.has_collection(self.collection):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_index", DataType.INT32)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": self.hnsw_m, "efConstruction": self.ef_construction},
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        rows = [
            {"id": c.id, "doc_id": c.doc_id, "chunk_index": c.chunk_index,
             "content": c.content, "embedding": c.embedding}
            for c in chunks
        ]
        self.client.upsert(collection_name=self.collection, data=rows)
        return len(rows)

    def search(self, embedding: List[float], top_k: int,
               doc_ids: Optional[List[str]] = None) -> List[SearchHit]:
        expr = None
        if doc_ids:
            quoted = ", ".join(f'"{d}"' for d in doc_ids)
            expr = f"doc_id in [{quoted}]"

        res = self.client.search(
            collection_name=self.collection,
            data=[embedding],
            limit=top_k,
            filter=expr,
            output_fields=["content", "doc_id", "chunk_index"],
            search_params={"metric_type": "IP", "params": {"ef": self.ef_search}},
        )
        hits: List[SearchHit] = []
        for row in (res[0] if res else []):
            hits.append(SearchHit(
                id=row["id"],
                content=row["entity"]["content"],
                score=float(row["distance"]),
                doc_id=row["entity"].get("doc_id", ""),
                chunk_index=int(row["entity"].get("chunk_index", -1)),
            ))
        return hits

    def delete_by_doc(self, doc_id: str) -> None:
        self.client.delete(collection_name=self.collection, filter=f'doc_id == "{doc_id}"')

    def list_documents(self) -> List[Dict]:
        rows = self.client.query(
            collection_name=self.collection,
            filter="id != ''",
            output_fields=["doc_id"],
            limit=16384,
        )
        agg: Dict[str, int] = {}
        for r in rows:
            agg[r["doc_id"]] = agg.get(r["doc_id"], 0) + 1
        return [{"doc_id": k, "chunks": v} for k, v in agg.items()]

    def count(self) -> int:
        res = self.client.query(
            collection_name=self.collection, filter="id != ''", output_fields=["count(*)"]
        )
        return int(res[0]["count(*)"]) if res else 0

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def build_store(backend: str, **kwargs) -> BaseVectorStore:
    """工厂。评测脚本靠它一行切换后端。"""
    backend = (backend or "memory").lower()
    if backend == "milvus":
        return MilvusStore(
            uri=kwargs.get("uri", "http://localhost:19530"),
            collection=kwargs.get("collection", "rag_chunks"),
            hnsw_m=kwargs.get("hnsw_m", 16),
            ef_construction=kwargs.get("ef_construction", 200),
            ef_search=kwargs.get("ef_search", 64),
        )
    if backend in ("milvus_rest", "milvusrest"):
        # 不依赖 pymilvus/grpcio 的实现，见 milvus_rest.py 顶部说明
        from .milvus_rest import MilvusRestStore
        return MilvusRestStore(
            uri=kwargs.get("uri", "http://localhost:19530"),
            collection=kwargs.get("collection", "rag_chunks"),
            hnsw_m=kwargs.get("hnsw_m", 16),
            ef_construction=kwargs.get("ef_construction", 200),
            ef_search=kwargs.get("ef_search", 64),
        )
    if backend in ("memory", "inmemory"):
        return InMemoryStore()
    raise ValueError(f"未知的向量后端: {backend}（可选 milvus / milvus_rest / memory）")

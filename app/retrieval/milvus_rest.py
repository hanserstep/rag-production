"""Milvus RESTful 实现。

为什么不用官方 pymilvus SDK：
    pymilvus 依赖 gRPC，在 Python 3.13 上 grpcio 还没有预编译 wheel，
    装的时候要走源码编译（几分钟且容易失败）。Milvus 2.4 起提供了完整的
    RESTful v2 接口，httpx 一个依赖就能覆盖建表、写入、检索、删除。

    代价是比 SDK 少一些高级能力（如 bulk insert、多向量字段），
    对本项目的用法足够。需要用 SDK 时把 VECTOR_BACKEND 换成 milvus，
    两个实现共用同一个 BaseVectorStore 接口，上层无感知。

接口文档：https://milvus.io/api-reference/restful/v2.4.x/
"""

from typing import Dict, List, Optional

from .vector_store import BaseVectorStore, Chunk, SearchHit


class MilvusRestStore(BaseVectorStore):

    def __init__(self, uri: str = "http://localhost:19530", collection: str = "rag_chunks",
                 hnsw_m: int = 16, ef_construction: int = 200, ef_search: int = 64,
                 timeout: float = 30.0):
        import httpx
        self.base = uri.rstrip("/") + "/v2/vectordb"
        self.collection = collection
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._client = httpx.Client(timeout=timeout)
        self._loaded = False

    # ---- 内部 ----
    def _post(self, path: str, payload: Dict) -> Dict:
        resp = self._client.post(f"{self.base}{path}", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Milvus {path} 失败: {data.get('message')}")
        return data

    def _has_collection(self) -> bool:
        data = self._post("/collections/list", {})
        return self.collection in (data.get("data") or [])

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._post("/collections/load", {"collectionName": self.collection})
        self._loaded = True

    # ---- 接口实现 ----
    def ensure_collection(self, dim: int, drop: bool = False) -> None:
        if drop and self._has_collection():
            self._post("/collections/drop", {"collectionName": self.collection})
            self._loaded = False
        if self._has_collection():
            self._ensure_loaded()
            return

        # 注意：REST 接口只支持 AutoIndex——传 indexType=HNSW 会被拒绝
        # （"only metric type can be passed when use AutoIndex"）。
        # AutoIndex 对 FloatVector 字段内部选用的就是 HNSW，只是 M 和
        # efConstruction 由 Milvus 按数据规模自动决定，不能手动指定。
        # 需要显式调参时走 pymilvus（VECTOR_BACKEND=milvus）。
        self._post("/collections/create", {
            "collectionName": self.collection,
            "schema": {
                "autoId": False,
                "enableDynamicField": False,
                "fields": [
                    {"fieldName": "id", "dataType": "VarChar", "isPrimary": True,
                     "elementTypeParams": {"max_length": 128}},
                    {"fieldName": "doc_id", "dataType": "VarChar",
                     "elementTypeParams": {"max_length": 64}},
                    {"fieldName": "chunk_index", "dataType": "Int32"},
                    {"fieldName": "content", "dataType": "VarChar",
                     "elementTypeParams": {"max_length": 8192}},
                    {"fieldName": "embedding", "dataType": "FloatVector",
                     "elementTypeParams": {"dim": dim}},
                ],
            },
        })
        self._create_auto_index()
        self._loaded = False
        self._ensure_loaded()

    def _create_auto_index(self) -> None:
        """建索引是 load 的前提，没有索引的 collection 不能加载。"""
        try:
            self._post("/indexes/create", {
                "collectionName": self.collection,
                "indexParams": [{
                    "fieldName": "embedding",
                    "indexName": "embedding_idx",
                    "metricType": "IP",
                }],
            })
        except RuntimeError as e:
            # 已经建过就跳过，其他错误照常抛
            if "already" not in str(e).lower() and "exist" not in str(e).lower():
                raise

    def upsert(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        self._ensure_loaded()
        # REST 的 upsert 在部分版本上对主键冲突处理不一致，
        # 这里显式走 upsert 接口，并按批切分避免单次请求过大
        batch = 200
        total = 0
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self._post("/entities/upsert", {
                "collectionName": self.collection,
                "data": [
                    {"id": c.id, "doc_id": c.doc_id, "chunk_index": c.chunk_index,
                     "content": c.content, "embedding": c.embedding}
                    for c in part
                ],
            })
            total += len(part)
        return total

    def search(self, embedding: List[float], top_k: int,
               doc_ids: Optional[List[str]] = None) -> List[SearchHit]:
        self._ensure_loaded()
        payload = {
            "collectionName": self.collection,
            "data": [embedding],
            "annsField": "embedding",
            "limit": top_k,
            "outputFields": ["content", "doc_id", "chunk_index"],
            "searchParams": {"metricType": "IP", "params": {"ef": self.ef_search}},
        }
        if doc_ids:
            quoted = ", ".join(f'"{d}"' for d in doc_ids)
            payload["filter"] = f"doc_id in [{quoted}]"

        data = self._post("/entities/search", payload)
        hits = []
        for row in (data.get("data") or []):
            hits.append(SearchHit(
                id=str(row.get("id", "")),
                content=row.get("content", ""),
                score=float(row.get("distance", 0.0)),
                doc_id=row.get("doc_id", ""),
                chunk_index=int(row.get("chunk_index", -1)),
            ))
        return hits

    def delete_by_doc(self, doc_id: str) -> None:
        if not self._has_collection():
            return
        self._ensure_loaded()
        self._post("/entities/delete", {
            "collectionName": self.collection,
            "filter": f'doc_id == "{doc_id}"',
        })

    def list_documents(self) -> List[Dict]:
        if not self._has_collection():
            return []
        self._ensure_loaded()
        data = self._post("/entities/query", {
            "collectionName": self.collection,
            "filter": "chunk_index >= 0",
            "outputFields": ["doc_id"],
            "limit": 16384,
        })
        agg: Dict[str, int] = {}
        for r in (data.get("data") or []):
            d = r.get("doc_id", "")
            agg[d] = agg.get(d, 0) + 1
        return [{"doc_id": k, "chunks": v} for k, v in agg.items()]

    def count(self) -> int:
        if not self._has_collection():
            return 0
        self._ensure_loaded()
        data = self._post("/entities/query", {
            "collectionName": self.collection,
            "filter": "chunk_index >= 0",
            "outputFields": ["count(*)"],
        })
        rows = data.get("data") or []
        return int(rows[0].get("count(*)", 0)) if rows else 0

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

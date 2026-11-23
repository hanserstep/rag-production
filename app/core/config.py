"""集中配置。

所有可变参数都从环境变量读，不在代码里写死——这样 docker-compose、
本地开发、评测脚本可以共用同一份代码，只切 .env。

面试点：为什么配置要集中？
    检索参数（chunk_size / top_k / rrf_k）是调优的对象。如果散落在
    各个模块里，做 A/B 评测时得改代码再重启；集中之后评测脚本可以
    用参数覆盖默认值，同一个进程里跑多组配置。
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:

    # ---- 向量存储 ----
    vector_backend: str = field(default_factory=lambda: os.getenv("VECTOR_BACKEND", "memory"))
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530"))
    collection_name: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION", "rag_chunks"))

    # ---- Embedding ----
    # api = 硅基流动（快、按量计费）；local = 本地 bge-m3（离线、慢、占 2.2G 显存/内存）
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "api"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 1024))
    siliconflow_api_key: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", ""))

    # ---- 重排序 ----
    rerank_backend: str = field(default_factory=lambda: os.getenv("RERANK_BACKEND", "api"))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))

    # ---- LLM ----
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))

    # ---- 分块 ----
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 50))

    # ---- 检索 ----
    top_k: int = field(default_factory=lambda: _int("TOP_K", 20))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    rerank_top_n: int = field(default_factory=lambda: _int("RERANK_TOP_N", 3))
    fusion_mode: str = field(default_factory=lambda: os.getenv("FUSION_MODE", "rrf"))

    # ---- Milvus 索引参数（面试会被问 M 和 efConstruction 的取舍） ----
    hnsw_m: int = field(default_factory=lambda: _int("HNSW_M", 16))
    hnsw_ef_construction: int = field(default_factory=lambda: _int("HNSW_EF_CONSTRUCTION", 200))
    hnsw_ef_search: int = field(default_factory=lambda: _int("HNSW_EF_SEARCH", 64))

    # ---- Redis ----
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    cache_ttl: int = field(default_factory=lambda: _int("CACHE_TTL", 3600))

    # ---- 成本核算（单位：元 / 1K tokens，用于 metric 里估算单次问答成本） ----
    price_prompt_per_1k: float = field(default_factory=lambda: _float("PRICE_PROMPT_PER_1K", 0.001))
    price_completion_per_1k: float = field(default_factory=lambda: _float("PRICE_COMPLETION_PER_1K", 0.002))
    price_embedding_per_1k: float = field(default_factory=lambda: _float("PRICE_EMBEDDING_PER_1K", 0.0005))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

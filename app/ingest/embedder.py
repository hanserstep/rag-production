"""Embedding 层。

两种后端：

api  —— 硅基流动（SiliconFlow）提供的 bge-m3，OpenAI 兼容协议。
        优点：不用下载 2.2G 权重，Mac 上 encode 一条 query 约 80ms。
        缺点：按量计费，依赖网络。

local —— 本地 sentence-transformers + bge-m3。
        优点：离线、免费、数据不出内网（私有化部署场景必须）。
        缺点：CPU 上 encode 一条 query 要 400ms+，冷启动加载模型 10s。

为什么两个都要：JD 里「私有化部署」出现 4 次，客户内网是连不了
外部 API 的。代码里留了切换开关，README 里说明离线环境怎么跑。

一个必须遵守的约定：所有输出向量都做 L2 归一化。
因为向量库用的是 IP（内积）索引，只有归一化之后内积才等于余弦相似度。
这个约束如果漏掉，检索结果会静默地变差——不报错，只是不准，很难查。
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n == 0:
        return arr.tolist()
    return (arr / n).tolist()


class BaseEmbedder(ABC):

    dim: int = 1024

    @abstractmethod
    def encode_documents(self, texts: List[str]) -> List[List[float]]: ...

    @abstractmethod
    def encode_query(self, text: str) -> List[float]: ...


class SiliconFlowEmbedder(BaseEmbedder):
    """硅基流动 bge-m3。

    批量策略：不能只按条数切。

    接口对单条条数有限制（32 条），但对单次请求的**总 token 数也有限制**。
    只按 32 条切会在文档块较长时超限，返回一个语焉不详的
    "The parameter is invalid"——这个 400 和参数格式无关，是总量超了，
    排查时容易误以为是模型名或字段写错。

    所以在条数之外再加一道总字符数约束。中文约 1 字 1 token，
    取 5000 字符作为保守上限，留出安全余量。
    """

    BATCH = 32
    MAX_CHARS = 5000

    def __init__(self, api_key: str, model: str = "BAAI/bge-m3", dim: int = 1024,
                 base_url: str = "https://api.siliconflow.cn/v1"):
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY 未配置，无法使用 api 后端")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dim = dim
        self._total_tokens = 0  # 累计 token，用于成本统计

    def _split_batches(self, texts: List[str]):
        batch: List[str] = []
        chars = 0
        for t in texts:
            # 单条就超上限时只能单独发，否则会死循环切不出来
            if batch and (len(batch) >= self.BATCH or chars + len(t) > self.MAX_CHARS):
                yield batch
                batch, chars = [], 0
            batch.append(t)
            chars += len(t)
        if batch:
            yield batch

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for batch in self._split_batches(texts):
            resp = self.client.embeddings.create(model=self.model, input=batch)
            self._total_tokens += getattr(resp.usage, "total_tokens", 0) or 0
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend(l2_normalize(d.embedding) for d in ordered)
        return out

    def encode_query(self, text: str) -> List[float]:
        return self.encode_documents([text])[0]


class LocalEmbedder(BaseEmbedder):
    """本地 bge-m3。

    Mac 上优先用 mps（Metal），比 cpu 快 2-3 倍；
    拿不到 mps 时自动退回 cpu。Linux 有 GPU 就填 cuda。
    """

    def __init__(self, model_path: str, dim: int = 1024, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        self.model = SentenceTransformer(model_path, device=device)
        self.dim = dim
        self.device = device

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> List[float]:
        return self.encode_documents([text])[0]


class CachedEmbedder(BaseEmbedder):
    """给任意 embedder 套一层缓存。

    评测时同一条 query 会被多组配置反复编码，缓存能省掉大量重复调用。
    没有 Redis 时退化为进程内字典。
    """

    def __init__(self, inner: BaseEmbedder, redis_client=None, ttl: int = 3600):
        self.inner = inner
        self.dim = inner.dim
        self.redis = redis_client
        self.ttl = ttl
        self._local: dict = {}
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"emb:{h}"

    def _get(self, text: str):
        key = self._key(text)
        if self.redis is not None:
            try:
                raw = self.redis.get(key)
                if raw:
                    import json
                    self.hits += 1
                    return json.loads(raw)
            except Exception:
                pass
        if key in self._local:
            self.hits += 1
            return self._local[key]
        self.misses += 1
        return None

    def _set(self, text: str, vec: List[float]) -> None:
        key = self._key(text)
        self._local[key] = vec
        if self.redis is not None:
            try:
                import json
                self.redis.setex(key, self.ttl, json.dumps(vec))
            except Exception:
                pass

    def encode_query(self, text: str) -> List[float]:
        cached = self._get(text)
        if cached is not None:
            return cached
        vec = self.inner.encode_query(text)
        self._set(text, vec)
        return vec

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [self._get(t) for t in texts]
        missing = [t for t, r in zip(texts, results) if r is None]
        if missing:
            fresh = self.inner.encode_documents(missing)
            for t, v in zip(missing, fresh):
                self._set(t, v)
            it = iter(fresh)
            results = [r if r is not None else next(it) for r in results]
        return results  # type: ignore[return-value]

    @property
    def cache_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def build_embedder(backend: str = "api", **kwargs) -> BaseEmbedder:
    backend = (backend or "api").lower()
    if backend == "api":
        return SiliconFlowEmbedder(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "BAAI/bge-m3"),
            dim=kwargs.get("dim", 1024),
        )
    if backend == "local":
        return LocalEmbedder(
            model_path=kwargs.get("model_path", "BAAI/bge-m3"),
            dim=kwargs.get("dim", 1024),
            device=kwargs.get("device"),
        )
    if backend == "ollama":
        return OllamaEmbedder(
            model=kwargs.get("ollama_model", "bge-m3"),
            dim=kwargs.get("dim", 1024),
            host=kwargs.get("ollama_host", "http://localhost:11434"),
        )
    if backend == "hashing":
        return HashingEmbedder(dim=kwargs.get("dim", 256))
    raise ValueError(f"未知的 embedding 后端: {backend}（可选 api / local / ollama / hashing）")


class OllamaEmbedder(BaseEmbedder):
    """本地 Ollama 的 embedding 接口。

    适合的场景：机器上已经有 Ollama 且拉过 bge-m3（约 1.2G），
    比 pip 一套 torch + sentence-transformers（400M+）轻，启动也快。
    数据不出本机，等同私有化部署。

    接口：POST /api/embed  {"model": "...", "input": [...]}
    返回：{"embeddings": [[...], [...]]}
    """

    def __init__(self, model: str = "bge-m3", dim: int = 1024,
                 host: str = "http://localhost:11434", timeout: float = 60.0):
        import httpx
        self._client = httpx.Client(timeout=timeout)
        self.host = host.rstrip("/")
        self.model = model
        self.dim = dim

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        return [l2_normalize(v) for v in resp.json()["embeddings"]]

    def encode_query(self, text: str) -> List[float]:
        return self.encode_documents([text])[0]


class HashingEmbedder(BaseEmbedder):
    """字符 n-gram 哈希向量，只给单测用。

    不能直接用随机向量：随机向量下任意两段文本都正交，测不出
    "相关文档排在前面"这件事，检索测试就退化成了流程测试。

    这里用 2-gram 做特征哈希（feature hashing）——共享字词的文本会
    落到相同的维度上，余弦相似度因此有了意义，足够验证检索链路。
    它当然不具备真实的语义能力，所以只用于单测，评测必须用真模型。
    """

    def __init__(self, dim: int = 256, n: int = 2):
        self.dim = dim
        self.n = n

    def _vec(self, text: str) -> List[float]:
        import hashlib

        v = np.zeros(self.dim, dtype=np.float32)
        s = text.replace(" ", "")
        grams = [s[i:i + self.n] for i in range(max(1, len(s) - self.n + 1))]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        # 次线性缩放，避免长文本被词频主导
        v = np.sign(v) * np.sqrt(np.abs(v))
        return l2_normalize(v.tolist())

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def encode_query(self, text: str) -> List[float]:
        return self._vec(text)

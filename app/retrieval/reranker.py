"""重排序（Cross-Encoder）。

为什么召回之后还要重排：
    Bi-Encoder（向量模型）把 query 和 doc 分别编码成向量再算相似度，
    doc 的向量是离线算好存起来的，所以能支撑百万级检索，但 query 和
    doc 之间没有交互，精度有上限。

    Cross-Encoder 把 [query, doc] 拼成一条输入一起过模型，token 级别
    充分交互，精度高很多，但每条候选都要跑一次前向，成本随候选数线性
    增长，不可能用来做全库检索。

    所以标准做法是两段式：Bi-Encoder 粗召回 20 条 → Cross-Encoder 精排
    取 3 条。这也是 RAG 里最标准的一层，JD 里写"重排"就是指它。

成本量级对比（1024 维、10 万 chunk 的库）：
    粗召回 20 条   1 次 ANN 查询      ~5ms
    精排 20 条     20 次 Cross-Encoder ~600ms（CPU）/ ~80ms（API）
    全库精排       10 万次           不可能
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .vector_store import SearchHit


class BaseReranker(ABC):

    @abstractmethod
    def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]: ...


class NoOpReranker(BaseReranker):
    """不做重排，用于评测对照——证明重排这一层到底带来了多少增益。"""

    def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]:
        return hits[:top_n]


class SiliconFlowReranker(BaseReranker):
    """硅基流动的 bge-reranker-v2-m3。

    返回的 relevance_score 是 0-1 的相关性，越大越相关。
    """

    def __init__(self, api_key: str, model: str = "BAAI/bge-reranker-v2-m3",
                 base_url: str = "https://api.siliconflow.cn/v1"):
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY 未配置，无法使用 api 重排序")
        import httpx
        self._client = httpx.Client(timeout=20.0)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]:
        if not hits:
            return []
        resp = self._client.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [h.content for h in hits],
                "top_n": min(top_n, len(hits)),
                "return_documents": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        out: List[SearchHit] = []
        for item in data.get("results", []):
            h = hits[item["index"]]
            h.meta["rerank_score"] = float(item["relevance_score"])
            out.append(h)
        return out[:top_n]


class LocalReranker(BaseReranker):
    """本地 CrossEncoder。CPU 上 20 条约 600ms，够用于离线评测。"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        if device is None:
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.model = CrossEncoder(model_path, device=device)

    def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]:
        if not hits:
            return []
        pairs = [[query, h.content] for h in hits]
        scores = self.model.predict(pairs)
        for h, s in zip(hits, scores):
            h.meta["rerank_score"] = float(s)
        ranked = sorted(hits, key=lambda h: -h.meta.get("rerank_score", 0.0))
        return ranked[:top_n]


class ScoreThresholdReranker(BaseReranker):
    """包一层相关性阈值。

    真实业务里，检索不到相关内容时应该让模型回答"知识库里没有"，
    而不是硬塞 3 条不相关的上下文让模型编。这是降低幻觉最有效的一招，
    比在 Prompt 里写"不要编造"管用得多。
    """

    def __init__(self, inner: BaseReranker, threshold: float = 0.3):
        self.inner = inner
        self.threshold = threshold

    def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]:
        ranked = self.inner.rerank(query, hits, top_n)
        kept = [h for h in ranked if h.meta.get("rerank_score", 1.0) >= self.threshold]
        return kept


def build_reranker(backend: str = "api", **kwargs) -> BaseReranker:
    backend = (backend or "none").lower()
    if backend == "api":
        return SiliconFlowReranker(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "BAAI/bge-reranker-v2-m3"),
        )
    if backend == "local":
        return LocalReranker(model_path=kwargs.get("model_path", "BAAI/bge-reranker-v2-m3"))
    if backend in ("none", "noop"):
        return NoOpReranker()
    raise ValueError(f"未知的 rerank 后端: {backend}（可选 api / local / none）")

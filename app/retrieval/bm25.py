"""BM25 关键词检索。

为什么还要保留 BM25（向量检索已经很强了）：
    向量检索对专有名词、错误码、函数名这类"字面量"经常召不回。
    问「ERR_CONN_REFUSED 怎么处理」，bge-m3 会把它编码成一个模糊的
    "连接报错"语义，反而召回一堆不相干的连接超时文档；BM25 靠精确
    词频能直接命中那个错误码。
    两者互补，这也是所有 JD 都写"混合检索"的原因。

相对旧版本改了三处：
    1. 索引可持久化。旧版存在内存 dict 里，进程重启索引全丢，
       必须重新上传所有文档才能检索——这是 demo 和线上的分界线。
    2. 支持 doc_id 过滤，不用再"每文档一个索引"。
    3. 返回统一的 SearchHit，让上层融合逻辑不关心数据来自哪一路。

分词：
    jieba 默认模式对技术术语切得不好（"内存泄漏"会被切成"内存/泄漏"，
    还行；但 "RAGFlow" 会被切碎）。这里额外加了用户词典兜底，
    并且对英文技术词做了保留——分词质量直接决定 BM25 的上限。
"""

import os
import pickle
import re
from typing import Dict, List, Optional

from .vector_store import Chunk, SearchHit


class BM25Index:
    """全局单索引 + doc_id 过滤，替代旧版"每文档一个索引"。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 persist_path: Optional[str] = None):
        self.k1 = k1
        self.b = b
        self.persist_path = persist_path

        self._ids: List[str] = []
        self._doc_ids: List[str] = []
        self._chunk_indexes: List[int] = []
        self._texts: List[str] = []
        self._tokens: List[List[str]] = []
        self._bm25 = None
        self._dirty = False

        self._jieba_ready = None  # None = 未检测；True/False = 检测结果
        self._jieba = None
        if persist_path and os.path.exists(persist_path):
            self.load()

    # ---- 分词 ----
    def _ensure_jieba(self) -> bool:
        """jieba 是可选的增强项，不是硬依赖。

        装了就用来切词（对多字词的切分更符合语言习惯）；
        没装就走 bigram 兜底，功能不受影响。
        这样部署时少一个 40MB 的词典，镜像也小一圈。
        """
        if self._jieba_ready is not None:
            return self._jieba_ready
        try:
            import jieba
            for w in [
                "RAGFlow", "LangGraph", "LangChain", "bge-m3", "bge-reranker",
                "efConstruction", "HNSW", "IVF_FLAT", "Cross-Encoder",
                "Bi-Encoder", "Embedding", "ChromaDB", "Milvus", "Qdrant",
                "召回率", "重排序", "混合检索", "向量数据库", "知识库",
            ]:
                jieba.add_word(w, freq=10000)
            jieba.setLogLevel(60)
            self._jieba = jieba
            self._jieba_ready = True
        except ImportError:
            self._jieba = None
            self._jieba_ready = False
        return self._jieba_ready

    def tokenize(self, text: str) -> List[str]:
        """英文/数字整体保留 + 中文分词。

        中文部分：有 jieba 用 jieba，否则用字符 bigram。
        bigram 是 Lucene 的 CJKAnalyzer 对中文的默认处理方式——
        不依赖词典，对 HNSW、efConstruction 这类未登录词天然友好，
        代价是索引体积比词级切分大约 1.5 倍（可接受）。
        """
        use_jieba = self._ensure_jieba()
        tokens: List[str] = []

        for seg in re.split(r"([A-Za-z][A-Za-z0-9_\-\.]*)", text):
            if not seg:
                continue
            if re.match(r"^[A-Za-z]", seg):
                tokens.append(seg.lower())
                continue

            cleaned = re.sub(r"\s+", "", seg)
            if not cleaned:
                continue

            if use_jieba:
                tokens.extend(t for t in self._jieba.cut(cleaned) if t.strip())
            else:
                if len(cleaned) == 1:
                    tokens.append(cleaned)
                else:
                    tokens.extend(cleaned[i:i + 2] for i in range(len(cleaned) - 1))

        return [t for t in tokens if not re.fullmatch(r"[\s\W]+", t)]

    # ---- 索引维护 ----
    def add(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        incoming = {c.id for c in chunks}
        # 同 id 覆盖
        if any(i in incoming for i in self._ids):
            keep = [i for i, cid in enumerate(self._ids) if cid not in incoming]
            self._ids = [self._ids[i] for i in keep]
            self._doc_ids = [self._doc_ids[i] for i in keep]
            self._chunk_indexes = [self._chunk_indexes[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._tokens = [self._tokens[i] for i in keep]

        for c in chunks:
            self._ids.append(c.id)
            self._doc_ids.append(c.doc_id)
            self._chunk_indexes.append(c.chunk_index)
            self._texts.append(c.content)
            self._tokens.append(self.tokenize(c.content))

        self._rebuild()
        self._dirty = True
        return len(chunks)

    def _rebuild(self) -> None:
        if not self._tokens:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokens, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 20,
               doc_ids: Optional[List[str]] = None) -> List[SearchHit]:
        if self._bm25 is None or not self._texts:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)

        allowed = set(doc_ids) if doc_ids else None
        candidates = []
        for i, s in enumerate(scores):
            # BM25 可能出现负分（idf 为负的常见词），这类结果没有排序意义
            if s <= 0:
                continue
            if allowed is not None and self._doc_ids[i] not in allowed:
                continue
            candidates.append((i, float(s)))

        candidates.sort(key=lambda x: -x[1])
        return [
            SearchHit(
                id=self._ids[i],
                content=self._texts[i],
                score=s,
                doc_id=self._doc_ids[i],
                chunk_index=self._chunk_indexes[i],
            )
            for i, s in candidates[:top_k]
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        keep = [i for i, d in enumerate(self._doc_ids) if d != doc_id]
        if len(keep) == len(self._doc_ids):
            return
        self._ids = [self._ids[i] for i in keep]
        self._doc_ids = [self._doc_ids[i] for i in keep]
        self._chunk_indexes = [self._chunk_indexes[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._tokens = [self._tokens[i] for i in keep]
        self._rebuild()
        self._dirty = True

    # ---- 持久化 ----
    def save(self) -> None:
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        tmp = self.persist_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({
                "ids": self._ids, "doc_ids": self._doc_ids,
                "chunk_indexes": self._chunk_indexes,
                "texts": self._texts, "tokens": self._tokens,
                "k1": self.k1, "b": self.b,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.persist_path)  # 原子替换，避免写一半崩掉留坏文件
        self._dirty = False

    def load(self) -> None:
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            return
        self._ids = data.get("ids", [])
        self._doc_ids = data.get("doc_ids", [])
        self._chunk_indexes = data.get("chunk_indexes", [])
        self._texts = data.get("texts", [])
        self._tokens = data.get("tokens", [])
        self.k1 = data.get("k1", self.k1)
        self.b = data.get("b", self.b)
        self._rebuild()

    @property
    def size(self) -> int:
        return len(self._ids)

    def stats(self) -> Dict:
        return {"chunks": len(self._ids), "docs": len(set(self._doc_ids))}

"""分块与内容过滤测试。

重点测过滤逻辑：真实文档里塞满了目录树和 ASCII 流程图，切块后它们会变成
一堆符号碎片。这些块没有语义但会参与相似度计算、占用 top_k 名额，
必须在入库前过滤掉。
"""

import pytest

from app.ingest.chunker import DocumentChunker


@pytest.fixture
def chunker():
    return DocumentChunker()


class TestContentFilter:

    def test_directory_tree_is_dropped(self, chunker):
        tree = """
rag-qa-system/
├── app.py
├── backend/
│   ├── main.py
│   └── retriever.py
└── requirements.txt
"""
        assert not chunker.split(tree, chunk_size=500, overlap=0)

    def test_ascii_diagram_is_dropped(self, chunker):
        diagram = """
┌─────────────┐     ┌─────────────┐
│  上传文档    │ ──▶ │  文本切块    │
└─────────────┘     └─────────────┘
        │                   │
        ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  向量化      │     │  BM25 索引   │
└─────────────┘     └─────────────┘
"""
        assert not chunker.split(diagram, chunk_size=500, overlap=0)

    def test_normal_paragraph_kept(self, chunker):
        text = ("向量检索把文本映射到高维空间，语义相近的文本距离也近。"
                "这样就能跨过字面差异找到相关内容，代价是失去了精确性。"
                "所以生产系统几乎都同时保留关键词检索，做混合召回。")
        assert len(chunker.split(text, chunk_size=500, overlap=0)) == 1

    def test_ratio_computation(self, chunker):
        assert chunker._content_ratio("中文abc123") == pytest.approx(1.0)
        assert chunker._content_ratio("────│┌┐└┘") == 0.0
        # 一半符号一半汉字
        assert chunker._content_ratio("中文──││") == pytest.approx(2 / 6)

    def test_short_chunk_dropped(self, chunker):
        assert not chunker.split("太短了", chunk_size=500, overlap=0)

    def test_code_block_survives(self, chunker):
        """代码块的有效字符比例偏低，但不能被误杀——这是阈值取 0.35 的原因。"""
        code = """
def retrieve(query, top_k=5):
    vector_hits = store.search(encode(query), top_k)
    bm25_hits = bm25.search(query, top_k)
    fused = reciprocal_rank_fusion({"vector": vector_hits, "bm25": bm25_hits})
    return rerank(query, fused, top_n=3)
"""
        chunks = chunker.split(code, chunk_size=500, overlap=0)
        assert len(chunks) == 1
        assert "reciprocal_rank_fusion" in chunks[0]

    def test_mixed_document_keeps_only_prose(self, chunker):
        mixed = """向量数据库的选型要看数据规模和过滤需求这两个维度，
规模到百万级才需要认真考虑索引结构，小规模用什么都够用。

rag-qa-system/
├── app.py
├── backend/
│   └── main.py

元数据过滤最容易被低估，真实业务几乎一定需要按租户或者时间范围过滤，
如果向量库只支持检索后过滤，条件一严召回就会归零。
"""
        chunks = chunker.split(mixed, chunk_size=500, overlap=0)
        assert len(chunks) == 1
        assert "选型" in chunks[0] and "元数据过滤" in chunks[0]
        assert "├──" not in chunks[0]


class TestSplitBehavior:

    def test_overlap_applied(self, chunker):
        text = "\n\n".join([
            "第一段讲了向量检索的基本原理和它在语义泛化上的优势。" * 3,
            "第二段讲了关键词检索在精确匹配上的不可替代性。" * 3,
            "第三段讲了两种检索方式如何互补形成混合召回。" * 3,
        ])
        chunks = chunker.split(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        # 后一块的开头应包含前一块的尾部内容
        assert chunks[1][:20] in chunks[0]

    def test_no_overlap_when_single_chunk(self, chunker):
        text = "只有一段话，长度足够通过过滤，所以应该完整保留下来不被切分。这句话用来把字符数凑到四十以上，确保它不会被最小长度规则误伤。"
        chunks = chunker.split(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

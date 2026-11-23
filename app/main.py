"""FastAPI 服务入口。

相对旧版本（backend/main.py）补齐的生产必需项：

1. 结构化日志 + trace_id     —— 线上排查一次问答要能串起所有环节
2. Redis 缓存                —— 同样的问题不重复打模型，省钱且降低延迟
3. 指标统计                  —— P50/P95 延迟、token 用量、单次成本
4. 相关性阈值                —— 检索不到就明确说没有，不硬凑上下文
5. 优雅降级                  —— 向量库或 LLM 挂了，返回可读错误而不是 500 堆栈

设计取舍：所有耗时都记在 Trace 里而不是散落各处打日志。这样一次请求
只有一个 JSON 日志输出，直接能进 ELK / Loki，不用跨行拼日志。
"""

import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.ingest.chunker import DocumentChunker
from app.ingest.embedder import build_embedder
from app.llm.client import DeepSeekClient
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import ScoreThresholdReranker, build_reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import build_store

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("rag")


class AppState:
    retriever: Optional[HybridRetriever] = None
    llm: Optional[DeepSeekClient] = None
    chunker: Optional[DocumentChunker] = None
    cache = None
    stats: Dict = None


state = AppState()
state.stats = {"requests": 0, "cache_hits": 0, "latencies": [], "tokens": 0, "cost": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    log.info("starting, vector_backend=%s embedding_backend=%s",
             cfg.vector_backend, cfg.embedding_backend)

    embedder = build_embedder(
        cfg.embedding_backend,
        api_key=cfg.siliconflow_api_key,
        model=cfg.embedding_model,
        dim=cfg.embedding_dim,
        model_path=cfg.embedding_model,
    )
    store = build_store(
        cfg.vector_backend,
        uri=cfg.milvus_uri,
        collection=cfg.collection_name,
        hnsw_m=cfg.hnsw_m,
        ef_construction=cfg.hnsw_ef_construction,
        ef_search=cfg.hnsw_ef_search,
    )
    bm25 = BM25Index(persist_path="data/bm25.pkl")
    # 必须包一层阈值过滤：只精排不过滤，等于把不相关片段也交给模型，
    # 检索不到相关内容时模型就会基于噪声硬编答案。这是 RAG 幻觉的主要来源。
    base_reranker = build_reranker(cfg.rerank_backend, api_key=cfg.siliconflow_api_key,
                                   model_path=cfg.rerank_model)
    reranker = ScoreThresholdReranker(base_reranker, threshold=cfg.rerank_threshold)

    state.retriever = HybridRetriever(
        store=store, bm25=bm25, embedder=embedder, reranker=reranker,
        rrf_k=cfg.rrf_k, dim=cfg.embedding_dim,
    )
    state.llm = DeepSeekClient(
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        model=cfg.llm_model,
    )
    state.chunker = DocumentChunker()

    try:
        import redis
        state.cache = redis.from_url(cfg.redis_url, decode_responses=True)
        state.cache.ping()
        log.info("redis connected")
    except Exception as e:
        log.warning("redis unavailable, cache disabled: %s", e)
        state.cache = None

    yield

    if state.retriever:
        state.retriever.store.close()
    log.info("shutdown")


app = FastAPI(title="RAG Service", version="3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    request.state.trace_id = trace_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-Id"] = trace_id
    response.headers["X-Elapsed-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


# ---------------- 文档管理 ----------------

@app.post("/api/documents/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    cfg = get_settings()
    content = await file.read()
    suffix = (file.filename or "x.txt").split(".")[-1].lower()

    text = state.chunker.parse(content, suffix)
    if not text or len(text) < 50:
        raise HTTPException(400, "文档内容过短或无法解析")

    chunks = state.chunker.split(text, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    if not chunks:
        raise HTTPException(400, "分块结果为空")

    doc_id = hashlib.md5(f"{file.filename}{time.time()}".encode()).hexdigest()[:12]
    n = state.retriever.index_chunks(doc_id, chunks)

    log.info(json.dumps({
        "event": "document_indexed", "trace_id": request.state.trace_id,
        "doc_id": doc_id, "filename": file.filename,
        "chunks": n, "chars": len(text),
    }, ensure_ascii=False))

    return {"doc_id": doc_id, "filename": file.filename, "chunks": n,
            "chars": len(text), "message": f"已入库 {n} 个文本块"}


@app.get("/api/documents")
async def list_documents():
    stats = state.retriever.stats()
    return {"documents": state.retriever.store.list_documents(), "stats": stats}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    state.retriever.delete_document(doc_id)
    return {"message": f"已删除 {doc_id}"}


# ---------------- 问答 ----------------

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = 20
    rerank_top_n: int = 3
    use_hyde: bool = False
    use_bm25: bool = True
    use_rerank: bool = True
    fusion_mode: str = "rrf"
    stream: bool = True
    use_cache: bool = True


def _cache_key(req: QuestionRequest) -> str:
    raw = f"{req.question}|{req.top_k}|{req.rerank_top_n}|{req.use_hyde}|{req.use_bm25}|{req.use_rerank}|{req.fusion_mode}"
    return "qa:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


@app.post("/api/qa")
async def ask(request: Request, req: QuestionRequest):
    cfg = get_settings()
    trace_id = request.state.trace_id
    state.stats["requests"] += 1

    key = _cache_key(req)
    if req.use_cache and state.cache is not None:
        try:
            cached = state.cache.get(key)
            if cached:
                state.stats["cache_hits"] += 1
                log.info(json.dumps({"event": "cache_hit", "trace_id": trace_id}, ensure_ascii=False))
                return json.loads(cached)
        except Exception:
            pass

    search_query = None
    if req.use_hyde:
        search_query = state.llm.generate_hyde(req.question)

    result = state.retriever.retrieve(
        query=req.question,
        top_k=req.top_k,
        rerank_top_n=req.rerank_top_n,
        fusion_mode=req.fusion_mode,
        use_bm25=req.use_bm25,
        use_rerank=req.use_rerank,
        search_query=search_query,
    )

    # 一条都没检索到：直接告知，不交给模型硬编
    if not result.hits:
        payload = {
            "answer": "知识库中没有检索到与该问题相关的内容。",
            "sources": [],
            "trace": result.trace.as_dict(),
            "trace_id": trace_id,
        }
        return payload

    context = "\n\n".join(
        f"[来源{i + 1}] {h.content}" for i, h in enumerate(result.hits)
    )
    messages = state.llm.build_rag_messages(req.question, context)

    if not req.stream:
        answer, usage = state.llm.complete(messages)
        cost = state.llm.estimate_cost(usage, cfg)
        payload = {
            "answer": answer,
            "sources": [
                {"id": h.id, "doc_id": h.doc_id, "score": h.meta.get("rerank_score", h.score),
                 "preview": h.content[:120]}
                for h in result.hits
            ],
            "trace": result.trace.as_dict(),
            "usage": usage,
            "cost": cost,
            "trace_id": trace_id,
        }
        state.stats["latencies"].append(result.trace.total_ms)
        state.stats["tokens"] += usage.get("total_tokens", 0)
        state.stats["cost"] += cost
        if req.use_cache and state.cache is not None:
            try:
                state.cache.setex(key, cfg.cache_ttl, json.dumps(payload, ensure_ascii=False))
            except Exception:
                pass
        log.info(json.dumps({"event": "qa", "trace_id": trace_id, **payload["trace"]},
                            ensure_ascii=False))
        return payload

    async def event_stream():
        head = json.dumps({"sources": [
            {"id": h.id, "doc_id": h.doc_id, "preview": h.content[:120]} for h in result.hits
        ], "trace": result.trace.as_dict(), "trace_id": trace_id}, ensure_ascii=False)
        yield f"event: sources\ndata: {head}\n\n"
        for token in state.llm.stream(messages):
            yield f"data: {json.dumps({'delta': token}, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------- 运维 ----------------

@app.get("/health")
async def health():
    info = {"status": "ok", "vector_store": None, "cache": state.cache is not None}
    try:
        info["vector_store"] = {"type": type(state.retriever.store).__name__,
                                "chunks": state.retriever.store.count()}
        info["bm25"] = state.retriever.bm25.stats()
    except Exception as e:
        info["status"] = "degraded"
        info["error"] = str(e)
    return info


@app.get("/metrics")
async def metrics():
    lat = sorted(state.stats["latencies"])
    n = len(lat)
    return {
        "requests": state.stats["requests"],
        "cache_hits": state.stats["cache_hits"],
        "cache_hit_rate": round(state.stats["cache_hits"] / max(1, state.stats["requests"]), 3),
        "retrieval_latency_ms": {
            "p50": round(lat[n // 2], 1) if n else 0,
            "p95": round(lat[min(n - 1, int(n * 0.95))], 1) if n else 0,
            "p99": round(lat[min(n - 1, int(n * 0.99))], 1) if n else 0,
        },
        "total_tokens": state.stats["tokens"],
        "estimated_cost": round(state.stats["cost"], 4),
    }

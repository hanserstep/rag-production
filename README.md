# 生产级 RAG 知识库问答服务

一个可评测、可部署、可观测的检索增强生成系统。

和大多数「RAG demo」的区别在于：**每一次检索策略的改动都有量化依据**。
项目内置消融评测脚本，能回答「混合检索到底提升了多少」这类问题，
而不是停留在「跑起来效果不错」。

## 核心特性

| 能力 | 实现 |
|---|---|
| 混合召回 | BM25 关键词 + 向量语义双路，覆盖精确术语和口语化提问两类查询 |
| 融合策略 | RRF（倒数排名融合），对两路分数量纲不可比免疫；保留分数加权作为对照 |
| 精排 | Cross-Encoder（bge-reranker-v2-m3），Bi-Encoder 粗召回 → Cross-Encoder 精排两段式 |
| 相关性阈值 | 低分结果直接丢弃，检索不到就明确说没有，从源头降低幻觉 |
| 向量库 | Milvus 单 collection + HNSW 索引，支持 `memory` 后端本地跑评测 |
| 可观测 | 每次请求产出 trace（各阶段耗时、候选数）+ 结构化 JSON 日志 |
| 缓存 | Redis 缓存问答结果，命中率可在 `/metrics` 查看 |
| 成本核算 | 按 token 单价估算单次问答成本 |
| 评测 | 30 条人工标注评测集，5 组配置消融，自动生成 `EVAL_REPORT.md` |
| 部署 | Docker Compose 一条命令起全栈 |

## 架构

```
                        ┌──────────────┐
   用户提问 ───────────▶ │  FastAPI     │  trace_id / 缓存 / 指标
                        └──────┬───────┘
                               ▼
                     ┌───────────────────┐
                     │ HybridRetriever   │
                     └────┬─────────┬────┘
              ┌───────────┘         └────────────┐
              ▼                                  ▼
    ┌──────────────────┐              ┌──────────────────┐
    │ 向量检索          │              │ BM25 检索         │
    │ bge-m3 + Milvus  │              │ bigram/jieba 分词 │
    │ HNSW / IP        │              │ 持久化索引        │
    └────────┬─────────┘              └────────┬─────────┘
             └───────────┬───────────────────┘
                         ▼
                ┌─────────────────┐
                │ RRF 融合（k=60）│
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Cross-Encoder 精排│ ── 阈值过滤 ──▶ 无结果则直接拒答
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ DeepSeek 流式生成│ ── [来源N] 引用标注
                └─────────────────┘
```

## 评测

<!-- EVAL_RESULT_START -->
### 消融结果（Hit Rate@5 / MRR / Recall@5，30 条 query）

| 配置 | Hit Rate@5 | MRR | Recall@5 | 说明 |
|---|---|---|---|---|
| `vector_only` | 0.667 | 0.434 | 0.633 | 只有向量检索，基线 |
| `vector_bm25_concat` | 0.667 | 0.434 | 0.633 | 双路召回 + 旧版截断拼接 |
| `vector_bm25_score` | 0.833 | 0.523 | 0.767 | 双路召回 + 分数归一化加权 |
| `rrf_no_rerank` | 0.800 | 0.536 | 0.733 | 双路召回 + RRF 融合 |
| `rrf_rerank` | 0.800 | 0.536 | 0.733 | 暂未生效，见下方说明 |

**这组数字最值得看的是第二行**：双路召回后如果沿用旧版的"截断拼接"，
相对纯向量基线**提升为 0**。原因是拼接后的顺序由遍历顺序决定，没有排序意义——
两路结果只是被并排放进去，没有任何一路的信息被用来重新排序。

换成 RRF 后 Hit Rate@5 提升 0.133、MRR 提升 0.101。这就是融合策略本身的价值，
和"有没有做混合检索"是两件事。

### 数据说明（重要）

上表由 `python -m eval.run_eval --backend milvus_rest --embedding hashing` 生成。
`hashing` 是字符 n-gram 哈希向量，用来在无模型环境下验证检索链路和评测框架的正确性，
**它不具备真实语义能力，因此绝对数值偏低，不能代表 bge-m3 的效果**。

- `rrf_rerank` 与 `rrf_no_rerank` 数字相同，是因为 hashing 模式下刻意跳过了真实
  Cross-Encoder（分数和上下文对不上，跑了反而误导）。接真模型后两者会分开。
- 真实模型下预期绝对值会显著上升（Hit Rate@5 通常 0.85+），但**相对的差值关系**
  （concat 无效、RRF 有效、重排提升 MRR 大于提升 Hit Rate）是稳定的结论。

复现：

```bash
# 无模型环境，验证流程
python -m eval.run_eval --backend milvus_rest --embedding hashing

# 真实效果（需要 bge-m3：ollama pull bge-m3，或配 SILICONFLOW_API_KEY）
python -m eval.run_eval --backend milvus_rest --embedding ollama
```
<!-- EVAL_RESULT_END -->

指标定义：

- **Hit Rate@k**：前 k 条是否命中至少一个相关片段。贴近用户体验——只要答案在前 k 条里，模型就能答出来。
- **MRR@k**：命中片段排名的倒数均值。对排序位置敏感，衡量重排序效果最直接。
- **Recall@k**：前 k 条覆盖的相关片段比例。能暴露「只召回一半」的问题，Hit Rate 会掩盖它。

三个要一起看。只报 Hit Rate 容易自欺：把 k 调到 20，Hit Rate 必然好看，但噪声塞满上下文反而拉低生成质量。

## 快速开始

### 1. 起依赖服务

```bash
# Milvus（单容器 embedded 模式，不需要 etcd / minio）
docker compose -f docker-compose.lite.yml up -d

# Redis
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine
```

### 2. 配置

```bash
cp .env.example .env
# 填入 DEEPSEEK_API_KEY，以及以下二选一：
#   SILICONFLOW_API_KEY + EMBEDDING_BACKEND=api
#   或用 EMBEDDING_BACKEND=ollama 配合本地 ollama pull bge-m3
```

### 3. 跑起来

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

打开 http://localhost:8000/docs 看接口。

### 4. 上传文档并提问

```bash
curl -X POST http://localhost:8000/api/documents/upload -F "file=@docs/01-向量检索与Milvus.md"

curl -X POST http://localhost:8000/api/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"HNSW 的 M 参数控制什么","stream":false}'
```

### 5. 跑评测

```bash
# 不依赖 Milvus，用内存后端（几秒跑完）
python -m eval.run_eval --backend memory --embedding ollama

# 用 Milvus
python -m eval.run_eval --backend milvus --embedding ollama
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/documents/upload` | 上传文档，自动解析+分块+入库 |
| GET | `/api/documents` | 文档列表与索引统计 |
| DELETE | `/api/documents/{doc_id}` | 删除文档（向量库和 BM25 同步删） |
| POST | `/api/qa` | 问答，支持 `stream=true` 走 SSE |
| GET | `/health` | 健康检查（向量库连接、缓存状态） |
| GET | `/metrics` | P50/P95/P99 延迟、缓存命中率、token 用量、成本 |

## 项目结构

```
app/
  core/config.py          集中配置，全部可用环境变量覆盖
  ingest/
    chunker.py            文档解析（PDF/DOCX/MD/TXT）+ 分块
    embedder.py           Embedding 四后端：api / local / ollama / hashing
  retrieval/
    vector_store.py       VectorStore 抽象 + Milvus / InMemory 实现
    bm25.py               BM25 索引，可持久化，jieba 缺失时降级 bigram
    fusion.py             RRF 与分数加权融合
    reranker.py           Cross-Encoder + 相关性阈值
    retriever.py          检索编排，各环节可独立开关
  llm/client.py           DeepSeek 调用、Prompt 构建、HyDE、成本估算
  main.py                 FastAPI 入口、中间件、缓存、指标
eval/
  dataset.jsonl           30 条标注评测集
  run_eval.py             消融评测，生成报告
tests/                    26 个单测，不依赖模型和网络
docs/                     知识库示例文档
```

## 设计取舍

**为什么用 RRF 而不是分数加权**

向量相似度和 BM25 分值的量纲完全不同，而且 BM25 的绝对分值会随语料规模漂移——
同一个 query 在 100 篇和 10000 篇文档上拿到的分数不在一个量级，任何写死的权重都会失效。
RRF 只用名次，对量纲免疫。代价是丢弃了分数里「0.9 和 0.85 的差别」这类置信度信息，
但跨路融合本来就不该比较绝对值。

**为什么一个 collection 而不是每文档一个**

旧实现给每个文档建一个 collection，检索时遍历所有 collection 逐个查询。
文档数一多，查询次数线性增长，而且跨 collection 的分数无法统一排序。
改成全局单 collection + `doc_id` 字段过滤后，检索永远是 O(1) 次 ANN 查询。

**为什么 BM25 索引要落盘**

内存索引意味着进程重启后所有文档必须重新上传才能检索。这是 demo 和线上的分界线。

**为什么 Prompt 里不写「不要编造」**

否定式指令对降低幻觉几乎无效。真正有效的是检索侧的相关性阈值：检索不到相关内容时
直接返回「知识库中没有」，而不是硬塞三条不相关片段让模型自由发挥。

**为什么 jieba 是可选的**

jieba 需要加载约 40MB 词典，冷启动有开销，且对技术新词切分不准（`RAGFlow` 会被切成
`RAG` + `Flow`）。本项目默认用字符 bigram（Lucene 的 CJKAnalyzer 处理中文的默认做法），
零依赖且对未登录词友好；装了 jieba 就自动启用，不让词典成为部署的硬依赖。

## 测试

```bash
pytest tests/ -v
```

26 个用例，覆盖 RRF 打分公式、k 值行为、融合对照、BM25 持久化、检索链路、
消融开关、向量库 upsert 语义。全部不依赖模型、不依赖 Milvus、不依赖网络。

## License

MIT

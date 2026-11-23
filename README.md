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
### 消融结果

知识库 6 篇文档 / 35 个 chunk，评测集 50 条人工标注问题，Embedding 为 bge-m3，向量库为 Milvus。

| 配置 | Hit Rate@5 | MRR | Recall@5 | P50 延迟 |
|---|---|---|---|---|
| `vector_only` | 0.880 | 0.637 | 0.823 | 95ms |
| `vector_bm25_concat` | 0.880 | 0.637 | 0.823 | 98ms |
| `vector_bm25_score` | 0.940 | 0.682 | 0.883 | 94ms |
| `rrf_no_rerank` | 0.940 | 0.669 | **0.903** | 91ms |
| `rrf_rerank` | 0.940 | **0.704** | **0.903** | 224ms |

### 三个结论

**1. 截断拼接等于没做融合。**
`vector_bm25_concat` 相对纯向量基线的 Hit Rate、MRR、Recall **三项完全一致（变化都是 0.000）**——
两路结果只是被并排放进去，排序由遍历顺序决定。这解释了为什么很多项目
「做了混合检索但没效果」：问题不在混合，在融合。

**2. 融合带来 +0.06 Hit Rate，RRF 的 Recall 最高。**
分数加权和 RRF 的 Hit Rate 一样（0.940），但 RRF 的 Recall@5 高 0.02——
说明 RRF 更容易把同一问题的多个相关片段一起召回。原因是 RRF 只按名次累加，
一个文档只要在两路都排得靠前就能拿高分，不像加权融合会被单路的极端分数主导。
代价是 MRR 略低于加权（0.669 vs 0.682），因为丢掉了分数的幅度信息。

**3. 重排序是最后一层，也是单层增益最大的。**
RRF 加 Cross-Encoder 后 MRR 从 0.669 提到 0.704（+0.035），Hit Rate 不变——
它不改变「召回了什么」，只改变「谁排第一」。代价是 P50 从 91ms 涨到 224ms，
翻了 2.4 倍。是否值得看业务：答案质量敏感就值得，延迟敏感可以只精排 top 5。

三层的分工可以这样记：**混合召回决定「能不能找到」（+0.06 Hit），
RRF 决定「多个相关片段能否一起进来」（+0.02 Recall），
重排决定「谁排第一」（+0.035 MRR）**。三者治的是不同的病。

### 评测过程中修正的三个问题

**语料规模不足会让结论完全失真。**
第一版只有 3 篇文档 13 个 chunk，而评测的 top_k 是 5——候选集已经接近全库，
Hit Rate 恒高，融合与否看不出差别，测出来的"RRF 不如基线"纯属噪声。
扩到 35 个 chunk 后结论才稳定。

**批量 embedding 不能只按条数切。**
接口限制 32 条/次，但还有单次总 token 上限。36 个 chunk 按 32 切后第一批超限，
返回的却是一个语焉不详的 `The parameter is invalid`——和参数格式无关。
现在按「条数 + 总字符数」双约束切分。

**改了分块策略，评测标注必须重新核对。**
加入目录树清洗后，文档的 chunk 编号整体前移，原来标注的 `04#11` 指向的
已经是另一段内容。如果不重新核对，测出来的分数下降会让人误以为是分块策略变差了，
实际是标错了。这件事没法自动化，只能每次改 chunker 后手工过一遍高编号的文档。

复现：

```bash
python -m eval.run_eval --backend milvus_rest --embedding api
```

> 需要 `SILICONFLOW_API_KEY`（embedding 和 rerank 用同一个 key）。
> 无 key 时用 `--embedding hashing` 可以验证流程正确性，但绝对数值不代表真实效果。
<!-- EVAL_RESULT_END -->

指标定义：

- **Hit Rate@k**：前 k 条是否命中至少一个相关片段。贴近用户体验——只要答案在前 k 条里，模型就能答出来。
- **MRR@k**：命中片段排名的倒数均值。对排序位置敏感，衡量重排序效果最直接。
- **Recall@k**：前 k 条覆盖的相关片段比例。能暴露「只召回一半」的问题，Hit Rate 会掩盖它。

三个要一起看。只报 Hit Rate 容易自欺：把 k 调到 20，Hit Rate 必然好看，但噪声塞满上下文反而拉低生成质量。

### 端到端实测

服务起来之后实测的四项（Milvus + Redis + bge-m3 + DeepSeek 全链路）：

| 场景 | 结果 |
|---|---|
| 正常提问「RRF 的 k 值起什么作用」 | 召回 3 条候选，阈值过滤后剩 1 条（rerank 0.979，其余两条 0.0007 / 0.0006 被丢弃） |
| 提问知识库未覆盖的「K8s HPA 扩缩容阈值」 | 返回「知识库中没有检索到与该问题相关的内容」，来源 0 条，未交给模型 |
| 相同问题第二次提问 | 1.04s → 0.02s（Redis 命中，快 52 倍） |
| 单次问答成本 | 419 prompt + 36 completion = 455 tokens，约 0.00049 元 |

第二条是这套设计最有价值的地方：**检索不到时明确拒答，而不是拿三条不相关片段
让模型硬编**。阈值实测把 0.0007 的噪声片段拦在外面，这正是幻觉率的主要来源。

## 性能

`scripts/bench.py` 对检索段做并发压测（Milvus 参与真实 ANN 查询，embedding 用哈希向量以排除模型耗时）：

```
200 请求 / 并发 8
QPS     844.6
P50     9.0 ms
P95     14.8 ms
P99     18.9 ms

分段 P50：vector 8.9ms │ bm25 0.1ms │ fusion 0.02ms │ rerank 0(未启用)
```

**分段数据比总延迟更有信息量**：向量检索占了 98% 的耗时，而 BM25 只花 0.1ms。
这说明混合检索的边际成本几乎可以忽略——多跑一路 BM25 换 Hit Rate +0.13，
这笔账在任何规模下都划算。反过来，如果某天延迟吃紧，该优化的也是向量库
（换 gRPC 替代 REST、调小 ef、加缓存），而不是砍掉 BM25 这一路。

注意：上表不含 embedding 调用。接真实模型后，embedding 会取代向量检索成为
主要瓶颈（API 约 80ms，本地 CPU 约 400ms），届时该优化的对象就变了。

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

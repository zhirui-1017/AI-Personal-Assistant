# 个人知识库智能问答 Agent

> 一款**轻量、完全私有化**的 RAG 智能体：导入你自己的资料（TXT / Markdown / PDF / Word / 网页），
> 基于**私有知识**问答，具备 **Agent 自主规划**、**长期对话记忆**、**知识库自动治理**与**逐句回答溯源**。

不是「简单 RAG Demo」——问题的复杂度判断、任务拆解、多轮检索、交叉验证、引用校验都在本项目里显式实现，
每一条结论都能点开看到**原文片段**与**所属文档**。

---

## 1. 快速开始

```bash
# 1）准备环境（Python 3.10+）
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate

# 2）安装依赖（纯 Python，无需 GPU、无需外部服务）
pip install -r requirements.txt

# 3）启动
python run.py                     # 打开 http://127.0.0.1:8000/
```

首次启动**零配置即可跑通全流程**：未填写 API Key 时自动启用内置的「离线抽取模式」
（基于检索片段做抽取式回答，天然不产生幻觉）；配置任意 OpenAI 兼容大模型后立即变为生成式问答。

```bash
# 可选：接上真实大模型（通义 / DeepSeek / 智谱 / 讯飞 / vLLM / Ollama / OpenAI 均可）
copy .env.example .env            # 然后编辑 .env
# LLM_API_KEY=sk-xxxx
# LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LLM_MODEL=qwen-plus
```

推荐第一次体验：把本目录下的 `个人知识库智能问答Agent 项目需求文档（PRD）.md` 拖进「知识库」页面，
然后在对话页问「**这个项目的核心技术栈是什么？**」「**Agent 自主规划和普通 RAG 有什么区别？**」。

其它入口：

| 入口 | 命令 | 说明 |
| --- | --- | --- |
| 内置网页版 | `python run.py` → `/` | 零构建单页前端：对话、流式输出、上传进度、溯源弹窗、记忆与统计（推荐） |
| 接口文档 | `python run.py` → `/docs` | Swagger UI，可直接调试全部接口 |
| 命令行问答 | `python scripts/ask.py "你的问题"` | 打印 Agent 思考链路与引用来源 |
| 批量导入 | `python scripts/ingest_dir.py <目录>` | 递归导入整个资料目录 |
| 检索评测 | `python scripts/eval_retrieval.py --ingest README.md --verbose` | 用黄金集量化 Hit@K / Top1 / MRR |
| Gradio 界面 | `pip install gradio` 后 `python -m app.ui.gradio_app` | 备用图形界面（PRD 中推荐的快速落地方案） |
| 测试 | `pytest` | 170 项单元 + 集成 + 端到端测试 |

---

## 2. 效果速览

向知识库导入本项目的 PRD 后（离线抽取模式，未配置大模型）：

```
问题： Agent 自主规划和普通 RAG 有什么区别？
意图： compare（complex）｜依据度 100%｜耗时 115 ms
思考链路：
   · 记忆加载：载入 0 条近期消息、0 条长期记忆
   · 意图识别：识别为 compare（complex），来源：规则；问题包含对比/差异类表述，需要分别检索后交叉比对
   · 任务拆解：Agent 自主规划 的核心内容与特点 → 普通 RAG 的核心内容与特点 → Agent 自主规划 与 普通 RAG 的异同与结论
   · 决策调度：按对比对象分别检索证据，再用表格或分点给出异同与结论
   · 检索：4 个子问题共命中 6 个去重片段
   · 迭代检索：补全命中片段的上下邻近片段，新增 3 条证据用于交叉验证
   · 证据整理：保留 9 条证据（约 2194 字）并完成重排去重
   · 溯源校验：引用 4 个片段，依据度 100%
回答： ……（每条结论后带 [1][2] 来源编号）
引用：[1] 《…PRD.md》3.2 Agent 智能规划与调度核心    [2] 《…PRD.md》1.3 核心亮点

问题： 请详细讲解 Rust 异步运行时的调度器实现原理
回答： 暂无相关资料。知识库中没有检索到与该问题相关的内容，可以尝试换一种问法，或先上传相关文档。
       （判定依据：知识库中没有检索到与问题相关的片段）
```

自我反思闭环开启后，思考链路里会多出可解释的反思步骤（下面是真实运行输出）：

```
问题： 切片策略和向量库选型的区别是什么？
思考链路（节选）：
   · 检索：1 个子问题共命中 2 个去重片段
   · 自我反思（第 1 轮）：自评可信度 80%；问题信息点覆盖 80%；未覆盖：区别；需要补充检索
   · 补充检索：改写查询「区别」没有发现新的片段
   · 反思终止：本轮没有检索到新增证据，停止迭代
   · 溯源校验：引用 1 个片段，依据度 100%
```

如果缺口确实能在知识库里找到，则会补进证据并**重新生成答案**（覆盖率 50% → 补充检索 →
换查询后再检索 → 覆盖率 100% → 判定信息充足后结束），相关行为有单元测试逐条覆盖。

实测性能（本机、单文档 32 片段、离线模式）：首次请求 ~0.9 s（含 BM25 索引构建），
之后单次问答 **30–150 ms**，远低于 PRD 要求的 2 s。

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│ 接入层  内置网页前端（对话 / 文件上传 / 溯源弹窗） · REST API · Gradio │
├──────────────────────────────────────────────────────────────────────┤
│ Agent 调度层（项目核心，app/agent/）                                  │
│   Planner   意图识别 → 复杂度判断 → 任务拆解 → 查询改写                │
│   Executor  决策调度 → 多轮检索 → 邻接补全 → 交叉验证 → 生成答案         │
│   Reflection 自我反思：校验幻觉/充足度 → 改写 Query → 补充检索 → 重生成  │
│   Tools     search_knowledge / search_many / expand_neighbors /       │
│             read_document / list_documents / kb_stats                 │
│   Guardrails 无资料兜底 · 引用编号校验 · 依据度评估                    │
├──────────────────────────────────────────────────────────────────────┤
│ RAG 检索层（app/rag/）                                                │
│   Embedding（hash / API / sentence-transformers 可插拔）              │
│   VectorStore（SQLite 暴力内积 · Chroma HNSW 可切换）                  │
│   Retriever  BM25 + 向量召回 → 加权融合 → 精排(rerank) → MMR 重排      │
├──────────────────────────────────────────────────────────────────────┤
│ 模型服务层（app/llm/）                                                │
│   OpenAI 兼容客户端（流式 / JSON 模式 / 重试 / 降级） + 离线抽取模型   │
├──────────────────────────────────────────────────────────────────────┤
│ 数据存储层（app/db/）                                                 │
│   SQLite：documents / chunks（含向量）/ sessions / messages /          │
│           memories / qa_logs / ingest_tasks                           │
└──────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
RAGagent/
├─ app/
│  ├─ config.py              全局配置（env / .env，零第三方依赖）
│  ├─ main.py                FastAPI 应用装配（lifespan 启动服务容器）
│  ├─ services.py            服务容器：各层组件的唯一装配点
│  ├─ text.py                分词 / 句子切分 / SimHash / 哈希（公共工具）
│  ├─ schemas.py             API 请求响应模型
│  ├─ ingest/                loader → cleaner → splitter → dedup → pipeline
│  ├─ rag/                   embeddings / vector_store / retriever / rerank / evaluate
│  ├─ agent/                 planner / executor / reflection / tools / guardrails / prompts
│  ├─ llm/                   OpenAI 兼容客户端 + 离线抽取模型
│  ├─ memory/                短期记忆（会话窗口 + 滚动摘要）/ 长期记忆
│  ├─ db/                    models / base / repo（SQLAlchemy 2.0）
│  ├─ api/                   documents / chat / sessions / system 路由 + auth 鉴权
│  └─ ui/                    内置网页版前端 + 可选 Gradio
├─ scripts/                  ask.py（命令行问答）/ ingest_dir.py（批量导入）/ eval_retrieval.py（检索评测）
├─ tests/                    170 项测试（单元 / 集成 / 端到端 / API）+ golden_qa.jsonl 黄金集
├─ data/                     运行时生成：uploads / chroma / knowledge.db
├─ requirements.txt          运行依赖（轻量，无 GPU）
├─ requirements-optional.txt 可选增强（sentence-transformers / chromadb / gradio）
└─ .env.example              配置模板
```

---

## 4. 核心设计说明

### 4.1 Agent 思考链路（PRD 3.2 / 阶段二）

```
用户提问 → 记忆加载 → 查询改写 → 意图识别 → 任务拆解 → 决策调度
        → 检索/推理/总结 → 交叉验证 → 生成答案
        → 自我反思（校验 → 改写 Query → 补充检索 → 重新生成）
        → 溯源校验 → 记忆更新
```

| 意图 | 触发方式 | 处理策略 |
| --- | --- | --- |
| `simple_qa` | 默认 | 一次混合检索 + 精准回答 |
| `complex_qa` | 长问题 / 多个诉求 / 并列连词 | 拆成 ≤4 个子问题，分别检索后合并、邻接补全、交叉验证 |
| `summarize` | 「总结 / 摘要 / 要点 / 思维导图」 | 扩大检索并**整篇读取**命中文档，输出分层要点 |
| `compare` | 「对比 / 区别 / 差异 / 优劣」 | 抽取对比对象 → 分别检索 → 表格化异同与结论 |
| `kb_stats` | 「有多少文档 / 知识库统计」 | 调用统计工具，只用真实数字回答 |
| `chitchat` | 寒暄 | 跳过检索，直接回应 |

**双通道规划**：规则通道（毫秒级、可解释、永不失灵）负责基础判断；
LLM 通道（`AGENT_LLM_PLANNER=true`）在规则拿不准时增强。
两者融合，**LLM 异常时自动回退规则**，Agent 不会因为模型问题瘫痪。

完整链路的每一步都会记录在返回值的 `steps` 字段里，前端可展开查看（答辩时可直观展示「自主规划」）。

### 4.2 防幻觉三层机制（PRD 3.2 / 4.2 / 3.5）

1. **检索门控**：每个片段计算「绝对相关度」
   `relevance = 0.6 × 语义相似度 + 0.4 × IDF 加权关键信息覆盖率`，
   低于 `MIN_RELEVANCE_SCORE`（默认 0.18）即判定知识库无资料，直接兜底回答"暂无相关资料"。
   覆盖率按 IDF 加权，`实现`、`系统` 这类高频词不会制造虚假命中；
   未登录词按「已知词的平均 IDF」计入分母，查询词**全部**不在语料中时覆盖率直接为 0，
   这样知识库越大，"知识库里没有"越不会被冷门词巧合顶上去。
2. **引用校验**：生成后逐句检查 `[n]` 是否指向真实片段、句子与所引片段是否有实质重合，
   无效编号在严格模式下会被移除；全文没有引用时自动补一份来源清单（保证"回答必有出处"）。
3. **依据度上报**：返回 `grounded` 与 `grounded_ratio`，前端可见；低依据度的回答会被标记提示复核。

### 4.3 Agent 自我反思闭环（Self-Reflection）

初次生成的答案**不直接返回**，先做一次「自我校验」；判定为不足时自动改写检索 Query 补充证据并重新生成 ——
也就是「思考 → 检索 → 校验 → 再检索」的闭环，而不是单次 RAG 问答。

**三个校验维度**

| 维度 | 判定方式 | 触发动作 |
| --- | --- | --- |
| 幻觉 | 引用校验得分 < `REFLECTION_MIN_GROUNDED`、存在无依据句子、或明明检索到片段却答「暂无相关资料」 | 补充检索后重新生成 |
| 充足 | 问题信息点被证据覆盖的比例 < `REFLECTION_MIN_COVERAGE`，未覆盖的词即「缺口」 | 用缺口词改写 Query 再检索 |
| 矛盾 | LLM 通道复核回答内部、回答与资料之间是否自相矛盾 | 同上 |

> 信息点 = 分词后去掉停用词的实词（「向量库选型和切片策略」→ 向量 / 选型 / 切片 / 策略）；
> 覆盖率同时接受「分词命中」与「子串命中」，避免中文分词口径不同造成误判。

**双通道设计**：规则通道毫秒级、可解释、离线可用，负责「发现缺口」；LLM 通道（`[[TASK:REFLECT]]`）
只在规则发现缺口后介入，负责更细的幻觉判定与更精准的 Query 改写。**两边都认为没问题才算充足**（保守，
避免漏检）；规则认为没问题时直接返回，所以简单问答不会因此多花一次模型调用。

**终止条件（防止无限检索）**：

- 反思判定「信息充足」；
- 达到 `REFLECTION_MAX_ROUNDS`（默认 2 轮）；
- 本轮没有检索到任何**新增**片段（继续检索也不会带来新信息）；
- 反思得分不再提升（检索收益已经见顶）；
- 超出 `REFLECTION_TIME_BUDGET_MS`（默认 6000ms）时间预算；
- 没有可用的补充查询。

补充检索同样要过 `MIN_RELEVANCE_SCORE` 相关度门槛，避免把噪声塞进上下文。每一轮反思都会写入 `steps`
（前端「Agent 思考链路」可直接看到），并在 `/api/chat` 返回的 `reflection` 字段给出
`rounds`（实际补充检索轮数）/ `assessments`（反思评估次数）/ `sufficient`（最终判定）。

### 4.4 混合检索与自动去重（PRD 3.1 / 3.4）

- **混合召回**：BM25（关键词，jieba 分词）与向量召回并行，各自归一化后按 `0.6 / 0.4` 加权融合。
- **精排（rerank）**：融合后的前 `RERANK_TOP_N` 条再走一次精排——默认 `lexical` 词面精排（零依赖：
  按词长加权的查询词覆盖 + 短语命中 + 标题命中）；装了 `sentence-transformers` 则自动升级为
  `cross-encoder` 句对重排，模型加载失败会**自动降级**回词面精排（`RERANK_PROVIDER=none` 可整体关闭）。
- **检索范围限定**：请求可带 `document_ids` 把召回限制在指定文档内（前端「检索范围」下拉即此参数）。
- **MMR 重排**：最大边际相关，兼顾相关性与多样性；与已选片段余弦相似度超过阈值（默认 0.96）的近似重复片段直接丢弃。
- **文档级去重**：内容哈希完全一致 → 重复；SimHash 汉明距离 ≤3 **且长度差 ≤2%** → 近似重复
  （长度约束保证「同一篇文档新增一章」这类正常更新不会被误判）。命中策略由 `DUPLICATE_POLICY` 控制（`skip` / `keep`）。
- **片段级去重**：SimHash + 分段 LSH 索引（鸽巢原理），千级片段下近似 O(1) 判定，避免重复入库。

### 4.5 智能切片（PRD 3.1）

先按语义块（Markdown / 中文学术标题、段落、列表、表格、代码块）解析原子单元，再在**标题层级内**贪心合并到目标长度；
过长段落按句子边界续切、表格 / 代码块按行续切（不破坏结构）；
最终每个片段都是**原文的连续切片**，因此 `start_offset / end_offset` 可精确定位到原文段落
（`/chunks/{id}/context` 提供上下文窗口）。

### 4.6 记忆系统（PRD 3.3）

- **短期记忆**：最近 N 轮对话 + 超过阈值后的滚动摘要（`sessions.summary`），支持追问与指代消解（查询改写）。
- **长期记忆**：每轮对话后抽取用户偏好 / 关注点 / 身份 / 常用场景，向量近似或文本高度重合时**合并去重**；
  条目超过上限按「重要度 × 时效」保留高价值记忆，低价值记忆自动停用（`active=false`，可追溯不丢失）。

### 4.7 知识库管理与会话（PRD 3.4 / 3.6）

- 文档列表 / 上传 / 删除（级联清理片段与向量）/ 重新解析（重新切片 + 重算向量）/ 刷新向量索引；
- 手动输入文本类文档没有源文件时，重新解析自动降级为**仅刷新向量索引**，不会直接失败；
- 入库进度落库（`ingest_tasks`），前端轮询展示阶段与百分比，**重启后仍可查询**；
- 会话新建 / 清空 / 删除 / 历史消息，各会话上下文互不干扰。

### 4.8 安全边界与访问控制

- **网页导入防 SSRF**（`app/ingest/url_guard.py`）：只允许 `http/https`，拒绝带用户名密码的 URL；
  逐跳校验每次重定向，默认拒绝回环 / 私网 / 链路本地 / 保留地址（导入内网 wiki 可显式
  `URL_ALLOW_PRIVATE_HOSTS=true`）；`Content-Length` 与流式累计做双重体积上限，超限立即中断。
- **可选接口鉴权**（`app/api/auth.py`）：配置 `API_TOKEN` 后 `/api/*`（`/api/health` 除外）要求
  `Authorization: Bearer <token>` 或 `X-API-Token` 头，比较使用 `hmac.compare_digest`；
  **不配置则完全放行**，保持本地零配置可用。
- **错误信息不外泄**：未捕获异常统一返回「服务器内部错误」+ `request_id`，堆栈只进服务端日志。
- **CORS 凭据保护**：`API_CORS_ORIGINS=*` 时不再下发 `Access-Control-Allow-Credentials`。

### 4.9 前端交互

- 用户提问**右对齐**气泡，与 AI 回答区分；窄屏（≤900px）侧栏自动收至顶部，气泡宽度自适应。
- **中途停止**：`AbortController` 取消在途流式请求，已生成内容保留。
- **检索范围下拉**：把问答限制在单篇文档，对应 `/api/chat` 的 `document_ids`。
- **回答操作条**：复制 / 重新生成 / 👍 / 👎，反馈写入 `qa_logs.feedback`。

---

## 5. 配置说明

复制 `.env.example` 为 `.env`。关键项：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 空 | 留空 → 离线抽取模式；填写 → OpenAI 兼容生成式问答 |
| `LLM_BASE_URL` / `LLM_MODEL` | 通义兼容地址 / `qwen-plus` | 任何 OpenAI 兼容服务均可 |
| `EMBEDDING_PROVIDER` | `auto` | `auto` / `hash`（离线默认）/ `api` / `sentence-transformers` |
| `ST_MODEL` | `BAAI/bge-small-zh-v1.5` | 使用本地模型时的模型名（首次会下载权重） |
| `VECTOR_BACKEND` | `sqlite` | `sqlite`（零部署，千级片段毫秒级）/ `chroma`（万级片段） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 600 / 80 | 切片长度与重叠 |
| `RETRIEVAL_TOP_K` | 6 | 注入提示词的证据条数 |
| `MIN_RELEVANCE_SCORE` | 0.18 | 兜底阈值，调高更保守（更严格地"宁可不答"） |
| `SEMANTIC_WEIGHT` / `KEYWORD_WEIGHT` | 0.6 / 0.4 | 语义与关键词的融合权重，也用于相关度计算 |
| `AGENT_LLM_PLANNER` | true | 是否启用 LLM 规划（MockLLM 下自动跳过） |
| `AGENT_MAX_SUB_QUESTIONS` / `AGENT_MAX_ROUNDS` | 4 / 2 | 任务拆解上限与迭代检索轮数 |
| `DUPLICATE_POLICY` | skip | 命中重复文档时跳过还是照常入库 |
| `SHORT_TERM_TURNS` / `LONG_TERM_MAX_ITEMS` | 6 / 60 | 记忆窗口与长期记忆上限 |
| `REFLECTION_ENABLED` | true | 是否开启生成后的自我反思闭环 |
| `REFLECTION_MAX_ROUNDS` | 2 | 最多补充检索几轮（0 = 只自检、不迭代） |
| `REFLECTION_MIN_GROUNDED` | 0.8 | 引用依据度低于该值即判定存在幻觉风险 |
| `REFLECTION_MIN_COVERAGE` | 0.6 | 问题信息点被检索覆盖的最低比例 |
| `REFLECTION_TIME_BUDGET_MS` | 6000 | 反思环节时间预算，超出立即停止，防止无限检索 |
| `RERANK_PROVIDER` | `auto` | `auto` / `lexical`（零依赖词面精排）/ `cross-encoder` / `none` |
| `RERANK_TOP_N` / `RERANK_WEIGHT` | 20 / 0.4 | 进入精排的候选条数 / 精排分与召回分的融合权重 |
| `URL_ALLOW_PRIVATE_HOSTS` | false | 是否允许抓取私网地址（导入内网 wiki 时打开） |
| `URL_FETCH_MAX_BYTES` / `URL_FETCH_MAX_REDIRECTS` | 10 MB / 3 | 单页体积上限与最大重定向跳数 |
| `API_TOKEN` | 空 | 填写后 `/api/*` 需带 Token（`/api/health` 除外）；留空不鉴权 |
| `API_CORS_ORIGINS` | `*` | 允许的跨域来源，逗号分隔；为 `*` 时不下发凭据 |

> 三种典型模式：
> ① **完全离线**（默认）：hash 向量 + SQLite 向量库 + 抽取式回答，无网络无 Key 可跑；
> ② **生产推荐**：`EMBEDDING_PROVIDER=api`（或 `sentence-transformers`）+ 任意 OpenAI 兼容 LLM；
> ③ **大规模**：`VECTOR_BACKEND=chroma`，片段上万时切换 HNSW 索引。

---

## 6. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 + 当前生效配置与组件状态 |
| POST | `/api/documents/upload` | 上传文件（多选批量，multipart） |
| POST | `/api/documents/url` | 导入网页链接 |
| POST | `/api/documents/text` | 导入一段文本 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/documents/{id}` | 文档详情（含标题层级） |
| DELETE | `/api/documents/{id}` | 删除文档（级联清理片段与向量） |
| POST | `/api/documents/{id}/reindex` | 重新解析并刷新索引 |
| GET | `/api/documents/{id}/chunks` | 片段列表（分页） |
| GET | `/api/documents/{id}/chunks/{cid}/context` | **溯源定位**：命中片段及其上下文窗口 |
| GET | `/api/documents/tasks`、`/tasks/{tid}` | 入库任务列表 / 进度查询 |
| POST | `/api/chat` | 知识库问答（返回答案、引用、计划、思考链路、反思过程；可传 `document_ids` 限定检索范围） |
| POST | `/api/chat/stream` | 同上的 SSE 流式版本（plan → steps → evidences → delta → **reflect** → final → done） |
| POST | `/api/chat/feedback` | 对最近一次回答点赞 / 点踩（写入 `qa_logs.feedback`） |
| GET / POST / DELETE / POST(`/clear`) | `/api/sessions...` | 会话列表 / 新建 / 删除 / 清空 / 历史消息 |
| GET / DELETE | `/api/memories` | 查看 / 遗忘长期记忆 |
| GET | `/api/stats` | 文档数、片段数、问答次数、平均耗时、近 7 天趋势、意图分布 |

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" -d "{\"question\":\"这个项目的核心技术栈是什么？\"}"
```

---

## 7. 测试

```bash
pytest                       # 170 项，约 5 秒
pytest tests/test_agent.py -v
```

覆盖范围：文本工具 / 清洗 / 切片偏移量 / 多格式解析（含 GBK、HTML、DOCX）/ 去重 / 向量化 / 向量库 /
混合检索 / 精排 / 规划器 / 防幻觉护栏 / **自我反思闭环（含全部终止条件）** / 入库流水线 /
Agent 端到端（含兜底、多轮记忆、文档范围限定）/ 全部 HTTP 接口 / 鉴权 / URL 安全 / 记忆过滤 /
反馈接口 / 评测指标 / 老库轻量迁移。
测试使用临时数据目录，不会污染真实知识库。

### 检索质量评测

```bash
python scripts/eval_retrieval.py --ingest README.md --verbose          # 先灌数据再评测
python scripts/eval_retrieval.py --top-k 6 --min-hit-rate 0.8          # 命中率跌破阈值即以非 0 退出（可接 CI）
```

黄金集 `tests/data/golden_qa.jsonl` 每行形如 `{"question": "...", "expect_keywords": ["..."]}`，
`#` 开头与空行会被跳过，可直接替换成你自己的问题集。
实测（README.md，离线 hash 向量 + 词面精排）：**Hit@6 = 100%、Top1 = 81.8%、MRR = 0.894**。

---

## 8. 与需求文档的对应关系

| PRD 需求 | 实现位置 | 状态 |
| --- | --- | --- |
| 3.1 多格式导入 / 清洗 / 智能切片 / 索引 | `app/ingest/`（TXT、MD、PDF、DOCX、HTML、网页链接） | 已完成 |
| 3.2 Agent 意图识别 / 任务拆解 / 多轮检索 / 兜底 | `app/agent/planner.py`、`executor.py`、`guardrails.py` | 已完成 |
| 3.3 会话短期记忆 + 用户长期记忆 + 记忆清理 | `app/memory/` | 已完成 |
| 3.4 知识库列表 / 增删 / 重新解析 / 去重 / 统计 | `app/api/documents.py`、`app/ingest/dedup.py` | 已完成 |
| 3.5 回答溯源（片段 + 文档名 + 点击定位原文） | `citations` 字段 + `/chunks/{cid}/context` + 前端弹窗 | 已完成 |
| 3.6 会话新建 / 清空 / 删除 / 持久化 | `app/api/sessions.py` | 已完成 |
| 4.1 响应 2s 内 / 千级片段不卡顿 / 解析进度可视化 | 实测 30–150ms；SQLite 内积毫秒级；`ingest_tasks` + 前端进度条 | 已完成 |
| 4.2 仅基于私有知识作答 / 异常文件跳过 / 数据持久化 | 相关性门控 + 引用校验；单文件失败不影响批量；SQLite 全量落库 | 已完成 |
| 4.3 极简交互 / 清晰的进度与错误提示 | 上传即用，无需配置；内置前端 + 明确错误文案 | 已完成 |
| 5.1 技术栈（FastAPI / Chroma·FAISS / PyPDF·docx·BS4 / LLM 兼容） | 全部实现；向量库默认 SQLite，可切 Chroma | 已完成 |
| 3.2 生成后自检 / 资料不足时改写 Query 再检索 | `app/agent/reflection.py`：三维修正 + 六条终止条件 | 已完成 |
| 非功能：接口不可被随意调用、不泄露内部细节 | `app/api/auth.py`（可选 Token）+ 全局异常兜底 + SSRF 防护 | 已完成 |
| 非功能：检索质量可量化、可回归 | `app/rag/evaluate.py` + `scripts/eval_retrieval.py` + 黄金集 | 已完成 |

---

## 9. 常见问题

- **PDF 解析不到内容？** 扫描版 PDF 需要先 OCR（本项目不含 OCR，会明确提示"可能是扫描件"）。
- **`.doc` 无法上传？** 请另存为 `.docx`（旧版二进制格式不被 python-docx 支持，会给出提示）。
- **换了 Embedding 模型后检索异常？** 向量维度变化会导致旧向量失效，请对文档执行「重新解析」重建索引。
- **回答太长 / 太短？** 调 `RETRIEVAL_TOP_K`（证据条数）与 `LLM_MAX_TOKENS`。
- **总是回答"暂无相关资料"？** 说明触发了兜底：可降低 `MIN_RELEVANCE_SCORE`（例如 0.12）观察效果，
  或换用语义更强的 Embedding（`sentence-transformers` / 云端 embedding 接口）。
- **想完全禁止任何编造？** 保持 `AGENT_STRICT_CITATION=true`，并把 `MIN_RELEVANCE_SCORE` 调高。

## 10. 后续可迭代方向

**已知未做**（按优先级排列，欢迎继续演进）：

- 引用校验与反思判定目前是「编号合法性 + 词面重合 + 覆盖率」启发式，尚未引入 NLI 蕴含判定（更贵但更准）。
- 鉴权为单 Token 模式，没有多用户、角色与 JWT 会话；多租户隔离建议配合外部网关。
- BM25 每次启动全量重建索引，未做增量索引 / SQLite FTS5；万级片段以上建议切 Chroma。
- Token 计数按字符估算，未接 tokenizer 级精确计数（影响长上下文截断精度）。
- 前端未做流式引用高亮，Markdown 渲染也仅覆盖常用语法。

更远的方向：文档标签与知识库分组、多模态（图片 / 表格 OCR）、知识图谱抽取、
RAGAS 风格自动评估、定时增量同步目录、多用户与权限隔离。

> 说明：本项目为教学 / 演示性质的完整工程实现，数据全部保存在本地 `data/` 目录，不会上传到任何第三方服务
> （除非你主动配置了云端大模型或 Embedding 接口）。

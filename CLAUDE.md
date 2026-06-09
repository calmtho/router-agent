# 项目概述
本项目实现一个基于两阶段 Reranker 意图识别的 Main-Sub Agent 架构智能代理系统，具备以下核心能力：

1. **智能意图识别**：主代理通过两阶段 Reranker 意图识别（Embedding 初筛 → Reranker 精排）优先决策分发至合适的子代理（闲聊 / RAG / MCP / Vision），置信度不足时由 LLM 意图识别兜底。携带 file_ids 时短路直接路由到 RAG Agent；携带 image_ids 时短路直接路由到 Vision Agent。
2. **错别字纠正**：预处理管道第一步，基于 macbert4csc 自动纠正中文错别字，纠正后保留原文到 `original_message`。
3. **内容安全过滤**：预处理管道第二步，检测并拦截包含敏感词的用户请求，防止不当内容进入系统。
4. **MCP 工具调用**：子代理遵循 Model Context Protocol 调用外部工具。MCP Server 通过 stdio 子进程启动，支持计算器、网页抓取等工具。自定义 Server 可放在 `app/mcp_servers/` 目录下，在 `config.yaml` 中注册即可。
5. **RAG 增强生成**：子代理基于 Milvus 向量库进行两阶段检索增强（Milvus 粗排 + Cross-Encoder 精排），支持文件上传与文档问答。
6. **自由闲聊**：子代理提供通用对话能力。
7. **会话上下文管理**：双层摘要机制——标题摘要（首次对话生成，用于 UI 展示）+ 滚动摘要（超阈值触发，用于上下文窗口管理），历史自动裁剪。
8. **可观测性**：集成 Langfuse 进行 LLM 调用链路追踪，支持 Trace、Span、Generation 等多种观测类型。
9. **配置文件驱动**：所有模型、连接、超参数等均通过配置文件管理。
10. **语音转写（STT）**：基于 FunASR Paraformer-zh 本地模型的中文语音转文字，前端一键录音上传，结果自动填入输入框。
11. **OpenAI 兼容模型**：支持任何提供 OpenAI API 协议的大模型（如 GPT‑4、DeepSeek、本地 vLLM 等）。
12. **图片理解问答**：小 VL + 大 LLM 模式，4 阶段架构（VL 结构化特征提取 → LLM 自检 → 补偿轮 → LLM 回答），支持 jpg/png/webp/gif/bmp/tiff。

# 技术栈

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.11 |
| Web 框架 | FastAPI |
| 大模型协议 | OpenAI API |
| 向量数据库 | Milvus (Standalone/Docker) |
| Embedding 模型 | HuggingFace + sentence-transformers (本地部署) |
| 编排与检索 | LangChain (langchain, langchain-openai, langchain-community) |
| MCP 客户端 | 自实现 (`mcp` SDK + stdio 传输) |
| 可观测性 | Langfuse (链路追踪 + 观测) |
| MCP 服务器 | `app/mcp_servers/` 目录（自定义工具 Server） |
| 配置管理 | Pydantic Settings + YAML |
| 异步支持 | asyncio + httpx |
| 错别字纠正 | macbert4csc-base-chinese (transformers + torch) |
| 语音识别 | FunASR (Paraformer-zh) + soundfile + ModelScope |
| VL 多模态 | LangChain ChatOpenAI (OpenAI 兼容 VL 模型，如 Qwen2.5-VL、InternVL2) |
| 重排序模型 | sentence-transformers Cross-Encoder (BAAI/bge-reranker-base，中文原生，同时用于 RAG 精排与路由分类) |
| 测试框架 | pytest + pytest-asyncio |

# 项目结构
```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置加载（YAML + 环境变量）
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── sub_agent_base.py   # 子代理抽象基类
│   │   ├── chat_agent.py       # 闲聊子代理
│   │   ├── rag_agent.py        # RAG 子代理（含 Milvus 检索）
│   │   ├── mcp_agent.py        # MCP 工具调用子代理
│   │   └── vision_agent.py     # 图片理解子代理（小VL+大LLM 4阶段架构）
│   ├── chains/
│   │   ├── cot_chain.py        # 意图识别链（Prompt + 解析）
│   │   └── rag_chain.py        # LangChain RAG 链
│   ├── data/                   # 数据文件
│   │   └── sensitive_words.py  # 敏感词词典（内容安全过滤）
│   ├── graph/                  # LangGraph 图定义
│   │   ├── __init__.py
│   │   ├── state.py            # State 定义
│   │   ├── nodes.py            # 图节点（router, chat, rag, mcp）
│   │   └── graph.py            # 图构建
│   ├── routers/
│   │   ├── chat.py             # /chat 接口（JSON 请求，引用已上传文件）
│   │   ├── upload.py           # /upload 接口（文件上传 + 向量化入库）
│   │   ├── upload_image.py     # /upload/image 接口（图片上传 + 持久化注册表）
│   │   └── stt.py              # /stt/transcribe 语音转文字接口
│   ├── services/
│   │   ├── milvus_service.py   # 向量库管理（插入/检索）
│   │   ├── mcp_client.py       # MCP 协议客户端（stdio 子进程通信）
│   │   ├── history_service.py  # 对话历史管理（标题/摘要生成、滑动窗口、历史裁剪）
│   │   ├── llm_client.py       # LLM 客户端 + 本地 Embedding
│   │   ├── stt_service.py      # STT 语音转写服务（FunASR Paraformer-zh 封装）
│   │   ├── reranker_service.py # 检索精排 + 两阶段意图识别（Embedding初筛→Reranker精排→LLM兜底）
│   │   ├── vl_client.py        # VL 多模态客户端（OpenAI 兼容，Base64 data URI 传图）
│   │   ├── session_context_service.py # Session Context（TTL 图片ID继承，跨轮保持）
│   │   └── typo_service.py     # 错别字纠正服务（macbert4csc 封装）
│   ├── mcp_servers/            # MCP 工具服务器
│   │   └── calculator_server.py # 计算器工具示例
│   └── utils/
│       ├── file_processor.py   # 文件解析（PDF, txt, md…）
│       └── logger.py
├── configs/
│   └── config.yaml             # 主配置文件（见 specs/spec.md）
├── static/                     # 静态文件（通过 /static 路径访问）
│   └── test_stream.html        # 流式聊天测试页面
├── fixtures/                   # 测试用静态资源（预生成，纳入 git）
│   ├── ai_learn.wav            # STT 测试用音频（中文语音样本）
│   ├── test_image.png          # Vision E2E 测试用图片
│   └── rag_test_document.pdf   # RAG / PaddleOCR 测试用 PDF
├── tests/                      # 测试模块
│   ├── conftest.py             # pytest 配置
│   ├── test_config.py          # 配置加载测试
│   ├── test_file_processor.py  # 文件解析测试
│   ├── test_graph.py           # LangGraph 路由自测脚本
│   ├── test_history.py         # 会话历史管理测试
│   ├── test_paddle.py          # PaddleOCR API 集成测试
│   ├── test_stt_local.py       # FunASR 语音转写本地测试
│   ├── test_preprocess.py       # 预处理管道测试（错字纠正 + 敏感词过滤）
│   ├── test_reranker.py         # 重排序功能测试
│   ├── test_vision_e2e.py       # Vision 端到端测试（自动启动服务→上传图片→图文问答）
│   └── test_imports.py          # 模块导入检查
├── requirements.txt
├── Dockerfile
├── docker-compose.yml          # 包含 Milvus, etcd, minio
├── CLAUDE.md                   # 项目说明文档
├── specs/
│   └── spec.md               # 技术规格说明书（事实标准）
```

# 快速开始

1. 启动 Milvus（推荐 Docker Compose）：
```bash
docker-compose up -d
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 配置模型：在 `.env` 中设置 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL_NAME`，`config.yaml` 通过 `${...}` 引用。
   - LLM 对话模型通过 OpenAI 兼容 API 调用。
   - Embedding 模型 (`BAAI/bge-small-zh-v1.5`) 使用 HuggingFace 本地运行，首次启动时自动下载，无需外部 API。
   - MCP Server 在 `mcp.servers` 下配置，通过 stdio 子进程启动。
4. 运行服务：
```bash
uvicorn app.main:app --reload
```

5. 测试页面：浏览器打开 `http://localhost:8000/static/test_stream.html`，即可在页面中测试流式聊天接口。

6. 运行测试：
```bash
# 运行单元测试（不需要外部服务）
python -m pytest tests/ -v

# 运行 PaddleOCR 集成测试（需要配置 PADDLEOCR_ACCESS_TOKEN）
python tests/test_paddle.py

# 运行 STT 本地测试（首次会下载 ~840MB 模型）
python tests/test_stt_local.py
```

7. 测试接口：
```bash
# 先上传文件
curl -X POST http://localhost:8000/upload \
     -F "file=@document.pdf" \
     -F "session_id=demo"

# 再通过 file_id 进行问答
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"总结这份文档","file_ids":["<file_id>"]}'

# 语音转文字
curl -X POST http://localhost:8000/stt/transcribe \
     -F "file=@fixtures/ai_learn.wav"
```

# 开发约定

1. **内容安全过滤**：敏感词词典位于 `app/data/sensitive_words.py`，预处理节点 `preprocess_node` 在图入口处进行错字纠正→敏感词过滤。
2. **LangGraph 意图识别**：使用 `app.graph.graph.app_graph` 执行请求意图识别，`router_node` 优先通过两阶段 Reranker（Embedding 初筛 → Reranker 精排）快速分类，置信度不足时回退 LLM 意图识别兜底。
3. **子代理注册**：所有子代理需继承 `SubAgentBase` 并实现 `can_handle()` 与 `handle()` 方法。
4. **配置优先级**：YAML 文件 → 环境变量（`APP_*`）→ 默认值。
5. **错误处理**：MCP 工具调用失败时降级为闲聊响应，RAG 检索为空时回退至生成式回答。
6. **Langfuse 配置**：在 `.env` 中设置 `LANGFUSE_ENABLED=true` 以及 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 即可启用链路追踪。
7. **文件上传异步处理**：文件上传接口立即返回 `file_id` 和 `status: "processing"`，后台异步执行切片和向量入库，前端通过 `/upload/status/{file_id}` 轮询状态。上传时显示 `⏳`，分析完成显示 `✅`，期间锁定发送按钮。
8. **会话上下文管理**：`app/services/history_service.py` 实现双层摘要机制：
   - **标题摘要**：首次对话后异步生成（5~15 字），用于 UI 展示和会话标识，生成一次后不再更新。
   - **滚动摘要**：对话超过阈值（默认 10 轮）时异步生成，压缩旧历史节省 token，每次超阈值时增量更新。
   - **Turn 计数器**：每会话维护递增 turn 号（`_session_turn_counter`），每轮 user/assistant 消息共享同一 turn 号，消息带 `"turn": N` 字段。`needs_summary()` 直接读计数器，窗口和裁剪按 turn 号操作，不再用 `len(history) // 2` 推算。
   - 三者独立存储（`_session_titles` / `_session_summaries` / `_session_turn_counter`），互不影响，`clear_history()` 同时清理。
   - 注入上下文时：标题以 `【会话主题】` 前缀注入，滚动摘要以 `【历史摘要】` 前缀注入。
9. **STT 语音转写服务**：`app/services/stt_service.py` 实现基于 FunASR Paraformer-zh 的本地语音识别：
   - **lifespan 预加载**：模型在 `app.main: lifespan` 启动时加载，避免首次请求等待。
   - **全局单例**：`get_stt_service()` 保证全应用共享一个模型实例（~840MB）。
   - **异步隔离**：使用 `asyncio.to_thread()` 将同步的 FunASR 推理丢到独立线程，不阻塞事件循环。
   - **格式兜底**：`_ensure_wav()` 通过 `soundfile` 将 mp3/webm 等转换为标准 16kHz mono WAV。
   - **前端集成**：`static/test_stream.html` 提供 🎤 录音按钮，Web Audio API 直出 PCM 并封装 WAV 上传。
10. **错别字纠正服务**：`app/services/typo_service.py` 实现基于 macbert4csc-base-chinese 的中文纠错：
   - **全局单例**：`get_typo_corrector()` 保证全应用共享一个模型实例。
   - **异步隔离**：使用 `asyncio.to_thread()` 将同步推理丢到独立线程。
   - **配置开关**：`preprocess.enable_typo_correction` 控制是否启用，关闭时跳过纠错步骤。
   - **状态保留**：纠错后原文存入 `original_message`，纠正后文本存入 `message`，供后续节点和 LLM 使用。
   - **配置**：`preprocess.typo_model` 指定模型名，默认 `shibing624/macbert4csc-base-chinese`。
11. **重排序与意图识别服务**：`app/services/reranker_service.py` 实现基于 Cross-Encoder 的两阶段检索精排与意图识别：
   - **全局单例**：`get_reranker_service()` 保证全应用共享一个模型实例。
   - **异步隔离**：使用 `asyncio.to_thread()` 将同步推理丢到独立线程，不阻塞事件循环。
   - **RAG 精排**：`rerank(query, documents)` 对检索结果重排序，`rag.rerank_enabled` 控制开关，加载失败自动降级。
   - **两阶段意图识别**：`classify_route(query)` 先由 Embedding 模型计算 query 与各 category description 的 cosine 相似度取 Top-K，再由 Reranker 精准打分 + margin→confidence sigmoid 映射。置信度 ≥ `router.confidence_threshold` 时直接返回目标；否则返回 "fallback" 交由 LLM 意图识别兜底。
   - **lifespan 预加载**：模型在 `app.main:lifespan` 启动时预加载，避免首次请求等待约 11s。
   - **配置**：`rag.rerank_model` 指定模型名（默认 `BAAI/bge-reranker-base`）；`rag.rerank_batch_size` 控制批处理大小；`rag.rerank_output_k` 控制精排后输出数量。
   - **路由参数**：`router.confidence_threshold`（置信度阈值）、`router.margin_temperature`（margin→confidence 锐度）、`router.embedding_top_k`（Embedding 初筛保留数）、`router.category_descriptions`（各代理类别描述）。
12. **Vision Agent 图片理解**：`app/agents/vision_agent.py` 实现小 VL + 大 LLM 模式的 4 阶段架构：
   - **Phase 1 — VL 特征提取**：VL 模型根据用户问题从图片提取结构化特征（JSON）。
   - **Phase 2 — LLM 自检**：文本 LLM 检查特征是否足以回答问题。
   - **Phase 3 — 补偿轮（可选）**：线索不足时 VL 定向补充提取。
   - **Phase 4 — LLM 回答**：文本 LLM 基于特征生成自然语言回答。
   - **配置**：`vision` 配置段指定 VL 模型的 `openai_base_url`/`api_key`/`model_name`/`temperature`/`max_tokens`，支持任何 OpenAI 兼容 VL 模型。
   - **VL 客户端**：`app/services/vl_client.py` 基于 `langchain-openai` ChatOpenAI，通过 Base64 data URI 传递图片。
13. **Session Context 图片 ID 继承**：`app/services/session_context_service.py` 为每个会话维护图片 ID 列表：
   - **TTL 过期**：默认 30 分钟，超时自动清理。
   - **自动继承**：Chat 请求未传 `image_ids` 时，自动继承同 session 上一轮的图片 ID。
   - **去重合并**：多轮图片 ID 按顺序去重合并。
   - **显式清除**：用户要求"不管那张图"时可清除。
14. **图片上传路由**：`app/routers/upload_image.py` 提供 `POST /upload/image` 接口：
   - **支持格式**：jpg、jpeg、png、webp、gif、bmp、tiff。
   - **持久化注册表**：图片元信息保存到 `uploads/images/.registry.json`，防止 uvicorn reload 丢失。
   - **启动时加载**：服务启动时从磁盘加载注册表，自动清理已不存在的文件记录。
   - **路径解析**：`resolve_image_paths(image_ids)` 供 `preprocess_node` 调用，将 image_ids 解析为文件路径。
15. **路由专用 LLM 配置**：`config.py` 中的 `RouterLLMConfig` 管理路由兜底所用的独立小模型参数，通过 `config.router_llm` 访问（`openai_base_url`/`api_key`/`model_name`），与聊天主 LLM 隔离。
16. **路由参数配置**：`config.py` 中的 `RouterConfig` 管理两阶段路由的超参：`confidence_threshold`（置信度阈值，默认 0.6）、`margin_temperature`（margin→confidence 映射锐度，默认 2.0）、`embedding_top_k`（Embedding 初筛保留数，默认 2）、`category_descriptions`（各代理类别的自然语言描述，用于 Embedding 相似度计算）。
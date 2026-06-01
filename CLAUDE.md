# 项目概述
本项目实现一个基于 Chain‑of‑Thought (CoT) 的 Main‑Sub Agent 架构智能代理系统，具备以下核心能力：

1. **智能路由**：主代理通过 CoT 推理决定将用户请求分发至合适的子代理（闲聊 / RAG / MCP 工具调用）。
2. **错别字纠正**：预处理管道第一步，基于 macbert4csc 自动纠正中文错别字，纠正后保留原文到 `original_message`。
3. **内容安全过滤**：预处理管道第二步，检测并拦截包含敏感词的用户请求，防止不当内容进入系统。
4. **MCP 工具调用**：子代理遵循 Model Context Protocol 调用外部工具。MCP Server 通过 stdio 子进程启动，支持计算器、网页抓取等工具。自定义 Server 可放在 `app/mcp_servers/` 目录下，在 `config.yaml` 中注册即可。
5. **RAG 增强生成**：子代理基于 Milvus 向量库进行检索增强，支持文件上传与文档问答。
6. **自由闲聊**：子代理提供通用对话能力。
7. **会话上下文管理**：双层摘要机制——标题摘要（首次对话生成，用于 UI 展示）+ 滚动摘要（超阈值触发，用于上下文窗口管理），历史自动裁剪。
8. **可观测性**：集成 Langfuse 进行 LLM 调用链路追踪，支持 Trace、Span、Generation 等多种观测类型。
9. **配置文件驱动**：所有模型、连接、超参数等均通过配置文件管理。
10. **语音转写（STT）**：基于 FunASR Paraformer-zh 本地模型的中文语音转文字，前端一键录音上传，结果自动填入输入框。
11. **OpenAI 兼容模型**：支持任何提供 OpenAI API 协议的大模型（如 GPT‑4、DeepSeek、本地 vLLM 等）。

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
│   │   └── mcp_agent.py        # MCP 工具调用子代理
│   ├── chains/
│   │   ├── cot_chain.py        # CoT 推理链（Prompt + 解析）
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
│   │   └── stt.py              # /stt/transcribe 语音转文字接口
│   ├── services/
│   │   ├── milvus_service.py   # 向量库管理（插入/检索）
│   │   ├── mcp_client.py       # MCP 协议客户端（stdio 子进程通信）
│   │   ├── history_service.py  # 对话历史管理（标题/摘要生成、滑动窗口、历史裁剪）
│   │   ├── llm_client.py       # LLM 客户端 + 本地 Embedding
│   │   ├── stt_service.py      # STT 语音转写服务（FunASR Paraformer-zh 封装）
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
│   └── test_imports.py          # 模块导入检查
├── requirements.txt
├── Dockerfile
├── docker-compose.yml          # 包含 Milvus, etcd, minio
├── CLAUDE.md                   # 本文件
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

6. 测试接口：
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
2. **LangGraph 路由**：使用 `app.graph.graph.app_graph` 执行请求路由，CoT 推理在 `router_node` 中完成。
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
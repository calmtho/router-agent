# Router Agent — 技术规格说明书

## 1. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI 入口                               │
│  /chat (POST)  /upload (POST)  /upload/image (POST)         │
│  /stt/transcribe (POST)                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │ Preprocess Node  │
              │(错字纠正→敏感词→图片路径解析)│
              └──┬──┬──┬──┬─────┘
                 │  │  │  │
        ┌────────┘  │  │  └────────┐
        ▼           ▼  ▼           ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │Chat Node │ │RAG Node  │ │MCP Node  │ │Vision Node│
  └──────────┘ └────┬─────┘ └────┬─────┘ └─────┬────┘
                     │             │              │
              ┌──────▼──────┐      │         ┌────▼─────┐
              │Milvus 向量库 │      │         │ VL 模型   │
              │ (粗排 top_k) │      │         │(特征提取) │
              └──────┬──────┘      │         └────┬─────┘
                     │             │              │
              ┌──────▼──────┐      │         ┌────▼─────┐
              │  Reranker   │      │         │ 文本 LLM  │
              │ (精排 top_n) │      │         │(自检+回答)│
              └──────┬──────┘      │         └──────────┘
                     │             │
              ┌──────▼──────┐      │
              │HuggingFace  │      │
              │Embedding    │      │
              │(本地CPU)    │      │
              └─────────────┘      │
                            ┌──────▼──────┐
                            │  MCP Server │
                            │ (stdio/SSE) │
                            └─────────────┘
```

### 1.1 文件上传交互流程

用户通过 `/upload` 上传文件后，系统采用「异步处理 + 状态轮询」模式：

1. 用户上传文件 → 服务端立即返回 `file_id`（status: processing）
2. 后台异步执行：解析文本 → 切片 → 向量化 → 写入 Milvus
3. 前端轮询 `/upload/status/{file_id}`，处理完成获得 status: ready
4. 用户发送消息并附带 `file_ids` → Router 意图识别到 RAG Agent → Milvus 检索相关 chunk → LLM 生成回答

### 1.2 图片上传交互流程

用户通过 `/upload/image` 上传图片后，系统采用「同步返回 + Session Context 继承」模式：

1. 用户上传图片 → 服务端立即返回 `image_id`、文件名、文件大小
2. 图片保存在 `uploads/images/` 目录，元信息持久化到 `.registry.json`
3. 用户发送消息并附带 `image_ids` → Preprocess 解析图片路径 → Router 优先意图识别到 Vision Agent
4. Vision Agent 执行 4 阶段处理（VL 特征提取 → LLM 自检 → 补偿轮 → LLM 回答）
5. Session Context 自动继承 image_ids，后续对话无需重复指定

> 该流程将 upload（用户侧，低延迟）与 chunk/index（计算密集）合在同一服务中，仅为简化演示。
> 生产环境建议拆分为独立的 upload 服务和 index 服务，详见 2.1 节。

## 2. 组件详述

### 2.1 文件上传异步处理

- **文件**: `app/routers/upload.py`
- **职责**: 文件上传 + 后台异步切片 + 向量化入库
- **接口**: 
  - `POST /upload` - 上传文件，立即返回 `file_id` + `status: "processing"`
  - `GET /upload/status/{file_id}` - 查询文件处理状态
- **处理流程**:
  1. 接收文件并保存到本地 `uploads/` 目录
  2. 注册文件状态为 `processing`
  3. 提交后台异步任务 `_process_file_background()`
  4. 立即返回响应，不阻塞用户
  5. 后台任务负责：解析文本 → 切片（500 字符，重叠 50）→ 向量化 → Milvus 入库
  6. 完成后更新状态为 `ready`，失败则标记为 `error`
- **前端集成**: 前端通过轮询 `/upload/status/{file_id}` 获取状态，上传时显示 `⏳`，分析完成显示 `✅`，处理期间锁定发送按钮
- **状态值**: `processing`（处理中） / `ready`（已就绪） / `error`（失败）
- **架构说明**: 当前 upload 与 chunk 在同一服务中，仅为简化演示。生产环境中建议拆分为独立的 upload 服务（用户侧，低延迟）和 index 服务（计算密集，可 GPU 加速、独立扩缩容），详见 1.1 节。

### 2.2 Preprocess Node（预处理节点）

- **文件**: `app/graph/nodes.py`
- **职责**: 错别字纠正 → 敏感内容过滤 → 图片路径解析（三步串行管道）
- **处理流程**:
  1. **错别字纠正**（可配置开关 `preprocess.enable_typo_correction`）：调用 `TypoCorrector` 对用户输入进行纠错，纠正后文本存入 `message`，原始输入存入 `original_message`
  2. **敏感词过滤**：检测用户输入是否包含敏感词，检测到则直接拒绝返回提示
  3. **图片路径解析**：将请求中的 `image_ids` 解析为实际文件路径，存入 `image_paths`，供 Vision Agent 使用
- **敏感词词典**: `app/data/sensitive_words.py`
- **检测方式**: 正则表达式匹配
- **处理方式**: 检测到敏感词时直接拒绝，返回提示信息
- **状态字段**: 若错字被纠正，`original_message` 保留原始输入，`message` 为纠正后文本，供后续节点和 LLM 使用
- **支持的敏感词类型**:
  - 政治相关（共产党、政府、总统等）
  - 色情相关（色、黄、裸、性等）
  - 暴力相关（杀、死、打、砸等）
  - 辱骂相关（傻逼、SB、渣男等）
  - 违法相关（贩毒、走私、拐卖等）
  - 其他敏感词（自杀、跳楼等）

### 2.2.1 TypoCorrector 错别字纠正服务

- **文件**: `app/services/typo_service.py`
- **模型**: `shibing624/macbert4csc-base-chinese`（MacBERT for Chinese Spelling Correction）
- **依赖**: `transformers>=4.40.0` + `torch`
- **架构特点**:
  - `get_typo_corrector()` 全局单例，全应用共享一个模型实例
  - `asyncio.to_thread()` 包装同步推理，不阻塞事件循环
  - 模型未加载时 `correct()` 原样返回文本，不抛异常
  - `load_model()` 在 lifespan 启动时预加载（与 STT 模型一起）
- **输入/输出**: `correct(text) -> (纠正后文本, [(错字, 正字, 位置), ...])`
- **配置**:
  ```yaml
  preprocess:
    enable_typo_correction: true
    typo_model: "shibing624/macbert4csc-base-chinese"
  ```

### 2.3 Router Node（意图识别节点）

- **文件**: `app/graph/nodes.py`
- **职责**: 优先通过两阶段 Reranker 快速分类，置信度不足时回退 LLM 意图识别兜底
- **意图识别决策流程**:
  1. **纯问候/自我介绍**：正则匹配短路，直接路由到 `chat`
  2. **带附件**：`file_ids` → `rag`、`image_ids` → `vision`（短路跳过所有推理）
  3. **场景不明**：先调用 `_strip_greeting_prefix()` 剥离问候前缀，再调用 `_classify_via_reranker()` 执行两阶段意图识别分类
  4. **Reranker 未命中**：回退 LLM 意图识别兜底（使用 `router_llm` 独立小模型）
- **输出格式**: `{"target": "rag|chat|mcp|vision", "reasoning": "..."}`
- **降级策略**: 目标代理执行失败时，回退至 `fallback_agent`（默认 `chat`）
- **流式意图识别**：`/chat/stream` 接口中意图识别决策在流式生成器内完成，不经过 LangGraph 图

### 2.4 Chat Agent（闲聊代理）

- **文件**: `app/agents/chat_agent.py`
- **职责**: 通用对话，直接将用户消息发送给 LLM 生成回复
- **特点**: 无外部依赖，纯 LLM 对话
- **上下文注入**: 构建消息时按顺序注入：
  1. System prompt（角色设定）
  2. `【会话主题】` + 标题（如有，5~15 字精炼标题）
  3. `【历史摘要】` + 滚动摘要（如有，压缩的旧对话摘要）
  4. 对话历史（滑动窗口内的近期消息）
  5. 当前用户消息

### 2.5 RAG Agent（检索增强代理）

- **文件**: `app/agents/rag_agent.py`
- **职责**: 基于 Milvus 向量检索 + LLM 生成的文档问答
- **流程**:
  1. 用户先通过 `/upload` 上传文件，文件被分块、向量化后存入 Milvus
  2. 用户通过 `/chat` 发送 query 并附带 `file_ids`
  3. RAG Agent 在 Milvus 中检索相关文档块（粗排，默认 `top_k=16`）
  4. Cross-Encoder 重排序模型对候选文档块进行精排（按相关性重新打分排序）
  5. 取精排后 top `rerank_output_k` 个最优结果（默认 4 个）
  6. 拼接精排结果作为上下文，由 LLM 生成最终回答
- **回退**: 检索为空时回退为生成式回答；重排序模型加载失败时自动降级，使用原始检索结果

> ⚠️ **架构说明 — 当前实现为简化演示**
>
> 本项目中「文件上传 → 切片 → 向量化 → RAG 检索」是一条龙流水线，**目的是简化路由逻辑、直观展示 RAG 交互流程，同时避免拆分成多个微服务增加示例复杂度**。
>
> 生产环境中，文档问答的架构选型取决于场景：
>
> | 场景 | 推荐方案 | 说明 |
> |------|---------|------|
> | 短文档即时问答 | **VLM + LLM 直接阅读** | 将文档页面截图或转 Markdown 后直接送入 LLM（如 GPT-4o-mini、Qwen-VL），无需切片和检索，延迟低、保真度高 |
> | 长文档/跨文档检索 | **RAG 流水线** | 文档超出 LLM 上下文窗口或需要跨文档语义检索时，才需要 chunk → embedding → vector DB → retrieve 的完整流程 |
> | 长期外挂知识库 | **RAG + 独立索引服务** | 知识需要持续更新、多用户共享时，upload 和 chunk/index 应拆为独立服务（upload 低延迟、限流；index 计算密集、可批量/GPU 加速、独立扩缩容） |

### 2.5.1 Reranker 重排序与意图识别服务

- **文件**: `app/services/reranker_service.py`
- **模型**: `BAAI/bge-reranker-base`（中文原生 Cross-Encoder，同时用于 RAG 精排与意图识别分类）
- **依赖**: `sentence-transformers>=2.7.0`
- **架构特点**:
  - `get_reranker_service()` 全局单例，全应用共享一个模型实例
  - lifespan 预加载：模型在服务启动时预加载，避免首次请求等待约 11s
  - 线程锁（`_lock`）防止并发加载
  - `asyncio.to_thread()` 包装同步推理，不阻塞事件循环
  - 模型加载失败时自动降级为原始检索结果，不中断流程
- **RAG 精排接口**: `rerank(query, documents) -> [(document, score), ...]`，按相关性降序
- **两阶段意图识别分类接口**: `classify_route(query) -> (target, confidence)`
  - **Stage 1 — Embedding 初筛**：计算 query 与各 `category_descriptions` 的 embedding cosine 相似度，取 `router.embedding_top_k` 个候选
  - **Stage 2 — Reranker 精排**：Cross-Encoder 对 Top-K 候选精准打分，取最高分与次高分差值 margin，通过 sigmoid 映射为 confidence（`margin_temperature` 控制锐度）
  - **置信度判断**：confidence ≥ `router.confidence_threshold` 时直接返回 target；否则返回 `"fallback"` 交由 LLM 意图识别兜底
- **配置**:
  ```yaml
  rag:
    rerank_enabled: true                      # 是否启用重排序
    rerank_model: "BAAI/bge-reranker-base"    # 重排序模型（中文原生 Cross-Encoder）
    rerank_batch_size: 4                      # 批处理大小
    rerank_output_k: 4                        # 精排后输出数量

  router:
    confidence_threshold: 0.6                 # 置信度阈值（低于此值回退 LLM 意图识别）
    margin_temperature: 2.0                   # margin→confidence sigmoid 锐度
    embedding_top_k: 2                        # Embedding 初筛保留候选数
    category_descriptions:                    # 各代理类别的自然语言描述
      chat:   "日常对话和闲聊..."
      rag:    "查询文档资料和知识库..."
      mcp:    "进行数学计算或使用工具..."
      vision: "分析识别用户上传的图片..."
  ```

### 2.6 MCP Agent（工具调用代理）

- **文件**: `app/agents/mcp_agent.py`
- **职责**: 调用 MCP 工具执行计算、数据获取等操作
- **协议**: 支持 stdio（子进程）和 SSE（HTTP）两种传输方式（当前仅实现 stdio）
- **客户端**: `app/services/mcp_client.py`
- **工作流**:
  1. 启动时从配置读取所有注册的 MCP Server
  2. 运行时通过 `can_handle()` 关键词匹配判断是否需要工具调用
  3. 调用所有 MCP Server 的 `list_tools()` 获取可用工具列表
  4. 将工具列表发送给 LLM，由 LLM 决策调用哪个工具及参数
  5. 通过 `MCPClient.call_tool()` 执行工具调用
  6. 将工具返回结果交给 LLM 生成自然语言回答
  7. 带重试机制（最多 3 次），JSON 解析失败自动重试

### 2.7 MCP Server（自定义工具服务器）

- **文件**: `app/mcp_servers/calculator_server.py`
- **协议**: 遵循 Model Context Protocol (MCP)
- **SDK**: 使用 `mcp>=1.0.0,<2.0.0` Python SDK
- **通信方式**: 通过 stdio（标准输入输出）与客户端进程通信
- **注册流程**:
  1. 使用 `@server.list_tools()` 装饰器注册可用工具（定义 name, description, inputSchema）
  2. 使用 `@server.call_tool()` 装饰器实现工具执行逻辑
  3. 通过 `stdio_server()` 建立双向通信通道
- **扩展方式**: 在 `app/mcp_servers/` 目录下新建服务器文件，并在 `config.yaml` 中注册

### 2.8 Embedding 服务

- **文件**: `app/services/llm_client.py`
- **方式**: HuggingFaceEmbeddings（本地 CPU 运行）
- **默认模型**: `BAAI/bge-small-zh-v1.5`（512 维，中文优化）
- **依赖**: `sentence-transformers>=2.7.0`
- **特点**: 无需外部 API，首次自动下载模型（约 100MB），之后离线可用

### 2.9 Milvus 向量库

- **文件**: `app/services/milvus_service.py`
- **版本**: Milvus 2.3.0 (Standalone)
- **连接**: localhost:19530
- **Collection**: `rag_docs`
- **索引**: IVF_FLAT + Inner Product (IP)
- **ID 策略**: 每条 chunk 生成 UUID hex 作为主键

### 2.10 会话上下文管理（标题摘要 + 滚动摘要）

- **文件**: `app/services/history_service.py`
- **职责**: 管理对话历史的滑动窗口、双层摘要生成与历史裁剪
- **两种摘要，独立管理**：

|| 类型 | 触发时机 | 目的 | 长度 | 更新频率 | 存储 |
|------|---------|------|------|---------|------|
| 标题摘要 | 首次对话后（异步） | UI 展示、会话标识 | 5~15 字 | 生成一次，不再更新 | `_session_titles` |
| 滚动摘要 | 对话超阈值（默认 10 轮，异步） | 上下文窗口管理，压缩旧历史 | 50~200 字 | 每次超阈值增量更新 | `_session_summaries` |
| turn 计数器 | 每轮对话时递增 | 精确追踪对话轮数，驱动窗口/裁剪 | — | 随对话持续递增 | `_session_turn_counter` |

- **标题摘要 (`generate_title`)**:
  1. 首次对话结束后异步触发（`chat.py` 中 `asyncio.create_task`）
  2. 已有标题则跳过，不重复生成
  3. 只取前 2 轮对话作为素材，prompt 要求直接输出标题
  4. 后处理：剥离推理泄漏（按分隔符取最后一段）、清理前缀/引号、限制 20 字
  5. 以 `【会话主题】{title}` 注入 LLM 上下文

- **滚动摘要 (`generate_summary`)**:
  1. 对话轮数 > 阈值（默认 10）时异步触发
  2. 带锁（`_summary_locks`），防止并发更新
  3. 将超出阈值的旧对话交给 LLM 生成摘要，已有旧摘要时追加为上下文
  4. 生成后裁剪历史，只保留最近 `keep_rounds` 轮
  5. 以 `【历史摘要】\n{summary}` 注入 LLM 上下文

- **Turn 计数器机制**:
  - 每个会话维护一个递增的 `turn` 号，存储在 `_session_turn_counter[sid]`
  - `append_history()` 时 turn +1，同一轮的 user 和 assistant 消息共享同一 turn 号
  - 每条消息带 `"turn": N` 字段（内部字段，不暴露给前端/API）
  - `needs_summary()` 直接读 `_session_turn_counter` 判断轮数，不再用 `len(history) // 2`
  - 滑动窗口（`get_history`）按 turn 号集合取最近 window 轮，不再用 `[-window * 2:]`
  - 历史裁剪按 `turn > cutoff_turn` 过滤，不再用条数索引
  - `clear_history()` 同时清理 `_session_turn_counter`

- **上下文注入顺序**（`chat_agent.py`）:
  ```
  System: 你是一个友好、专业的 AI 助手...
  System: 【会话主题】关于高等数学学习的对话
  System: 【历史摘要】用户叫 AI 小张，正在学习高数...
  User: (近期对话历史)
  User: (当前消息)
  ```

- **清理**: `clear_history()` 同时清理标题、摘要、turn 计数器和历史

### 2.11 STT 语音转写服务

- **文件**: `app/services/stt_service.py` + `app/routers/stt.py`
- **模型**: FunASR Paraformer-zh（~840MB，含 VAD + 标点恢复子模型）
- **依赖**: `funasr>=1.0.0`, `modelscope>=1.0.0`, `soundfile>=0.12.0`
- **输入**: 16kHz mono WAV（意图识别层通过 `_ensure_wav()` 兜底转换 mp3/webm/ogg 等）
- **输出**: 带标点的中文文本
- **架构特点**:
  - lifespan 启动时预加载模型，避免首次请求等待
  - `get_stt_service()` 全局单例，全应用共享一个模型实例
  - `asyncio.to_thread()` 包装同步推理，不阻塞事件循环
- **前端集成**: `static/test_stream.html` 通过 Web Audio API 直出 PCM 并手动封装 WAV，保证标准格式

### 2.12 Vision Agent（图片理解代理）

- **文件**: `app/agents/vision_agent.py`
- **职责**: 基于小 VL + 大 LLM 模式的图片理解问答
- **架构（4 阶段，两模型职责分离）**:
  1. **Phase 1 — VL 结构化特征提取**：VL 模型根据用户问题从图片提取结构化特征（JSON），包含 `features`、`scene_description`、`text_content`、`objects` 字段
  2. **Phase 2 — LLM 自检**：文本 LLM 检查提取的特征是否足以回答用户问题，输出 `sufficient`/`missing_info`/`reason`
  3. **Phase 3 — 补偿轮（可选）**：自检结果为 `insufficient` 时，VL 模型根据缺失信息定向补充提取 `additional_features`，合并到已有特征中
  4. **Phase 4 — LLM 回答**：文本 LLM 基于特征生成自然语言回答，要求不臆测图片中不存在的内容
- **模型选择**：配置问题，架构保持职责分离——VL 负责视觉感知→结构化特征，文本 LLM 负责推理检查+最终回答
- **降级策略**：无图片时返回提示"请先上传图片"；处理异常时返回错误提示
- **流式支持**：`handle_stream()` 暂不支持流式 VL，回退到非流式一次性输出
- **JSON 解析兼容**：`_try_parse_json()` 支持直接解析、````json```代码块提取、`{...}`花括号提取三种方式
- **配置**:
  ```yaml
  vision:
    openai_base_url: "${VL_BASE_URL}"
    api_key: "${VL_API_KEY}"
    model_name: "${VL_MODEL_NAME}"
    temperature: 0.1
    max_tokens: 2048
    phases: "simple"    # "simple"=VL提取+LLM回答(快) / "full"=4阶段含自检+补偿(慢但准确)
  ```

### 2.13 VL 多模态客户端

- **文件**: `app/services/vl_client.py`
- **模型**: 任何兼容 OpenAI API 格式的 VL 模型（Qwen2.5-VL、InternVL2、gpt-4o-mini 等）
- **依赖**: `langchain-openai`（ChatOpenAI）
- **架构特点**:
  - `get_vl_client()` 全局单例，延迟初始化
  - 图片通过 Base64 data URI 编码后放入 `HumanMessage.content` 多模态数组
  - 支持图片格式：jpg、jpeg、png、webp、gif、bmp
  - 编码失败时降级为文本占位 `[图片加载失败: path]`
- **接口**: `analyze(prompt, image_paths) -> str`：发送提示词+图片列表，返回 VL 模型文本响应
- **配置**: 从 `config.vision` 读取 `openai_base_url`/`api_key`/`model_name`/`temperature`/`max_tokens`

### 2.14 Session Context 服务

- **文件**: `app/services/session_context_service.py`
- **职责**: 为每个会话维护上下文包，自动继承上一轮用到的图片 ID
- **核心功能**:
  - **图片 ID 继承**：Chat 请求未传 `image_ids` 时，自动继承同 session 上一轮的图片 ID，支持多轮连续问答
  - **TTL 过期清理**：默认 30 分钟，超时后 `get_recent_image_ids()` 返回空列表
  - **去重合并**：多轮图片 ID 按顺序去重合并，避免重复
  - **显式清除**：`clear_image_ids()` 清除指定会话的图片 ID；`clear_context()` 清除整个上下文包
- **存储**：内存存储（`_session_context` 字典），服务重启后清空
- **调试**：`get_context_stats()` 返回各会话的图片 ID 数量

## 3. API 接口

### 3.1 POST /upload

上传文件，立即返回，后台异步切片并向量化存入 Milvus。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | UploadFile | 要上传的文档（PDF/TXT/MD/DOCX） |
| `session_id` | Form string | 会话标识（可选，默认 "default"） |

**响应**:
```json
{
  "file_id": "a1b2c3d4e5f6",
  "filename": "document.pdf",
  "status": "processing"
}
```

**处理流程**:
1. 接收文件并保存到本地 `uploads/` 目录
2. 注册文件状态为 `processing`
3. 提交后台异步任务（切片 + 向量化）
4. 立即返回 `file_id` 和 `status: "processing"`

**状态查询**: 通过 `GET /upload/status/{file_id}` 查询文件处理状态（`processing` / `ready` / `error`）

**前端集成**: 前端通过轮询方式获取文件处理状态，上传时显示 `⏳`，分析完成显示 `✅`，处理期间锁定发送按钮。

### 3.2 GET /upload/status/{file_id}

查询文件处理状态。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_id` | path string | 文件 ID |

**响应**:
```json
{
  "file_id": "a1b2c3d4e5f6",
  "filename": "document.pdf",
  "session_id": "default",
  "status": "ready",
  "chunks": 5
}
```

### 3.3 POST /chat

发送消息，由 Main Agent 意图识别到合适的子代理处理。

| 参数 | 类型 | 说明 |
|------|------|------|
| `message` | string | 用户消息 |
| `session_id` | string | 会话标识（可选） |
| `file_ids` | string[] | 引用已上传文件的 ID 列表（可选） |
| `image_ids` | string[] | 引用已上传图片的 ID 列表（可选，未传时自动继承 Session Context） |

**响应**:
```json
{
  "reply": "回答内容",
  "agent_used": "rag",
  "cot_reasoning": "用户询问上传的文档内容",
  "sources": ["相关片段1", "相关片段2"]
}
```

### 3.4 GET /health

健康检查，返回服务状态。

### 3.5 /static（静态文件服务）

- **路径前缀**: `/static`
- **目录**: `static/`
- **实现**: FastAPI `StaticFiles` 中间件，启用 `html=True` 模式
- **用途**: 提供开发测试页面等静态资源
- **示例**: `http://localhost:8000/static/test_stream.html` — 流式聊天测试页面
- **注意**: `mount` 必须放在所有路由定义之后，否则会截获该路径前缀下的所有请求

### 3.6 POST /stt/transcribe

语音转文字接口，支持 wav、mp3、webm、ogg、flac、m4a 等格式。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | UploadFile | 音频文件 |

**响应**:
```json
{
  "text": "转写结果文本",
  "model": "paraformer-zh",
  "duration_ms": 1250
}
```

**前端集成**: 前端通过 Web Audio API 录制 16kHz mono WAV，以 FormData 上传。

### 3.7 POST /upload/image

上传图片文件，返回 `image_id` 用于后续 Chat 请求引用。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | UploadFile | 要上传的图片（支持 jpg/jpeg/png/webp/gif/bmp/tiff） |

**响应**:
```json
{
  "image_id": "a1b2c3d4e5f6",
  "filename": "photo.jpg",
  "size_bytes": 123456
}
```

**处理流程**:
1. 校验文件后缀是否为支持的图片格式
2. 校验文件内容长度（至少 20 字节）
3. 生成 `image_id`（UUID hex 前 12 位），保存图片到 `uploads/images/` 目录
4. 注册图片元信息到 `uploads/images/.registry.json`（持久化，防止 reload 丢失）
5. 返回 `image_id`、原始文件名和文件大小

**持久化注册表**:
- 图片元信息保存到 `uploads/images/.registry.json`
- 服务启动时自动加载注册表，清理已不存在的文件记录
- `resolve_image_paths(image_ids)` 函数将 image_ids 解析为文件路径列表

**使用方式**:
```bash
# 上传图片
curl -X POST http://localhost:8000/upload/image -F "file=@photo.jpg"

# 在 Chat 请求中引用图片
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"这张图片里有什么？","image_ids":["<image_id>"]}'
```

## 4. 配置文件

### 4.1 config.yaml 结构

```yaml
llm:                          # LLM 对话模型（OpenAI 兼容 API）
  openai_base_url: "..."      # API 地址
  api_key: "${OPENAI_API_KEY}" # API 密钥（支持环境变量）
  model_name: "..."           # 模型名
  temperature: 0.1            # 温度参数
  max_tokens: 4096            # 最大输出 token

router_llm:                   # 路由专用 LLM（小模型，Reranker 置信度不足时LLLM兜底）
  openai_base_url: "${ROUTER_LLM_BASE_URL}"     # 路由模型 API 地址
  api_key: "${ROUTER_LLM_API_KEY}"              # 路由模型 API 密钥
  model_name: "${ROUTER_LLM_MODEL_NAME}"        # 路由模型名（小模型即可）
  temperature: 0.1                              # 路由需确定性输出
  max_tokens: 256                               # 路由只需短 JSON

router:                       # 两阶段路由参数（Embedding 初筛 → Reranker 精排 → Intent Recognition 置信度）
  confidence_threshold: 0.6                     # 低于此分数 fallback LLM
  margin_temperature: 2.0                       # margin→confidence 映射锐度
  embedding_top_k: 2                            # Embedding 初筛保留候选数
  category_descriptions:                        # 各代理类别的自然语言描述（用于 Intent Recognition）
    chat:   "日常对话和闲聊，比如打招呼、问候、聊天..."
    rag:    "查询文档资料和知识库..."
    mcp:    "进行数学计算或使用工具服务..."
    vision: "分析识别用户上传的图片..."

milvus:                       # 向量数据库 + Embedding
  host: "localhost"           # Milvus 地址
  port: 19530                 # Milvus 端口
  collection_name: "rag_docs" # 集合名
  embedding_model: "BAAI/bge-small-zh-v1.5"  # HuggingFace 模型
  embedding_dim: 512          # 向量维度
  index_params:
    metric_type: "IP"         # 相似度度量（IP = 内积，等价余弦）
    index_type: "IVF_FLAT"    # 索引类型

mcp:                          # MCP 工具服务器
  servers:
    - name: "calculator"        # 服务器名称
      command: "python"         # 启动命令
      args: ["-m", "app.mcp_servers.calculator_server"]  # 启动参数
      env: {}                   # 环境变量（可选）
    - name: "fetch"
      command: "uvx"
      args: ["mcp-server-fetch"]
      env: {}
  # url 字段用于 SSE 传输（当前未实现）

rag:                          # RAG 参数
  chunk_size: 500             # 文档分块大小
  chunk_overlap: 50           # 分块重叠
  top_k: 16                   # 粗排召回候选数量
  rerank_enabled: true        # 是否启用重排序
  rerank_model: "BAAI/bge-reranker-base"  # 重排序模型（中文原生 Cross-Encoder）
  rerank_batch_size: 4        # 批处理大小
  rerank_output_k: 4          # 精排后输出数量

paddle_ocr:                   # 飞桨 PaddleOCR 云端 OCR
  endpoint: "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
  token: "${PADDLEOCR_ACCESS_TOKEN}"
  model: "PaddleOCR-VL-1.6"

vision:                       # VL 多模态模型（图片理解）
  openai_base_url: "${VL_BASE_URL}"   # VL 模型 API 地址
  api_key: "${VL_API_KEY}"            # VL 模型 API 密钥
  model_name: "${VL_MODEL_NAME}"      # VL 模型名称
  temperature: 0.1                    # 温度参数
  max_tokens: 2048                    # 最大输出 token
  phases: "simple"                    # "simple"=提取+回答 / "full"=4阶段含自检+补偿

main_agent:                   # 主代理配置（LangGraph 使用）
  cot_prompt_template: |      # 意图识别推理提示模板（用于LLM兜底）
    ...
  fallback_agent: "chat"      # 降级代理

preprocess:                   # 预处理管道配置
  enable_typo_correction: true                    # 是否启用错别字纠正
  typo_model: "shibing624/macbert4csc-base-chinese" # 纠错模型名

server:                       # 服务器配置
  host: "0.0.0.0"
  port: 8000
  max_file_size_mb: 10

langfuse:                     # Langfuse 可观测性配置（可选）
  enabled: true
  host: "${LANGFUSE_BASE_URL}"
  public_key: "${LANGFUSE_PUBLIC_KEY}"
  secret_key: "${LANGFUSE_SECRET_KEY}"
```

### 4.2 配置优先级

YAML 文件 → 环境变量（`APP_*` 前缀）→ Pydantic 默认值

环境变量示例:
```bash
export APP_LLM_MODEL_NAME="gpt-4o"
export APP_MILVUS_HOST="192.168.1.100"
export APP_SERVER_PORT=8080
```

## 5. 依赖清单

| 包 | 版本 | 用途 |
|------|------|------|
| fastapi | >=0.115.0 | Web 框架 |
| uvicorn[standard] | 0.34.0 | ASGI 服务器 |
| openai | >=1.58.1,<2.0.0 | LLM API 客户端 |
| pymilvus | 2.5.0 | Milvus SDK |
| langchain | 0.3.0 | 编排框架 |
| langchain-openai | 0.3.0 | LangChain OpenAI 集成 |
| langchain-community | 0.3.0 | LangChain 社区集成（HuggingFace Embedding 等） |
| sentence-transformers | >=2.7.0 | 本地 Embedding 模型 |
| python-docx | 1.1.0 | DOCX 解析 |
| httpx | 0.27.2 | HTTP 客户端（PaddleOCR API 调用等） |
| mcp | >=1.0.0,<2.0.0 | MCP 协议 SDK |
| pyyaml | 6.0.1 | YAML 配置解析 |
| pydantic | 2.11.0 | 数据验证 |
| pydantic-settings | 2.9.0 | 配置管理 |
| langgraph | >=0.2.0,<0.3.0 | LangGraph 图编排 |
| langfuse | >=2.0,<3.0 | 链路追踪与可观测性 |
| funasr | >=1.0.0 | FunASR 语音识别引擎（Paraformer-zh） |
| modelscope | >=1.0.0 | ModelScope Hub（模型下载） |
| soundfile | >=0.12.0 | 音频读取 / WAV 转换 |
| transformers | >=4.40.0 | HuggingFace Transformers（macbert4csc 纠错模型） |
| torch | >=2.0.0 | PyTorch（TypoCorrector 模型推理） |

> **已移除的依赖：** `pypdf`（PDF 改用飞桨 PaddleOCR-VL 云端 OCR）、`markdown` + `beautifulsoup4`（MD 不再转 HTML，直接保留原文）、`reportlab`（测试 mock PDF 改用 `fixtures/` 预生成文件）、`torchaudio`（改用 soundfile 处理音频）。

## 6. Docker 服务

| 服务 | 镜像 | 端口 |
|------|------|------|
| etcd | quay.io/coreos/etcd:v3.5.5 | 2379 |
| minio | minio/minio:RELEASE.2023-03-20 | 9000, 9001 |
| milvus-standalone | milvusdb/milvus:v2.3.0 | 19530, 9091 |

## 7. 文件解析支持

| 格式 | 解析库 | 说明 |
|------|------|------|
| .txt | 内置 | 纯文本直接读取 |
| .pdf | 飞桨 PaddleOCR-VL 云端 API | OCR 识别转 Markdown，需配置 `PADDLEOCR_ACCESS_TOKEN` |
| .md | 内置 | 直接保留原文，不转 HTML |
| .docx | python-docx | 结构化提取标题层级 + 表格 → Markdown |

**测试资源：** 测试用静态文件统一存放在 `fixtures/` 目录（如 `rag_test_document.pdf`），纳入 git 版本管理，无需运行时生成。

## 8. 子代理注册规范

所有子代理必须继承 `SubAgentBase` 并实现：

```python
class SubAgentBase:
    name: str                                  # 代理名称
    async def can_handle(self, query, context) -> bool  # 是否可处理
    async def handle(self, query, context) -> dict       # 处理请求
```

`handle()` 返回格式:
```python
{
    "answer": "响应文本",
    "agent_used": "代理名",
    "sources": ["可选", "引用来源"]
}
```

## 9. Langfuse 可观测性

### 9.1 概述

项目集成 Langfuse 进行 LLM 调用链路追踪，自动记录：
- **Trace**：单次请求的完整调用链（Main Agent → 子代理 → LLM）
- **Span**：关键步骤的耗时（意图识别推理、RAG 检索、MCP 工具调用）
- **Generation**：LLM 调用详情（输入/输出 tokens、耗时、模型名）

### 9.2 配置方式

在 `.env` 文件中设置：

```bash
# Langfuse 配置
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

配置优先级：`.env` → `config.yaml` → 默认值

### 9.3 使用说明

启用后，以下操作会自动被追踪：

| 操作 | Trace 名称 | Span 名称 |
|------|------------|-----------|
| Main Agent 意图识别 | `main_agent` | `intent_routing` |
| Chat Agent | `chat_agent` | `chat_agent` |
| RAG Agent | `rag_agent` | `rag_agent` |
| MCP Agent | `mcp_agent` | `mcp_agent` |
| Vision Agent | `vision_agent` | `vision_agent` |
| LangChain LLM 调用 | 自动创建 | `LLM` |

### 9.4 查看追踪数据

1. 访问 [Langfuse Dashboard](https://cloud.langfuse.com)
2. 使用 `LANGFUSE_PUBLIC_KEY` 对应的项目
3. 在 Traces 页面查看请求链路
4. 点击 Trace 可查看详细调用信息、耗时、输入输出等

### 9.5 关闭可观测性

将 `LANGFUSE_ENABLED` 设置为 `false` 或直接删除相关配置即可。

# 📚 项目代码学习路线图

## 第一阶段：理解整体架构（1-2天）

先看这些文件，建立全局认知：

| 文件 | 作用 | 学习重点 |
| --- | --- | --- |
| claude.md | 项目说明 | 了解项目目标和架构设计 |
| specs/spec.md | 技术规格 | 了解技术选型和模块关系（事实标准，详情见 `specs/spec.md`） |
| configs/config.yaml | 配置文件 | 了解系统如何配置（LLM、MCP、Milvus） |
| app/main.py | 入口文件 | 看懂 FastAPI 启动流程、Langfuse 初始化 |
| app/config.py | 配置加载 | Pydantic + YAML 配置管理方式 |

**学习目标：**

- 理解 Main Agent 如何路由到 Sub Agents（chat/rag/mcp）
- 了解数据流向：用户请求 → 预处理 → 路由 → 子代理 → 返回

## 第二阶段：核心路由机制（2-3天）

| 文件 | 作用 | 学习重点 |
| --- | --- | --- |
| app/graph/state.py | 状态定义 | LangGraph 的 AgentState 结构 |
| app/graph/graph.py | 图构建 | 节点和边的连接关系 |
| app/graph/nodes.py | 图节点 | preprocess（错字纠正→敏感词过滤）/router/chat/rag/mcp 节点实现 |
| app/chains/cot_chain.py | CoT 推理链 | LLM 如何决定路由目标 |

**关键流程：**

```
用户请求 → preprocess_node(错字纠正→敏感词过滤)
        → router_node(CoT 推理决定 target_agent)
        → 条件边路由到 chat/rag/mcp
        → 返回结果
```

## 第三阶段：子代理实现（3-4天）

按优先级顺序学习：

| 文件 | 作用 | 学习重点 |
| --- | --- | --- |
| app/agents/sub_agent_base.py | 抽象基类 | can_handle() + handle() 接口 |
| app/agents/chat_agent.py | 闲聊代理 | 最简单的子代理实现 |
| app/agents/mcp_agent.py | 工具调用代理 | MCP 协议调用、工具列表获取 |
| app/agents/rag_agent.py | RAG 代理 | 调用 RAGChain 进行检索生成 |

**学习目标：**

- 理解子代理如何判断是否能处理请求（can_handle）
- 理解子代理如何处理请求并返回结果（handle）
- 理解 Langfuse Span 追踪的实现

## 第四阶段：底层服务（3-4天）

| 文件 | 作用 | 学习重点 |
| --- | --- | --- |
| app/services/llm_client.py | LLM 客户端 | LangChain ChatOpenAI + Embeddings |
| app/services/milvus_service.py | 向量服务 | Milvus 检索 + LangChain vectorstore |
| app/services/mcp_client.py | MCP 客户端 | stdio 子进程通信、mcp SDK |
| app/services/typo_service.py | 错别字纠正 | macbert4csc 单例、asyncio.to_thread 异步推理 |
| app/services/history_service.py | 对话历史管理 | 标题摘要 / 滚动摘要双层机制、滑动窗口、历史裁剪 |
| app/chains/rag_chain.py | RAG 链 | 检索 + 生成的完整流程 |

**学习重点 — history_service.py 双层摘要机制：**

```
首次对话 → generate_title() → 5~15字标题 → 存入 _session_titles → 不再更新
每轮对话 → append_history() → turn 计数器 +1 → 消息带 turn 号
对话超阈值 → needs_summary() 读 turn 计数器 → generate_summary() → 压缩旧历史 → 存入 _session_summaries → 增量更新
滑动窗口 → get_history() → 按 turn 号取最近 N 轮
历史裁剪 → generate_summary() → 按 turn > cutoff_turn 过滤
Chat Agent 构建上下文 → 标题(【会话主题】) + 摘要(【历史摘要】) + 历史窗口 + 当前消息
```

## 第五阶段：API 接口（2天）

| 文件 | 作用 | 学习重点 |
| --- | --- | --- |
| app/routers/chat.py | 聊天接口 | /chat 和 /chat/stream 实现 |
| app/routers/upload.py | 上传接口 | 文件异步处理 + 后台切片入库 |
| app/routers/stt.py | 语音转写接口 | /stt/transcribe 接收音频返回文字 |
| app/utils/file_processor.py | 文件处理 | PDF（PaddleOCR 云端 OCR）/MD/TXT/DOCX 解析 |
| app/main.py | 静态文件挂载 | StaticFiles 中间件，`/static` 路径提供测试页面 |
| static/test_stream.html | 流式测试页面 | 浏览器访问 `http://localhost:8000/static/test_stream.html` |

## 第六阶段：测试体系（1-2天）

| 文件 | 作用 | 学习重点 |
| --- | --- | --- |
| tests/conftest.py | pytest 配置 | 测试用配置 fixture |
| tests/test_config.py | 配置测试 | Pydantic 配置验证 |
| tests/test_file_processor.py | 文件解析测试 | 各格式解析 + 分块 + mock PDF OCR |
| tests/test_graph.py | LangGraph 自测 | 图结构 + 路由逻辑 |
| tests/test_history.py | 历史管理测试 | 标题摘要 + 滚动摘要 + 裁剪 |
| tests/test_paddle.py | PaddleOCR 集成测试 | 真实 API 调用（需 token） |
| tests/test_stt_local.py | FunASR 语音转写测试 | 本地模型推理，不依赖外部服务 |
| tests/test_preprocess.py | 预处理管道测试 | 错字纠正 + 敏感词过滤集成测试 |
| tests/test_imports.py | 模块导入检查 | 验证所有模块可正常导入 |
| fixtures/ai_learn.wav | 测试静态资源 | STT 测试用中文音频 |
| fixtures/rag_test_document.pdf | 测试静态资源 | 预生成 PDF，无需运行时生成 |

**测试资源约定：** 测试用静态文件统一存放在 `fixtures/` 目录，纳入 git 版本管理。不使用运行时生成（已移除 `reportlab` 依赖）。

## 第七阶段：辅助功能（1-2天）

| 文件 | 作用 |
| --- | --- |
| app/data/sensitive_words.py | 敏感词过滤 |
| app/utils/logger.py | 日志工具 |
| app/mcp_servers/calculator_server.py | MCP Server 示例 |
---

## 🎯 学习建议

1. **先运行项目**：按照 CLAUDE.md 快速启动，测试接口
2. **边看边测**：用 Postman 或 curl 测试每个接口，观察返回结果
3. **画图理解**：画出图结构、数据流向、类继承关系
4. **调试跟踪**：在关键节点打断点，看状态如何传递

## 🔄 推荐阅读顺序（快速版）

```
configs/config.yaml
→ app/main.py
→ app/graph/state.py
→ app/graph/graph.py
→ app/graph/nodes.py
→ app/chains/cot_chain.py
→ app/agents/chat_agent.py
→ app/agents/mcp_agent.py
→ app/agents/rag_agent.py
→ app/services/llm_client.py
→ app/services/milvus_service.py
→ app/services/history_service.py
→ app/services/typo_service.py
→ app/routers/chat.py
→ app/routers/stt.py
→ app/services/stt_service.py
→ app/utils/file_processor.py
→ tests/conftest.py
→ tests/test_file_processor.py
→ tests/test_history.py
```

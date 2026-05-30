# 1. 系统架构
## 1.1 总体组件
```
[用户] ──► /chat (FastAPI) ──► MainAgent (CoT Router)
                                      │
          ┌──────────────┬────────────┼────────────┬──────────────┐
          ▼              ▼            ▼            ▼              ▼
   [ChatAgent]   [RAGAgent]   [MCPAgent]   [Other...]    [回退闲聊]
       │              │            │
       └──────────────┴────────────┘
               │
          [LLM (OpenAI API)]
```

## 1.2 数据流（以 RAG 为例）

1. 用户上传文件 + 提问 → POST /chat
2. 服务端解析文件 → 切割文本 → 生成向量 → 存入 Milvus
3. MainAgent 使用 CoT 推理 → 输出 `{ "agent": "rag", "query": "..." }`
4. RAGAgent 从 Milvus 检索 top‑k 文档 → 构建上下文 → 调用 LLM 生成最终回答
5. 返回结果给用户


# 2. API 规范
## 2.1 聊天接口

**端点：** POST /chat

**请求：** multipart/form-data

| 参数名 | 类型 | 是否必填 | 描述 |
|--------|------|----------|------|
| message | string | 是 | 用户输入的消息 |
| file | file | 否 | 上传的文档（支持 .txt, .pdf, .md） |
| session_id | string | 否 | 会话标识，用于维持 Milvus 分区或记忆 |

**响应：** JSON
```
{
  "reply": "字符串，代理的最终回答",
  "agent_used": "rag | chat | mcp",
  "cot_reasoning": "主代理的思考链（可选）"
}
```

## 2.2 健康检查
端点：GET /health → {"status": "ok"}

# 3. 配置规范

**配置文件** `configs/config.yaml` 示例：
```
llm:
  openai_base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"        # 支持环境变量替换
  model_name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1024

milvus:
  host: "localhost"
  port: 19530
  collection_name: "rag_docs"
  embedding_model: "text-embedding-3-small"   # 使用 OpenAI 嵌入
  embedding_dim: 1536
  index_params:
    metric_type: "IP"
    index_type: "IVF_FLAT"

mcp:
  servers:
    - name: "calculator"
      command: "python"
      args: ["-m", "mcp_server_calc"]
      env: {}
    - name: "weather"
      url: "http://localhost:5000/mcp"   # 支持 SSE/HTTP 传输

rag:
  chunk_size: 500
  chunk_overlap: 50
  top_k: 4

main_agent:
  cot_prompt_template: |
    你是一个智能路由器。用户问题是：{query}
    可用的子代理：chat（一般对话）、rag（需要外部知识/文档）、mcp（需要工具计算/实时数据）。
    请用 JSON 格式输出你的决策，包含 "target" 和 "reasoning" 字段。
    例如：{"target": "rag", "reasoning": "用户询问上传的文档内容"}
  fallback_agent: "chat"

server:
  host: "0.0.0.0"
  port: 8000
  max_file_size_mb: 10
```

# 4. 关键模块设计
## 4.1 MainAgent (CoT)

**输入：** user_query, uploaded_files_info

1. 构造 CoT Prompt，调用 LLM 获取路由决策
2. 解析 JSON 结果，分发到对应子代理
3. 若解析失败或子代理不存在，则调用 fallback_agent

## 4.2 RAGAgent

使用 LangChain 的 Milvus 向量存储 + OpenAIEmbeddings

**文件处理流水线：**

1. 根据后缀选择解析器（PyPDF2, textract, 等）
2. 分块：RecursiveCharacterTextSplitter
3. 生成嵌入并插入 Milvus（每个会话独立 partition）
4. 检索相关块（similarity_search_with_score）
5. 构建 Prompt 并调用 LLM 生成回答

## 4.3 MCPAgent

使用 mcp 客户端连接配置中定义的 MCP 服务器

支持两种传输：stdio（子进程）和 sse（HTTP）

**工具调用流程：**

1. 主代理识别需要调用工具
2. 提取工具名称和参数
3. 通过 MCP 协议调用对应服务器
4. 返回结果给主代理或直接输出

## 4.4 ChatAgent

简单的对话生成，可携带对话历史（基于 session_id 的内存缓存）

不涉及工具或外部检索

# 5. 异常处理与日志

1. 所有外部调用（LLM, Milvus, MCP）捕获异常并记录结构化日志
2. 当 LLM 调用超时或返回错误时，返回友好错误信息
3. 文件上传失败时仅处理文本消息，忽略文件部分

# 6. 部署要求

1. **Milvus**：Standalone 模式（推荐 Docker Compose 一键启动）
2. **Python 依赖**：`requirements.txt` 需包含 fastapi, langchain, pymilvus, mcp, openai, python-multipart, pyyaml, uvicorn
3. **环境变量**：`OPENAI_API_KEY` 必须设置

# 7. 测试要点

**单元测试：**

1. CoT 路由解析逻辑
2. 文件分块与嵌入流程
3. MCP 客户端模拟调用

**集成测试：**

1. 完整聊天流程（含文件）
2. Milvus 插入 / 检索正确性
3. 路由到正确子代理

# 8. 扩展计划

1. 支持更多文件类型（.docx, .html）
2. 增加会话记忆（LangChain ConversationBufferMemory）
3. 多模态输入（图像 → 描述 → 检索）
4. 异步 MCP 工具流式输出
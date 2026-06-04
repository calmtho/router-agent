# Router Agent - 智能代理系统

基于 Chain-of-Thought (CoT) 的 Main-Sub Agent 架构智能代理，支持智能路由、文档问答、工具调用和对话交互。

## 🌟 功能特性

| 功能 | 描述 |
|------|------|
| **智能路由** | 主代理通过 CoT 推理自动分发用户请求到合适的子代理 |
| **错别字纠正** | 基于 macbert4csc 的中文错别字自动纠正，预处理管道第一步 |
| **文档问答** | 基于 Milvus 向量库 + 两阶段检索（粗排→Cross-Encoder 精排）的 RAG 增强，支持 PDF、TXT、MD、DOCX 文件 |
| **工具调用** | 遵循 Model Context Protocol 调用外部工具（计算器、API 等） |
| **自由对话** | 提供通用对话能力，支持 OpenAI 兼容的大模型 |
| **会话上下文管理** | 双层摘要机制——标题摘要（UI 展示）+ 滚动摘要（上下文窗口管理），历史自动裁剪 |
| **语音输入** | 基于 FunASR Paraformer-zh 的本地语音转文字，前端一键录音 |
| **配置驱动** | 所有关键参数通过 YAML 配置文件管理 |

## 📋 系统架构

```
用户录音（可选）→ POST /stt/transcribe → 填入输入框
                                    ↓
用户请求（文字/转写结果）
    ↓
[FastAPI /chat 接口]
    ↓
[Main Agent - CoT 推理路由]
    ↓
    ├── Chat Agent  → 直接对话
    ├── RAG Agent   → 文档检索 + 问答
    └── MCP Agent   → 工具调用
```

## 🚀 快速开始

### 1. 环境准备

**系统要求：**
- Python 3.11+
- Docker & Docker Compose（用于运行 Milvus）

### 2. 启动 Milvus 向量数据库

使用 Docker Compose 启动 Milvus（包含 etcd、MinIO）：

```bash
docker-compose up -d
```

检查服务状态：
```bash
docker-compose ps
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 LLM 模型

编辑 `configs/config.yaml` 文件，修改以下配置：

#### 4.1 配置 LLM 服务

```yaml
llm:
  openai_base_url: "${OPENAI_BASE_URL}"          # 从环境变量加载 API 地址
  api_key: "${OPENAI_API_KEY}"                   # 从环境变量加载 API 密钥
  model_name: "${OPENAI_MODEL_NAME}"             # 从环境变量加载模型名称
  temperature: 0.7                               # 温度参数（0-1，越低越确定）
  max_tokens: 1024                               # 最大输出 token 数
```

**支持的 API 来源：**
- **OpenAI 官方**: `https://api.openai.com/v1`
- **本地 vLLM**: `http://localhost:8000/v1`
- **DeepSeek**: `https://api.deepseek.com/v1`
- **其他 OpenAI 兼容服务**

#### 4.2 配置 Embedding 模型（本地运行）

```yaml
milvus:
  host: "localhost"
  port: 19530
  collection_name: "rag_docs"
  embedding_model: "BAAI/bge-small-zh-v1.5"  # HuggingFace 模型，首次自动下载
  embedding_dim: 768                          # 向量维度（与模型匹配）
```

Embedding 使用 HuggingFace + sentence-transformers 在本地 CPU 运行，无需外部 API。
首次启动时自动下载模型（约 100MB），之后离线可用。

#### 4.3 配置向量检索参数

```yaml
rag:
  chunk_size: 500      # 文档分块大小（字符数）
  chunk_overlap: 50    # 分块重叠大小
  top_k: 16            # 先召回候选文档数量
  rerank_enabled: true             # 是否启用重排序
  rerank_model: "cross-encoder/ms-marco-MiniLM-L12-v2"  # Cross-Encoder 精排模型
  rerank_batch_size: 4             # 批处理大小（内存控制）
  rerank_output_k: 4               # 重排序后输出数量
```

#### 4.4 配置 MCP 工具服务器（可选）

如需使用工具调用功能，在 `config.yaml` 中添加 MCP 服务器配置：

```yaml
mcp:
  servers:
    - name: "calculator"          # 服务器名称
      command: "python"           # 启动命令
      args: ["-m", "app.mcp_servers.calculator_server"]  # 启动参数
      env: {}                     # 环境变量（可选）
    - name: "fetch"
      command: "uvx"
      args: ["mcp-server-fetch"]
      env: {}
```

**添加自定义 MCP 工具：**

1. 在 `app/mcp_servers/` 目录下创建新的服务器文件
2. 使用 `mcp.server.Server` 定义工具：
   ```python
   from mcp.server import Server
   import mcp.types as types

   server = Server("my_tools")

   @server.list_tools()
   async def list_tools() -> list[types.Tool]:
       return [
           types.Tool(
               name="my_tool",
               description="工具描述",
               inputSchema={"type": "object", "properties": {...}, "required": [...]}
           )
       ]

   @server.call_tool()
   async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
       # 实现工具逻辑
       return [types.TextContent(type="text", text="结果")]
   ```
3. 在 `config.yaml` 的 `mcp.servers` 下注册该服务器

> 当前仅支持 **stdio** 传输方式（子进程通信），SSE（HTTP）传输方式待后续支持。

### 4.5 配置服务器（API 服务）

```yaml
server:
  host: "0.0.0.0"              # 监听地址（0.0.0.0 表示所有网络接口）
  port: 8000                   # 监听端口
  max_file_size_mb: 10        # 最大上传文件大小
```

### 5. 设置环境变量

创建 `.env` 文件或设置系统环境变量：

```bash
# 设置 OpenAI API 配置
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL_NAME="gpt-4o-mini"

# 或者在 PowerShell 中（Windows）
setx OPENAI_API_KEY "your-api-key-here"
setx OPENAI_BASE_URL "https://api.openai.com/v1"
setx OPENAI_MODEL_NAME "gpt-4o-mini"
```

### 6. 启动服务

```bash
uvicorn app.main:app --reload
```

服务将在 `http://localhost:8000` 启动。

## 📖 交互流程

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户上传文件（可选）                        │
│                          ↓                                   │
├─────────────────────────────────────────────────────────────┤
│                  用户发送问题/message                         │
│                          ↓                                   │
├─────────────────────────────────────────────────────────────┤
│              FastAPI 接收请求 → /chat 端点                    │
│                          ↓                                   │
├─────────────────────────────────────────────────────────────┤
│      Main Agent 执行 CoT 推理，决定分发目标：                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CoT 推理提示：                                       │   │
│  │  "你是一个智能路由器。用户问题是：{query}               │   │
│  │   可用的子代理：chat、rag、mcp"                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│              ┌─────┬─────┬─────┐                            │
│              │目标1│目标2│目标3│                            │
│              ↓     ↓     ↓     ↓                            │
├──────────────┴──┬──┴──┬──┴──────┴───────────────────────────┤
│                 │     │                                        │
│         ┌───────▼──┐  │                                        │
│         │Chat Agent│  │                                        │
│         └──────────┘  │                                        │
│               ↓       │                                        │
│         直接对话响应    │                                        │
│                                                      ↓         │
│                                         ┌─────────────────┐   │
│                                         │    RAG Agent     │   │
│                                         └─────────────────┘   │
│                                           ↓                   │
│                             ┌────────────────────────────┐   │
│                             │  1. 检索候选文档 (Milvus top_k=16)      │   │
│                             │  2. Cross-Encoder 重排序精排            │   │
│                             │  3. 构造提示词 (RAG Chain)              │   │
│                             │  4. LLM 生成回答                        │   │
│                             └────────────────────────────┘   │
│                                           ↓                   │
│                                    返回文档相关回答            │
│                                                              │
│                                                       ↓         │
│                                          ┌─────────────────┐   │
│                                          │   MCP Agent     │   │
│                                          └─────────────────┘   │
│                                            ↓                   │
│                              ┌────────────────────────────┐   │
│                              │  1. 选择工具 (Calculator) │   │
│                              │  2. 通过 MCP 调用工具      │   │
│                              │  3. 处理工具返回结果       │   │
│                              └────────────────────────────┘   │
│                                            ↓                   │
│                                    返回工具操作结果              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
                      返回用户响应
```

### 决策逻辑

Main Agent 根据以下规则进行路由决策：

| 查询类型 | 目标代理 | 示例 |
|---------|---------|------|
| 一般对话 | `chat` | "你好"、"讲个笑话" |
| 文档问题 | `rag` | "这份文档讲了什么"、"总结一下上传的文件" |
| 计算/工具 | `mcp` | "计算 1+1"、"今天的天气" |

**降级机制：** 如果选中的代理执行失败，系统会自动降级到 `chat` 代理，确保总是有响应返回。

### 会话上下文管理

系统采用双层摘要机制，自动管理长对话的上下文窗口：

| 阶段 | 触发条件 | 生成内容 | 用途 |
|------|---------|---------|------|
| 首次对话 | 第 1 轮结束后 | 标题摘要（5~15 字） | UI 会话列表展示、会话标识 |
| 长对话 | 超过 10 轮阈值 | 滚动摘要（50~200 字） | 压缩旧历史、节省 token |

**上下文注入顺序：**
```
System: 你是一个友好、专业的 AI 助手...
System: 【会话主题】关于高等数学学习的对话
System: 【历史摘要】用户叫 AI 小张，正在学习高数...
User:  (近期对话历史，滑动窗口 10 轮)
User:  (当前消息)
```

标题和摘要独立存储、独立更新，互不影响。`clear_history()` 同时清理两者。

## 🎯 使用示例

### 示例 1: 简单对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"你好，请介绍一下你自己"}'
```

**响应示例：**
```json
{
  "reply": "你好！我是 Router Agent，一个基于 CoT 的智能代理系统...",
  "agent_used": "chat",
  "cot_reasoning": "一般问候对话"
}
```

### 示例 2: 文档问答

先上传文件，再用返回的 `file_id` 进行问答：

```bash
# 步骤 1: 上传文件
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf" \
  -F "session_id=demo"
```

**上传响应：**
```json
{
  "file_id": "a1b2c3d4e5f6",
  "filename": "document.pdf",
  "status": "ready"
}
```

```bash
# 步骤 2: 用 file_id 进行问答
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"这份文档主要讲了什么？","file_ids":["a1b2c3d4e5f6"]}'
```

**响应示例：**
```json
{
  "reply": "根据文档内容，主要讲述了...",
  "agent_used": "rag",
  "sources": ["片段1...", "片段2..."],
  "cot_reasoning": "用户询问上传文档内容"
}
```

### 示例 3: 工具调用

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"计算 123 + 456"}'
```
```

**响应示例：**
```json
{
  "reply": "123 + 456 = 579",
  "agent_used": "mcp",
  "cot_reasoning": "需要执行计算操作"
}
```

### 示例 4: 语音转写

```bash
curl -X POST http://localhost:8000/stt/transcribe \
  -F "file=@fixtures/ai_learn.wav"
```

**响应示例：**
```json
{
  "text": "人工智能技术正在快速发展",
  "model": "paraformer-zh",
  "duration_ms": 1250
}
```

## 🔧 配置文件说明

### 核心配置 (`configs/config.yaml`)

```yaml
├── llm              # 大语言模型配置
│   ├── openai_base_url      # API 基础 URL
│   ├── api_key              # API 密钥（支持环境变量）
│   ├── model_name           # 模型名称
│   ├── temperature          # 温度参数
│   └── max_tokens           # 最大生成长度
│
├── milvus           # 向量数据库配置
│   ├── host                  # Milvus 主机地址
│   ├── port                  # Milvus 端口
│   ├── collection_name       # 集合名称
│   ├── embedding_model       # HuggingFace Embedding 模型名称（本地运行）
│   ├── embedding_dim         # 嵌入维度
│   └── index_params          # 索引参数
│
├── mcp               # MCP 服务器配置
│   └── servers               # MCP 服务器列表
│       ├── name              # 服务器名称
│       ├── command           # 启动命令
│       ├── args              # 启动参数
│       └── url               # HTTP 服务器 URL
│
├── rag              # RAG 检索配置
│   ├── chunk_size            # 文档分块大小
│   ├── chunk_overlap         # 分块重叠大小
│   ├── top_k                 # 候选召回数量
│   ├── rerank_enabled        # 是否启用重排序
│   ├── rerank_model          # Cross-Encoder 精排模型
│   ├── rerank_batch_size     # 批处理大小
│   └── rerank_output_k       # 重排序后输出数量
│
├── main_agent       # 主代理配置
│   ├── cot_prompt_template   # CoT 推理提示模板
│   └── fallback_agent        # 降级代理
│
├── preprocess       # 预处理管道配置
│   ├── enable_typo_correction # 是否启用错别字纠正
│   └── typo_model            # 纠错模型名称
│
└── server           # 服务器配置
    ├── host                  # 监听地址
    ├── port                  # 监听端口
    └── max_file_size_mb      # 最大文件大小
```

## 🧪 测试

运行测试套件：

```bash
# 运行所有单元测试
pytest

# 运行特定测试文件
pytest tests/test_config.py

# 查看测试覆盖率
pytest --cov=app --cov-report=html

# 运行 PaddleOCR 集成测试（需要配置 PADDLEOCR_ACCESS_TOKEN）
python tests/test_paddle.py
```

**测试覆盖：**
- ✅ 配置加载和验证（`test_config.py`）
- ✅ 文件处理 TXT、MD、DOCX（`test_file_processor.py`）
- ✅ PDF 解析通过 mock PaddleOCR API（`test_file_processor.py`）
- ✅ 文本分块算法（`test_file_processor.py`）
- ✅ LangGraph 路由与图结构（`test_graph.py`）
- ✅ 会话历史管理：标题摘要、滚动摘要、裁剪、共存（`test_history.py`）
- ✅ PaddleOCR API 集成测试（`test_paddle.py`，需 token）
- ✅ FunASR 语音转写本地测试（`tests/test_stt_local.py`）
- ✅ 预处理管道：错字纠正 + 敏感词过滤（`tests/test_preprocess.py`）
- ✅ Cross-Encoder 重排序精排（`tests/test_reranker.py`）

**测试资源：** 静态测试文件统一存放在 `fixtures/` 目录，纳入 git 版本管理，无需运行时生成。

## 📁 项目结构

```
router_agent/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理
│   ├── agents/                 # 代理模块
│   │   ├── main_agent.py       # CoT 主代理
│   │   ├── sub_agent_base.py   # 子代理基类
│   │   ├── chat_agent.py       # 闲聊代理
│   │   ├── rag_agent.py        # RAG 代理
│   │   └── mcp_agent.py        # MCP 工具代理
│   ├── chains/                 # LangChain 链
│   │   ├── cot_chain.py        # CoT 推理链
│   │   └── rag_chain.py        # RAG 检索链
│   ├── routers/                # API 路由
│   │   ├── chat.py             # 聊天接口
│   │   ├── upload.py           # 文件上传接口
│   │   └── stt.py              # 语音转文字接口
│   ├── services/               # 服务层
│   │   ├── llm_client.py       # LLM + 本地 Embedding
│   │   ├── mcp_client.py       # MCP 客户端（stdio 子进程通信）
│   │   ├── milvus_service.py   # 向量库服务
│   │   ├── history_service.py  # 对话历史管理（标题/摘要生成、滑动窗口、历史裁剪）
│   │   ├── stt_service.py      # STT 语音转写服务（FunASR 封装）
│   │   ├── typo_service.py     # 错别字纠正服务（macbert4csc 封装）
│   │   └── reranker_service.py # 检索重排序（Cross-Encoder 精排）
│   ├── mcp_servers/            # MCP 工具服务器
│   │   └── calculator_server.py # 计算器工具示例
│   └── utils/                  # 工具函数
│       ├── file_processor.py   # 文件处理
│       └── logger.py           # 日志管理
├── configs/
│   └── config.yaml             # 主配置文件
├── static/                     # 静态文件（通过 /static 路径访问）
│   └── test_stream.html        # 流式聊天测试页面
├── fixtures/                   # 测试用静态资源（预生成，纳入 git）
│   ├── ai_learn.wav            # STT 测试用音频
│   └── rag_test_document.pdf   # RAG / PaddleOCR 测试用 PDF
├── tests/                      # 测试模块
│   ├── conftest.py             # pytest 配置
│   ├── test_config.py          # 配置测试
│   ├── test_file_processor.py  # 文件处理测试
│   ├── test_graph.py           # LangGraph 路由自测脚本
│   ├── test_history.py         # 会话历史管理测试
│   ├── test_paddle.py          # PaddleOCR API 集成测试
│   ├── test_stt_local.py       # FunASR 语音转写本地测试
│   ├── test_preprocess.py       # 预处理管道测试（错字纠正 + 敏感词过滤）
│   ├── test_reranker.py         # Cross-Encoder 重排序功能测试
│   └── test_imports.py          # 模块导入检查
├── docker-compose.yml          # Docker 编排文件
├── Dockerfile                  # Docker 镜像构建文件
├── requirements.txt            # Python 依赖
├── CLAUDE.md                   # 项目说明文档
├── README.md                   # 本文件
└── specs/
    └── spec.md                 # 技术规格说明书
```

## 🐳 Docker 部署

### 构建镜像

```bash
docker build -t router-agent:latest .
```

### 使用 Docker Compose 启动

```bash
# 启动所有服务（包括 Milvus）
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 停止服务

```bash
docker-compose down
```

## ⚙️ 故障排查

### 1. Milvus 连接失败

**问题：** 启动时报错 "Failed to connect to Milvus"

**解决方案：**
```bash
# 检查 Docker 容器状态
docker-compose ps

# 重启 Milvus
docker-compose restart milvus-standalone

# 查看日志
docker-compose logs milvus-standalone
```

### 2. API 密钥未设置

**问题：** 返回 "API key not found" 错误

**解决方案：**
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 或在 .env 文件中设置
echo "OPENAI_API_KEY=your-key" > .env
```

### 3. Embedding 模型下载失败

**问题：** 启动时 HuggingFace 模型下载缓慢或失败

**解决方案：**
```bash
# 设置 HuggingFace 镜像（国内用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地缓存目录
# 模型默认缓存路径: ~/.cache/huggingface/hub/
```

### 4. 文件上传失败

**问题：** 文件上传时返回 413 错误

**解决方案：** 检查 `max_file_size_mb` 配置：
```yaml
server:
  max_file_size_mb: 10  # 增加此值
```

## 🔄 API 接口文档

启动服务后，访问以下地址查看完整 API 文档：

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 主要接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 聊天接口（JSON 请求，引用已上传文件） |
| `/upload` | POST | 文件上传接口（提取文本并向量化入库） |
| `/health` | GET | 健康检查 |
| `/stt/transcribe` | POST | 语音转文字（支持 wav/mp3/webm 等） |
| `/static` | GET | 静态文件服务（测试页面等） |

## 📝 开发指南

### 添加新的子代理

1. 在 `app/agents/` 创建新的代理类，继承 `SubAgentBase`

```python
from app.agents.sub_agent_base import SubAgentBase

class CustomAgent(SubAgentBase):
    def __init__(self):
        super().__init__("custom")

    async def can_handle(self, query: str, context: dict) -> bool:
        return "关键词" in query

    async def handle(self, query: str, context: dict) -> dict:
        # 实现你的逻辑
        return {"answer": "响应内容", "agent": self.name}
```

2. 在 `MainAgent` 中注册新代理

### 添加新的 MCP 工具

编辑 `configs/config.yaml`，添加新的 MCP 服务器配置：

```yaml
mcp:
  servers:
    - name: "your-tool"
      command: "python"
      args: ["-m", "your_mcp_server"]
      env: {}
```

## 📄 许可证

本项目为学习和演示项目，请根据实际需求选择合适的开源许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📮 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**最后更新:** 2026-06-05

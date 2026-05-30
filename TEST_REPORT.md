# 代码审查与测试报告

> 最后更新: 2026-05-30

## 代码审查发现的问题

### 1. 类型注解问题
- **文件**: `app/config.py`
- **问题**: 使用了旧的类型注解格式 `Dict[str, str]` 而不是 `dict[str, str]`
- **状态**: ✅ 已修复

### 2. Pydantic 配置过时
- **文件**: `app/config.py`
- **问题**: 使用了已弃用的内部 `Config` 类
- **状态**: ✅ 已修复，改用 `ConfigDict`

### 3. 递归函数缺少深度限制
- **文件**: `app/config.py`
- **问题**: `replace_env_vars` 递归函数没有最大深度限制
- **影响**: 可能导致栈溢出
- **状态**: ⚠️ 待后续优化

### 4. MilvusService 初始化问题
- **文件**: `app/services/milvus_service.py`
- **问题**: 实例化时立即连接数据库，导致导入失败
- **状态**: ✅ 已修复，使用延迟初始化

### 5. FileProcessor.process() 缺少 @classmethod
- **文件**: `app/utils/file_processor.py`
- **问题**: `process` 方法参数名为 `cls` 并使用 `cls.read_txt` 等，但缺少 `@classmethod` 装饰器，导致调用时 TypeError
- **状态**: ✅ 已修复

### 6. 依赖缺失
- **问题**: `requirements.txt` 缺少 `pytest` 等测试依赖
- **状态**: ✅ 已修复

### 7. 测试文件散落在根目录
- **问题**: `test_paddle.py` 放在项目根目录，mock PDF 运行时动态生成（依赖 `reportlab`），存放于 `uploads/` 混淆业务目录
- **状态**: ✅ 已修复 — `test_paddle.py` 移入 `tests/`，mock PDF 改用 `fixtures/rag_test_document.pdf` 预生成文件，移除 `reportlab` 依赖

## 测试结果

### 测试套件概述
- 总测试数: 43 个测试用例
- 已通过: 43/43 ✅
- 集成测试: 1 个（`test_paddle.py`，需 PaddleOCR token，单独运行）

### 测试文件清单

| 文件 | 测试数 | 说明 |
|------|--------|------|
| `tests/test_config.py` | 3 | 配置加载、环境变量替换、默认值验证 |
| `tests/test_file_processor.py` | 18 | TXT/MD/DOCX 解析、PDF mock OCR、分块算法 |
| `tests/test_history.py` | 22 | 标题摘要生成、滚动摘要生成与裁剪、标题与摘要独立共存 |
| `tests/test_graph.py` | - | LangGraph 自测脚本（需 LLM 服务，非 pytest 用例） |
| `tests/test_paddle.py` | - | PaddleOCR API 集成测试（需 token，独立运行） |
| `tests/test_stt_local.py` | - | FunASR 语音转写本地测试（首次启动下载模型） |

### 测试分类明细

#### ✅ 配置测试 (3/3)
- ✅ `test_load_config_basic` - 基础配置加载
- ✅ `test_env_var_replacement` - 环境变量替换
- ✅ `test_config_validation_defaults` - 配置默认值验证

#### ✅ 文件处理器测试 (18/18)
- ✅ `test_read_txt` - TXT 文件读取
- ✅ `test_read_md_preserves_structure` - MD 保留排版结构（标题、表格）
- ✅ `test_read_docx_preserves_headings` - DOCX 结构化提取保留标题层级
- ✅ `test_read_docx_preserves_tables` - DOCX 结构化提取保留表格
- ✅ `test_split_md_by_headers` - Markdown 按标题层级切片
- ✅ `test_split_md_no_headers_fallback` - 无标题 Markdown 退回字符分割
- ✅ `test_split_text_basic` - 基础文本分块
- ✅ `test_split_text_with_overlap` - 带重叠的文本分块
- ✅ `test_split_text_small_text` - 小文本处理
- ✅ `test_process_txt_file` - TXT 文件处理
- ✅ `test_process_md_file` - MD 文件处理
- ✅ `test_process_unsupported_file_type` - 不支持的文件类型
- ✅ `test_process_file_case_insensitive` - 文件扩展名大小写不敏感
- ✅ `test_split_docx_with_headings` - DOCX 按标题切片完整流程
- ✅ `test_split_long_section_sub_split` - 超长章节二次字符分割
- ✅ `test_read_pdf_success` - PDF 通过 mock PaddleOCR API 解析
- ✅ `test_read_pdf_no_token` - 未配置 PaddleOCR token 时报错
- ✅ `test_read_pdf_job_failed` - PaddleOCR 任务失败时报错

#### ✅ 会话历史测试 (22/22)
- ✅ `test_no_history` ~ `test_above_threshold` - needs_summary 阈值判断 (4)
- ✅ `test_empty` ~ `test_exceeds_window` - 滑动窗口 (3)
- ✅ `test_no_summary_needed_below_threshold` ~ `test_summary_includes_old_summary` - 摘要生成与裁剪 (5)
- ✅ `test_full_pruning_flow` - 完整剪枝流程 (1)
- ✅ `test_title_generated_from_first_rounds` ~ `test_title_cleared_with_clear_history` - 标题生成 (9)
- ✅ `test_title_and_summary_independent` - 标题与摘要共存 (1)

## 测试资源

| 资源 | 路径 | 说明 |
|------|------|------|
| 测试 PDF | `fixtures/rag_test_document.pdf` | 预生成，约 642KB，纳入 git 版本管理 |
| 测试音频 | `fixtures/ai_learn.wav` | STT 测试用中文语音样本，纳入 git 版本管理 |

**约定**: 测试用静态文件统一存放在 `fixtures/` 目录，不使用运行时生成。已移除 `reportlab` 依赖。

## 依赖清单

### 当前依赖 (requirements.txt)

| 包 | 版本 | 用途 |
|------|------|------|
| fastapi | >=0.115.0 | Web 框架 |
| uvicorn[standard] | 0.34.0 | ASGI 服务器 |
| python-multipart | 0.0.20 | 文件上传 |
| pyyaml | 6.0.1 | YAML 配置解析 |
| pydantic | 2.11.0 | 数据验证 |
| pydantic-settings | 2.9.0 | 配置管理 |
| python-dotenv | 1.0.1 | .env 文件加载 |
| openai | >=1.58.1,<2.0.0 | LLM API 客户端 |
| httpx | 0.27.2 | HTTP 客户端（PaddleOCR API 等） |
| mcp | >=1.0.0,<2.0.0 | MCP 协议 SDK |
| pymilvus | 2.5.0 | Milvus SDK |
| langchain | 0.3.0 | 编排框架 |
| langchain-openai | 0.3.0 | LangChain OpenAI 集成 |
| langchain-community | 0.3.0 | LangChain 社区集成 |
| langgraph | >=0.2.0,<0.3.0 | LangGraph 图编排 |
| python-docx | 1.1.0 | DOCX 解析 |
| sentence-transformers | >=2.7.0 | 本地 Embedding 模型 |
| langfuse | >=2.0,<3.0 | 链路追踪 |
| pytest | 7.4.4 | 测试框架 |
| pytest-asyncio | 0.23.3 | 异步测试支持 |
| pytest-cov | 4.1.0 | 覆盖率测试 |
| pytest-mock | 3.12.0 | Mock 工具 |
| funasr | >=1.0.0 | FunASR 语音识别引擎 |
| modelscope | >=1.0.0 | ModelScope Hub（模型下载） |
| soundfile | >=0.12.0 | 音频读取 / WAV 转换 |

### 已移除的依赖
- `pypdf` — PDF 改用飞桨 PaddleOCR-VL 云端 OCR 处理
- `markdown` + `beautifulsoup4` — MD 不再转 HTML，直接保留原文
- `reportlab` — 测试 mock PDF 改用 `fixtures/` 预生成文件

## 测试覆盖率分析

### 当前覆盖范围
- ✅ 配置加载和验证逻辑
- ✅ 文件处理核心功能（TXT、MD、DOCX、PDF mock OCR）
- ✅ 文本分块算法（Markdown 标题层级切片 + 字符分割）
- ✅ 会话历史管理（标题摘要、滚动摘要、滑动窗口、历史裁剪、标题与摘要独立共存）
- ⚠️ CoT 推理链（需 LLM 服务）
- ⚠️ 代理系统（需 LLM / Milvus 服务）
- ⚠️ 主代理路由逻辑（需 LLM 服务）
- ⚠️ STT 语音转写（需 FunASR 模型，首次下载后本地运行）

### 建议增加的测试
1. **CoT 链 mock 测试** - 使用 mock LLM 测试路由推理逻辑
2. **代理系统 mock 测试** - mock 外部服务，覆盖各代理分支
3. **API 端点测试** - 使用 FastAPI TestClient 测试 /chat、/upload 接口
4. **错误处理边界测试** - 异常输入、超大文件、空内容等

## 代码质量建议

### 结构建议
1. **依赖注入** - 改用依赖注入模式，便于测试
2. **接口抽象** - 为外部服务定义接口，便于 mock
3. **配置验证** - 加强配置项的业务逻辑验证

### 安全建议
1. **输入验证** - 加强对用户输入的验证和清理
2. **文件安全** - 限制文件大小和类型
3. **API 密钥管理** - 使用环境变量管理敏感信息

### 性能建议
1. **连接池** - 为数据库和 API 调用使用连接池
2. **缓存机制** - 对频繁访问的数据添加缓存
3. **异步优化** - 确保所有 I/O 操作都是异步的

## 总结

### 代码质量评级: A- (优秀)
**优点**:
- 架构清晰，职责分离明确
- 配置管理规范（Pydantic + YAML + 环境变量）
- 测试体系完善，43/43 全部通过
- 测试资源规范，fixtures/ 预生成文件纳入 git
- PDF 解析升级为 PaddleOCR 云端 OCR，依赖精简

**待改进**:
- CoT 链和代理系统缺少 mock 测试
- API 端点缺少集成测试
- 递归函数缺深度限制

### 下一步行动计划
1. ✅ 修复已发现的代码问题
2. ✅ 测试文件归入 `tests/` 目录，mock 资源归入 `fixtures/`
3. ✅ 文档归档同步更新（CLAUDE.md / README.md / spec.md / Learning Guide.md）
4. ✅ 静态文件服务：`test_stream.html` 移入 `static/` 目录，FastAPI 挂载 `/static` 路径，浏览器访问 `http://localhost:8000/static/test_stream.html`
5. 🔄 增加 CoT 链和代理的 mock 测试
6. 🔄 增加 API 端点集成测试（FastAPI TestClient）
7. ✅ FunASR 本地 STT 功能实现与归档（`stt_service.py`、`stt.py`、`test_stt_local.py`）
8. 🔄 增加 CoT 链和代理的 mock 测试
9. 🔄 增加 API 端点集成测试（FastAPI TestClient）
10. 📊 优化测试覆盖目标至 80%+

# Course RAG 前端

这是课程 RAG 项目的 Vue 3 前端。页面默认进入“智能问答”，通过 FastAPI `POST /ask` 展示最终答案、原始引用来源、Embedding/LLM 模型和各阶段耗时；“语义检索”完整保留原有 `POST /search` Top-K 原文展示能力；“知识库管理”通过 `GET /documents`、`POST /documents` 和 `DELETE /documents/{id}` 支持上传、查看和删除 TXT、Markdown、文本型 PDF 文档。

`/ask` 返回 `answered` 或 `insufficient_context`。达到后端相关性阈值时页面显示“AI 回答”和“引用来源”；资料不足时显示“资料不足”、最高相关性、回答阈值和“检索候选”，生成耗时为 `0 ms`。前端只展示后端的结构化判断，不在浏览器中重新计算阈值。

问答和检索的来源都会展示文件名、Chunk 编号和相似度；PDF 来源带有页码时显示“第 N 页”，TXT/Markdown 不显示页码。文件名一律以普通文本渲染，不使用 `v-html`，也不会被当作链接执行。

三种模式共用同一个问题输入框（知识库管理模式下输入框隐藏但保留状态），切换不会自动请求，也不会清除另一模式最近的结果。智能问答处理 422 输入错误、502 生成失败、503 LLM 未配置、503 RAG 阈值配置无效和网络连接失败；问答不可用时仍可切换到语义检索。知识库管理处理 413 文件过大、415 类型不支持、422 空文件或解析失败、404 文档不存在、409 内置文档不可删除和网络错误；上传失败不影响已有文档，上传或删除成功后列表会自动刷新。上传和删除成功后无需重启应用即可在问答与检索中使用新内容。

## 技术栈

- Vue 3 Composition API
- Vite
- TypeScript
- 原生 CSS
- Fetch API
- Vitest
- Vue Test Utils

## 安装依赖

在 `course-rag/frontend/` 中运行：

```powershell
npm.cmd install
```

本机 PowerShell 可能优先执行受策略限制的 `npm.ps1`，因此文档统一使用 `npm.cmd`。

## 启动完整应用

终端一在项目根目录启动 FastAPI：

```powershell
uvicorn src.api:app --reload
```

终端二启动前端：

```powershell
Set-Location frontend
npm.cmd run dev
```

浏览器打开终端显示的地址，通常为 `http://127.0.0.1:5173`。前端分别请求 `/api/ask`、`/api/search` 和 `/api/documents`，Vite 会移除 `/api` 前缀并代理到 `http://127.0.0.1:8000/ask`、`/search` 和 `/documents`，无需额外配置 CORS。

配置项目根目录的后端 LLM 环境变量后，智能问答可以调用真实模型。`RAG_MIN_RELEVANCE_SCORE` 默认是 `0.35`；后端只在最高检索分数大于或等于阈值时调用 LLM，低于阈值时以 HTTP 200 返回资料不足和检索候选。该默认值只是初始启发式设置，后续需要用评估集调整。LLM 未配置或阈值无效时，`/ask` 返回安全的 503 页面提示；`/search` 和文档上传不受问答阈值影响，仍可正常使用。上传支持 `.txt`、`.md` 和文本型 PDF，默认最大 10 MB（`MAX_UPLOAD_BYTES`），不支持扫描 PDF 的 OCR。包含密钥的真实 `.env` 不得提交。

## 使用模拟服务

如果只想离线查看三种界面、不启动 FastAPI，可以在启动 Vite 前设置：

```powershell
$env:VITE_USE_MOCK_SEARCH="true"
$env:VITE_USE_MOCK_ASK="true"
$env:VITE_USE_MOCK_DOCUMENTS="true"
npm.cmd run dev
```

对应的环境变量模板是：

```env
VITE_USE_MOCK_SEARCH=false
VITE_USE_MOCK_ASK=false
VITE_USE_MOCK_DOCUMENTS=false
```

恢复真实服务模式可以关闭当前终端，或删除这两个进程环境变量。

## 验证

```powershell
npm.cmd run test
npm.cmd run type-check
npm.cmd run build
```

自动化测试注入 Fake Service 或 Fake fetch，不访问真实 FastAPI、网络或 LLM。

## 前端调用链

```text
智能问答：App.vue → AskService → useKnowledgeAsk → POST /api/ask
          → answer_status 校验
          → AskStatus → AnswerCard + AnswerSources（引用来源或检索候选）

语义检索：App.vue → SearchService → useKnowledgeSearch → POST /api/search
          → SearchStatus + SearchResults

知识库管理：App.vue → DocumentService → useDocuments → KnowledgeBasePanel
          → DocumentUpload + DocumentList（GET/POST/DELETE /api/documents）
```

`main.ts` 默认注入真实 HTTP Service；`VITE_USE_MOCK_ASK=true`、`VITE_USE_MOCK_SEARCH=true` 和 `VITE_USE_MOCK_DOCUMENTS=true` 可分别替换为确定性的 Mock Service。成功响应会先经过运行时结构校验，再进入页面状态。文档上传使用 `FormData`，不手动设置 `Content-Type`，由浏览器生成 multipart boundary。

## 当前未实现

本前端不支持扫描 PDF 的 OCR、图片识别、Word、PowerPoint、Excel、数据库、pgvector、对象存储、用户登录与多用户隔离、后台任务队列、评估集、多轮记忆、流式输出、LangChain、LangGraph 或 Agent。上传进度百分比等后端未提供的信息不会被虚构。

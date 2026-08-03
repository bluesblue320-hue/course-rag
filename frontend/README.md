# Course RAG 前端

这是课程 RAG 项目的 Vue 3 前端。页面默认进入“智能问答”，通过 FastAPI `POST /ask` 展示最终答案、原始引用来源、Embedding/LLM 模型和各阶段耗时；“语义检索”完整保留原有 `POST /search` Top-K 原文展示能力。

两种模式共用同一个问题输入框，切换不会自动请求，也不会清除另一模式最近的结果。智能问答处理 422 输入错误、502 生成失败、503 LLM 未配置和网络连接失败；LLM 未配置时仍可切换到语义检索。

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

浏览器打开终端显示的地址，通常为 `http://127.0.0.1:5173`。前端分别请求 `/api/ask` 和 `/api/search`，Vite 会移除 `/api` 前缀并代理到 `http://127.0.0.1:8000/ask` 和 `/search`，无需额外配置 CORS。

配置项目根目录的后端 LLM 环境变量后，智能问答可以调用真实模型。未配置时，`/ask` 返回 503，页面给出可切换到语义检索的提示；`/search` 不依赖 LLM，仍可正常使用。包含密钥的真实 `.env` 不得提交。

## 使用模拟服务

如果只想离线查看两种界面、不启动 FastAPI，可以在启动 Vite 前设置：

```powershell
$env:VITE_USE_MOCK_SEARCH="true"
$env:VITE_USE_MOCK_ASK="true"
npm.cmd run dev
```

对应的环境变量模板是：

```env
VITE_USE_MOCK_SEARCH=false
VITE_USE_MOCK_ASK=false
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
          → AskStatus → AnswerCard + AnswerSources

语义检索：App.vue → SearchService → useKnowledgeSearch → POST /api/search
          → SearchStatus + SearchResults
```

`main.ts` 默认注入真实 HTTP Service；`VITE_USE_MOCK_ASK=true` 和 `VITE_USE_MOCK_SEARCH=true` 可分别替换为确定性的 Mock Service。成功响应会先经过运行时结构校验，再进入页面状态。

## 当前未实现

本前端不支持 PDF、文件上传、数据库、pgvector、评估集、多轮记忆、流式输出、LangChain、LangGraph 或 Agent。

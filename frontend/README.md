# Course RAG 前端

这是课程语义检索项目的学习型 Vue 3 前端。当前版本默认通过 Vite `/api` 代理调用真实 FastAPI，展示问题输入、加载状态、Top 3 原文、相似度、Chunk 编号、检索耗时和索引元数据。它不会生成 AI 答案。

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

本机 PowerShell 禁止执行 `npm.ps1`，因此文档统一使用 `npm.cmd`。

## 启动完整应用

先在项目根目录启动 FastAPI：

```powershell
uvicorn src.api:app --reload
```

再进入 `course-rag/frontend/` 启动开发服务器：

```powershell
npm.cmd run dev
```

浏览器打开终端显示的本地地址，通常为 `http://127.0.0.1:5173`。前端请求 `/api/search`，Vite 会把请求代理到 `http://127.0.0.1:8000/search`。

## 使用模拟服务

如果只想查看界面、不启动 FastAPI，可以在启动 Vite 前设置：

```powershell
$env:VITE_USE_MOCK_SEARCH="true"
npm.cmd run dev
```

恢复真实服务模式可以关闭当前终端，或执行 `Remove-Item Env:VITE_USE_MOCK_SEARCH`。

## 运行测试

```powershell
npm.cmd run test
```

## 类型检查

```powershell
npm.cmd run type-check
```

## 生产构建

```powershell
npm.cmd run build
```

## 建议体验的问题

- 业务逻辑应该写在哪一层？
- 哪个模块负责访问数据库？
- 如何把文本转换成向量？
- 今天午饭吃什么？（用于查看空结果状态）

## 学习顺序

1. 阅读 `src/types/search.ts`，理解前后端数据契约。
2. 阅读 `src/services/httpSearchService.ts`，理解真实 HTTP 请求和运行时响应校验。
3. 阅读 `src/services/mockSearchService.ts`，理解备用模拟服务和环境切换。
4. 阅读 `src/composables/useKnowledgeSearch.ts`，理解五种页面状态。
5. 阅读 `src/components/SearchForm.vue`，理解 Props、事件和表单校验。
6. 阅读 `src/App.vue`，理解组件组合和依赖注入。

## 前后端连接

`main.ts` 默认注入 `HttpSearchService`，并在 `VITE_USE_MOCK_SEARCH=true` 时注入 `MockSearchService`。两者实现同一个 `SearchService` 接口，因此页面组件和状态机不需要关心数据来自真实 API 还是模拟数据。

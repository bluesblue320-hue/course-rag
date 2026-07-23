# Course RAG 前端

这是课程语义检索项目的学习型 Vue 3 前端。当前版本使用模拟服务展示问题输入、加载状态、Top 3 原文、相似度和 Chunk 编号，不会调用真实 Python 后端，也不会生成 AI 答案。

## 技术栈

- Vue 3 Composition API
- Vite
- TypeScript
- 原生 CSS
- Vitest
- Vue Test Utils

## 安装依赖

在 `course-rag/frontend/` 中运行：

```powershell
npm.cmd install
```

本机 PowerShell 禁止执行 `npm.ps1`，因此文档统一使用 `npm.cmd`。

## 启动开发服务器

```powershell
npm.cmd run dev
```

浏览器打开终端显示的本地地址，通常为 `http://localhost:5173`。

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
2. 阅读 `src/services/mockSearchService.ts`，理解 Promise 和模拟延迟。
3. 阅读 `src/composables/useKnowledgeSearch.ts`，理解五种页面状态。
4. 阅读 `src/components/SearchForm.vue`，理解 Props、事件和表单校验。
5. 阅读 `src/App.vue`，理解组件组合和依赖注入。

## 后续接入 FastAPI

保留 `SearchService` 接口，新建真实 API 实现并在 `main.ts` 中替换注入对象即可。页面组件和状态机不需要重写。

# Vue 3 语义检索前端设计

## 1. 背景

现有项目已经用 Python 实现了本地中文资料加载、Chunk 切分、Embedding 和 Top 3 语义检索，但入口仍是命令行，没有 Web API 或前端。本阶段先完成一个学习型 Vue 3 前端，用模拟服务还原真实异步检索流程；后续增加 FastAPI 接口时，仅替换服务实现，不重写页面组件。

## 2. 目标与非目标

### 目标

- 使用 Vue 3、Vite 和 TypeScript 实现单页语义检索界面。
- 帮助初学者理解组件、Props、事件、组合函数、异步状态和响应式布局。
- 展示问题输入、加载状态、Top 3 原文、相似度、Chunk 编号和索引说明。
- 使用前后端共享语义的接口契约，降低后续接入 FastAPI 的改动量。
- 桌面端优先；窗口变窄时自动切换为单列布局。
- 覆盖初始、加载、成功、空结果和错误五种页面状态。

### 非目标

- 不调用真实 Python 后端或真实 Embedding 模型。
- 不生成 AI 答案，不实现聊天界面。
- 不实现资料上传、删除、重新索引或知识库管理。
- 不实现真实的“知识资料”和“学习说明”页面。
- 不引入 Vue Router、Pinia、Element Plus、Axios 或大型 UI 组件库。
- 不在第一阶段增加端到端测试框架。

## 3. 已确认的技术决策

- 技术栈：Vue 3 + Vite + TypeScript。
- 项目定位：学习型实现，核心组件和 CSS 自己编写。
- 页面范围：只实现一个完整可交互的语义检索页。
- 数据策略：接口契约 + 可替换的模拟服务。
- 响应式策略：桌面端三栏优先，约 900px 以下切换为单列。
- 样式策略：CSS 变量、全局基础样式和组件 `scoped` 样式；图标使用简单的项目内 SVG，不增加图标库。

## 4. 目录与模块边界

前端位于现有项目根目录下的 `frontend/`，与 Python `src/` 并列：

```text
course-rag/
├── src/                         # 现有 Python 检索逻辑
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── AppSidebar.vue
    │   │   ├── SearchForm.vue
    │   │   ├── SearchStatus.vue
    │   │   ├── SearchResults.vue
    │   │   ├── SearchResultCard.vue
    │   │   ├── IndexSummary.vue
    │   │   └── RetrievalExplanation.vue
    │   ├── composables/
    │   │   └── useKnowledgeSearch.ts
    │   ├── services/
    │   │   ├── searchService.ts
    │   │   └── mockSearchService.ts
    │   ├── mocks/
    │   │   └── searchResponses.ts
    │   ├── types/
    │   │   └── search.ts
    │   ├── styles/
    │   │   ├── tokens.css
    │   │   └── global.css
    │   ├── App.vue
    │   └── main.ts
    ├── package.json
    ├── tsconfig.json
    └── vite.config.ts
```

`App.vue` 只组合布局和连接事件，不直接实现搜索请求。`SearchForm.vue` 负责输入和提交事件；`useKnowledgeSearch.ts` 负责状态转换；搜索服务负责数据来源；结果组件只负责展示。每个单元都能在不了解内部实现的情况下通过明确的 Props、事件或 TypeScript 接口使用。

## 5. 组件职责

### `App.vue`

组合左侧导航、中间检索内容和右侧说明面板。它从 `useKnowledgeSearch` 获取当前状态，将提交事件交给组合函数，并把响应数据传给展示组件。

### `AppSidebar.vue`

展示产品名称、“语义检索”选中状态、“知识资料”和“学习说明”占位项，以及“本地知识库 · 已就绪”状态。占位项使用不可交互展示，不伪装成已经实现的页面链接。

### `SearchForm.vue`

包含可访问的文本标签、多行问题输入框、提示文本和提交按钮。组件保留用户输入，去掉提交值两端空格并发出 `submit` 事件。`Enter` 提交，`Shift + Enter` 换行；加载期间禁用输入和按钮。

### `SearchStatus.vue`

根据状态展示加载提示、成功摘要、空结果说明或错误信息。成功状态显示结果数量和检索耗时；错误状态提供重新检索按钮。

### `SearchResults.vue` 与 `SearchResultCard.vue`

`SearchResults.vue` 管理结果列表或加载骨架。`SearchResultCard.vue` 只展示一条结果的排名、四位小数相似度、Chunk 编号和原文。

### `IndexSummary.vue`

展示响应中的文本块数量、Top K 和模型名称。初始状态使用与模拟索引一致的本地元数据；检索成功后以响应元数据为准。

### `RetrievalExplanation.vue`

用三个简短步骤解释“问题转为向量 → 计算余弦相似度 → 返回 Top 3 原文”，不承担交互职责。

## 6. 服务接口与数据契约

页面依赖 `SearchService`，而不是直接依赖某个模拟函数：

```ts
export interface SearchService {
  search(request: SearchRequest): Promise<SearchResponse>
}
```

请求和响应类型为：

```ts
export interface SearchRequest {
  query: string
  top_k: number
}

export interface SearchResult {
  rank: number
  score: number
  chunk_index: number
  text: string
}

export interface SearchResponse {
  query: string
  elapsed_ms: number
  indexed_chunks: number
  model: string
  results: SearchResult[]
}
```

字段使用与 Python 一致的 `snake_case`，避免后续接口映射。未来真实实现保留同一接口，并向 `POST /api/search` 发送：

```json
{
  "query": "业务逻辑应该写在哪一层？",
  "top_k": 3
}
```

第一阶段的 `mockSearchService` 等待约 500ms 后返回类型正确的 Promise。它根据已知关键词提供确定性结果：业务逻辑相关问题优先返回 service Chunk，数据库相关问题优先返回 repository Chunk，向量或 Embedding 相关问题优先返回 Embedding Chunk；无法匹配的输入返回空结果。服务会回显清理后的问题，并返回与现有知识库一致的模型和索引元数据。

## 7. 状态与数据流

组合函数使用可判别联合类型表示页面状态，避免多个布尔变量出现冲突：

```ts
type SearchState =
  | { status: "idle" }
  | { status: "loading"; query: string }
  | { status: "success"; response: SearchResponse }
  | { status: "empty"; response: SearchResponse }
  | { status: "error"; query: string; message: string }
```

完整流程为：

```text
用户输入问题
  → SearchForm 发出 submit
  → useKnowledgeSearch 校验输入
  → 状态切换为 loading
  → SearchService.search
  → mockSearchService 返回 SearchResponse
  → 根据 results 长度进入 success 或 empty
  → 展示状态摘要、索引元数据和结果卡片
```

加载期间禁止重复提交，因此第一阶段不需要请求取消或竞态处理。以后接入真实 API 时，可以在服务层增加 `AbortController`，不改变组件接口。

## 8. 页面布局与视觉系统

宽屏页面采用三栏布局：约 220px 左侧导航、可伸缩的主内容区、约 300px 右侧信息栏。主内容依次展示标题、检索表单、状态摘要和结果列表。视觉采用暖白背景、深蓝文字、靛蓝主操作色、青绿色成功状态和少量琥珀色分数强调。

`tokens.css` 定义颜色、字体、间距、圆角、阴影、内容宽度和断点变量；`global.css` 提供重置、基础排版和焦点样式。组件仅保留与自身结构相关的 `scoped` 样式。

约 900px 以下：

- 左侧栏变为顶部简化区域。
- 中间内容和右侧信息栏按单列排列。
- 结果卡片的元数据允许换行。
- 更窄屏幕下，检索按钮占满宽度。
- 不实现移动抽屉导航，因为第一版只有一个页面。

## 9. 交互与页面状态

- `idle`：展示输入提示，不展示虚构结果。
- `loading`：禁用表单，按钮显示“检索中…”，结果区显示三个骨架卡片。
- `success`：展示耗时、结果数量、索引信息和按排名排列的结果。
- `empty`：提示“没有找到相关内容，请换一种问法”。
- `error`：保留问题，显示可理解的错误信息和重新检索按钮。

空白输入不会进入服务层，输入框附近显示“请输入问题”。结果只展示检索原文，不出现聊天气泡或 AI 生成答案。

## 10. 错误处理与可访问性

输入错误由表单就地处理；服务拒绝 Promise 时，由组合函数把未知异常转换为稳定的中文错误信息；响应的 `results` 不是数组时视为数据格式错误。错误不导致应用崩溃，也不清空用户的问题。

可访问性要求：

- 输入框使用真实 `<label>`。
- 交互元素使用原生 `<button>` 和可见焦点样式。
- 加载和成功提示使用 `aria-live="polite"`。
- 错误提示使用 `role="alert"`。
- 相似度同时显示文字和数字，不只依赖颜色。
- 使用 `nav`、`main`、标题和 `article` 等语义化元素。
- 骨架动画尊重 `prefers-reduced-motion`。

## 11. 测试策略

使用 Vitest 和 Vue Test Utils：

- `mockSearchService`：验证延迟后返回契约数据、已知关键词顺序和未知问题空结果。
- `useKnowledgeSearch`：验证 `idle → loading → success/empty/error` 转换、输入清理和错误归一化。
- `SearchForm`：验证空输入、按钮提交、Enter 提交、Shift + Enter 换行和加载禁用。
- `SearchResultCard`：验证四位小数、Chunk 编号和原文展示。
- `App`：注入可控服务，验证提交后出现加载状态并最终展示三条结果。

测试断言面向用户可见的 DOM 和行为，不依赖组件内部变量，不把快照作为主要正确性依据。响应式样式通过构建后在约 1440px、900px 和 390px 三种宽度人工检查。

## 12. 完成标准

- 开发服务器能够启动。
- TypeScript 类型检查通过。
- 页面视觉方向与已确认的概念图一致。
- 五种页面状态均有明确渲染。
- 键盘提交、换行、焦点和错误提示有效。
- 约 900px 以下自动切换为单列。
- 单元测试和组件测试全部通过。
- 生产构建成功。
- 没有真实后端请求、知识管理功能或 AI 生成答案。

## 13. 后续接入 FastAPI

后端增加 `POST /api/search` 后，新建实现 `SearchService` 的 `apiSearchService`，内部使用原生 `fetch` 调用接口并校验基础响应结构。应用入口把依赖从 `mockSearchService` 切换为 `apiSearchService`；`App.vue`、组合函数状态模型、表单和结果组件保持不变。真实网络环境下再增加超时、请求取消、CORS 配置和端到端测试。

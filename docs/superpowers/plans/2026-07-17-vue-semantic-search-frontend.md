# Vue 3 Semantic Search Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a beginner-readable Vue 3 semantic-search page that uses a typed mock service to demonstrate the existing Python retriever's Top 3 results without calling a backend.

**Architecture:** Create a standalone `frontend/` Vite application beside the existing Python `src/`. Components render the page, `useKnowledgeSearch` owns the state machine, and an injected `SearchService` provides data so the mock can later be replaced by a FastAPI client without changing the UI.

**Tech Stack:** Vue 3, Vite, TypeScript, native CSS, Vitest, Vue Test Utils, jsdom, Node.js 24.18.0, npm 11.16.0.

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-07-17-vue-semantic-search-frontend-design.md`.
- Use Composition API and `<script setup lang="ts">` in every Vue component.
- Do not add Vue Router, Pinia, Element Plus, Axios, Tailwind, an icon package, or an end-to-end test framework.
- Keep the first release to one semantic-search page; “知识资料” and “学习说明” remain non-interactive labels.
- Use `snake_case` for transport fields: `top_k`, `elapsed_ms`, `indexed_chunks`, and `chunk_index`.
- Always request `top_k: 3`; show scores with exactly four decimal places.
- The mock service waits 500ms and returns deterministic business-logic, database, embedding, or empty results.
- Use a three-column desktop layout and switch to a single column at `max-width: 900px`.
- Use native `fetch` only in the future API service; this plan does not make a real network request.
- Use `npm.cmd` rather than `npm` in PowerShell because this machine blocks `npm.ps1`.
- Existing backend files are already staged. Never run an unscoped `git commit`; every commit in this plan must use `git commit --only ... -- <explicit paths>`.
- Work test-first: observe the expected RED failure before adding each implementation, then rerun the focused test and the complete frontend suite.

---

## File Structure

```text
course-rag/
├── .gitignore
└── frontend/
    ├── README.md
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── tsconfig.json
    ├── tsconfig.app.json
    ├── tsconfig.node.json
    ├── vite.config.ts
    └── src/
        ├── components/
        │   ├── AppSidebar.vue
        │   ├── IndexSummary.vue
        │   ├── RetrievalExplanation.vue
        │   ├── SearchForm.vue
        │   ├── SearchResultCard.vue
        │   ├── SearchResults.vue
        │   └── SearchStatus.vue
        ├── composables/
        │   └── useKnowledgeSearch.ts
        ├── mocks/
        │   └── searchResponses.ts
        ├── services/
        │   ├── mockSearchService.ts
        │   └── searchService.ts
        ├── styles/
        │   ├── global.css
        │   └── tokens.css
        ├── types/
        │   └── search.ts
        ├── App.vue
        └── main.ts
```

Tests live next to the unit they cover with `.test.ts` suffixes. This keeps each implementation and its behavioral contract together.

---

### Task 1: Vite Foundation and Search Contract

**Files:**
- Modify: `.gitignore`
- Create through the Vite scaffold: `frontend/package.json`, `frontend/package-lock.json`, `frontend/index.html`, `frontend/tsconfig.json`, `frontend/tsconfig.app.json`, `frontend/tsconfig.node.json`, `frontend/vite.config.ts`, `frontend/src/App.vue`, `frontend/src/main.ts`
- Create: `frontend/src/types/search.ts`
- Create: `frontend/src/services/searchService.ts`
- Create: `frontend/src/services/searchService.test.ts`
- Create: `frontend/src/styles/tokens.css`
- Create: `frontend/src/styles/global.css`

**Interfaces:**
- Consumes: no application interfaces.
- Produces: `SearchRequest`, `SearchResult`, `SearchResponse`, `SearchState`, `SearchService`, `searchServiceKey`, and `isSearchResponse(value: unknown): value is SearchResponse`.

- [ ] **Step 1: Scaffold the Vue TypeScript app and install test dependencies**

Run from `course-rag/`:

```powershell
npm.cmd create vite@latest frontend -- --template vue-ts --no-interactive
Set-Location frontend
npm.cmd install
npm.cmd install --save-dev vitest @vue/test-utils jsdom
```

Expected: `frontend/` is created, both installs exit 0, and `frontend/package-lock.json` exists. Vite's current Node floor is satisfied by the installed Node.js 24.18.0.

- [ ] **Step 2: Replace the generated demo with a buildable minimal shell**

Run from `course-rag/frontend/`:

```powershell
Remove-Item -LiteralPath "src\components\HelloWorld.vue" -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "src\assets\vue.svg" -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "public\vite.svg" -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "src\style.css" -Force -ErrorAction SilentlyContinue
```

Replace `frontend/src/App.vue` with:

```vue
<template>
  <main>
    <h1>Course RAG</h1>
    <p>Vue 3 语义检索前端正在构建中。</p>
  </main>
</template>
```

Replace `frontend/src/main.ts` with:

```ts
import { createApp } from "vue"

import App from "./App.vue"
import "./styles/tokens.css"
import "./styles/global.css"

createApp(App).mount("#app")
```

Expected: only the four explicit generated demo assets are removed if present, and the minimal app no longer imports any removed file.

- [ ] **Step 3: Add deterministic scripts and Vitest configuration**

Replace the `scripts` object in `frontend/package.json` with:

```json
"scripts": {
  "dev": "vite",
  "build": "vue-tsc -b && vite build",
  "preview": "vite preview",
  "type-check": "vue-tsc --noEmit",
  "test": "vitest run",
  "test:watch": "vitest"
}
```

Replace `frontend/vite.config.ts` with:

```ts
import { defineConfig } from "vitest/config"
import vue from "@vitejs/plugin-vue"

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: "jsdom",
  },
})
```

Append the following to the repository-root `.gitignore`:

```gitignore

# Frontend dependencies and test output
node_modules/
coverage/
```

- [ ] **Step 4: Write the failing response-guard test**

Create `frontend/src/services/searchService.test.ts`:

```ts
import { describe, expect, it } from "vitest"

import { isSearchResponse } from "./searchService"

const validResponse = {
  query: "业务逻辑应该写在哪一层？",
  elapsed_ms: 420,
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  results: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "service 层负责业务逻辑。",
    },
  ],
}

describe("isSearchResponse", () => {
  it("accepts a complete search response", () => {
    expect(isSearchResponse(validResponse)).toBe(true)
  })

  it("rejects a response whose results are not an array", () => {
    expect(isSearchResponse({ ...validResponse, results: null })).toBe(false)
  })

  it("rejects malformed result items", () => {
    expect(
      isSearchResponse({
        ...validResponse,
        results: [{ rank: 1, score: "high" }],
      }),
    ).toBe(false)
  })
})
```

- [ ] **Step 5: Run the focused test and verify RED**

Run:

```powershell
npm.cmd run test -- src/services/searchService.test.ts
```

Expected: FAIL because `./searchService` does not exist.

- [ ] **Step 6: Add the exact search types**

Create `frontend/src/types/search.ts`:

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

export type SearchState =
  | { status: "idle" }
  | { status: "loading"; query: string }
  | { status: "success"; response: SearchResponse }
  | { status: "empty"; response: SearchResponse }
  | { status: "error"; query: string; message: string }
```

- [ ] **Step 7: Implement the service contract and runtime response guard**

Create `frontend/src/services/searchService.ts`:

```ts
import type { InjectionKey } from "vue"

import type {
  SearchRequest,
  SearchResponse,
  SearchResult,
} from "../types/search"

export interface SearchService {
  search(request: SearchRequest): Promise<SearchResponse>
}

export const searchServiceKey: InjectionKey<SearchService> = Symbol(
  "searchService",
)

function isSearchResult(value: unknown): value is SearchResult {
  if (typeof value !== "object" || value === null) {
    return false
  }

  const result = value as Record<string, unknown>
  return (
    typeof result.rank === "number" &&
    typeof result.score === "number" &&
    typeof result.chunk_index === "number" &&
    typeof result.text === "string"
  )
}

export function isSearchResponse(value: unknown): value is SearchResponse {
  if (typeof value !== "object" || value === null) {
    return false
  }

  const response = value as Record<string, unknown>
  return (
    typeof response.query === "string" &&
    typeof response.elapsed_ms === "number" &&
    typeof response.indexed_chunks === "number" &&
    typeof response.model === "string" &&
    Array.isArray(response.results) &&
    response.results.every(isSearchResult)
  )
}
```

- [ ] **Step 8: Add design tokens and global base styles**

Create `frontend/src/styles/tokens.css`:

```css
:root {
  font-family:
    Inter, "PingFang SC", "Microsoft YaHei", system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #17213d;
  background: #f7f8fb;
  font-synthesis: none;
  text-rendering: optimizeLegibility;

  --color-bg: #f7f8fb;
  --color-surface: #ffffff;
  --color-surface-soft: #f8faff;
  --color-text: #17213d;
  --color-muted: #697386;
  --color-border: #dfe3eb;
  --color-primary: #3155c6;
  --color-primary-dark: #233f9e;
  --color-success: #128568;
  --color-success-soft: #eefaf6;
  --color-warning: #a66a00;
  --color-warning-soft: #fff8e8;
  --color-danger: #b42318;
  --color-danger-soft: #fff2f0;
  --shadow-card: 0 8px 24px rgb(23 33 61 / 8%);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
}
```

Create `frontend/src/styles/global.css`:

```css
* {
  box-sizing: border-box;
}

html {
  min-width: 320px;
  background: var(--color-bg);
}

body {
  min-width: 320px;
  min-height: 100vh;
  margin: 0;
  background: var(--color-bg);
}

button,
textarea {
  font: inherit;
}

button {
  cursor: pointer;
}

button:disabled,
textarea:disabled {
  cursor: not-allowed;
}

:focus-visible {
  outline: 3px solid rgb(49 85 198 / 28%);
  outline-offset: 2px;
}

#app {
  min-height: 100vh;
}
```

- [ ] **Step 9: Run focused tests, type checking, and the initial build**

Run:

```powershell
npm.cmd run test -- src/services/searchService.test.ts
npm.cmd run type-check
npm.cmd run build
```

Expected: 3 focused tests pass; type checking exits 0; Vite creates `frontend/dist/` and exits 0.

- [ ] **Step 10: Commit only the frontend foundation**

Run from `course-rag/`:

```powershell
git add -- .gitignore frontend
git commit --only -m "chore: scaffold Vue semantic search frontend" -- .gitignore frontend
```

Expected: the commit contains only `.gitignore` and `frontend/`; the pre-existing staged backend files remain outside the commit.

---

### Task 2: Deterministic Mock Search Service

**Files:**
- Create: `frontend/src/mocks/searchResponses.ts`
- Create: `frontend/src/services/mockSearchService.ts`
- Create: `frontend/src/services/mockSearchService.test.ts`

**Interfaces:**
- Consumes: `SearchService`, `SearchRequest`, `SearchResponse`, and `SearchResult` from Task 1.
- Produces: `DEFAULT_INDEX_METADATA`, three deterministic result collections, `createMockSearchService(delayMs?: number): SearchService`, and `mockSearchService`.

- [ ] **Step 1: Write failing mock-service tests**

Create `frontend/src/services/mockSearchService.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest"

import { createMockSearchService } from "./mockSearchService"

afterEach(() => {
  vi.useRealTimers()
})

async function finishSearch(query: string, topK = 3) {
  vi.useFakeTimers()
  const service = createMockSearchService(500)
  const pending = service.search({ query, top_k: topK })
  await vi.advanceTimersByTimeAsync(500)
  return pending
}

describe("createMockSearchService", () => {
  it("returns service-first results for a business-logic question", async () => {
    const response = await finishSearch(" 业务逻辑应该写在哪一层？ ")

    expect(response.query).toBe("业务逻辑应该写在哪一层？")
    expect(response.results).toHaveLength(3)
    expect(response.results[0].text).toContain("service 层")
    expect(response.indexed_chunks).toBe(8)
  })

  it("returns repository-first results for a database question", async () => {
    const response = await finishSearch("哪个模块负责访问数据库？")

    expect(response.results[0].text).toContain("repository 层")
  })

  it("returns embedding-first results for a vector question", async () => {
    const response = await finishSearch("如何把文本转换成向量？", 2)

    expect(response.results).toHaveLength(2)
    expect(response.results[0].text).toContain("Embedding")
  })

  it("returns an empty result list for an unrelated question", async () => {
    const response = await finishSearch("今天午饭吃什么？")

    expect(response.results).toEqual([])
  })

  it("rejects an invalid top_k before waiting", async () => {
    const service = createMockSearchService(500)

    await expect(
      service.search({ query: "业务逻辑", top_k: 0 }),
    ).rejects.toThrow("top_k 必须大于 0")
  })
})
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
npm.cmd run test -- src/services/mockSearchService.test.ts
```

Expected: FAIL because `./mockSearchService` does not exist.

- [ ] **Step 3: Add exact mock index metadata and result collections**

Create `frontend/src/mocks/searchResponses.ts`:

```ts
import type { SearchResult } from "../types/search"

export const DEFAULT_INDEX_METADATA = {
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  top_k: 3,
} as const

export const BUSINESS_RESULTS: SearchResult[] = [
  {
    rank: 1,
    score: 0.8421,
    chunk_index: 2,
    text: "service 层负责业务逻辑，例如校验状态、计算价格和协调多个数据操作。",
  },
  {
    rank: 2,
    score: 0.7168,
    chunk_index: 1,
    text: "router 层负责接收 HTTP 请求、读取参数，并调用 service 层。",
  },
  {
    rank: 3,
    score: 0.6234,
    chunk_index: 3,
    text: "repository 层负责数据访问，使 service 层无需了解具体 SQL。",
  },
]

export const DATABASE_RESULTS: SearchResult[] = [
  {
    rank: 1,
    score: 0.8652,
    chunk_index: 3,
    text: "repository 层负责查询、新增、修改和删除等数据访问操作。",
  },
  {
    rank: 2,
    score: 0.7521,
    chunk_index: 4,
    text: "PostgreSQL 用于持久化用户、课程和订单等结构化数据。",
  },
  {
    rank: 3,
    score: 0.5987,
    chunk_index: 2,
    text: "service 层通过 repository 协调数据操作，不直接编写 SQL。",
  },
]

export const EMBEDDING_RESULTS: SearchResult[] = [
  {
    rank: 1,
    score: 0.8913,
    chunk_index: 5,
    text: "Embedding 是把文本转换成一组数字向量的过程。",
  },
  {
    rank: 2,
    score: 0.8144,
    chunk_index: 6,
    text: "余弦相似度衡量两个向量方向有多接近。",
  },
  {
    rank: 3,
    score: 0.6742,
    chunk_index: 7,
    text: "RAG 会根据问题检索 Top K 相关 Chunk，再把原文交给生成模型。",
  },
]
```

- [ ] **Step 4: Implement the delayed mock service**

Create `frontend/src/services/mockSearchService.ts`:

```ts
import {
  BUSINESS_RESULTS,
  DATABASE_RESULTS,
  DEFAULT_INDEX_METADATA,
  EMBEDDING_RESULTS,
} from "../mocks/searchResponses"
import type { SearchResult } from "../types/search"
import type { SearchService } from "./searchService"

const BUSINESS_KEYWORDS = ["业务", "service", "逻辑"]
const DATABASE_KEYWORDS = ["数据库", "repository", "postgresql", "sql"]
const EMBEDDING_KEYWORDS = ["向量", "embedding", "余弦"]

function includesKeyword(query: string, keywords: string[]): boolean {
  const normalized = query.toLowerCase()
  return keywords.some((keyword) => normalized.includes(keyword))
}

function selectResults(query: string): SearchResult[] {
  if (includesKeyword(query, BUSINESS_KEYWORDS)) {
    return BUSINESS_RESULTS
  }
  if (includesKeyword(query, DATABASE_KEYWORDS)) {
    return DATABASE_RESULTS
  }
  if (includesKeyword(query, EMBEDDING_KEYWORDS)) {
    return EMBEDDING_RESULTS
  }
  return []
}

function wait(delayMs: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, delayMs)
  })
}

export function createMockSearchService(delayMs = 500): SearchService {
  return {
    async search(request) {
      const query = request.query.trim()
      if (request.top_k <= 0) {
        throw new Error("top_k 必须大于 0")
      }
      if (!query) {
        throw new Error("请输入问题")
      }

      await wait(delayMs)
      const results = selectResults(query)
        .slice(0, request.top_k)
        .map((result, index) => ({ ...result, rank: index + 1 }))

      return {
        query,
        elapsed_ms: 420,
        indexed_chunks: DEFAULT_INDEX_METADATA.indexed_chunks,
        model: DEFAULT_INDEX_METADATA.model,
        results,
      }
    },
  }
}

export const mockSearchService = createMockSearchService()
```

- [ ] **Step 5: Run focused and complete frontend tests**

Run:

```powershell
npm.cmd run test -- src/services/mockSearchService.test.ts
npm.cmd run test
npm.cmd run type-check
```

Expected: 5 mock-service tests pass; all frontend tests pass; type checking exits 0.

- [ ] **Step 6: Commit only the mock search layer**

Run from `course-rag/`:

```powershell
git add -- frontend/src/mocks frontend/src/services/mockSearchService.ts frontend/src/services/mockSearchService.test.ts
git commit --only -m "feat: add deterministic mock search service" -- frontend/src/mocks frontend/src/services/mockSearchService.ts frontend/src/services/mockSearchService.test.ts
```

---

### Task 3: Search State Composable

**Files:**
- Create: `frontend/src/composables/useKnowledgeSearch.ts`
- Create: `frontend/src/composables/useKnowledgeSearch.test.ts`

**Interfaces:**
- Consumes: `SearchService`, `isSearchResponse`, and `SearchState`.
- Produces: `useKnowledgeSearch(service: SearchService)`, returning `state`, `isLoading`, `search(rawQuery: string)`, and `retry()`.

- [ ] **Step 1: Write failing state-transition tests**

Create `frontend/src/composables/useKnowledgeSearch.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest"

import type { SearchResponse } from "../types/search"
import type { SearchService } from "../services/searchService"
import { useKnowledgeSearch } from "./useKnowledgeSearch"

const successResponse: SearchResponse = {
  query: "业务逻辑",
  elapsed_ms: 420,
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  results: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "service 层负责业务逻辑。",
    },
  ],
}

describe("useKnowledgeSearch", () => {
  it("moves from idle through loading to success", async () => {
    let resolveResponse!: (response: SearchResponse) => void
    const service: SearchService = {
      search: vi.fn(
        () =>
          new Promise<SearchResponse>((resolve) => {
            resolveResponse = resolve
          }),
      ),
    }
    const search = useKnowledgeSearch(service)

    const pending = search.search(" 业务逻辑 ")
    expect(search.state.value).toEqual({
      status: "loading",
      query: "业务逻辑",
    })
    expect(search.isLoading.value).toBe(true)

    resolveResponse(successResponse)
    await pending

    expect(search.state.value.status).toBe("success")
    expect(search.isLoading.value).toBe(false)
  })

  it("uses the empty state for a valid response with no results", async () => {
    const service: SearchService = {
      search: vi.fn().mockResolvedValue({
        ...successResponse,
        results: [],
      }),
    }
    const search = useKnowledgeSearch(service)

    await search.search("未知问题")

    expect(search.state.value.status).toBe("empty")
  })

  it("normalizes unknown failures into a stable message", async () => {
    const service: SearchService = {
      search: vi.fn().mockRejectedValue("offline"),
    }
    const search = useKnowledgeSearch(service)

    await search.search("业务逻辑")

    expect(search.state.value).toEqual({
      status: "error",
      query: "业务逻辑",
      message: "检索失败，请稍后重试",
    })
  })

  it("rejects a malformed service response", async () => {
    const service = {
      search: vi.fn().mockResolvedValue({
        ...successResponse,
        results: null,
      }),
    } as unknown as SearchService
    const search = useKnowledgeSearch(service)

    await search.search("业务逻辑")

    expect(search.state.value).toEqual({
      status: "error",
      query: "业务逻辑",
      message: "检索结果格式不正确",
    })
  })

  it("retries the last submitted query", async () => {
    const searchMethod = vi
      .fn()
      .mockRejectedValueOnce(new Error("服务暂时不可用"))
      .mockResolvedValueOnce(successResponse)
    const search = useKnowledgeSearch({ search: searchMethod })

    await search.search("业务逻辑")
    await search.retry()

    expect(searchMethod).toHaveBeenCalledTimes(2)
    expect(search.state.value.status).toBe("success")
  })
})
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
npm.cmd run test -- src/composables/useKnowledgeSearch.test.ts
```

Expected: FAIL because `./useKnowledgeSearch` does not exist.

- [ ] **Step 3: Implement the state machine and retry behavior**

Create `frontend/src/composables/useKnowledgeSearch.ts`:

```ts
import { computed, ref } from "vue"

import type { SearchService } from "../services/searchService"
import { isSearchResponse } from "../services/searchService"
import type { SearchState } from "../types/search"

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }
  return "检索失败，请稍后重试"
}

export function useKnowledgeSearch(service: SearchService) {
  const state = ref<SearchState>({ status: "idle" })
  const lastQuery = ref("")
  const isLoading = computed(() => state.value.status === "loading")

  async function search(rawQuery: string): Promise<void> {
    const query = rawQuery.trim()
    if (!query) {
      state.value = {
        status: "error",
        query: "",
        message: "请输入问题",
      }
      return
    }

    lastQuery.value = query
    state.value = { status: "loading", query }

    try {
      const response = await service.search({ query, top_k: 3 })
      if (!isSearchResponse(response)) {
        throw new Error("检索结果格式不正确")
      }
      state.value =
        response.results.length > 0
          ? { status: "success", response }
          : { status: "empty", response }
    } catch (error) {
      state.value = {
        status: "error",
        query,
        message: errorMessage(error),
      }
    }
  }

  async function retry(): Promise<void> {
    if (lastQuery.value) {
      await search(lastQuery.value)
    }
  }

  return {
    state,
    isLoading,
    search,
    retry,
  }
}
```

- [ ] **Step 4: Run focused and complete tests**

Run:

```powershell
npm.cmd run test -- src/composables/useKnowledgeSearch.test.ts
npm.cmd run test
npm.cmd run type-check
```

Expected: 5 composable tests pass; all frontend tests pass; type checking exits 0.

- [ ] **Step 5: Commit only the composable**

Run from `course-rag/`:

```powershell
git add -- frontend/src/composables
git commit --only -m "feat: add semantic search state machine" -- frontend/src/composables
```

---

### Task 4: Accessible Search Form

**Files:**
- Create: `frontend/src/components/SearchForm.vue`
- Create: `frontend/src/components/SearchForm.test.ts`

**Interfaces:**
- Consumes: `loading: boolean` from the parent.
- Produces: a `submit` event carrying one trimmed, non-empty query string.

- [ ] **Step 1: Write failing form-behavior tests**

Create `frontend/src/components/SearchForm.test.ts`:

```ts
import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import SearchForm from "./SearchForm.vue"

describe("SearchForm", () => {
  it("shows a local validation error for a blank query", async () => {
    const wrapper = mount(SearchForm)

    await wrapper.get("form").trigger("submit")

    expect(wrapper.get('[role="alert"]').text()).toBe("请输入问题")
    expect(wrapper.emitted("submit")).toBeUndefined()
  })

  it("emits a trimmed query from the submit button", async () => {
    const wrapper = mount(SearchForm)
    await wrapper.get("textarea").setValue(" 业务逻辑应该写在哪一层？ ")

    await wrapper.get("form").trigger("submit")

    expect(wrapper.emitted("submit")).toEqual([
      ["业务逻辑应该写在哪一层？"],
    ])
  })

  it("submits with Enter", async () => {
    const wrapper = mount(SearchForm)
    await wrapper.get("textarea").setValue("业务逻辑")

    await wrapper.get("textarea").trigger("keydown", {
      key: "Enter",
      shiftKey: false,
    })

    expect(wrapper.emitted("submit")).toEqual([["业务逻辑"]])
  })

  it("does not submit with Shift+Enter", async () => {
    const wrapper = mount(SearchForm)
    await wrapper.get("textarea").setValue("第一行")

    await wrapper.get("textarea").trigger("keydown", {
      key: "Enter",
      shiftKey: true,
    })

    expect(wrapper.emitted("submit")).toBeUndefined()
  })

  it("disables both controls while loading", () => {
    const wrapper = mount(SearchForm, {
      props: { loading: true },
    })

    expect(wrapper.get("textarea").attributes("disabled")).toBeDefined()
    expect(wrapper.get("button").attributes("disabled")).toBeDefined()
    expect(wrapper.get("button").text()).toBe("检索中…")
  })
})
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
npm.cmd run test -- src/components/SearchForm.test.ts
```

Expected: FAIL because `./SearchForm.vue` does not exist.

- [ ] **Step 3: Implement the form, keyboard rules, and local validation**

Create `frontend/src/components/SearchForm.vue`:

```vue
<script setup lang="ts">
import { computed, ref } from "vue"

const props = withDefaults(
  defineProps<{
    loading?: boolean
  }>(),
  {
    loading: false,
  },
)

const emit = defineEmits<{
  submit: [query: string]
}>()

const query = ref("")
const validationMessage = ref("")
const describedBy = computed(() =>
  validationMessage.value ? "query-hint query-error" : "query-hint",
)

function submit(): void {
  const trimmed = query.value.trim()
  if (!trimmed) {
    validationMessage.value = "请输入问题"
    return
  }
  validationMessage.value = ""
  emit("submit", trimmed)
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}
</script>

<template>
  <form class="search-form" @submit.prevent="submit">
    <label for="knowledge-query">向知识库提问</label>
    <textarea
      id="knowledge-query"
      v-model="query"
      name="query"
      rows="4"
      placeholder="例如：业务逻辑应该写在哪一层？"
      :disabled="props.loading"
      :aria-describedby="describedBy"
      :aria-invalid="Boolean(validationMessage)"
      @input="validationMessage = ''"
      @keydown="handleKeydown"
    />

    <div class="form-footer">
      <span id="query-hint">按 Enter 检索，Shift + Enter 换行</span>
      <button type="submit" :disabled="props.loading">
        {{ props.loading ? "检索中…" : "开始检索" }}
      </button>
    </div>

    <p v-if="validationMessage" id="query-error" role="alert">
      {{ validationMessage }}
    </p>
  </form>
</template>

<style scoped>
.search-form {
  display: grid;
  gap: var(--space-3);
  padding: var(--space-6);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  box-shadow: var(--shadow-card);
}

label {
  font-size: 1.05rem;
  font-weight: 700;
}

textarea {
  width: 100%;
  min-height: 128px;
  resize: vertical;
  padding: var(--space-4);
  border: 1px solid #b9c2d3;
  border-radius: var(--radius-sm);
  color: var(--color-text);
  background: var(--color-surface);
  line-height: 1.6;
}

textarea:focus {
  border-color: var(--color-primary);
}

.form-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
}

.form-footer span {
  color: var(--color-muted);
  font-size: 0.9rem;
}

button {
  min-width: 136px;
  padding: 12px 22px;
  border: 0;
  border-radius: var(--radius-sm);
  color: white;
  background: var(--color-primary);
  font-weight: 700;
}

button:hover:not(:disabled) {
  background: var(--color-primary-dark);
}

button:disabled {
  opacity: 0.65;
}

[role="alert"] {
  margin: 0;
  color: var(--color-danger);
}

@media (max-width: 560px) {
  .form-footer {
    align-items: stretch;
    flex-direction: column;
  }

  button {
    width: 100%;
  }
}
</style>
```

- [ ] **Step 4: Run focused and complete tests**

Run:

```powershell
npm.cmd run test -- src/components/SearchForm.test.ts
npm.cmd run test
npm.cmd run type-check
```

Expected: 5 form tests pass; all frontend tests pass; type checking exits 0.

- [ ] **Step 5: Commit only the search form**

Run from `course-rag/`:

```powershell
git add -- frontend/src/components/SearchForm.vue frontend/src/components/SearchForm.test.ts
git commit --only -m "feat: add accessible knowledge search form" -- frontend/src/components/SearchForm.vue frontend/src/components/SearchForm.test.ts
```

---

### Task 5: Ranked Result Presentation

**Files:**
- Create: `frontend/src/components/SearchResultCard.vue`
- Create: `frontend/src/components/SearchResultCard.test.ts`
- Create: `frontend/src/components/SearchResults.vue`
- Create: `frontend/src/components/SearchResults.test.ts`

**Interfaces:**
- Consumes: one `SearchResult` for a card and one `SearchState` for the list.
- Produces: ranked result cards and exactly three loading skeletons.

- [ ] **Step 1: Write failing result-component tests**

Create `frontend/src/components/SearchResultCard.test.ts`:

```ts
import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import SearchResultCard from "./SearchResultCard.vue"

describe("SearchResultCard", () => {
  it("renders rank, four-decimal score, chunk index, and source text", () => {
    const wrapper = mount(SearchResultCard, {
      props: {
        result: {
          rank: 1,
          score: 0.8,
          chunk_index: 2,
          text: "service 层负责业务逻辑。",
        },
      },
    })

    expect(wrapper.text()).toContain("TOP 1")
    expect(wrapper.text()).toContain("相似度 0.8000")
    expect(wrapper.text()).toContain("Chunk #2")
    expect(wrapper.text()).toContain("service 层负责业务逻辑。")
  })
})
```

Create `frontend/src/components/SearchResults.test.ts`:

```ts
import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import type { SearchState } from "../types/search"
import SearchResults from "./SearchResults.vue"

const loadingState: SearchState = {
  status: "loading",
  query: "业务逻辑",
}

const successState: SearchState = {
  status: "success",
  response: {
    query: "业务逻辑",
    elapsed_ms: 420,
    indexed_chunks: 8,
    model: "paraphrase-multilingual-MiniLM-L12-v2",
    results: [
      {
        rank: 1,
        score: 0.8421,
        chunk_index: 2,
        text: "service 层负责业务逻辑。",
      },
      {
        rank: 2,
        score: 0.7168,
        chunk_index: 1,
        text: "router 层接收 HTTP 请求。",
      },
    ],
  },
}

describe("SearchResults", () => {
  it("renders exactly three skeletons while loading", () => {
    const wrapper = mount(SearchResults, {
      props: { state: loadingState },
    })

    expect(wrapper.findAll(".result-skeleton")).toHaveLength(3)
  })

  it("renders one card per successful result", () => {
    const wrapper = mount(SearchResults, {
      props: { state: successState },
    })

    expect(wrapper.findAll("article")).toHaveLength(2)
    expect(wrapper.text()).toContain("TOP 2")
  })

  it("renders no result region for idle state", () => {
    const wrapper = mount(SearchResults, {
      props: { state: { status: "idle" } },
    })

    expect(wrapper.find("section").exists()).toBe(false)
  })
})
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
npm.cmd run test -- src/components/SearchResultCard.test.ts src/components/SearchResults.test.ts
```

Expected: FAIL because both Vue components are missing.

- [ ] **Step 3: Implement the result card**

Create `frontend/src/components/SearchResultCard.vue`:

```vue
<script setup lang="ts">
import type { SearchResult } from "../types/search"

defineProps<{
  result: SearchResult
}>()
</script>

<template>
  <article class="result-card">
    <div class="rank" aria-hidden="true">{{ result.rank }}</div>
    <div class="result-content">
      <div class="metadata">
        <span class="top-label">TOP {{ result.rank }}</span>
        <span class="score">相似度 {{ result.score.toFixed(4) }}</span>
        <span class="chunk">Chunk #{{ result.chunk_index }}</span>
      </div>
      <p>{{ result.text }}</p>
    </div>
  </article>
</template>

<style scoped>
.result-card {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr);
  gap: var(--space-4);
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  box-shadow: 0 4px 16px rgb(23 33 61 / 6%);
}

.rank {
  display: grid;
  width: 36px;
  height: 36px;
  place-items: center;
  border-radius: 999px;
  color: white;
  background: var(--color-primary);
  font-weight: 800;
}

.result-content {
  min-width: 0;
}

.metadata {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.metadata span {
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.88rem;
}

.top-label {
  color: var(--color-primary);
  background: #eef3ff;
}

.score {
  color: var(--color-warning);
  background: var(--color-warning-soft);
}

.chunk {
  color: var(--color-muted);
  background: #f1f3f7;
}

p {
  margin: var(--space-3) 0 0;
  line-height: 1.65;
}
</style>
```

- [ ] **Step 4: Implement the results list and reduced-motion skeletons**

Create `frontend/src/components/SearchResults.vue`:

```vue
<script setup lang="ts">
import type { SearchState } from "../types/search"
import SearchResultCard from "./SearchResultCard.vue"

defineProps<{
  state: SearchState
}>()
</script>

<template>
  <section
    v-if="state.status === 'loading' || state.status === 'success'"
    class="results"
    aria-label="语义检索结果"
  >
    <template v-if="state.status === 'loading'">
      <div
        v-for="position in 3"
        :key="position"
        class="result-skeleton"
        aria-hidden="true"
      />
    </template>
    <template v-else-if="state.status === 'success'">
      <SearchResultCard
        v-for="result in state.response.results"
        :key="result.rank"
        :result="result"
      />
    </template>
  </section>
</template>

<style scoped>
.results {
  display: grid;
  gap: var(--space-3);
}

.result-skeleton {
  min-height: 128px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background:
    linear-gradient(
      90deg,
      rgb(240 242 247) 25%,
      rgb(248 249 252) 50%,
      rgb(240 242 247) 75%
    );
  background-size: 200% 100%;
  animation: shimmer 1.2s infinite linear;
}

@keyframes shimmer {
  to {
    background-position-x: -200%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .result-skeleton {
    animation: none;
  }
}
</style>
```

- [ ] **Step 5: Run focused and complete tests**

Run:

```powershell
npm.cmd run test -- src/components/SearchResultCard.test.ts src/components/SearchResults.test.ts
npm.cmd run test
npm.cmd run type-check
```

Expected: 4 result-component tests pass; all frontend tests pass; type checking exits 0.

- [ ] **Step 6: Commit only the result presentation**

Run from `course-rag/`:

```powershell
git add -- frontend/src/components/SearchResultCard.vue frontend/src/components/SearchResultCard.test.ts frontend/src/components/SearchResults.vue frontend/src/components/SearchResults.test.ts
git commit --only -m "feat: render ranked semantic search results" -- frontend/src/components/SearchResultCard.vue frontend/src/components/SearchResultCard.test.ts frontend/src/components/SearchResults.vue frontend/src/components/SearchResults.test.ts
```

---

### Task 6: Status and Educational Panels

**Files:**
- Create: `frontend/src/components/SearchStatus.vue`
- Create: `frontend/src/components/SearchStatus.test.ts`
- Create: `frontend/src/components/AppSidebar.vue`
- Create: `frontend/src/components/IndexSummary.vue`
- Create: `frontend/src/components/RetrievalExplanation.vue`
- Create: `frontend/src/components/InformationPanels.test.ts`

**Interfaces:**
- Consumes: `SearchState`; index props `indexedChunks`, `topK`, and `model`.
- Produces: live status text, a `retry` event, static navigation labels, current-index information, and a three-step retrieval explanation.

- [ ] **Step 1: Write failing status and panel tests**

Create `frontend/src/components/SearchStatus.test.ts`:

```ts
import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import type { SearchState } from "../types/search"
import SearchStatus from "./SearchStatus.vue"

describe("SearchStatus", () => {
  it("announces loading", () => {
    const state: SearchState = { status: "loading", query: "业务逻辑" }
    const wrapper = mount(SearchStatus, { props: { state } })

    expect(wrapper.get('[role="status"]').text()).toContain("正在检索")
  })

  it("shows result count and elapsed time for success", () => {
    const state: SearchState = {
      status: "success",
      response: {
        query: "业务逻辑",
        elapsed_ms: 420,
        indexed_chunks: 8,
        model: "MiniLM",
        results: [
          { rank: 1, score: 0.8, chunk_index: 2, text: "service 层" },
        ],
      },
    }
    const wrapper = mount(SearchStatus, { props: { state } })

    expect(wrapper.text()).toContain("找到 1 条相关内容")
    expect(wrapper.text()).toContain("0.42 秒")
  })

  it("shows the empty-state guidance", () => {
    const state: SearchState = {
      status: "empty",
      response: {
        query: "午饭",
        elapsed_ms: 420,
        indexed_chunks: 8,
        model: "MiniLM",
        results: [],
      },
    }
    const wrapper = mount(SearchStatus, { props: { state } })

    expect(wrapper.text()).toContain("请换一种问法")
  })

  it("announces errors and emits retry", async () => {
    const state: SearchState = {
      status: "error",
      query: "业务逻辑",
      message: "服务暂时不可用",
    }
    const wrapper = mount(SearchStatus, { props: { state } })

    expect(wrapper.get('[role="alert"]').text()).toContain("服务暂时不可用")
    await wrapper.get("button").trigger("click")
    expect(wrapper.emitted("retry")).toHaveLength(1)
  })
})
```

Create `frontend/src/components/InformationPanels.test.ts`:

```ts
import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import AppSidebar from "./AppSidebar.vue"
import IndexSummary from "./IndexSummary.vue"
import RetrievalExplanation from "./RetrievalExplanation.vue"

describe("information panels", () => {
  it("shows the active search area and disabled placeholders", () => {
    const wrapper = mount(AppSidebar)

    expect(wrapper.get('[aria-current="page"]').text()).toBe("语义检索")
    expect(wrapper.findAll('[aria-disabled="true"]')).toHaveLength(2)
    expect(wrapper.text()).toContain("本地知识库 · 已就绪")
  })

  it("shows current index metadata", () => {
    const wrapper = mount(IndexSummary, {
      props: {
        indexedChunks: 8,
        topK: 3,
        model: "paraphrase-multilingual-MiniLM-L12-v2",
      },
    })

    expect(wrapper.text()).toContain("8 个文本块")
    expect(wrapper.text()).toContain("Top K · 3")
    expect(wrapper.text()).toContain("MiniLM-L12-v2")
  })

  it("explains the three retrieval steps", () => {
    const wrapper = mount(RetrievalExplanation)

    expect(wrapper.findAll("li")).toHaveLength(3)
    expect(wrapper.text()).toContain("计算余弦相似度")
  })
})
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
npm.cmd run test -- src/components/SearchStatus.test.ts src/components/InformationPanels.test.ts
```

Expected: FAIL because the four Vue components do not exist.

- [ ] **Step 3: Implement the five-state status component**

Create `frontend/src/components/SearchStatus.vue`:

```vue
<script setup lang="ts">
import type { SearchState } from "../types/search"

defineProps<{
  state: SearchState
}>()

const emit = defineEmits<{
  retry: []
}>()
</script>

<template>
  <div
    v-if="state.status === 'loading'"
    class="status status-loading"
    role="status"
    aria-live="polite"
  >
    正在检索“{{ state.query }}”…
  </div>

  <div
    v-else-if="state.status === 'success'"
    class="status status-success"
    role="status"
    aria-live="polite"
  >
    <strong>找到 {{ state.response.results.length }} 条相关内容</strong>
    <span>检索耗时 {{ (state.response.elapsed_ms / 1000).toFixed(2) }} 秒</span>
  </div>

  <div
    v-else-if="state.status === 'empty'"
    class="status status-empty"
    role="status"
    aria-live="polite"
  >
    没有找到相关内容，请换一种问法。
  </div>

  <div
    v-else-if="state.status === 'error'"
    class="status status-error"
    role="alert"
  >
    <span>{{ state.message }}</span>
    <button type="button" @click="emit('retry')">重新检索</button>
  </div>
</template>

<style scoped>
.status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  min-height: 58px;
  padding: var(--space-4) var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}

.status-loading,
.status-empty {
  color: var(--color-muted);
  background: var(--color-surface);
}

.status-success {
  color: var(--color-success);
  border-color: #8dd5c0;
  background: var(--color-success-soft);
}

.status-error {
  color: var(--color-danger);
  border-color: #f2b8b5;
  background: var(--color-danger-soft);
}

button {
  padding: 8px 12px;
  border: 1px solid currentcolor;
  border-radius: var(--radius-sm);
  color: inherit;
  background: transparent;
  font-weight: 700;
}

@media (max-width: 560px) {
  .status {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
```

- [ ] **Step 4: Implement the sidebar**

Create `frontend/src/components/AppSidebar.vue`:

```vue
<template>
  <aside class="sidebar">
    <div class="brand">
      <svg viewBox="0 0 48 48" aria-hidden="true">
        <path d="M7 9h14c5 0 8 3 8 8v18c-2-3-5-4-9-4H7V9Z" />
        <circle cx="34" cy="31" r="7" />
        <path d="m39 36 5 5" />
      </svg>
      <strong>Course RAG</strong>
    </div>

    <nav aria-label="主导航">
      <span class="nav-item active" aria-current="page">语义检索</span>
      <span class="nav-item" aria-disabled="true">知识资料</span>
      <span class="nav-item" aria-disabled="true">学习说明</span>
    </nav>

    <p class="ready"><span aria-hidden="true" />本地知识库 · 已就绪</p>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  min-height: 100vh;
  padding: var(--space-8) var(--space-4);
  border-right: 1px solid var(--color-border);
  background: var(--color-surface);
  flex-direction: column;
}

.brand {
  display: grid;
  justify-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-8);
  color: var(--color-text);
  font-size: 1.2rem;
}

.brand svg {
  width: 52px;
  fill: none;
  stroke: var(--color-primary);
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 3;
}

nav {
  display: grid;
  gap: var(--space-2);
}

.nav-item {
  padding: 13px 15px;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  color: var(--color-muted);
}

.nav-item.active {
  color: var(--color-primary);
  border-color: #b9c7f0;
  background: #f5f7ff;
  font-weight: 700;
}

.nav-item[aria-disabled="true"] {
  opacity: 0.72;
}

.ready {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: auto 0 0;
  color: var(--color-muted);
  font-size: 0.85rem;
}

.ready span {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #2fb277;
}

@media (max-width: 900px) {
  .sidebar {
    min-height: auto;
    padding: var(--space-4);
    border-right: 0;
    border-bottom: 1px solid var(--color-border);
  }

  .brand {
    display: flex;
    margin-bottom: var(--space-3);
  }

  .brand svg {
    width: 36px;
  }

  nav {
    display: flex;
    flex-wrap: wrap;
  }

  .ready {
    margin-top: var(--space-3);
  }
}
</style>
```

- [ ] **Step 5: Implement the index summary and retrieval explanation**

Create `frontend/src/components/IndexSummary.vue`:

```vue
<script setup lang="ts">
defineProps<{
  indexedChunks: number
  topK: number
  model: string
}>()

function shortModelName(model: string): string {
  return model.replace("paraphrase-multilingual-", "")
}
</script>

<template>
  <section class="panel" aria-labelledby="index-heading">
    <h2 id="index-heading">当前索引</h2>
    <dl>
      <div>
        <dt>文本块</dt>
        <dd>{{ indexedChunks }} 个文本块</dd>
      </div>
      <div>
        <dt>返回数量</dt>
        <dd>Top K · {{ topK }}</dd>
      </div>
      <div>
        <dt>向量模型</dt>
        <dd>{{ shortModelName(model) }}</dd>
      </div>
    </dl>
  </section>
</template>

<style scoped>
.panel {
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
}

h2 {
  margin: 0 0 var(--space-4);
  font-size: 1.2rem;
}

dl {
  display: grid;
  gap: var(--space-3);
  margin: 0;
}

dl div {
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface-soft);
}

dt {
  color: var(--color-muted);
  font-size: 0.8rem;
}

dd {
  margin: var(--space-1) 0 0;
  overflow-wrap: anywhere;
  font-weight: 700;
}
</style>
```

Create `frontend/src/components/RetrievalExplanation.vue`:

```vue
<template>
  <section class="panel" aria-labelledby="explanation-heading">
    <h2 id="explanation-heading">这一步发生了什么？</h2>
    <ol>
      <li><span>1</span>问题转为向量</li>
      <li><span>2</span>计算余弦相似度</li>
      <li><span>3</span>返回 Top 3 原文</li>
    </ol>
  </section>
</template>

<style scoped>
.panel {
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
}

h2 {
  margin: 0 0 var(--space-5);
  color: var(--color-success);
  font-size: 1.05rem;
}

ol {
  display: grid;
  gap: var(--space-5);
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  font-weight: 650;
}

li span {
  display: grid;
  width: 30px;
  height: 30px;
  place-items: center;
  border-radius: 999px;
  color: white;
  background: var(--color-success);
}
</style>
```

- [ ] **Step 6: Run focused and complete tests**

Run:

```powershell
npm.cmd run test -- src/components/SearchStatus.test.ts src/components/InformationPanels.test.ts
npm.cmd run test
npm.cmd run type-check
```

Expected: 7 status/panel tests pass; all frontend tests pass; type checking exits 0.

- [ ] **Step 7: Commit only the status and information panels**

Run from `course-rag/`:

```powershell
git add -- frontend/src/components/SearchStatus.vue frontend/src/components/SearchStatus.test.ts frontend/src/components/AppSidebar.vue frontend/src/components/IndexSummary.vue frontend/src/components/RetrievalExplanation.vue frontend/src/components/InformationPanels.test.ts
git commit --only -m "feat: add retrieval status and learning panels" -- frontend/src/components/SearchStatus.vue frontend/src/components/SearchStatus.test.ts frontend/src/components/AppSidebar.vue frontend/src/components/IndexSummary.vue frontend/src/components/RetrievalExplanation.vue frontend/src/components/InformationPanels.test.ts
```

---

### Task 7: Application Composition and Responsive Shell

**Files:**
- Modify: `frontend/src/App.vue`
- Create: `frontend/src/App.test.ts`
- Modify: `frontend/src/main.ts`
- Modify: `frontend/index.html`

**Interfaces:**
- Consumes: `searchServiceKey`, `mockSearchService`, `useKnowledgeSearch`, all page components, and `DEFAULT_INDEX_METADATA`.
- Produces: the complete single-page application and the dependency-injection seam for a future FastAPI service.

- [ ] **Step 1: Write failing application integration tests**

Create `frontend/src/App.test.ts`:

```ts
import { flushPromises, mount } from "@vue/test-utils"
import { describe, expect, it, vi } from "vitest"

import App from "./App.vue"
import { searchServiceKey } from "./services/searchService"
import type { SearchResponse } from "./types/search"

const response: SearchResponse = {
  query: "业务逻辑应该写在哪一层？",
  elapsed_ms: 420,
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  results: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "service 层负责业务逻辑。",
    },
    {
      rank: 2,
      score: 0.7168,
      chunk_index: 1,
      text: "router 层接收 HTTP 请求。",
    },
    {
      rank: 3,
      score: 0.6234,
      chunk_index: 3,
      text: "repository 层负责数据访问。",
    },
  ],
}

describe("App", () => {
  it("submits a query, shows loading, and renders three results", async () => {
    let resolveResponse!: (value: SearchResponse) => void
    const service = {
      search: vi.fn(
        () =>
          new Promise<SearchResponse>((resolve) => {
            resolveResponse = resolve
          }),
      ),
    }
    const wrapper = mount(App, {
      global: {
        provide: {
          [searchServiceKey as symbol]: service,
        },
      },
    })

    await wrapper.get("textarea").setValue("业务逻辑应该写在哪一层？")
    await wrapper.get("form").trigger("submit")

    expect(wrapper.text()).toContain("正在检索")
    expect(service.search).toHaveBeenCalledWith({
      query: "业务逻辑应该写在哪一层？",
      top_k: 3,
    })

    resolveResponse(response)
    await flushPromises()

    expect(wrapper.findAll("article")).toHaveLength(3)
    expect(wrapper.text()).toContain("找到 3 条相关内容")
    expect(wrapper.text()).not.toContain("AI 生成答案")
  })

  it("renders the empty state without fake result cards", async () => {
    const service = {
      search: vi.fn().mockResolvedValue({
        ...response,
        query: "今天午饭吃什么？",
        results: [],
      }),
    }
    const wrapper = mount(App, {
      global: {
        provide: {
          [searchServiceKey as symbol]: service,
        },
      },
    })

    await wrapper.get("textarea").setValue("今天午饭吃什么？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(wrapper.text()).toContain("请换一种问法")
    expect(wrapper.findAll("article")).toHaveLength(0)
  })
})
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
npm.cmd run test -- src/App.test.ts
```

Expected: FAIL because the generated `App.vue` does not provide the specified page or injected search behavior.

- [ ] **Step 3: Compose the complete application**

Replace `frontend/src/App.vue` with:

```vue
<script setup lang="ts">
import { computed, inject } from "vue"

import AppSidebar from "./components/AppSidebar.vue"
import IndexSummary from "./components/IndexSummary.vue"
import RetrievalExplanation from "./components/RetrievalExplanation.vue"
import SearchForm from "./components/SearchForm.vue"
import SearchResults from "./components/SearchResults.vue"
import SearchStatus from "./components/SearchStatus.vue"
import { useKnowledgeSearch } from "./composables/useKnowledgeSearch"
import { DEFAULT_INDEX_METADATA } from "./mocks/searchResponses"
import { searchServiceKey } from "./services/searchService"

const searchService = inject(searchServiceKey)
if (!searchService) {
  throw new Error("SearchService 未注入")
}

const { state, isLoading, search, retry } =
  useKnowledgeSearch(searchService)

const responseMetadata = computed(() => {
  if (state.value.status === "success" || state.value.status === "empty") {
    return {
      indexed_chunks: state.value.response.indexed_chunks,
      model: state.value.response.model,
      top_k: DEFAULT_INDEX_METADATA.top_k,
    }
  }
  return DEFAULT_INDEX_METADATA
})
</script>

<template>
  <div class="app-shell">
    <AppSidebar />

    <main>
      <header class="page-header">
        <p class="eyebrow">SEMANTIC RETRIEVAL</p>
        <h1>课程知识检索</h1>
        <p>输入问题，查看语义最相关的课程原文。</p>
      </header>

      <SearchForm :loading="isLoading" @submit="search" />
      <SearchStatus :state="state" @retry="retry" />
      <SearchResults :state="state" />
    </main>

    <aside class="context-panel" aria-label="索引与检索说明">
      <IndexSummary
        :indexed-chunks="responseMetadata.indexed_chunks"
        :top-k="responseMetadata.top_k"
        :model="responseMetadata.model"
      />
      <RetrievalExplanation />
    </aside>
  </div>
</template>

<style scoped>
.app-shell {
  display: grid;
  min-height: 100vh;
  grid-template-columns: 220px minmax(0, 1fr) 300px;
}

main {
  display: grid;
  align-content: start;
  gap: var(--space-5);
  min-width: 0;
  padding: 56px clamp(24px, 4vw, 64px);
}

.page-header {
  margin-bottom: var(--space-2);
}

.page-header h1 {
  margin: var(--space-1) 0 var(--space-2);
  font-size: clamp(2rem, 4vw, 3rem);
  line-height: 1.15;
}

.page-header p {
  margin: 0;
  color: var(--color-muted);
}

.page-header .eyebrow {
  color: var(--color-primary);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.12em;
}

.context-panel {
  display: grid;
  align-content: start;
  gap: var(--space-5);
  min-width: 0;
  padding: 56px var(--space-6) 56px 0;
}

@media (max-width: 1180px) {
  .app-shell {
    grid-template-columns: 200px minmax(0, 1fr) 260px;
  }

  main {
    padding-inline: var(--space-6);
  }
}

@media (max-width: 900px) {
  .app-shell {
    grid-template-columns: minmax(0, 1fr);
  }

  main {
    padding: var(--space-8) var(--space-5);
  }

  .context-panel {
    padding: 0 var(--space-5) var(--space-8);
  }
}
</style>
```

- [ ] **Step 4: Provide the mock service at the application entry point**

Replace `frontend/src/main.ts` with:

```ts
import { createApp } from "vue"

import App from "./App.vue"
import { mockSearchService } from "./services/mockSearchService"
import { searchServiceKey } from "./services/searchService"
import "./styles/tokens.css"
import "./styles/global.css"

createApp(App)
  .provide(searchServiceKey, mockSearchService)
  .mount("#app")
```

In `frontend/index.html`, replace the generated title with:

```html
<title>Course RAG - 课程知识检索</title>
```

- [ ] **Step 5: Run integration tests and all automated checks**

Run:

```powershell
npm.cmd run test -- src/App.test.ts
npm.cmd run test
npm.cmd run type-check
npm.cmd run build
```

Expected: 2 application tests pass; the complete suite reports 31 passing tests and zero failures; type checking and production build exit 0.

- [ ] **Step 6: Commit only the application composition**

Run from `course-rag/`:

```powershell
git add -- frontend/src/App.vue frontend/src/App.test.ts frontend/src/main.ts frontend/index.html
git commit --only -m "feat: compose responsive Course RAG search page" -- frontend/src/App.vue frontend/src/App.test.ts frontend/src/main.ts frontend/index.html
```

---

### Task 8: Learning Guide and End-to-End Verification

**Files:**
- Modify: `frontend/README.md`
- Modify only if verification exposes a defect: the specific frontend source file and its colocated test.

**Interfaces:**
- Consumes: the completed Vue application and all npm scripts.
- Produces: beginner setup documentation and fresh evidence that frontend tests, types, build, responsive layout, accessibility, and the existing backend suite remain valid.

- [ ] **Step 1: Replace the scaffold README with the exact learning guide**

Replace `frontend/README.md` with:

````markdown
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
````

- [ ] **Step 2: Run the fresh complete frontend verification**

Run from `course-rag/frontend/`:

```powershell
npm.cmd run test
npm.cmd run type-check
npm.cmd run build
```

Expected: 31 tests pass with zero failures; type checking exits 0; production build exits 0 and writes `frontend/dist/`.

- [ ] **Step 3: Verify the existing backend suite was not regressed**

Run from `course-rag/`:

```powershell
python -m pytest -v
```

Expected baseline: 33 backend tests pass with zero failures and no model download.

- [ ] **Step 4: Run the app and perform the visual/responsive checks**

Run from `course-rag/frontend/`:

```powershell
npm.cmd run dev -- --host 127.0.0.1
```

Expected: Vite prints a local URL and the page loads without console errors. Check the page at these viewport widths:

- 1440px: left navigation, main search content, and right information panel form three columns.
- 900px: layout switches to one column and the sidebar becomes a compact top section.
- 390px: form, metadata badges, results, and information panels remain readable; the search button uses full width.

Submit the four README questions and verify:

- Business logic returns service first.
- Database access returns repository first.
- Vector conversion returns Embedding first.
- The unrelated lunch question shows the empty-state guidance.
- Loading disables the form and shows three skeletons.
- Results display four-decimal scores and Chunk numbers.
- No chat bubble or generated answer appears.

- [ ] **Step 5: Perform keyboard and accessibility checks**

With the development server still running:

- Tab through the page and verify every interactive element has a visible focus ring.
- Press Enter in the textarea and verify it submits.
- Press Shift + Enter and verify it inserts a line break without submitting.
- Submit an empty value and verify the inline `role="alert"` message appears.
- In browser accessibility tools, verify the input has the “向知识库提问” label and the status region announces loading/success.
- Enable reduced motion and verify skeleton shimmer stops.

Expected: every check passes without mouse-only interaction or color-only meaning.

- [ ] **Step 6: Check repository scope and whitespace**

Stop the development server with Ctrl+C, then run from `course-rag/`:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` prints nothing. Status lists the intended frontend README change plus the repository's pre-existing staged backend files; no `node_modules/`, `dist/`, or `coverage/` paths appear.

- [ ] **Step 7: Commit only the frontend learning guide**

Run:

```powershell
git add -- frontend/README.md
git commit --only -m "docs: add Vue frontend learning guide" -- frontend/README.md
```

- [ ] **Step 8: Record final evidence**

Run:

```powershell
git log --oneline -8
git status --short --branch
```

Expected: the frontend task commits appear in order; the branch contains no uncommitted frontend work; the unrelated staged backend files remain exactly as they were before implementation.

---

## Final Deliverable Checklist

- [ ] Vue 3 + Vite + TypeScript application exists under `frontend/`.
- [ ] Search contract uses the exact `snake_case` fields from the design.
- [ ] Mock service supports three relevant question families and one empty-result path.
- [ ] State machine covers idle, loading, success, empty, and error.
- [ ] Form supports local validation, Enter submit, Shift + Enter newline, and loading disablement.
- [ ] Ranked cards show four-decimal scores, Chunk numbers, and original text.
- [ ] Sidebar placeholders are visibly non-interactive.
- [ ] Live status, labels, semantic landmarks, focus states, and reduced motion are present.
- [ ] Three-column desktop layout becomes one column at 900px.
- [ ] All 31 frontend tests, type checking, production build, and 33 backend tests pass.
- [ ] No real backend call, AI answer generation, router, global store, or UI component framework was added.
- [ ] Every commit is path-limited so pre-existing staged backend files are not included.

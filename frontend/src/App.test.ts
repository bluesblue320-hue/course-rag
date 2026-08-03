import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { describe, expect, it, vi } from "vitest"

import App from "./App.vue"
import type { AskService } from "./services/askService"
import { askServiceKey } from "./services/askService"
import { AskServiceError } from "./services/httpAskService"
import type { SearchService } from "./services/searchService"
import { searchServiceKey } from "./services/searchService"
import type { AskResponse } from "./types/ask"
import type { SearchResponse } from "./types/search"

const askResponse: AskResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责核心业务逻辑。[来源1]",
  retrieval_elapsed_ms: 12.4,
  generation_elapsed_ms: 680.7,
  total_elapsed_ms: 693.1,
  embedding_model: "embedding-model",
  llm_model: "llm-model",
  sources: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "Service 层负责核心业务逻辑。",
    },
  ],
}

const searchResponse: SearchResponse = {
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

function mountApp(
  askService: AskService = { ask: vi.fn().mockResolvedValue(askResponse) },
  searchService: SearchService = {
    search: vi.fn().mockResolvedValue(searchResponse),
  },
): VueWrapper {
  return mount(App, {
    global: {
      provide: {
        [askServiceKey as symbol]: askService,
        [searchServiceKey as symbol]: searchService,
      },
    },
  })
}

function modeButtons(wrapper: VueWrapper) {
  return wrapper.findAll(".mode-switch button")
}

describe("App", () => {
  it("defaults to intelligent Q&A and submits a trimmed top-3 ask request", async () => {
    const askMethod = vi.fn().mockResolvedValue(askResponse)
    const searchMethod = vi.fn().mockResolvedValue(searchResponse)
    const wrapper = mountApp({ ask: askMethod }, { search: searchMethod })

    expect(wrapper.text()).toContain("课程知识问答")
    expect(wrapper.text()).toContain("RAG QUESTION ANSWERING")
    expect(modeButtons(wrapper)[0].attributes("aria-pressed")).toBe("true")

    await wrapper.get("textarea").setValue("  Service 层负责什么？  ")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(askMethod).toHaveBeenCalledWith({
      question: "Service 层负责什么？",
      top_k: 3,
    })
    expect(searchMethod).not.toHaveBeenCalled()
  })

  it("shows ask loading, blocks switching, then renders answer details", async () => {
    let resolveAsk!: (value: AskResponse) => void
    const askMethod = vi.fn(
      () =>
        new Promise<AskResponse>((resolve) => {
          resolveAsk = resolve
        }),
    )
    const wrapper = mountApp({ ask: askMethod })

    await wrapper.get("textarea").setValue("Service 层负责什么？")
    await wrapper.get("form").trigger("submit")

    expect(wrapper.text()).toContain("正在检索课程资料并生成回答")
    expect(wrapper.get("textarea").attributes("disabled")).toBeDefined()
    expect(modeButtons(wrapper).every((button) => button.attributes("disabled") !== undefined)).toBe(
      true,
    )
    await modeButtons(wrapper)[1].trigger("click")
    expect(wrapper.text()).toContain("课程知识问答")

    resolveAsk(askResponse)
    await flushPromises()

    expect(wrapper.text()).toContain(askResponse.answer)
    expect(wrapper.text()).toContain("引用来源（1）")
    expect(wrapper.text()).toContain("Service 层负责核心业务逻辑。")
    expect(wrapper.text()).toContain("embedding-model")
    expect(wrapper.text()).toContain("llm-model")
  })

  it("switches to retrieval without requesting and keeps the shared input", async () => {
    const askMethod = vi.fn().mockResolvedValue(askResponse)
    const searchMethod = vi.fn().mockResolvedValue(searchResponse)
    const wrapper = mountApp({ ask: askMethod }, { search: searchMethod })

    await wrapper.get("textarea").setValue("业务逻辑应该写在哪一层？")
    await modeButtons(wrapper)[1].trigger("click")

    expect(wrapper.text()).toContain("课程知识检索")
    expect(wrapper.get("textarea").element.value).toBe("业务逻辑应该写在哪一层？")
    expect(askMethod).not.toHaveBeenCalled()
    expect(searchMethod).not.toHaveBeenCalled()

    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(searchMethod).toHaveBeenCalledWith({
      query: "业务逻辑应该写在哪一层？",
      top_k: 3,
    })
    expect(wrapper.findAll(".result-card")).toHaveLength(3)
    expect(wrapper.text()).toContain("找到 3 条相关内容")
  })

  it("preserves independent ask and search results across mode switches", async () => {
    const wrapper = mountApp()
    await wrapper.get("textarea").setValue("Service 层负责什么？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()
    expect(wrapper.text()).toContain(askResponse.answer)

    await modeButtons(wrapper)[1].trigger("click")
    await wrapper.get("textarea").setValue("业务逻辑应该写在哪一层？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()
    expect(wrapper.findAll(".result-card")).toHaveLength(3)
    expect(wrapper.text()).not.toContain(askResponse.answer)

    await modeButtons(wrapper)[0].trigger("click")
    expect(wrapper.text()).toContain(askResponse.answer)
    expect(wrapper.findAll(".result-card")).toHaveLength(0)
  })

  it("shows the 503 guidance and allows switching to retrieval", async () => {
    const askMethod = vi.fn().mockRejectedValue(
      new AskServiceError(
        "问答服务尚未完成配置，你仍可使用语义检索",
        "LLM_NOT_CONFIGURED",
        503,
      ),
    )
    const wrapper = mountApp({ ask: askMethod })

    await wrapper.get("textarea").setValue("问题")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(wrapper.get('[role="alert"]').text()).toContain(
      "仍可以切换到“语义检索”查看相关原文",
    )
    await modeButtons(wrapper)[1].trigger("click")
    expect(wrapper.text()).toContain("课程知识检索")
  })

  it("renders the retrieval empty state without fake result cards", async () => {
    const searchService = {
      search: vi.fn().mockResolvedValue({
        ...searchResponse,
        query: "今天午饭吃什么？",
        results: [],
      }),
    }
    const wrapper = mountApp(undefined, searchService)

    await modeButtons(wrapper)[1].trigger("click")
    await wrapper.get("textarea").setValue("今天午饭吃什么？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(wrapper.text()).toContain("请换一种问法")
    expect(wrapper.findAll(".result-card")).toHaveLength(0)
  })
})

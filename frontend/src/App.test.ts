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

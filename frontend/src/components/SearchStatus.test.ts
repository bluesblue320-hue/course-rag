import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import type { SearchState } from "../types/search"
import SearchStatus from "./SearchStatus.vue"

describe("SearchStatus", () => {
  it("renders no status panel while idle", () => {
    const state: SearchState = { status: "idle" }
    const wrapper = mount(SearchStatus, { props: { state } })

    expect(wrapper.find('[role="status"]').exists()).toBe(false)
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect(wrapper.find("div").exists()).toBe(false)
  })

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
          {
            rank: 1,
            score: 0.8,
            chunk_index: 2,
            text: "service 层",
            document_id: "builtin-knowledge",
            filename: "knowledge.txt",
            page_number: null,
          },
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

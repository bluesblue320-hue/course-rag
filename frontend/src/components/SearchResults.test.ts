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

  it("renders the result count and response metadata", () => {
    const wrapper = mount(SearchResults, {
      props: { state: successState },
    })

    expect(wrapper.text()).toContain("找到 2 条相关结果")
    expect(wrapper.text()).toContain("420 ms")
    expect(wrapper.text()).toContain("paraphrase-multilingual-MiniLM-L12-v2")
    expect(wrapper.text()).toContain("8")
  })

  it("renders an empty-result message", () => {
    const wrapper = mount(SearchResults, {
      props: {
        state: {
          status: "empty",
          response: {
            ...successState.response,
            results: [],
          },
        },
      },
    })

    expect(wrapper.find("section").exists()).toBe(true)
    expect(wrapper.text()).toContain("未找到相关结果")
  })

  it("keeps empty-result guidance out of a live region", () => {
    const wrapper = mount(SearchResults, {
      props: {
        state: {
          status: "empty",
          response: {
            ...successState.response,
            results: [],
          },
        },
      },
    })

    expect(wrapper.find(".empty-message").attributes("role")).toBeUndefined()
  })

  it("renders no result region for idle state", () => {
    const wrapper = mount(SearchResults, {
      props: { state: { status: "idle" } },
    })

    expect(wrapper.find("section").exists()).toBe(false)
  })
})

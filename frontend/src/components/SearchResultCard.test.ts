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

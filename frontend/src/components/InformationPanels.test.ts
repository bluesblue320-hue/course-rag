import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import AppSidebar from "./AppSidebar.vue"
import IndexSummary from "./IndexSummary.vue"
import RetrievalExplanation from "./RetrievalExplanation.vue"

describe("information panels", () => {
  it("shows the active search area and disabled placeholders", () => {
    const wrapper = mount(AppSidebar)

    expect(wrapper.get('[aria-current="page"]').text()).toBe("知识问答")
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

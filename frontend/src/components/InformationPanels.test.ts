import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import AppSidebar from "./AppSidebar.vue"
import IndexSummary from "./IndexSummary.vue"
import RetrievalExplanation from "./RetrievalExplanation.vue"

describe("information panels", () => {
  it("renders three ready navigation actions and emits selected areas", async () => {
    const wrapper = mount(AppSidebar, {
      props: { modelValue: "ask" },
    })
    const buttons = wrapper.findAll("nav button")

    expect(buttons).toHaveLength(3)
    expect(wrapper.get('[aria-current="page"]').text()).toBe("知识问答")
    expect(wrapper.findAll("button:disabled")).toHaveLength(0)
    expect(wrapper.text()).toContain("本地知识库 · 已就绪")

    await buttons[1].trigger("click")
    await buttons[2].trigger("click")

    expect(wrapper.emitted("update:modelValue")).toEqual([
      ["documents"],
      ["guide"],
    ])
  })

  it("marks the guide as current and disables navigation while busy", () => {
    const wrapper = mount(AppSidebar, {
      props: { modelValue: "guide", disabled: true },
    })

    expect(wrapper.get('[aria-current="page"]').text()).toBe("学习说明")
    expect(wrapper.findAll("nav button:disabled")).toHaveLength(3)
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

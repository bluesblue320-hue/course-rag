import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import type { AskSource } from "../types/ask"
import AnswerSources from "./AnswerSources.vue"

const sources: AskSource[] = [
  {
    rank: 2,
    score: 0.7,
    text: "原始顺序中的第一段",
    chunk_index: 8,
  },
  {
    rank: 1,
    score: 0.8421,
    text: "原始顺序中的第二段",
    chunk_index: 2,
  },
]

describe("AnswerSources", () => {
  it("shows the source count and preserves backend order", () => {
    const wrapper = mount(AnswerSources, { props: { sources } })
    const cards = wrapper.findAll("details")

    expect(wrapper.text()).toContain("引用来源（2）")
    expect(cards).toHaveLength(2)
    expect(cards[0].text()).toContain("原始顺序中的第一段")
    expect(cards[1].text()).toContain("原始顺序中的第二段")
  })

  it("shows source rank, four-decimal score, chunk index, and text", () => {
    const wrapper = mount(AnswerSources, { props: { sources } })

    expect(wrapper.text()).toContain("[来源1]")
    expect(wrapper.text()).toContain("相似度：0.8421")
    expect(wrapper.text()).toContain("相似度：0.7000")
    expect(wrapper.text()).toContain("Chunk #2")
    expect(wrapper.text()).toContain("原始顺序中的第二段")
    expect(wrapper.text()).not.toContain("页码")
  })

  it("shows a clear empty state", () => {
    const wrapper = mount(AnswerSources, { props: { sources: [] } })

    expect(wrapper.text()).toContain("引用来源（0）")
    expect(wrapper.text()).toContain("本次回答没有返回引用来源")
    expect(wrapper.findAll("details")).toHaveLength(0)
  })
})

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
          document_id: "builtin-knowledge",
          filename: "knowledge.txt",
          page_number: null,
        },
      },
    })

    expect(wrapper.text()).toContain("TOP 1")
    expect(wrapper.text()).toContain("相似度 0.8000")
    expect(wrapper.text()).toContain("Chunk #2")
    expect(wrapper.text()).toContain("knowledge.txt")
    expect(wrapper.text()).not.toContain("第 ")
    expect(wrapper.text()).toContain("service 层负责业务逻辑。")
  })

  it("renders the PDF page number when present", () => {
    const wrapper = mount(SearchResultCard, {
      props: {
        result: {
          rank: 1,
          score: 0.8,
          chunk_index: 0,
          text: "上传的讲义内容。",
          document_id: "doc-9",
          filename: "课程讲义.pdf",
          page_number: 12,
        },
      },
    })

    expect(wrapper.text()).toContain("课程讲义.pdf")
    expect(wrapper.text()).toContain("第 12 页")
  })

  it("renders the filename as plain text, never as a link", () => {
    const wrapper = mount(SearchResultCard, {
      props: {
        result: {
          rank: 1,
          score: 0.8,
          chunk_index: 0,
          text: "内容",
          document_id: "doc-9",
          filename: "<img src=x onerror=alert(1)>讲义.pdf",
          page_number: null,
        },
      },
    })

    expect(wrapper.findAll("a")).toHaveLength(0)
    expect(wrapper.findAll("img")).toHaveLength(0)
    expect(wrapper.text()).toContain("讲义.pdf")
  })
})

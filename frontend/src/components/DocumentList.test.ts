import { mount } from "@vue/test-utils"
import { describe, expect, it, vi } from "vitest"

import type { Document } from "../types/documents"
import DocumentList from "./DocumentList.vue"
import DocumentListItem from "./DocumentListItem.vue"

const builtin: Document = {
  document_id: "builtin-knowledge",
  filename: "knowledge.txt",
  content_type: "text/plain",
  size_bytes: 1800,
  text_length: 1800,
  chunk_count: 8,
  created_at: "2026-01-01T00:00:00Z",
  is_builtin: true,
}

const uploaded: Document = {
  document_id: "doc-1",
  filename: "课程讲义.pdf",
  content_type: "application/pdf",
  size_bytes: 2 * 1024 * 1024,
  text_length: 18000,
  chunk_count: 64,
  created_at: "2026-08-03T08:00:00Z",
  is_builtin: false,
}

describe("DocumentListItem", () => {
  it("shows the filename, type, size, chunk count, and upload time", () => {
    const wrapper = mount(DocumentListItem, {
      props: { document: uploaded },
    })

    expect(wrapper.text()).toContain("课程讲义.pdf")
    expect(wrapper.text()).toContain("application/pdf")
    expect(wrapper.text()).toContain("2.0 MB")
    expect(wrapper.text()).toContain("64 个 Chunk")
    expect(wrapper.text()).toContain("2026")
  })

  it("marks built-in documents and hides the delete button", () => {
    const wrapper = mount(DocumentListItem, {
      props: { document: builtin },
    })

    expect(wrapper.text()).toContain("内置")
    expect(wrapper.find("button").exists()).toBe(false)
  })

  it("emits delete with the document id", async () => {
    const wrapper = mount(DocumentListItem, {
      props: { document: uploaded },
    })

    await wrapper.get("button").trigger("click")

    expect(wrapper.emitted("delete")).toEqual([["doc-1"]])
  })

  it("disables the delete button while deleting", async () => {
    const wrapper = mount(DocumentListItem, {
      props: { document: uploaded, deleting: true },
    })

    expect(wrapper.text()).toContain("删除中…")
    expect(
      (wrapper.get("button").attributes("disabled") !== undefined),
    ).toBe(true)
  })

  it("renders the filename as plain text, never as a link", () => {
    const wrapper = mount(DocumentListItem, {
      props: {
        document: {
          ...uploaded,
          filename: "<img src=x onerror=alert(1)>讲义.pdf",
        },
      },
    })

    expect(wrapper.findAll("a")).toHaveLength(0)
    expect(wrapper.findAll("img")).toHaveLength(0)
    expect(wrapper.text()).toContain("讲义.pdf")
  })
})

describe("DocumentList", () => {
  it("shows the loading state", () => {
    const wrapper = mount(DocumentList, {
      props: {
        documents: [],
        loading: true,
        error: null,
        deletingDocumentId: null,
        deleteError: null,
      },
    })

    expect(wrapper.text()).toContain("正在加载文档列表")
  })

  it("shows an error with a retry button", async () => {
    const wrapper = mount(DocumentList, {
      props: {
        documents: [],
        loading: false,
        error: "无法连接后端服务",
        deletingDocumentId: null,
        deleteError: null,
      },
    })

    expect(wrapper.get('[role="alert"]').text()).toContain("无法连接后端服务")
    await wrapper.get("button").trigger("click")
    expect(wrapper.emitted("retry")).toHaveLength(1)
  })

  it("renders one item per document", () => {
    const wrapper = mount(DocumentList, {
      props: {
        documents: [builtin, uploaded],
        loading: false,
        error: null,
        deletingDocumentId: null,
        deleteError: null,
      },
    })

    expect(wrapper.findAll("li")).toHaveLength(2)
    expect(wrapper.text()).toContain("内置")
  })

  it("asks for confirmation before deleting a document", async () => {
    const confirmMethod = vi.fn().mockReturnValue(true)
    vi.stubGlobal("confirm", confirmMethod)
    const wrapper = mount(DocumentList, {
      props: {
        documents: [uploaded],
        loading: false,
        error: null,
        deletingDocumentId: null,
        deleteError: null,
      },
    })

    await wrapper.get("li button").trigger("click")

    expect(confirmMethod).toHaveBeenCalledWith(
      "删除后，该文档将不再用于检索和问答。",
    )
    expect(wrapper.emitted("delete")).toEqual([["doc-1"]])
    vi.unstubAllGlobals()
  })

  it("does not delete when the confirmation is cancelled", async () => {
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(false))
    const wrapper = mount(DocumentList, {
      props: {
        documents: [uploaded],
        loading: false,
        error: null,
        deletingDocumentId: null,
        deleteError: null,
      },
    })

    await wrapper.get("li button").trigger("click")

    expect(wrapper.emitted("delete")).toBeUndefined()
    vi.unstubAllGlobals()
  })

  it("shows a delete error without hiding the list", () => {
    const wrapper = mount(DocumentList, {
      props: {
        documents: [uploaded],
        loading: false,
        error: null,
        deletingDocumentId: null,
        deleteError: "内置文档不能删除",
      },
    })

    expect(wrapper.get('[role="alert"]').text()).toContain("内置文档不能删除")
    expect(wrapper.findAll("li")).toHaveLength(1)
  })
})

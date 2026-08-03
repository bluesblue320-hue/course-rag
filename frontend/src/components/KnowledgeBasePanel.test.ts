import { flushPromises, mount } from "@vue/test-utils"
import { describe, expect, it, vi } from "vitest"

import KnowledgeBasePanel from "./KnowledgeBasePanel.vue"
import {
  documentServiceKey,
  type DocumentService,
} from "../services/documentService"
import type {
  DeleteDocumentResponse,
  Document,
  DocumentListResponse,
} from "../types/documents"

const builtinDocument: Document = {
  document_id: "builtin-knowledge",
  filename: "knowledge.txt",
  content_type: "text/plain",
  size_bytes: 1800,
  text_length: 1800,
  chunk_count: 8,
  created_at: "2026-01-01T00:00:00Z",
  is_builtin: true,
}

const uploadedDocument: Document = {
  document_id: "doc-1",
  filename: "notes.txt",
  content_type: "text/plain",
  size_bytes: 1024,
  text_length: 512,
  chunk_count: 3,
  created_at: "2026-08-03T08:00:00Z",
  is_builtin: false,
}

const listResponse: DocumentListResponse = {
  documents: [builtinDocument],
  document_count: 1,
  chunk_count: 8,
}

const afterUploadListResponse: DocumentListResponse = {
  documents: [builtinDocument, uploadedDocument],
  document_count: 2,
  chunk_count: 11,
}

const deleteResponse: DeleteDocumentResponse = {
  document_id: "doc-1",
  deleted: true,
  document_count: 1,
  chunk_count: 8,
}

function makeService(overrides: Partial<DocumentService> = {}): DocumentService {
  return {
    listDocuments: vi.fn().mockResolvedValue(listResponse),
    uploadDocument: vi.fn().mockResolvedValue(uploadedDocument),
    deleteDocument: vi.fn().mockResolvedValue(deleteResponse),
    ...overrides,
  }
}

function mountPanel(service: DocumentService) {
  return mount(KnowledgeBasePanel, {
    global: {
      provide: {
        [documentServiceKey as symbol]: service,
      },
    },
  })
}

async function selectAndUpload(wrapper: ReturnType<typeof mount>): Promise<void> {
  const file = new File(["alpha content"], "notes.txt", {
    type: "text/plain",
  })
  const input = wrapper.get('input[type="file"]').element as HTMLInputElement
  Object.defineProperty(input, "files", { value: [file] })
  await wrapper.get('input[type="file"]').trigger("change")
  await wrapper.get("form").trigger("submit")
  await flushPromises()
}

describe("KnowledgeBasePanel", () => {
  it("loads the document list on mount", async () => {
    const service = makeService()
    mountPanel(service)
    await flushPromises()

    expect(service.listDocuments).toHaveBeenCalledOnce()
    expect(mountPanel(service).text()).toBeDefined()
  })

  it("refreshes the list after a successful upload", async () => {
    const service = makeService()
    const listMethod = service.listDocuments as ReturnType<typeof vi.fn>
    listMethod.mockResolvedValueOnce(listResponse).mockResolvedValueOnce(
      afterUploadListResponse,
    )
    const wrapper = mountPanel(service)
    await flushPromises()
    expect(wrapper.text()).not.toContain("notes.txt")

    await selectAndUpload(wrapper)

    expect(wrapper.text()).toContain("上传成功")
    expect(wrapper.text()).toContain("notes.txt")
    expect(listMethod).toHaveBeenCalledTimes(2)
  })

  it("shows a stable error when the upload fails", async () => {
    const service = makeService({
      uploadDocument: vi.fn().mockRejectedValue(
        new Error("仅支持 .txt、.md 和文本型 .pdf 文件"),
      ),
    })
    const wrapper = mountPanel(service)
    await flushPromises()

    await selectAndUpload(wrapper)

    expect(wrapper.get('[role="alert"]').text()).toContain("仅支持")
  })

  it("removes the document from the list after a confirmed deletion", async () => {
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(true))
    const service = makeService()
    const listMethod = service.listDocuments as ReturnType<typeof vi.fn>
    listMethod
      .mockResolvedValueOnce(afterUploadListResponse)
      .mockResolvedValueOnce(listResponse)
    const wrapper = mountPanel(service)
    await flushPromises()

    await wrapper.get("li button").trigger("click")
    await flushPromises()

    expect(wrapper.text()).not.toContain("notes.txt")
    expect(service.deleteDocument).toHaveBeenCalledWith("doc-1")
    vi.unstubAllGlobals()
  })

  it("keeps the document listed when deletion fails", async () => {
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(true))
    const service = makeService({
      listDocuments: vi.fn().mockResolvedValue(afterUploadListResponse),
      deleteDocument: vi.fn().mockRejectedValue(
        new Error("文档不存在"),
      ),
    })
    const wrapper = mountPanel(service)
    await flushPromises()

    await wrapper.get("li button").trigger("click")
    await flushPromises()

    expect(wrapper.get('[role="alert"]').text()).toContain("文档不存在")
    expect(wrapper.text()).toContain("knowledge.txt")
    expect(wrapper.text()).toContain("notes.txt")
    vi.unstubAllGlobals()
  })
})

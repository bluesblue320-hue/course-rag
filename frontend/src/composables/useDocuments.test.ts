import { describe, expect, it, vi } from "vitest"

import {
  DocumentServiceError,
  type DocumentService,
} from "../services/documentService"
import type {
  DeleteDocumentResponse,
  Document,
  DocumentListResponse,
} from "../types/documents"
import { useDocuments } from "./useDocuments"

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

describe("useDocuments", () => {
  it("loads the initial document list", async () => {
    const controller = useDocuments(makeService())

    const pending = controller.load()
    expect(controller.isListLoading.value).toBe(true)

    await pending

    expect(controller.listState.value).toEqual({
      status: "success",
      documents: [builtinDocument],
      documentCount: 1,
      chunkCount: 8,
    })
  })

  it("moves through uploading to success and refreshes the list", async () => {
    const service = makeService()
    const controller = useDocuments(service)

    const pending = controller.upload(new File(["x"], "notes.txt"))
    expect(controller.uploadState.value).toEqual({ status: "uploading" })
    expect(controller.isUploading.value).toBe(true)

    await pending

    expect(controller.uploadState.value).toEqual({
      status: "success",
      document: uploadedDocument,
    })
    expect(service.listDocuments).toHaveBeenCalledOnce()
  })

  it("captures upload errors with a stable message", async () => {
    const controller = useDocuments(
      makeService({
        uploadDocument: vi.fn().mockRejectedValue(
          new DocumentServiceError(
            "仅支持 .txt、.md 和文本型 .pdf 文件",
            "UNSUPPORTED_DOCUMENT_TYPE",
            415,
          ),
        ),
      }),
    )

    await controller.upload(new File(["x"], "notes.docx"))

    expect(controller.uploadState.value).toEqual({
      status: "error",
      message: "仅支持 .txt、.md 和文本型 .pdf 文件",
    })
  })

  it("blocks a second upload while uploading", async () => {
    let resolveUpload!: (value: Document) => void
    const uploadMethod = vi.fn(
      () =>
        new Promise<Document>((resolve) => {
          resolveUpload = resolve
        }),
    )
    const controller = useDocuments(
      makeService({ uploadDocument: uploadMethod }),
    )

    const first = controller.upload(new File(["a"], "a.txt"))
    await controller.upload(new File(["b"], "b.txt"))

    expect(uploadMethod).toHaveBeenCalledOnce()
    resolveUpload(uploadedDocument)
    await first
  })

  it("deletes a document and refreshes the list", async () => {
    const service = makeService()
    const controller = useDocuments(service)

    const pending = controller.deleteDocument("doc-1")
    expect(controller.deleteState.value).toEqual({
      status: "deleting",
      documentId: "doc-1",
    })
    expect(controller.isDeleting.value).toBe(true)

    await pending

    expect(controller.deleteState.value).toEqual({
      status: "success",
      documentId: "doc-1",
    })
    expect(service.deleteDocument).toHaveBeenCalledWith("doc-1")
    expect(service.listDocuments).toHaveBeenCalledOnce()
  })

  it("captures delete errors and keeps the list available", async () => {
    const controller = useDocuments(
      makeService({
        deleteDocument: vi.fn().mockRejectedValue(
          new DocumentServiceError("内置文档不能删除", "BUILTIN_DOCUMENT_CANNOT_BE_DELETED", 409),
        ),
      }),
    )
    await controller.load()

    await controller.deleteDocument("builtin-knowledge")

    expect(controller.deleteState.value).toEqual({
      status: "error",
      documentId: "builtin-knowledge",
      message: "内置文档不能删除",
    })
    expect(controller.listState.value.status).toBe("success")
  })

  it("retries a failed list load", async () => {
    const listMethod = vi
      .fn()
      .mockRejectedValueOnce(new DocumentServiceError("网络错误", "NETWORK_ERROR"))
      .mockResolvedValueOnce(listResponse)
    const controller = useDocuments(makeService({ listDocuments: listMethod }))

    await controller.load()
    expect(controller.listState.value).toEqual({
      status: "error",
      message: "网络错误",
    })

    await controller.load()

    expect(controller.listState.value.status).toBe("success")
  })

  it("resets the upload status to idle", async () => {
    const controller = useDocuments(makeService())

    controller.resetUpload()

    expect(controller.uploadState.value).toEqual({ status: "idle" })
  })
})

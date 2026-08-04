import { describe, expect, it, vi } from "vitest"

import { DocumentServiceError } from "./documentService"
import { createHttpDocumentService } from "./httpDocumentService"

const validDocument = {
  document_id: "abc123",
  filename: "notes.txt",
  content_type: "text/plain",
  size_bytes: 1024,
  text_length: 512,
  chunk_count: 3,
  created_at: "2026-08-03T08:00:00Z",
  is_builtin: false,
  index_status: "ready",
}

const validListResponse = {
  documents: [validDocument],
  document_count: 1,
  chunk_count: 4,
}

const validDeleteResponse = {
  document_id: "abc123",
  deleted: true,
  document_count: 1,
  chunk_count: 4,
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("createHttpDocumentService", () => {
  it("lists documents via GET /documents and returns guarded data", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(validListResponse))
    const service = createHttpDocumentService("/api/", fetchImpl)

    await expect(service.listDocuments()).resolves.toEqual(validListResponse)

    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(fetchImpl).toHaveBeenCalledWith("/api/documents")
  })

  it("rejects a malformed list response", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse({ ...validListResponse, documents: [{ bad: true }] }),
    )
    const service = createHttpDocumentService("/api", fetchImpl)

    await expect(service.listDocuments()).rejects.toMatchObject({
      code: "INVALID_RESPONSE",
    })
  })

  it("uploads with multipart/form-data without a manual Content-Type", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(validDocument, 201))
    const service = createHttpDocumentService("/api", fetchImpl)
    const file = new File(["content"], "notes.txt", { type: "text/plain" })

    await expect(service.uploadDocument(file)).resolves.toEqual(validDocument)

    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("/api/documents")
    expect(init.method).toBe("POST")
    expect(init.body).toBeInstanceOf(FormData)
    expect(init.headers).toBeUndefined()
    expect((init.body as FormData).get("file")).toBe(file)
  })

  it("rejects an upload whose filename is blank", async () => {
    const fetchImpl = vi.fn()
    const service = createHttpDocumentService("/api", fetchImpl)

    await expect(
      service.uploadDocument(new File(["x"], "  ")),
    ).rejects.toMatchObject({ code: "INVALID_FILE" })
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it.each([
    [413, "文件超过大小限制", "UPLOAD_TOO_LARGE"],
    [415, "仅支持 .txt、.md 和文本型 .pdf 文件", "UNSUPPORTED_DOCUMENT_TYPE"],
    [422, "文档内容为空", "EMPTY_DOCUMENT"],
    [503, "文档上传服务尚未完成配置", "UPLOAD_NOT_CONFIGURED"],
  ])("maps HTTP %i to a stable upload error", async (status, message, code) => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse({ code, message: "internal details" }, status),
    )
    const service = createHttpDocumentService("/api", fetchImpl)
    const file = new File(["x"], "notes.txt", { type: "text/plain" })

    const error = (await service
      .uploadDocument(file)
      .catch((caught: unknown) => caught)) as DocumentServiceError

    expect(error).toBeInstanceOf(DocumentServiceError)
    expect(error).toMatchObject({ message, code, httpStatus: status })
    expect(String(error)).not.toContain("internal details")
  })

  it("maps a generic 422 to a parse error without trusting backend text", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse({ code: "DOCUMENT_PARSE_FAILED", message: "leaked path" }, 422),
    )
    const service = createHttpDocumentService("/api", fetchImpl)

    const error = (await service
      .uploadDocument(new File(["x"], "broken.pdf", { type: "application/pdf" }))
      .catch((caught: unknown) => caught)) as DocumentServiceError

    expect(error).toMatchObject({
      message: "文档解析失败，请确认文件内容有效",
      code: "DOCUMENT_PARSE_FAILED",
    })
    expect(String(error)).not.toContain("leaked path")
  })

  it("normalizes upload network failures", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("secret host"))
    const service = createHttpDocumentService("/api", fetchImpl)

    await expect(
      service.uploadDocument(new File(["x"], "notes.txt")),
    ).rejects.toMatchObject({
      message: "无法连接后端服务，请确认 FastAPI 已启动",
      code: "NETWORK_ERROR",
    })
  })

  it("deletes via DELETE /documents/{id} and returns guarded data", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(validDeleteResponse))
    const service = createHttpDocumentService("/api", fetchImpl)

    await expect(service.deleteDocument("abc/123")).resolves.toEqual(
      validDeleteResponse,
    )

    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(fetchImpl).toHaveBeenCalledWith("/api/documents/abc%2F123", {
      method: "DELETE",
    })
  })

  it.each([
    [404, "文档不存在", "DOCUMENT_NOT_FOUND"],
    [409, "内置文档不能删除", "BUILTIN_DOCUMENT_CANNOT_BE_DELETED"],
  ])("maps HTTP %i to a stable delete error", async (status, message, code) => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({}, status))
    const service = createHttpDocumentService("/api", fetchImpl)

    await expect(service.deleteDocument("abc")).rejects.toMatchObject({
      message,
      code,
      httpStatus: status,
    })
  })

  it("normalizes delete network failures", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("offline"))
    const service = createHttpDocumentService("/api", fetchImpl)

    await expect(service.deleteDocument("abc")).rejects.toMatchObject({
      code: "NETWORK_ERROR",
    })
  })
})

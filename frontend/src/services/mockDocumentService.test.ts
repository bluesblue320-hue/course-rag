import { afterEach, describe, expect, it, vi } from "vitest"

import { createMockDocumentService } from "./mockDocumentService"

afterEach(() => {
  vi.useRealTimers()
})

async function finish<T>(pending: Promise<T>, delayMs: number): Promise<T> {
  await vi.advanceTimersByTimeAsync(delayMs)
  return pending
}

describe("createMockDocumentService", () => {
  it("lists the built-in document without network access", async () => {
    vi.useFakeTimers()
    const service = createMockDocumentService(50)

    const response = await finish(service.listDocuments(), 50)

    expect(response.document_count).toBe(1)
    expect(response.documents[0].filename).toBe("knowledge.txt")
    expect(response.documents[0].is_builtin).toBe(true)
  })

  it("returns deep copies so callers cannot mutate the store", async () => {
    vi.useFakeTimers()
    const service = createMockDocumentService(50)
    const first = await finish(service.listDocuments(), 50)
    first.documents[0].filename = "tampered"

    const second = await finish(service.listDocuments(), 50)

    expect(second.documents[0].filename).toBe("knowledge.txt")
  })

  it("uploads a txt file and lists it after the built-in document", async () => {
    vi.useFakeTimers()
    const service = createMockDocumentService(50)
    const file = new File(["alpha content"], "notes.txt", { type: "text/plain" })

    const uploaded = await finish(service.uploadDocument(file), 50)
    const list = await finish(service.listDocuments(), 50)

    expect(uploaded.filename).toBe("notes.txt")
    expect(uploaded.is_builtin).toBe(false)
    expect(list.documents.map((document) => document.filename)).toEqual([
      "knowledge.txt",
      "notes.txt",
    ])
  })

  it("returns deep copies of uploaded documents", async () => {
    vi.useFakeTimers()
    const service = createMockDocumentService(50)
    const uploaded = await finish(
      service.uploadDocument(new File(["x"], "notes.txt")),
      50,
    )
    uploaded.filename = "tampered"

    const list = await finish(service.listDocuments(), 50)

    expect(list.documents[1].filename).toBe("notes.txt")
  })

  it.each([
    ["docx", "notes.docx", "too long", 415, "UNSUPPORTED_DOCUMENT_TYPE"],
    ["empty", "empty.txt", "", 422, "EMPTY_DOCUMENT"],
    ["oversized", "big.txt", "too long", 413, "UPLOAD_TOO_LARGE"],
  ])(
    "rejects a %s file with a stable error",
    async (_name, filename, content, status, code) => {
      vi.useFakeTimers()
      const service = createMockDocumentService(50, 5)
      const file = new File([content], filename, { type: "text/plain" })

      const pending = service.uploadDocument(file)
      const assertion = expect(pending).rejects.toMatchObject({
        code,
        httpStatus: status,
      })
      await vi.advanceTimersByTimeAsync(50)
      await assertion
    },
  )

  it("deletes an uploaded document and updates totals", async () => {
    vi.useFakeTimers()
    const service = createMockDocumentService(50)
    const uploaded = await finish(
      service.uploadDocument(new File(["x"], "notes.txt")),
      50,
    )

    const deleted = await finish(service.deleteDocument(uploaded.document_id), 50)

    expect(deleted.deleted).toBe(true)
    expect(deleted.document_count).toBe(1)
  })

  it("rejects deleting the built-in document with 409", async () => {
    vi.useFakeTimers()
    const service = createMockDocumentService(50)

    const pending = service.deleteDocument("builtin-knowledge")
    const assertion = expect(pending).rejects.toMatchObject({
      code: "BUILTIN_DOCUMENT_CANNOT_BE_DELETED",
      httpStatus: 409,
    })
    await vi.advanceTimersByTimeAsync(50)
    await assertion
  })

  it("rejects deleting an unknown document with 404", async () => {
    vi.useFakeTimers()
    const service = createMockDocumentService(50)

    const pending = service.deleteDocument("missing")
    const assertion = expect(pending).rejects.toMatchObject({
      code: "DOCUMENT_NOT_FOUND",
      httpStatus: 404,
    })
    await vi.advanceTimersByTimeAsync(50)
    await assertion
  })
})

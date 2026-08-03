import { describe, expect, it } from "vitest"

import {
  isDeleteDocumentResponse,
  isDocument,
  isDocumentListResponse,
} from "./documentService"

const validDocument = {
  document_id: "abc123",
  filename: "notes.txt",
  content_type: "text/plain",
  size_bytes: 1024,
  text_length: 512,
  chunk_count: 3,
  created_at: "2026-08-03T08:00:00Z",
  is_builtin: false,
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

describe("documentService runtime guards", () => {
  it("accepts a complete document", () => {
    expect(isDocument(validDocument)).toBe(true)
  })

  it("accepts the built-in document", () => {
    expect(isDocument({ ...validDocument, is_builtin: true })).toBe(true)
  })

  it("rejects null and non-object values", () => {
    expect(isDocument(null)).toBe(false)
    expect(isDocument("text")).toBe(false)
    expect(isDocument([])).toBe(false)
  })

  it("rejects missing fields", () => {
    const { document_id: _id, ...withoutId } = validDocument
    const { is_builtin: _builtin, ...withoutBuiltin } = validDocument
    expect(isDocument(withoutId)).toBe(false)
    expect(isDocument(withoutBuiltin)).toBe(false)
  })

  it("rejects non-integer sizes and counts", () => {
    expect(isDocument({ ...validDocument, size_bytes: 1.5 })).toBe(false)
    expect(isDocument({ ...validDocument, size_bytes: -1 })).toBe(false)
    expect(isDocument({ ...validDocument, text_length: "512" })).toBe(false)
    expect(isDocument({ ...validDocument, chunk_count: 2.5 })).toBe(false)
  })

  it("rejects a boolean built-in flag as a string", () => {
    expect(isDocument({ ...validDocument, is_builtin: "true" })).toBe(false)
  })

  it("accepts a complete list response", () => {
    expect(isDocumentListResponse(validListResponse)).toBe(true)
  })

  it("rejects a list response with invalid documents", () => {
    expect(
      isDocumentListResponse({
        ...validListResponse,
        documents: [{ filename: "only-name" }],
      }),
    ).toBe(false)
  })

  it("rejects a list response with wrong counts", () => {
    expect(
      isDocumentListResponse({ ...validListResponse, document_count: -1 }),
    ).toBe(false)
    expect(
      isDocumentListResponse({ ...validListResponse, chunk_count: "8" }),
    ).toBe(false)
  })

  it("accepts a complete delete response", () => {
    expect(isDeleteDocumentResponse(validDeleteResponse)).toBe(true)
  })

  it("rejects a delete response with deleted as a string", () => {
    expect(
      isDeleteDocumentResponse({ ...validDeleteResponse, deleted: "yes" }),
    ).toBe(false)
  })

  it("rejects a delete response without counts", () => {
    const { chunk_count: _count, ...withoutCount } = validDeleteResponse
    expect(isDeleteDocumentResponse(withoutCount)).toBe(false)
  })
})

import type {
  DeleteDocumentResponse,
  Document,
  DocumentListResponse,
} from "../types/documents"
import {
  DocumentServiceError,
  type DocumentService,
} from "./documentService"

const BUILTIN_DOCUMENT: Document = {
  document_id: "builtin-knowledge",
  filename: "knowledge.txt",
  content_type: "text/plain",
  size_bytes: 1800,
  text_length: 1800,
  chunk_count: 8,
  created_at: "2026-01-01T00:00:00Z",
  is_builtin: true,
}

const ALLOWED_SUFFIXES = [".txt", ".md", ".pdf"]

function deepCopy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function wait(delayMs: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, delayMs)
  })
}

export function createMockDocumentService(
  delayMs = 400,
  maxUploadBytes = 10 * 1024 * 1024,
): DocumentService {
  const documents: Document[] = [deepCopy(BUILTIN_DOCUMENT)]
  let nextId = 1

  return {
    async listDocuments(): Promise<DocumentListResponse> {
      await wait(delayMs)
      return {
        documents: documents.map((document) => deepCopy(document)),
        document_count: documents.length,
        chunk_count: documents.reduce(
          (total, document) => total + document.chunk_count,
          0,
        ),
      }
    },

    async uploadDocument(file: File): Promise<Document> {
      const suffix = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`
      if (!ALLOWED_SUFFIXES.includes(suffix)) {
        throw new DocumentServiceError(
          "仅支持 .txt、.md 和文本型 .pdf 文件",
          "UNSUPPORTED_DOCUMENT_TYPE",
          415,
        )
      }
      if (file.size === 0) {
        throw new DocumentServiceError("文档内容为空", "EMPTY_DOCUMENT", 422)
      }
      if (file.size > maxUploadBytes) {
        throw new DocumentServiceError(
          "文件超过大小限制",
          "UPLOAD_TOO_LARGE",
          413,
        )
      }

      await wait(delayMs)
      const document: Document = {
        document_id: `mock-document-${nextId}`,
        filename: file.name,
        content_type: file.type || "application/octet-stream",
        size_bytes: file.size,
        text_length: file.size,
        chunk_count: 2,
        created_at: new Date().toISOString(),
        is_builtin: false,
      }
      nextId += 1
      documents.push(document)
      return deepCopy(document)
    },

    async deleteDocument(documentId: string): Promise<DeleteDocumentResponse> {
      await wait(delayMs)
      const index = documents.findIndex(
        (document) => document.document_id === documentId,
      )
      if (index < 0) {
        throw new DocumentServiceError(
          "文档不存在",
          "DOCUMENT_NOT_FOUND",
          404,
        )
      }
      if (documents[index].is_builtin) {
        throw new DocumentServiceError(
          "内置文档不能删除",
          "BUILTIN_DOCUMENT_CANNOT_BE_DELETED",
          409,
        )
      }
      documents.splice(index, 1)
      return {
        document_id: documentId,
        deleted: true,
        document_count: documents.length,
        chunk_count: documents.reduce(
          (total, document) => total + document.chunk_count,
          0,
        ),
      }
    },
  }
}

export const mockDocumentService = createMockDocumentService()

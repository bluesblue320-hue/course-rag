import type { InjectionKey } from "vue"

import type {
  DeleteDocumentResponse,
  Document,
  DocumentListResponse,
} from "../types/documents"

export interface DocumentService {
  listDocuments(): Promise<DocumentListResponse>
  uploadDocument(file: File): Promise<Document>
  deleteDocument(documentId: string): Promise<DeleteDocumentResponse>
}

export const documentServiceKey: InjectionKey<DocumentService> = Symbol(
  "documentService",
)

export class DocumentServiceError extends Error {
  public readonly code?: string
  public readonly httpStatus?: number

  constructor(
    message: string,
    code?: string,
    httpStatus?: number,
  ) {
    super(message)
    this.name = "DocumentServiceError"
    this.code = code
    this.httpStatus = httpStatus
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0
}

export function isDocument(value: unknown): value is Document {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.document_id === "string" &&
    typeof value.filename === "string" &&
    typeof value.content_type === "string" &&
    isNonNegativeInteger(value.size_bytes) &&
    isNonNegativeInteger(value.text_length) &&
    isNonNegativeInteger(value.chunk_count) &&
    typeof value.created_at === "string" &&
    typeof value.is_builtin === "boolean"
  )
}

export function isDocumentListResponse(
  value: unknown,
): value is DocumentListResponse {
  return (
    isRecord(value) &&
    Array.isArray(value.documents) &&
    value.documents.every(isDocument) &&
    isNonNegativeInteger(value.document_count) &&
    isNonNegativeInteger(value.chunk_count)
  )
}

export function isDeleteDocumentResponse(
  value: unknown,
): value is DeleteDocumentResponse {
  return (
    isRecord(value) &&
    typeof value.document_id === "string" &&
    typeof value.deleted === "boolean" &&
    isNonNegativeInteger(value.document_count) &&
    isNonNegativeInteger(value.chunk_count)
  )
}

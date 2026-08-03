export interface Document {
  document_id: string
  filename: string
  content_type: string
  size_bytes: number
  text_length: number
  chunk_count: number
  created_at: string
  is_builtin: boolean
}

export interface DocumentListResponse {
  documents: Document[]
  document_count: number
  chunk_count: number
}

export interface DeleteDocumentResponse {
  document_id: string
  deleted: boolean
  document_count: number
  chunk_count: number
}

export type DocumentListState =
  | { status: "idle" }
  | { status: "loading" }
  | {
      status: "success"
      documents: Document[]
      documentCount: number
      chunkCount: number
    }
  | { status: "error"; message: string }

export type DocumentUploadState =
  | { status: "idle" }
  | { status: "uploading" }
  | { status: "success"; document: Document }
  | { status: "error"; message: string }

export type DocumentDeleteState =
  | { status: "idle" }
  | { status: "deleting"; documentId: string }
  | { status: "success"; documentId: string }
  | { status: "error"; documentId: string; message: string }

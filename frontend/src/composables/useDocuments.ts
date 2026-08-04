import { computed, ref } from "vue"

import {
  DocumentServiceError,
  isDeleteDocumentResponse,
  isDocument,
  isDocumentListResponse,
  type DocumentService,
} from "../services/documentService"
import type {
  DocumentDeleteState,
  DocumentListState,
  DocumentUploadState,
} from "../types/documents"

function errorMessage(error: unknown): string {
  if (error instanceof DocumentServiceError) {
    return error.message
  }
  if (error instanceof Error && error.message) {
    return error.message
  }
  return "文档服务请求失败，请稍后重试"
}

export function useDocuments(service: DocumentService) {
  const listState = ref<DocumentListState>({ status: "idle" })
  const uploadState = ref<DocumentUploadState>({ status: "idle" })
  const deleteState = ref<DocumentDeleteState>({ status: "idle" })

  const isListLoading = computed(() => listState.value.status === "loading")
  const isUploading = computed(() => uploadState.value.status === "uploading")
  const isDeleting = computed(() => deleteState.value.status === "deleting")

  async function load(): Promise<void> {
    listState.value = { status: "loading" }
    try {
      const response = await service.listDocuments()
      if (!isDocumentListResponse(response)) {
        throw new DocumentServiceError(
          "文档服务返回的数据格式不正确",
          "INVALID_RESPONSE",
        )
      }
      listState.value = {
        status: "success",
        documents: response.documents,
        documentCount: response.document_count,
        chunkCount: response.chunk_count,
      }
    } catch (error) {
      listState.value = { status: "error", message: errorMessage(error) }
    }
  }

  async function upload(file: File): Promise<void> {
    if (isUploading.value) {
      return
    }
    uploadState.value = { status: "uploading" }
    try {
      const document = await service.uploadDocument(file)
      if (!isDocument(document)) {
        throw new DocumentServiceError(
          "文档服务返回的数据格式不正确",
          "INVALID_RESPONSE",
        )
      }
      uploadState.value = { status: "success", document }
      await load()
    } catch (error) {
      uploadState.value = { status: "error", message: errorMessage(error) }
    }
  }

  async function deleteDocument(documentId: string): Promise<void> {
    if (isDeleting.value) {
      return
    }
    deleteState.value = { status: "deleting", documentId }
    try {
      const response = await service.deleteDocument(documentId)
      if (!isDeleteDocumentResponse(response)) {
        throw new DocumentServiceError(
          "文档服务返回的数据格式不正确",
          "INVALID_RESPONSE",
        )
      }
      deleteState.value = { status: "success", documentId }
      await load()
    } catch (error) {
      deleteState.value = {
        status: "error",
        documentId,
        message: errorMessage(error),
      }
    }
  }

  function resetUpload(): void {
    uploadState.value = { status: "idle" }
  }

  return {
    listState,
    uploadState,
    deleteState,
    isListLoading,
    isUploading,
    isDeleting,
    load,
    upload,
    deleteDocument,
    resetUpload,
  }
}

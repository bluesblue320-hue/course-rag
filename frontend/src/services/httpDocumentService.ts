import type {
  DeleteDocumentResponse,
  Document,
  DocumentListResponse,
} from "../types/documents"
import {
  DocumentServiceError,
  isDeleteDocumentResponse,
  isDocument,
  isDocumentListResponse,
  type DocumentService,
} from "./documentService"

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "")
}

function parsedErrorCode(payload: unknown): string | undefined {
  if (typeof payload !== "object" || payload === null) {
    return undefined
  }
  const code = (payload as Record<string, unknown>).code
  return typeof code === "string" ? code : undefined
}

function errorForStatus(status: number, payload: unknown): DocumentServiceError {
  const parsedCode = parsedErrorCode(payload)
  if (status === 413) {
    return new DocumentServiceError(
      "文件超过大小限制",
      "UPLOAD_TOO_LARGE",
      status,
    )
  }
  if (status === 415) {
    return new DocumentServiceError(
      "仅支持 .txt、.md 和文本型 .pdf 文件",
      "UNSUPPORTED_DOCUMENT_TYPE",
      status,
    )
  }
  if (status === 422 && parsedCode === "EMPTY_DOCUMENT") {
    return new DocumentServiceError("文档内容为空", "EMPTY_DOCUMENT", status)
  }
  if (status === 422) {
    return new DocumentServiceError(
      "文档解析失败，请确认文件内容有效",
      "DOCUMENT_PARSE_FAILED",
      status,
    )
  }
  if (status === 404) {
    return new DocumentServiceError("文档不存在", "DOCUMENT_NOT_FOUND", status)
  }
  if (status === 409) {
    return new DocumentServiceError(
      "内置文档不能删除",
      "BUILTIN_DOCUMENT_CANNOT_BE_DELETED",
      status,
    )
  }
  if (status === 503) {
    return new DocumentServiceError(
      "文档上传服务尚未完成配置",
      "UPLOAD_NOT_CONFIGURED",
      status,
    )
  }
  return new DocumentServiceError(
    "文档服务暂时不可用",
    parsedCode ?? "SERVICE_UNAVAILABLE",
    status,
  )
}

async function readJsonSafely(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return undefined
  }
}

export function createHttpDocumentService(
  baseUrl = "/api",
  fetchImpl: typeof fetch = fetch,
): DocumentService {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl)

  return {
    async listDocuments(): Promise<DocumentListResponse> {
      let response: Response
      try {
        response = await fetchImpl(`${normalizedBaseUrl}/documents`)
      } catch {
        throw new DocumentServiceError(
          "无法连接后端服务，请确认 FastAPI 已启动",
          "NETWORK_ERROR",
        )
      }

      const payload = await readJsonSafely(response)
      if (!response.ok) {
        throw errorForStatus(response.status, payload)
      }
      if (!isDocumentListResponse(payload)) {
        throw new DocumentServiceError(
          "文档服务返回的数据格式不正确",
          "INVALID_RESPONSE",
          response.status,
        )
      }
      return payload
    },

    async uploadDocument(file: File): Promise<Document> {
      const trimmedName = file.name.trim()
      if (!trimmedName) {
        throw new DocumentServiceError(
          "请选择要上传的文件",
          "INVALID_FILE",
        )
      }

      const form = new FormData()
      form.append("file", file)

      let response: Response
      try {
        response = await fetchImpl(`${normalizedBaseUrl}/documents`, {
          method: "POST",
          body: form,
        })
      } catch {
        throw new DocumentServiceError(
          "无法连接后端服务，请确认 FastAPI 已启动",
          "NETWORK_ERROR",
        )
      }

      const payload = await readJsonSafely(response)
      if (!response.ok) {
        throw errorForStatus(response.status, payload)
      }
      if (!isDocument(payload)) {
        throw new DocumentServiceError(
          "文档服务返回的数据格式不正确",
          "INVALID_RESPONSE",
          response.status,
        )
      }
      return payload
    },

    async deleteDocument(documentId: string): Promise<DeleteDocumentResponse> {
      let response: Response
      try {
        response = await fetchImpl(
          `${normalizedBaseUrl}/documents/${encodeURIComponent(documentId)}`,
          { method: "DELETE" },
        )
      } catch {
        throw new DocumentServiceError(
          "无法连接后端服务，请确认 FastAPI 已启动",
          "NETWORK_ERROR",
        )
      }

      const payload = await readJsonSafely(response)
      if (!response.ok) {
        throw errorForStatus(response.status, payload)
      }
      if (!isDeleteDocumentResponse(payload)) {
        throw new DocumentServiceError(
          "文档服务返回的数据格式不正确",
          "INVALID_RESPONSE",
          response.status,
        )
      }
      return payload
    },
  }
}

export const httpDocumentService = createHttpDocumentService()

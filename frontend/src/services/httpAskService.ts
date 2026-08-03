import type { AskRequest, AskResponse } from "../types/ask"
import {
  AskServiceError,
  isAskApiError,
  isAskResponse,
  type AskService,
} from "./askService"

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "")
}

function validateRequest(request: AskRequest): AskRequest {
  const question = request.question.trim()
  if (!question) {
    throw new AskServiceError("请输入问题", "INVALID_QUESTION")
  }
  if (
    !Number.isInteger(request.top_k) ||
    request.top_k < 1 ||
    request.top_k > 10
  ) {
    throw new AskServiceError("top_k 必须是 1 到 10 之间的整数", "INVALID_TOP_K")
  }
  return { question, top_k: request.top_k }
}

function errorForStatus(status: number, payload: unknown): AskServiceError {
  const parsedCode = isAskApiError(payload) ? payload.code : undefined
  if (status === 422) {
    return new AskServiceError(
      "问题格式不正确，请检查后重试",
      "VALIDATION_ERROR",
      status,
    )
  }
  if (status === 502) {
    return new AskServiceError(
      "回答生成失败，请稍后重试",
      "GENERATION_FAILED",
      status,
    )
  }
  if (status === 503 && parsedCode === "RAG_NOT_CONFIGURED") {
    return new AskServiceError(
      "问答相关性配置无效，请检查后端 RAG 配置",
      "RAG_NOT_CONFIGURED",
      status,
    )
  }
  if (status === 503) {
    return new AskServiceError(
      "问答服务尚未完成配置，你仍可使用语义检索",
      "LLM_NOT_CONFIGURED",
      status,
    )
  }
  return new AskServiceError(
    "问答服务暂时不可用",
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

export function createHttpAskService(
  baseUrl = "/api",
  fetchImpl: typeof fetch = fetch,
): AskService {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl)

  return {
    async ask(request: AskRequest): Promise<AskResponse> {
      const validatedRequest = validateRequest(request)
      let response: Response

      try {
        response = await fetchImpl(`${normalizedBaseUrl}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(validatedRequest),
        })
      } catch {
        throw new AskServiceError(
          "无法连接后端服务，请确认 FastAPI 已启动",
          "NETWORK_ERROR",
        )
      }

      const payload = await readJsonSafely(response)
      if (!response.ok) {
        throw errorForStatus(response.status, payload)
      }
      if (!isAskResponse(payload)) {
        throw new AskServiceError(
          "问答服务返回的数据格式不正确",
          "INVALID_RESPONSE",
          response.status,
        )
      }
      return payload
    },
  }
}

export const httpAskService = createHttpAskService()

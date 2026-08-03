import type { InjectionKey } from "vue"

import type {
  AskApiError,
  AskRequest,
  AskResponse,
  AskSource,
} from "../types/ask"

export interface AskService {
  ask(request: AskRequest): Promise<AskResponse>
}

export const askServiceKey: InjectionKey<AskService> = Symbol("askService")

export class AskServiceError extends Error {
  public readonly code?: string
  public readonly httpStatus?: number

  constructor(
    message: string,
    code?: string,
    httpStatus?: number,
  ) {
    super(message)
    this.name = "AskServiceError"
    this.code = code
    this.httpStatus = httpStatus
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function isFiniteNonNegative(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
}

function isRelevanceScore(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= -1 && value <= 1
}

function isRelevanceThreshold(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
}

export function isAskSource(value: unknown): value is AskSource {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.rank === "number" &&
    Number.isInteger(value.rank) &&
    value.rank > 0 &&
    typeof value.score === "number" &&
    Number.isFinite(value.score) &&
    typeof value.text === "string" &&
    typeof value.chunk_index === "number" &&
    Number.isInteger(value.chunk_index) &&
    value.chunk_index >= 0
  )
}

export function isAskResponse(value: unknown): value is AskResponse {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.question === "string" &&
    typeof value.answer === "string" &&
    (value.answer_status === "answered" ||
      value.answer_status === "insufficient_context") &&
    (value.max_relevance_score === null ||
      isRelevanceScore(value.max_relevance_score)) &&
    isRelevanceThreshold(value.relevance_threshold) &&
    isFiniteNonNegative(value.retrieval_elapsed_ms) &&
    isFiniteNonNegative(value.generation_elapsed_ms) &&
    isFiniteNonNegative(value.total_elapsed_ms) &&
    typeof value.embedding_model === "string" &&
    typeof value.llm_model === "string" &&
    Array.isArray(value.sources) &&
    value.sources.every(isAskSource)
  )
}

export function isAskApiError(value: unknown): value is AskApiError {
  return (
    isRecord(value) &&
    typeof value.code === "string" &&
    typeof value.message === "string"
  )
}

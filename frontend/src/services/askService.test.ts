import { describe, expect, it } from "vitest"

import {
  AskServiceError,
  isAskApiError,
  isAskResponse,
  isAskSource,
} from "./askService"

const validSource = {
  rank: 1,
  score: 0.8421,
  text: "Service 层负责核心业务逻辑。",
  chunk_index: 2,
}

const validResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责核心业务逻辑。[来源1]",
  answer_status: "answered",
  max_relevance_score: 0.8421,
  relevance_threshold: 0.35,
  retrieval_elapsed_ms: 12.4,
  generation_elapsed_ms: 680.7,
  total_elapsed_ms: 693.1,
  embedding_model: "embedding-model",
  llm_model: "llm-model",
  sources: [validSource],
}

describe("AskService runtime guards", () => {
  it("exposes a shared error type with code and HTTP status", () => {
    const error = new AskServiceError(
      "回答生成失败",
      "GENERATION_FAILED",
      502,
    )

    expect(error).toBeInstanceOf(Error)
    expect(error.name).toBe("AskServiceError")
    expect(error.message).toBe("回答生成失败")
    expect(error.code).toBe("GENERATION_FAILED")
    expect(error.httpStatus).toBe(502)
  })

  it("accepts a complete AskResponse", () => {
    expect(isAskResponse(validResponse)).toBe(true)
  })

  it("rejects null", () => {
    expect(isAskResponse(null)).toBe(false)
  })

  it("rejects a response without answer", () => {
    const { answer: _answer, ...incomplete } = validResponse
    expect(isAskResponse(incomplete)).toBe(false)
  })

  it("rejects a response whose sources are not an array", () => {
    expect(isAskResponse({ ...validResponse, sources: null })).toBe(false)
  })

  it("rejects a string source rank", () => {
    expect(isAskSource({ ...validSource, rank: "1" })).toBe(false)
  })

  it("rejects NaN and Infinity source scores", () => {
    expect(isAskSource({ ...validSource, score: Number.NaN })).toBe(false)
    expect(isAskSource({ ...validSource, score: Number.POSITIVE_INFINITY })).toBe(
      false,
    )
  })

  it("rejects a fractional chunk index", () => {
    expect(isAskSource({ ...validSource, chunk_index: 1.5 })).toBe(false)
  })

  it("rejects string, non-finite, and negative timings", () => {
    expect(
      isAskResponse({ ...validResponse, retrieval_elapsed_ms: "12.4" }),
    ).toBe(false)
    expect(
      isAskResponse({ ...validResponse, generation_elapsed_ms: Number.NaN }),
    ).toBe(false)
    expect(isAskResponse({ ...validResponse, total_elapsed_ms: -1 })).toBe(false)
  })

  it("accepts a complete AskApiError", () => {
    expect(
      isAskApiError({ code: "LLM_NOT_CONFIGURED", message: "未配置" }),
    ).toBe(true)
  })

  it("rejects an AskApiError without code", () => {
    expect(isAskApiError({ message: "未配置" })).toBe(false)
  })

  it("accepts an insufficient-context response without a maximum score", () => {
    expect(
      isAskResponse({
        ...validResponse,
        answer_status: "insufficient_context",
        max_relevance_score: null,
        generation_elapsed_ms: 0,
      }),
    ).toBe(true)
  })

  it("rejects invalid answer statuses", () => {
    expect(isAskResponse({ ...validResponse, answer_status: "unknown" })).toBe(false)
  })

  it("rejects invalid maximum relevance scores", () => {
    expect(isAskResponse({ ...validResponse, max_relevance_score: Number.NaN })).toBe(false)
    expect(isAskResponse({ ...validResponse, max_relevance_score: 1.1 })).toBe(false)
    expect(isAskResponse({ ...validResponse, max_relevance_score: -1.1 })).toBe(false)
  })

  it("rejects invalid relevance thresholds", () => {
    expect(isAskResponse({ ...validResponse, relevance_threshold: Number.POSITIVE_INFINITY })).toBe(false)
    expect(isAskResponse({ ...validResponse, relevance_threshold: -0.1 })).toBe(false)
    expect(isAskResponse({ ...validResponse, relevance_threshold: 1.1 })).toBe(false)
  })

  it("rejects a string maximum relevance score", () => {
    expect(isAskResponse({ ...validResponse, max_relevance_score: "0.8" })).toBe(false)
  })

  it("rejects missing relevance fields", () => {
    const {
      relevance_threshold: _threshold,
      ...withoutThreshold
    } = validResponse
    const {
      answer_status: _status,
      ...withoutStatus
    } = validResponse

    expect(isAskResponse(withoutThreshold)).toBe(false)
    expect(isAskResponse(withoutStatus)).toBe(false)
  })

  it("rejects boolean answer status and infinite maximum score", () => {
    expect(isAskResponse({ ...validResponse, answer_status: true })).toBe(false)
    expect(
      isAskResponse({
        ...validResponse,
        max_relevance_score: Number.POSITIVE_INFINITY,
      }),
    ).toBe(false)
  })
})

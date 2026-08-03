import { describe, expect, it } from "vitest"

import { isAskApiError, isAskResponse, isAskSource } from "./askService"

const validSource = {
  rank: 1,
  score: 0.8421,
  text: "Service 层负责核心业务逻辑。",
  chunk_index: 2,
}

const validResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责核心业务逻辑。[来源1]",
  retrieval_elapsed_ms: 12.4,
  generation_elapsed_ms: 680.7,
  total_elapsed_ms: 693.1,
  embedding_model: "embedding-model",
  llm_model: "llm-model",
  sources: [validSource],
}

describe("AskService runtime guards", () => {
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
})

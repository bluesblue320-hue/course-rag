import { describe, expect, it } from "vitest"

import { isSearchResponse } from "./searchService"

const validResponse = {
  query: "业务逻辑应该写在哪一层？",
  elapsed_ms: 420,
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  results: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "service 层负责业务逻辑。",
      document_id: "builtin-knowledge",
      filename: "knowledge.txt",
      page_number: null,
    },
  ],
}

describe("isSearchResponse", () => {
  it("accepts a complete search response", () => {
    expect(isSearchResponse(validResponse)).toBe(true)
  })

  it("rejects a response whose results are not an array", () => {
    expect(isSearchResponse({ ...validResponse, results: null })).toBe(false)
  })

  it("rejects malformed result items", () => {
    expect(
      isSearchResponse({
        ...validResponse,
        results: [{ rank: 1, score: "high" }],
      }),
    ).toBe(false)
  })
})

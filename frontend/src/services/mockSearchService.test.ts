import { afterEach, describe, expect, it, vi } from "vitest"

import { createMockSearchService } from "./mockSearchService"

afterEach(() => {
  vi.useRealTimers()
})

async function finishSearch(query: string, topK = 3) {
  vi.useFakeTimers()
  const service = createMockSearchService(500)
  const pending = service.search({ query, top_k: topK })
  await vi.advanceTimersByTimeAsync(500)
  return pending
}

describe("createMockSearchService", () => {
  it("returns service-first results for a business-logic question", async () => {
    const response = await finishSearch(" 业务逻辑应该写在哪一层？ ")

    expect(response.query).toBe("业务逻辑应该写在哪一层？")
    expect(response.results).toHaveLength(3)
    expect(response.results[0].text).toContain("service 层")
    expect(response.indexed_chunks).toBe(8)
  })

  it("returns repository-first results for a database question", async () => {
    const response = await finishSearch("哪个模块负责访问数据库？")

    expect(response.results[0].text).toContain("repository 层")
  })

  it("returns embedding-first results for a vector question", async () => {
    const response = await finishSearch("如何把文本转换成向量？", 2)

    expect(response.results).toHaveLength(2)
    expect(response.results[0].text).toContain("Embedding")
  })

  it("returns an empty result list for an unrelated question", async () => {
    const response = await finishSearch("今天午饭吃什么？")

    expect(response.results).toEqual([])
  })

  it("rejects an invalid top_k before waiting", async () => {
    const service = createMockSearchService(500)

    await expect(
      service.search({ query: "业务逻辑", top_k: 0 }),
    ).rejects.toThrow("top_k 必须大于 0")
  })
})

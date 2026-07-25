import { afterEach, describe, expect, it, vi } from "vitest"

import { createHttpSearchService } from "./httpSearchService"

const validResponse = {
  query: "业务逻辑应该写在哪一层？",
  elapsed_ms: 12.5,
  indexed_chunks: 4,
  model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  results: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "service 层负责业务逻辑。",
    },
  ],
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("createHttpSearchService", () => {
  it("posts the search request and returns a validated response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(validResponse), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    vi.stubGlobal("fetch", fetchMock)

    const service = createHttpSearchService("/api/")
    const request = { query: validResponse.query, top_k: 3 }

    await expect(service.search(request)).resolves.toEqual(validResponse)
    expect(fetchMock).toHaveBeenCalledWith("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    })
  })

  it("rejects unsuccessful HTTP responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 503 })),
    )

    const service = createHttpSearchService()

    await expect(
      service.search({ query: "测试问题", top_k: 3 }),
    ).rejects.toThrow("检索请求失败（503）")
  })

  it("rejects malformed JSON responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ query: "缺少字段" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    )

    const service = createHttpSearchService()

    await expect(
      service.search({ query: "测试问题", top_k: 3 }),
    ).rejects.toThrow("检索服务返回了无效数据")
  })

  it("normalizes network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")))

    const service = createHttpSearchService()

    await expect(
      service.search({ query: "测试问题", top_k: 3 }),
    ).rejects.toThrow("无法连接检索服务")
  })
})

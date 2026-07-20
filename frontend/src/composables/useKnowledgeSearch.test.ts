import { describe, expect, it, vi } from "vitest"

import type { SearchService } from "../services/searchService"
import type { SearchResponse } from "../types/search"
import { useKnowledgeSearch } from "./useKnowledgeSearch"

const successResponse: SearchResponse = {
  query: "业务逻辑",
  elapsed_ms: 420,
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  results: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "service 层负责业务逻辑。",
    },
  ],
}

describe("useKnowledgeSearch", () => {
  it("moves from idle through loading to success", async () => {
    let resolveResponse!: (response: SearchResponse) => void
    const service: SearchService = {
      search: vi.fn(
        () =>
          new Promise<SearchResponse>((resolve) => {
            resolveResponse = resolve
          }),
      ),
    }
    const search = useKnowledgeSearch(service)

    const pending = search.search(" 业务逻辑 ")
    expect(search.state.value).toEqual({
      status: "loading",
      query: "业务逻辑",
    })
    expect(search.isLoading.value).toBe(true)

    resolveResponse(successResponse)
    await pending

    expect(search.state.value.status).toBe("success")
    expect(search.isLoading.value).toBe(false)
  })

  it("uses the empty state for a valid response with no results", async () => {
    const service: SearchService = {
      search: vi.fn().mockResolvedValue({
        ...successResponse,
        results: [],
      }),
    }
    const search = useKnowledgeSearch(service)

    await search.search("未知问题")

    expect(search.state.value.status).toBe("empty")
  })

  it("normalizes unknown failures into a stable message", async () => {
    const service: SearchService = {
      search: vi.fn().mockRejectedValue("offline"),
    }
    const search = useKnowledgeSearch(service)

    await search.search("业务逻辑")

    expect(search.state.value).toEqual({
      status: "error",
      query: "业务逻辑",
      message: "检索失败，请稍后重试",
    })
  })

  it("rejects a malformed service response", async () => {
    const service = {
      search: vi.fn().mockResolvedValue({
        ...successResponse,
        results: null,
      }),
    } as unknown as SearchService
    const search = useKnowledgeSearch(service)

    await search.search("业务逻辑")

    expect(search.state.value).toEqual({
      status: "error",
      query: "业务逻辑",
      message: "检索结果格式不正确",
    })
  })

  it("retries the last submitted query", async () => {
    const searchMethod = vi
      .fn()
      .mockRejectedValueOnce(new Error("服务暂时不可用"))
      .mockResolvedValueOnce(successResponse)
    const search = useKnowledgeSearch({ search: searchMethod })

    await search.search("业务逻辑")
    await search.retry()

    expect(searchMethod).toHaveBeenCalledTimes(2)
    expect(search.state.value.status).toBe("success")
  })
})

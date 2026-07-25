import type { SearchRequest, SearchResponse } from "../types/search"
import { isSearchResponse, type SearchService } from "./searchService"

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "")
}

export function createHttpSearchService(baseUrl = "/api"): SearchService {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl)

  return {
    async search(request: SearchRequest): Promise<SearchResponse> {
      let response: Response

      try {
        response = await fetch(`${normalizedBaseUrl}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
        })
      } catch {
        throw new Error("无法连接检索服务")
      }

      if (!response.ok) {
        throw new Error(`检索请求失败（${response.status}）`)
      }

      let payload: unknown
      try {
        payload = await response.json()
      } catch {
        throw new Error("检索服务返回了无效数据")
      }

      if (!isSearchResponse(payload)) {
        throw new Error("检索服务返回了无效数据")
      }

      return payload
    },
  }
}

export const httpSearchService = createHttpSearchService()

import {
  BUSINESS_RESULTS,
  DATABASE_RESULTS,
  DEFAULT_INDEX_METADATA,
  EMBEDDING_RESULTS,
  UPLOADED_RESULTS,
} from "../mocks/searchResponses"
import type { SearchResult } from "../types/search"
import type { SearchService } from "./searchService"

const BUSINESS_KEYWORDS = ["业务", "service", "逻辑"]
const DATABASE_KEYWORDS = ["数据库", "repository", "postgresql", "sql"]
const EMBEDDING_KEYWORDS = ["向量", "embedding", "余弦"]
const UPLOADED_KEYWORDS = ["讲义", "上传"]

function includesKeyword(query: string, keywords: string[]): boolean {
  const normalized = query.toLowerCase()
  return keywords.some((keyword) => normalized.includes(keyword))
}

function selectResults(query: string): SearchResult[] {
  if (includesKeyword(query, UPLOADED_KEYWORDS)) {
    return UPLOADED_RESULTS
  }
  if (includesKeyword(query, BUSINESS_KEYWORDS)) {
    return BUSINESS_RESULTS
  }
  if (includesKeyword(query, DATABASE_KEYWORDS)) {
    return DATABASE_RESULTS
  }
  if (includesKeyword(query, EMBEDDING_KEYWORDS)) {
    return EMBEDDING_RESULTS
  }
  return []
}

function wait(delayMs: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, delayMs)
  })
}

export function createMockSearchService(delayMs = 500): SearchService {
  return {
    async search(request) {
      const query = request.query.trim()
      if (request.top_k <= 0) {
        throw new Error("top_k 必须大于 0")
      }
      if (!query) {
        throw new Error("请输入问题")
      }

      await wait(delayMs)
      const results = selectResults(query)
        .slice(0, request.top_k)
        .map((result, index) => ({ ...result, rank: index + 1 }))

      return {
        query,
        elapsed_ms: 420,
        indexed_chunks: DEFAULT_INDEX_METADATA.indexed_chunks,
        model: DEFAULT_INDEX_METADATA.model,
        results,
      }
    },
  }
}

export const mockSearchService = createMockSearchService()

import type { InjectionKey } from "vue"

import type {
  SearchRequest,
  SearchResponse,
  SearchResult,
} from "../types/search"

export interface SearchService {
  search(request: SearchRequest): Promise<SearchResponse>
}

export const searchServiceKey: InjectionKey<SearchService> = Symbol(
  "searchService",
)

function isPageNumber(value: unknown): value is number | null {
  return (
    value === null ||
    (typeof value === "number" && Number.isInteger(value) && value > 0)
  )
}

function isSearchResult(value: unknown): value is SearchResult {
  if (typeof value !== "object" || value === null) {
    return false
  }

  const result = value as Record<string, unknown>
  return (
    typeof result.rank === "number" &&
    typeof result.score === "number" &&
    typeof result.chunk_index === "number" &&
    typeof result.text === "string" &&
    typeof result.document_id === "string" &&
    typeof result.filename === "string" &&
    isPageNumber(result.page_number)
  )
}

export function isSearchResponse(value: unknown): value is SearchResponse {
  if (typeof value !== "object" || value === null) {
    return false
  }

  const response = value as Record<string, unknown>
  return (
    typeof response.query === "string" &&
    typeof response.elapsed_ms === "number" &&
    typeof response.indexed_chunks === "number" &&
    typeof response.model === "string" &&
    Array.isArray(response.results) &&
    response.results.every(isSearchResult)
  )
}

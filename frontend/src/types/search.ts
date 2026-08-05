export interface SearchRequest {
  query: string
  top_k: number
}

export interface SearchResult {
  rank: number
  score: number
  chunk_index: number
  text: string
  document_id: string
  filename: string
  page_number: number | null
  /** Original vector retrieval rank (before optional reranking). */
  retrieval_rank?: number | null
  /** Second-stage reranker score (null when reranker not applied). */
  rerank_score?: number | null
  /** Whether the reranker was applied to this result. */
  reranker_applied?: boolean
}

export interface SearchResponse {
  query: string
  elapsed_ms: number
  indexed_chunks: number
  model: string
  results: SearchResult[]
  /** Whether this request actually used the reranker ordering. */
  reranker_applied?: boolean
  /** Whether the reranker fell back to vector-only ordering for this request. */
  reranker_fallback?: boolean
}

export type SearchState =
  | { status: "idle" }
  | { status: "loading"; query: string }
  | { status: "success"; response: SearchResponse }
  | { status: "empty"; response: SearchResponse }
  | { status: "error"; query: string; message: string }

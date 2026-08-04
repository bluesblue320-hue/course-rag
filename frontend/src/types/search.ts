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
}

export interface SearchResponse {
  query: string
  elapsed_ms: number
  indexed_chunks: number
  model: string
  results: SearchResult[]
}

export type SearchState =
  | { status: "idle" }
  | { status: "loading"; query: string }
  | { status: "success"; response: SearchResponse }
  | { status: "empty"; response: SearchResponse }
  | { status: "error"; query: string; message: string }

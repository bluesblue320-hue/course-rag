export type AnswerStatus = "answered" | "insufficient_context"

export interface AskRequest {
  question: string
  top_k: number
}

export interface AskSource {
  rank: number
  score: number
  text: string
  chunk_index: number
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

export interface AskResponse {
  question: string
  answer: string
  answer_status: AnswerStatus
  max_relevance_score: number | null
  relevance_threshold: number
  retrieval_elapsed_ms: number
  generation_elapsed_ms: number
  total_elapsed_ms: number
  embedding_model: string
  llm_model: string
  sources: AskSource[]
  /** Whether the reranker was applied during retrieval. */
  reranker_applied?: boolean
  /** Whether the reranker fell back to vector-only ordering. */
  reranker_fallback?: boolean
}

export interface AskApiError {
  code: string
  message: string
}

export type AskState =
  | { status: "idle" }
  | { status: "loading"; question: string }
  | { status: "success"; response: AskResponse }
  | {
      status: "error"
      question: string
      message: string
      code?: string
      httpStatus?: number
    }

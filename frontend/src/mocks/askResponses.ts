import type { AskResponse } from "../types/ask"

export const MOCK_ANSWERED_RESPONSE: AskResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责组织和执行核心业务逻辑。[来源1]",
  answer_status: "answered",
  max_relevance_score: 0.8421,
  relevance_threshold: 0.35,
  retrieval_elapsed_ms: 12.4,
  generation_elapsed_ms: 680.7,
  total_elapsed_ms: 693.1,
  embedding_model: "mock/embedding-model",
  llm_model: "mock/llm-model",
  sources: [
    {
      rank: 1,
      score: 0.8421,
      text: "Service 层负责核心业务逻辑，并协调数据访问流程。",
      chunk_index: 2,
      document_id: "builtin-knowledge",
      filename: "knowledge.txt",
      page_number: null,
    },
  ],
}

export const MOCK_INSUFFICIENT_CONTEXT_RESPONSE: AskResponse = {
  question: "今天天气怎么样？",
  answer:
    "当前课程资料中没有足够信息回答这个问题。请尝试换一种问法，或切换到语义检索查看最接近的课程原文。",
  answer_status: "insufficient_context",
  max_relevance_score: 0.121,
  relevance_threshold: 0.35,
  retrieval_elapsed_ms: 8.2,
  generation_elapsed_ms: 0,
  total_elapsed_ms: 8.2,
  embedding_model: "mock/embedding-model",
  llm_model: "mock/llm-model",
  sources: [],
}

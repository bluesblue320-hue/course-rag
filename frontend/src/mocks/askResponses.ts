import type { AskResponse } from "../types/ask"

export const MOCK_ASK_RESPONSE: AskResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责组织和执行核心业务逻辑。[来源1]",
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
    },
  ],
}

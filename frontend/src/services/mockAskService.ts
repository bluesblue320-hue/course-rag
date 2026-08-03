import { MOCK_ASK_RESPONSE } from "../mocks/askResponses"
import type { AskService } from "./askService"

function wait(delayMs: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, delayMs)
  })
}

export function createMockAskService(delayMs = 500): AskService {
  return {
    async ask(request) {
      const question = request.question.trim()
      if (!question) {
        throw new Error("请输入问题")
      }
      if (!Number.isInteger(request.top_k) || request.top_k < 1 || request.top_k > 10) {
        throw new Error("top_k 必须是 1 到 10 之间的整数")
      }

      await wait(delayMs)
      return {
        ...MOCK_ASK_RESPONSE,
        question,
        sources: MOCK_ASK_RESPONSE.sources.map((source) => ({ ...source })),
      }
    },
  }
}

export const mockAskService = createMockAskService()

import {
  MOCK_ANSWERED_RESPONSE,
  MOCK_INSUFFICIENT_CONTEXT_RESPONSE,
} from "../mocks/askResponses"
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
      const normalizedQuestion = question.toLowerCase()
      const shouldRefuse = [
        "天气", "weather", "股票", "stock", "足球", "football",
      ].some((keyword) => normalizedQuestion.includes(keyword))
      const response = shouldRefuse
        ? MOCK_INSUFFICIENT_CONTEXT_RESPONSE
        : MOCK_ANSWERED_RESPONSE

      return {
        ...response,
        question,
        sources: response.sources.map((source) => ({ ...source })),
      }
    },
  }
}

export const mockAskService = createMockAskService()

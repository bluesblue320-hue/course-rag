import { afterEach, describe, expect, it, vi } from "vitest"

import { createMockAskService } from "./mockAskService"

afterEach(() => {
  vi.useRealTimers()
})

describe("createMockAskService", () => {
  it("returns deterministic cloned responses without network access", async () => {
    vi.useFakeTimers()
    const service = createMockAskService(100)
    const firstPending = service.ask({ question: "  自定义问题  ", top_k: 3 })
    await vi.advanceTimersByTimeAsync(100)
    const first = await firstPending
    first.sources[0].text = "changed"

    const secondPending = service.ask({ question: "自定义问题", top_k: 3 })
    await vi.advanceTimersByTimeAsync(100)
    const second = await secondPending

    expect(first.question).toBe("自定义问题")
    expect(first.answer_status).toBe("answered")
    expect(second.sources[0].text).toContain("Service 层")
  })

  it.each(["今天天气如何", "stock prices", "football news"])(
    "returns a deterministic refusal for unrelated question %s",
    async (question) => {
      vi.useFakeTimers()
      const service = createMockAskService(25)
      const pending = service.ask({ question, top_k: 3 })
      await vi.advanceTimersByTimeAsync(25)

      const response = await pending

      expect(response.answer_status).toBe("insufficient_context")
      expect(response.generation_elapsed_ms).toBe(0)
      expect(response.relevance_threshold).toBe(0.35)
    },
  )
})

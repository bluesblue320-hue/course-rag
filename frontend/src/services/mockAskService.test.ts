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
    expect(second.sources[0].text).toContain("Service 层")
  })
})

import { describe, expect, it, vi } from "vitest"

import type { AskService } from "../services/askService"
import { AskServiceError } from "../services/httpAskService"
import type { AskResponse } from "../types/ask"
import { useKnowledgeAsk } from "./useKnowledgeAsk"

const response: AskResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责核心业务逻辑。[来源1]",
  retrieval_elapsed_ms: 12.4,
  generation_elapsed_ms: 680.7,
  total_elapsed_ms: 693.1,
  embedding_model: "embedding-model",
  llm_model: "llm-model",
  sources: [
    {
      rank: 1,
      score: 0.8421,
      text: "相关课程原文",
      chunk_index: 2,
    },
  ],
}

describe("useKnowledgeAsk", () => {
  it("starts idle", () => {
    const controller = useKnowledgeAsk({ ask: vi.fn() })
    expect(controller.state.value).toEqual({ status: "idle" })
    expect(controller.isLoading.value).toBe(false)
  })

  it("moves from idle through loading to success with a trimmed question", async () => {
    let resolveResponse!: (value: AskResponse) => void
    const askMethod = vi.fn(
      () =>
        new Promise<AskResponse>((resolve) => {
          resolveResponse = resolve
        }),
    )
    const controller = useKnowledgeAsk({ ask: askMethod })

    const pending = controller.ask("  Service 层负责什么？  ")
    expect(controller.state.value).toEqual({
      status: "loading",
      question: "Service 层负责什么？",
    })
    expect(controller.isLoading.value).toBe(true)

    resolveResponse(response)
    await pending

    expect(askMethod).toHaveBeenCalledWith({
      question: "Service 层负责什么？",
      top_k: 3,
    })
    expect(controller.state.value).toEqual({ status: "success", response })
  })

  it("moves from loading to error and preserves details", async () => {
    const error = new AskServiceError(
      "问答服务尚未完成配置，你仍可使用语义检索",
      "LLM_NOT_CONFIGURED",
      503,
    )
    const controller = useKnowledgeAsk({
      ask: vi.fn().mockRejectedValue(error),
    })

    await controller.ask("Service 层负责什么？")

    expect(controller.state.value).toEqual({
      status: "error",
      question: "Service 层负责什么？",
      message: error.message,
      code: "LLM_NOT_CONFIGURED",
      httpStatus: 503,
    })
  })

  it("rejects a blank question without calling the service", async () => {
    const askMethod = vi.fn()
    const controller = useKnowledgeAsk({ ask: askMethod })

    await controller.ask("   ")

    expect(askMethod).not.toHaveBeenCalled()
    expect(controller.state.value).toEqual({
      status: "error",
      question: "",
      message: "请输入问题",
    })
  })

  it("blocks a second request while loading", async () => {
    let resolveResponse!: (value: AskResponse) => void
    const askMethod = vi.fn(
      () =>
        new Promise<AskResponse>((resolve) => {
          resolveResponse = resolve
        }),
    )
    const controller = useKnowledgeAsk({ ask: askMethod })

    const first = controller.ask("第一个问题")
    await controller.ask("第二个问题")

    expect(askMethod).toHaveBeenCalledOnce()
    expect(askMethod).toHaveBeenCalledWith({ question: "第一个问题", top_k: 3 })
    resolveResponse({ ...response, question: "第一个问题" })
    await first
  })

  it("normalizes unknown failures without an unhandled rejection", async () => {
    const controller = useKnowledgeAsk({
      ask: vi.fn().mockRejectedValue("offline"),
    })

    await expect(controller.ask("问题")).resolves.toBeUndefined()
    expect(controller.state.value).toEqual({
      status: "error",
      question: "问题",
      message: "问答失败，请稍后重试",
    })
  })

  it("rejects malformed successful service data", async () => {
    const service = {
      ask: vi.fn().mockResolvedValue({ ...response, total_elapsed_ms: Infinity }),
    } as unknown as AskService
    const controller = useKnowledgeAsk(service)

    await controller.ask("问题")

    expect(controller.state.value).toMatchObject({
      status: "error",
      question: "问题",
      code: "INVALID_RESPONSE",
    })
  })

  it("retries the last valid question", async () => {
    const askMethod = vi
      .fn()
      .mockRejectedValueOnce(new Error("暂时失败"))
      .mockResolvedValueOnce(response)
    const controller = useKnowledgeAsk({ ask: askMethod })

    await controller.ask("  Service 层负责什么？  ")
    await controller.retry()

    expect(askMethod).toHaveBeenCalledTimes(2)
    expect(askMethod).toHaveBeenNthCalledWith(2, {
      question: "Service 层负责什么？",
      top_k: 3,
    })
    expect(controller.state.value.status).toBe("success")
  })

  it("does not retry without a previous valid question", async () => {
    const askMethod = vi.fn()
    const controller = useKnowledgeAsk({ ask: askMethod })

    await controller.retry()

    expect(askMethod).not.toHaveBeenCalled()
  })

  it("resets to idle", async () => {
    const controller = useKnowledgeAsk({
      ask: vi.fn().mockResolvedValue(response),
    })
    await controller.ask("问题")

    controller.reset()

    expect(controller.state.value).toEqual({ status: "idle" })
  })
})

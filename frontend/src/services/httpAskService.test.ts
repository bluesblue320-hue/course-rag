import { describe, expect, it, vi } from "vitest"

import { AskServiceError } from "./askService"
import { createHttpAskService } from "./httpAskService"

const validResponse = {
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

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("createHttpAskService", () => {
  it("posts a trimmed request to /api/ask and returns guarded data", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(validResponse))
    const service = createHttpAskService("/api/", fetchImpl)

    await expect(
      service.ask({ question: "  Service 层负责什么？  ", top_k: 3 }),
    ).resolves.toEqual(validResponse)

    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(fetchImpl).toHaveBeenCalledWith("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: "Service 层负责什么？",
        top_k: 3,
      }),
    })
  })

  it.each([
    ["blank question", { question: "   ", top_k: 3 }],
    ["zero top_k", { question: "问题", top_k: 0 }],
    ["top_k above ten", { question: "问题", top_k: 11 }],
    ["fractional top_k", { question: "问题", top_k: 1.5 }],
  ])("rejects %s before fetch", async (_name, request) => {
    const fetchImpl = vi.fn()
    const service = createHttpAskService("/api", fetchImpl)

    await expect(service.ask(request)).rejects.toBeInstanceOf(AskServiceError)
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it.each([
    [422, "问题格式不正确，请检查后重试", "VALIDATION_ERROR"],
    [502, "回答生成失败，请稍后重试", "GENERATION_FAILED"],
    [503, "问答服务尚未完成配置，你仍可使用语义检索", "LLM_NOT_CONFIGURED"],
    [500, "问答服务暂时不可用", "SERVICE_UNAVAILABLE"],
  ])("maps HTTP %i to a stable error", async (status, message, code) => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({}, status))
    const service = createHttpAskService("/api", fetchImpl)

    const error = await service
      .ask({ question: "问题", top_k: 3 })
      .catch((caught: unknown) => caught)

    expect(error).toBeInstanceOf(AskServiceError)
    expect(error).toMatchObject({ message, code, httpStatus: status })
  })

  it("maps fetch failures without exposing the underlying error", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new TypeError("secret endpoint"))
    const service = createHttpAskService("/api", fetchImpl)

    await expect(
      service.ask({ question: "问题", top_k: 3 }),
    ).rejects.toMatchObject({
      message: "无法连接后端服务，请确认 FastAPI 已启动",
      code: "NETWORK_ERROR",
    })
  })

  it("rejects a non-JSON success response", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      new Response("not json", {
        status: 200,
        headers: { "Content-Type": "text/plain" },
      }),
    )
    const service = createHttpAskService("/api", fetchImpl)

    await expect(
      service.ask({ question: "问题", top_k: 3 }),
    ).rejects.toThrow("问答服务返回的数据格式不正确")
  })

  it("rejects malformed JSON success data", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse({ ...validResponse, sources: [{ rank: "1" }] }),
    )
    const service = createHttpAskService("/api", fetchImpl)

    await expect(
      service.ask({ question: "问题", top_k: 3 }),
    ).rejects.toThrow("问答服务返回的数据格式不正确")
  })

  it("does not expose an HTML error page", async () => {
    const html = "<html>private proxy stack trace</html>"
    const fetchImpl = vi.fn().mockResolvedValue(
      new Response(html, {
        status: 502,
        headers: { "Content-Type": "text/html" },
      }),
    )
    const service = createHttpAskService("/api", fetchImpl)

    const error = await service
      .ask({ question: "问题", top_k: 3 })
      .catch((caught: unknown) => caught)

    expect(error).toMatchObject({ message: "回答生成失败，请稍后重试" })
    expect(String(error)).not.toContain("private proxy stack trace")
  })

  it("does not trust an internal backend message", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse(
        { code: "GENERATION_FAILED", message: "api key abc123 leaked" },
        502,
      ),
    )
    const service = createHttpAskService("/api", fetchImpl)

    const error = await service
      .ask({ question: "问题", top_k: 3 })
      .catch((caught: unknown) => caught)

    expect(error).toMatchObject({ message: "回答生成失败，请稍后重试" })
    expect(String(error)).not.toContain("abc123")
  })
})

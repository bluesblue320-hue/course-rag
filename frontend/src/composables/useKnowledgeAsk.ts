import { computed, ref } from "vue"

import { isAskResponse, type AskService } from "../services/askService"
import { AskServiceError } from "../services/httpAskService"
import type { AskState } from "../types/ask"

function normalizedError(error: unknown): {
  message: string
  code?: string
  httpStatus?: number
} {
  if (error instanceof AskServiceError) {
    return {
      message: error.message,
      code: error.code,
      httpStatus: error.httpStatus,
    }
  }
  if (error instanceof Error && error.message) {
    return { message: error.message }
  }
  return { message: "问答失败，请稍后重试" }
}

export function useKnowledgeAsk(service: AskService) {
  const state = ref<AskState>({ status: "idle" })
  const lastValidQuestion = ref("")
  const isLoading = computed(() => state.value.status === "loading")

  async function ask(rawQuestion: string): Promise<void> {
    if (isLoading.value) {
      return
    }

    const question = rawQuestion.trim()
    if (!question) {
      state.value = {
        status: "error",
        question: "",
        message: "请输入问题",
      }
      return
    }

    lastValidQuestion.value = question
    state.value = { status: "loading", question }

    try {
      const response = await service.ask({ question, top_k: 3 })
      if (!isAskResponse(response)) {
        throw new AskServiceError(
          "问答服务返回的数据格式不正确",
          "INVALID_RESPONSE",
        )
      }
      state.value = { status: "success", response }
    } catch (error) {
      state.value = {
        status: "error",
        question,
        ...normalizedError(error),
      }
    }
  }

  async function retry(): Promise<void> {
    if (lastValidQuestion.value) {
      await ask(lastValidQuestion.value)
    }
  }

  function reset(): void {
    state.value = { status: "idle" }
  }

  return {
    state,
    isLoading,
    ask,
    retry,
    reset,
  }
}

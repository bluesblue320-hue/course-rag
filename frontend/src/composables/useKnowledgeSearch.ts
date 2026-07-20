import { computed, ref } from "vue"

import {
  isSearchResponse,
  type SearchService,
} from "../services/searchService"
import type { SearchState } from "../types/search"

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }
  return "搜索失败，请稍后重试"
}

export function useKnowledgeSearch(service: SearchService) {
  const state = ref<SearchState>({ status: "idle" })
  const lastQuery = ref("")
  const isLoading = computed(() => state.value.status === "loading")

  async function search(rawQuery: string): Promise<void> {
    const query = rawQuery.trim()
    if (!query) {
      state.value = {
        status: "error",
        query: "",
        message: "请输入问题",
      }
      return
    }

    lastQuery.value = query
    state.value = { status: "loading", query }

    try {
      const response = await service.search({ query, top_k: 3 })
      if (!isSearchResponse(response)) {
        throw new Error("搜索结果格式不正确")
      }
      state.value =
        response.results.length > 0
          ? { status: "success", response }
          : { status: "empty", response }
    } catch (error) {
      state.value = {
        status: "error",
        query,
        message: errorMessage(error),
      }
    }
  }

  async function retry(): Promise<void> {
    if (lastQuery.value) {
      await search(lastQuery.value)
    }
  }

  return {
    state,
    isLoading,
    search,
    retry,
  }
}

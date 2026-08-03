<script setup lang="ts">
import { computed } from "vue"

import type { AskState } from "../types/ask"

const props = defineProps<{
  state: AskState
}>()

const emit = defineEmits<{
  retry: []
}>()

const errorMessage = computed(() => {
  if (props.state.status !== "error") {
    return ""
  }
  if (props.state.code === "LLM_NOT_CONFIGURED") {
    return "问答服务尚未完成配置，你仍可以切换到“语义检索”查看相关原文。"
  }
  if (props.state.code === "GENERATION_FAILED") {
    return "回答生成失败，请稍后重试。"
  }
  if (props.state.code === "NETWORK_ERROR") {
    return "无法连接后端服务，请确认 FastAPI 已启动。"
  }
  return props.state.message
})
</script>

<template>
  <div
    v-if="state.status === 'loading'"
    class="status status-loading"
    role="status"
    aria-live="polite"
  >
    正在检索课程资料并生成回答……
  </div>

  <div
    v-else-if="state.status === 'error'"
    class="status status-error"
    role="alert"
  >
    <span>{{ errorMessage }}</span>
    <button type="button" @click="emit('retry')">重试</button>
  </div>
</template>

<style scoped>
.status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  min-height: 58px;
  padding: var(--space-4) var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}

.status-loading {
  color: var(--color-muted);
  background: var(--color-surface);
}

.status-error {
  color: var(--color-danger);
  border-color: #f2b8b5;
  background: var(--color-danger-soft);
}

button {
  flex: 0 0 auto;
  padding: 8px 12px;
  border: 1px solid currentcolor;
  border-radius: var(--radius-sm);
  color: inherit;
  background: transparent;
  font-weight: 700;
}

@media (max-width: 560px) {
  .status {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>

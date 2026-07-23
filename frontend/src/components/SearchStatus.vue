<script setup lang="ts">
import type { SearchState } from "../types/search"

defineProps<{
  state: SearchState
}>()

const emit = defineEmits<{
  retry: []
}>()
</script>

<template>
  <div
    v-if="state.status === 'loading'"
    class="status status-loading"
    role="status"
    aria-live="polite"
  >
    正在检索“{{ state.query }}”…
  </div>

  <div
    v-else-if="state.status === 'success'"
    class="status status-success"
    role="status"
    aria-live="polite"
  >
    <strong>找到 {{ state.response.results.length }} 条相关内容</strong>
    <span>检索耗时 {{ (state.response.elapsed_ms / 1000).toFixed(2) }} 秒</span>
  </div>

  <div
    v-else-if="state.status === 'empty'"
    class="status status-empty"
    role="status"
    aria-live="polite"
  >
    没有找到相关内容，请换一种问法。
  </div>

  <div
    v-else-if="state.status === 'error'"
    class="status status-error"
    role="alert"
  >
    <span>{{ state.message }}</span>
    <button type="button" @click="emit('retry')">重新检索</button>
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

.status-loading,
.status-empty {
  color: var(--color-muted);
  background: var(--color-surface);
}

.status-success {
  color: var(--color-success);
  border-color: #8dd5c0;
  background: var(--color-success-soft);
}

.status-error {
  color: var(--color-danger);
  border-color: #f2b8b5;
  background: var(--color-danger-soft);
}

button {
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

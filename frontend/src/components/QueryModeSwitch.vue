<script setup lang="ts">
import type { QueryMode } from "../types/queryMode"

const props = withDefaults(
  defineProps<{
    modelValue: QueryMode
    disabled?: boolean
  }>(),
  { disabled: false },
)

const emit = defineEmits<{
  "update:modelValue": [mode: QueryMode]
}>()

function select(mode: QueryMode): void {
  if (!props.disabled && mode !== props.modelValue) {
    emit("update:modelValue", mode)
  }
}
</script>

<template>
  <div class="mode-switch" role="group" aria-label="问题处理模式">
    <button
      type="button"
      :class="{ active: modelValue === 'ask' }"
      :aria-pressed="modelValue === 'ask'"
      :disabled="disabled"
      @click="select('ask')"
    >
      <strong>智能问答</strong>
      <span>生成带来源的答案</span>
    </button>
    <button
      type="button"
      :class="{ active: modelValue === 'search' }"
      :aria-pressed="modelValue === 'search'"
      :disabled="disabled"
      @click="select('search')"
    >
      <strong>语义检索</strong>
      <span>查看 Top-K 课程原文</span>
    </button>
  </div>
</template>

<style scoped>
.mode-switch {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
}

button {
  display: grid;
  gap: var(--space-1);
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-text);
  background: var(--color-surface);
  text-align: left;
}

button.active {
  border: 2px solid var(--color-primary);
  background: #f5f7ff;
  box-shadow: 0 0 0 2px rgb(49 85 198 / 8%);
}

button.active strong::before {
  content: "✓ ";
}

button span {
  color: var(--color-muted);
  font-size: 0.86rem;
}

button:disabled {
  opacity: 0.65;
}

@media (max-width: 560px) {
  .mode-switch {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>

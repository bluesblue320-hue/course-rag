<script setup lang="ts">
import { computed, ref } from "vue"

import type { QueryMode } from "../types/queryMode"

const props = withDefaults(
  defineProps<{
    mode?: QueryMode
    loading?: boolean
  }>(),
  {
    mode: "ask",
    loading: false,
  },
)

const emit = defineEmits<{
  submit: [query: string]
}>()

const query = ref("")
const validationMessage = ref("")
const describedBy = computed(() =>
  validationMessage.value ? "query-hint query-error" : "query-hint",
)
const copy = computed(() =>
  props.mode === "ask"
    ? {
        label: "向课程知识库提问",
        hint: "按 Enter 生成回答，Shift + Enter 换行",
        idleButton: "生成回答",
        loadingButton: "生成中…",
      }
    : {
        label: "检索课程知识",
        hint: "按 Enter 检索，Shift + Enter 换行",
        idleButton: "开始检索",
        loadingButton: "检索中…",
      },
)

function submit(): void {
  const trimmed = query.value.trim()
  if (!trimmed) {
    validationMessage.value = "请输入问题"
    return
  }
  validationMessage.value = ""
  emit("submit", trimmed)
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}
</script>

<template>
  <form class="search-form" @submit.prevent="submit">
    <label for="knowledge-query">{{ copy.label }}</label>
    <textarea
      id="knowledge-query"
      v-model="query"
      name="query"
      rows="4"
      placeholder="例如：业务逻辑应该写在哪一层？"
      :disabled="props.loading"
      :aria-describedby="describedBy"
      :aria-invalid="Boolean(validationMessage)"
      @input="validationMessage = ''"
      @keydown="handleKeydown"
    />

    <div class="form-footer">
      <span id="query-hint">{{ copy.hint }}</span>
      <button type="submit" :disabled="props.loading">
        {{ props.loading ? copy.loadingButton : copy.idleButton }}
      </button>
    </div>

    <p v-if="validationMessage" id="query-error" role="alert">
      {{ validationMessage }}
    </p>
  </form>
</template>

<style scoped>
.search-form {
  display: grid;
  gap: var(--space-3);
  padding: var(--space-6);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  box-shadow: var(--shadow-card);
}

label {
  font-size: 1.05rem;
  font-weight: 700;
}

textarea {
  width: 100%;
  min-height: 128px;
  resize: vertical;
  padding: var(--space-4);
  border: 1px solid #b9c2d3;
  border-radius: var(--radius-sm);
  color: var(--color-text);
  background: var(--color-surface);
  line-height: 1.6;
}

textarea:focus {
  border-color: var(--color-primary);
}

.form-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
}

.form-footer span {
  color: var(--color-muted);
  font-size: 0.9rem;
}

button {
  min-width: 136px;
  padding: 12px 22px;
  border: 0;
  border-radius: var(--radius-sm);
  color: white;
  background: var(--color-primary);
  font-weight: 700;
}

button:hover:not(:disabled) {
  background: var(--color-primary-dark);
}

button:disabled {
  opacity: 0.65;
}

[role="alert"] {
  margin: 0;
  color: var(--color-danger);
}

@media (max-width: 560px) {
  .form-footer {
    align-items: stretch;
    flex-direction: column;
  }

  button {
    width: 100%;
  }
}
</style>

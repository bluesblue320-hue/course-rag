<script setup lang="ts">
import { computed } from "vue"

import type { AskResponse } from "../types/ask"

const props = defineProps<{
  response: AskResponse
}>()

const isInsufficient = computed(
  () => props.response.answer_status === "insufficient_context",
)

function formatMilliseconds(value: number): string {
  return `${value} ms`
}

function formatRelevance(value: number | null): string {
  return value === null ? "无检索结果" : value.toFixed(4)
}
</script>

<template>
  <article class="answer-card" aria-labelledby="answer-heading">
    <header>
      <p class="answer-label">
        {{ isInsufficient ? "未调用 LLM" : "已基于课程资料生成" }}
      </p>
      <h2 id="answer-heading">{{ isInsufficient ? "当前资料不足" : "AI 回答" }}</h2>
      <p class="question">{{ response.question }}</p>
    </header>

    <p class="answer-text">{{ response.answer }}</p>

    <dl class="timings" aria-label="回答耗时">
      <div>
        <dt>检索耗时</dt>
        <dd>{{ formatMilliseconds(response.retrieval_elapsed_ms) }}</dd>
      </div>
      <div>
        <dt>生成耗时</dt>
        <dd>{{ formatMilliseconds(response.generation_elapsed_ms) }}</dd>
      </div>
      <div>
        <dt>总耗时</dt>
        <dd>{{ formatMilliseconds(response.total_elapsed_ms) }}</dd>
      </div>
    </dl>

    <dl class="models" aria-label="回答模型">
      <div>
        <dt>Embedding 模型</dt>
        <dd>{{ response.embedding_model }}</dd>
      </div>
      <div>
        <dt>LLM 模型</dt>
        <dd>{{ response.llm_model }}</dd>
      </div>
    </dl>

    <dl class="relevance" aria-label="相关性判断">
      <div>
        <dt>最高相似度</dt>
        <dd>{{ formatRelevance(response.max_relevance_score) }}</dd>
      </div>
      <div>
        <dt>生成阈值</dt>
        <dd>{{ response.relevance_threshold.toFixed(4) }}</dd>
      </div>
    </dl>
  </article>
</template>

<style scoped>
.answer-card {
  display: grid;
  gap: var(--space-5);
  padding: var(--space-6);
  border: 1px solid #b9c7f0;
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  box-shadow: var(--shadow-card);
}

header p,
header h2 {
  margin: 0;
}

.answer-label {
  color: var(--color-primary);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.1em;
}

h2 {
  margin-top: var(--space-1);
}

.question {
  margin-top: var(--space-2);
  color: var(--color-muted);
}

.answer-text {
  margin: 0;
  padding: var(--space-5);
  border-left: 4px solid var(--color-primary);
  border-radius: var(--radius-sm);
  background: var(--color-surface-soft);
  line-height: 1.75;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

dl {
  margin: 0;
}

.timings {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--space-3);
}

.timings div,
.models div,
.relevance div {
  min-width: 0;
  padding: var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
}

.models,
.relevance {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
}

dt {
  color: var(--color-muted);
  font-size: 0.8rem;
}

dd {
  margin: var(--space-1) 0 0;
  overflow-wrap: anywhere;
  font-weight: 700;
}

@media (max-width: 560px) {
  .timings,
  .models,
  .relevance {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>

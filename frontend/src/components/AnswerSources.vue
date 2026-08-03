<script setup lang="ts">
import type { AnswerStatus, AskSource } from "../types/ask"
import AnswerSourceCard from "./AnswerSourceCard.vue"

defineProps<{
  sources: AskSource[]
  answerStatus: AnswerStatus
}>()
</script>

<template>
  <section class="answer-sources" aria-labelledby="sources-heading">
    <h2 id="sources-heading">
      {{ answerStatus === "answered" ? "引用来源" : "检索候选" }}（{{ sources.length }}）
    </h2>
    <p v-if="answerStatus === 'insufficient_context'" class="source-note">
      以下内容是最接近的检索结果，但最高相似度未达到生成阈值，因此未交给 LLM 生成答案。
    </p>
    <div v-if="sources.length > 0" class="source-list">
      <AnswerSourceCard
        v-for="(source, index) in sources"
        :key="`${source.rank}-${source.chunk_index}-${index}`"
        :source="source"
      />
    </div>
    <p v-else class="empty-sources">
      {{ answerStatus === "answered" ? "本次回答没有返回引用来源。" : "没有检索到课程资料候选。" }}
    </p>
  </section>
</template>

<style scoped>
.answer-sources {
  display: grid;
  gap: var(--space-3);
}

h2 {
  margin: 0;
  font-size: 1.2rem;
}

.source-list {
  display: grid;
  gap: var(--space-3);
}

.source-note {
  margin: 0;
  padding: var(--space-3);
  border-left: 3px solid var(--color-primary);
  color: var(--color-muted);
  background: var(--color-surface-soft);
}

.empty-sources {
  margin: 0;
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-muted);
  background: var(--color-surface);
  text-align: center;
}
</style>

<script setup lang="ts">
import type { AskSource } from "../types/ask"
import AnswerSourceCard from "./AnswerSourceCard.vue"

defineProps<{
  sources: AskSource[]
}>()
</script>

<template>
  <section class="answer-sources" aria-labelledby="sources-heading">
    <h2 id="sources-heading">引用来源（{{ sources.length }}）</h2>
    <div v-if="sources.length > 0" class="source-list">
      <AnswerSourceCard
        v-for="(source, index) in sources"
        :key="`${source.rank}-${source.chunk_index}-${index}`"
        :source="source"
      />
    </div>
    <p v-else class="empty-sources">本次回答没有返回引用来源。</p>
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

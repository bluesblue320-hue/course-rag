<script setup lang="ts">
import type { AskSource } from "../types/ask"

defineProps<{
  source: AskSource
}>()
</script>

<template>
  <details class="source-card" open>
    <summary>
      <strong>[来源{{ source.rank }}]</strong>
      <span>相似度：{{ source.score.toFixed(4) }}</span>
      <span>Chunk #{{ source.chunk_index }}</span>
    </summary>
    <p>{{ source.text }}</p>
  </details>
</template>

<style scoped>
.source-card {
  padding: var(--space-4) var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
}

summary {
  cursor: pointer;
}

summary strong,
summary span {
  margin-right: var(--space-3);
}

summary strong {
  color: var(--color-primary);
}

summary span {
  color: var(--color-muted);
  font-size: 0.88rem;
}

p {
  margin: var(--space-4) 0 0;
  line-height: 1.65;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}
</style>

<script setup lang="ts">
import type { SearchResult } from "../types/search"

defineProps<{
  result: SearchResult
}>()
</script>

<template>
  <article class="result-card">
    <div class="rank" aria-hidden="true">{{ result.rank }}</div>
    <div class="result-content">
      <div class="metadata">
        <span class="top-label">TOP {{ result.rank }}</span>
        <span class="file">{{ result.filename }}</span>
        <span v-if="result.page_number !== null" class="page">
          第 {{ result.page_number }} 页
        </span>
        <span class="score">相似度 {{ result.score.toFixed(4) }}</span>
        <span class="chunk">Chunk #{{ result.chunk_index }}</span>
      </div>
      <p>{{ result.text }}</p>
    </div>
  </article>
</template>

<style scoped>
.result-card {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr);
  gap: var(--space-4);
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  box-shadow: 0 4px 16px rgb(23 33 61 / 6%);
}

.rank {
  display: grid;
  width: 36px;
  height: 36px;
  place-items: center;
  border-radius: 999px;
  color: white;
  background: var(--color-primary);
  font-weight: 800;
}

.result-content {
  min-width: 0;
}

.metadata {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.metadata span {
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.88rem;
}

.top-label {
  color: var(--color-primary);
  background: #eef3ff;
}

.file {
  color: var(--color-text);
  background: #f1f3f7;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.page {
  color: var(--color-text);
  background: #f1f3f7;
}

.score {
  color: var(--color-warning);
  background: var(--color-warning-soft);
}

.chunk {
  color: var(--color-muted);
  background: #f1f3f7;
}

p {
  margin: var(--space-3) 0 0;
  line-height: 1.65;
}
</style>

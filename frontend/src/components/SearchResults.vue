<script setup lang="ts">
import type { SearchState } from "../types/search"
import SearchResultCard from "./SearchResultCard.vue"

defineProps<{
  state: SearchState
}>()
</script>

<template>
  <section
    v-if="state.status === 'loading' || state.status === 'success' || state.status === 'empty'"
    class="results"
    aria-label="语义检索结果"
  >
    <template v-if="state.status === 'loading'">
      <div
        v-for="position in 3"
        :key="position"
        class="result-skeleton"
        aria-hidden="true"
      />
    </template>
    <template v-else-if="state.status === 'success'">
      <header class="result-summary">
        <p>找到 {{ state.response.results.length }} 条相关结果</p>
        <dl>
          <div>
            <dt>耗时</dt>
            <dd>{{ state.response.elapsed_ms }} ms</dd>
          </div>
          <div>
            <dt>模型</dt>
            <dd>{{ state.response.model }}</dd>
          </div>
          <div>
            <dt>已索引分块</dt>
            <dd>{{ state.response.indexed_chunks }}</dd>
          </div>
        </dl>
      </header>
      <SearchResultCard
        v-for="result in state.response.results"
        :key="result.rank"
        :result="result"
      />
    </template>
    <p v-else class="empty-message">未找到相关结果，请尝试调整查询内容。</p>
  </section>
</template>

<style scoped>
.results {
  display: grid;
  gap: var(--space-3);
}

.result-summary {
  display: grid;
  gap: var(--space-2);
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
}

.result-summary p {
  margin: 0;
  font-weight: 700;
}

dl {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  margin: 0;
}

dl div {
  display: flex;
  gap: var(--space-1);
  color: var(--color-muted);
  font-size: 0.88rem;
}

dt::after {
  content: "：";
}

dd {
  margin: 0;
  color: inherit;
}

.result-skeleton {
  min-height: 128px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background:
    linear-gradient(
      90deg,
      rgb(240 242 247) 25%,
      rgb(248 249 252) 50%,
      rgb(240 242 247) 75%
    );
  background-size: 200% 100%;
  animation: shimmer 1.2s infinite linear;
}

.empty-message {
  margin: 0;
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-muted);
  background: var(--color-surface);
  text-align: center;
}

@keyframes shimmer {
  to {
    background-position-x: -200%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .result-skeleton {
    animation: none;
  }
}
</style>

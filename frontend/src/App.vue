<script setup lang="ts">
import { computed, inject } from "vue"

import AppSidebar from "./components/AppSidebar.vue"
import IndexSummary from "./components/IndexSummary.vue"
import RetrievalExplanation from "./components/RetrievalExplanation.vue"
import SearchForm from "./components/SearchForm.vue"
import SearchResults from "./components/SearchResults.vue"
import SearchStatus from "./components/SearchStatus.vue"
import { useKnowledgeSearch } from "./composables/useKnowledgeSearch"
import { DEFAULT_INDEX_METADATA } from "./mocks/searchResponses"
import { searchServiceKey } from "./services/searchService"

const searchService = inject(searchServiceKey)
if (!searchService) {
  throw new Error("SearchService 未注入")
}

const { state, isLoading, search, retry } =
  useKnowledgeSearch(searchService)

const responseMetadata = computed(() => {
  if (state.value.status === "success" || state.value.status === "empty") {
    return {
      indexed_chunks: state.value.response.indexed_chunks,
      model: state.value.response.model,
      top_k: DEFAULT_INDEX_METADATA.top_k,
    }
  }
  return DEFAULT_INDEX_METADATA
})
</script>

<template>
  <div class="app-shell">
    <AppSidebar />

    <main>
      <header class="page-header">
        <p class="eyebrow">SEMANTIC RETRIEVAL</p>
        <h1>课程知识检索</h1>
        <p>输入问题，查看语义最相关的课程原文。</p>
      </header>

      <SearchForm :loading="isLoading" @submit="search" />
      <SearchStatus :state="state" @retry="retry" />
      <SearchResults :state="state" />
    </main>

    <aside class="context-panel" aria-label="索引与检索说明">
      <IndexSummary
        :indexed-chunks="responseMetadata.indexed_chunks"
        :top-k="responseMetadata.top_k"
        :model="responseMetadata.model"
      />
      <RetrievalExplanation />
    </aside>
  </div>
</template>

<style scoped>
.app-shell {
  display: grid;
  min-height: 100vh;
  grid-template-columns: 220px minmax(0, 1fr) 300px;
}

main {
  display: grid;
  align-content: start;
  gap: var(--space-5);
  min-width: 0;
  padding: 56px clamp(24px, 4vw, 64px);
}

.page-header {
  margin-bottom: var(--space-2);
}

.page-header h1 {
  margin: var(--space-1) 0 var(--space-2);
  font-size: clamp(2rem, 4vw, 3rem);
  line-height: 1.15;
}

.page-header p {
  margin: 0;
  color: var(--color-muted);
}

.page-header .eyebrow {
  color: var(--color-primary);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.12em;
}

.context-panel {
  display: grid;
  align-content: start;
  gap: var(--space-5);
  min-width: 0;
  padding: 56px var(--space-6) 56px 0;
}

@media (max-width: 1180px) {
  .app-shell {
    grid-template-columns: 200px minmax(0, 1fr) 260px;
  }

  main {
    padding-inline: var(--space-6);
  }
}

@media (max-width: 900px) {
  .app-shell {
    grid-template-columns: minmax(0, 1fr);
  }

  main {
    padding: var(--space-8) var(--space-5);
  }

  .context-panel {
    padding: 0 var(--space-5) var(--space-8);
  }
}
</style>

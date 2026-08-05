<script setup lang="ts">
import { computed, inject, ref } from "vue"

import AnswerCard from "./components/AnswerCard.vue"
import AnswerSources from "./components/AnswerSources.vue"
import AppSidebar from "./components/AppSidebar.vue"
import AskStatus from "./components/AskStatus.vue"
import IndexSummary from "./components/IndexSummary.vue"
import KnowledgeBasePanel from "./components/KnowledgeBasePanel.vue"
import QueryModeSwitch from "./components/QueryModeSwitch.vue"
import RetrievalExplanation from "./components/RetrievalExplanation.vue"
import SearchForm from "./components/SearchForm.vue"
import SearchResults from "./components/SearchResults.vue"
import SearchStatus from "./components/SearchStatus.vue"
import StudyGuide from "./components/StudyGuide.vue"
import { useKnowledgeAsk } from "./composables/useKnowledgeAsk"
import { useKnowledgeSearch } from "./composables/useKnowledgeSearch"
import { DEFAULT_INDEX_METADATA } from "./mocks/searchResponses"
import { askServiceKey } from "./services/askService"
import { searchServiceKey } from "./services/searchService"
import type { QueryMode } from "./types/queryMode"

const searchService = inject(searchServiceKey)
if (!searchService) {
  throw new Error("SearchService 未注入")
}
const askService = inject(askServiceKey)
if (!askService) {
  throw new Error("AskService 未注入")
}

const mode = ref<QueryMode>("ask")
const {
  state: searchState,
  isLoading: isSearchLoading,
  search,
  retry: retrySearch,
} = useKnowledgeSearch(searchService)
const {
  state: askState,
  isLoading: isAskLoading,
  ask,
  retry: retryAsk,
} = useKnowledgeAsk(askService)

const isBusy = computed(
  () => isSearchLoading.value || isAskLoading.value,
)

const pageCopy = computed(() => {
  if (mode.value === "ask") {
    return {
      eyebrow: "RAG QUESTION ANSWERING",
      title: "课程知识问答",
      description: "根据课程资料生成带引用来源的回答。",
    }
  }
  if (mode.value === "documents") {
    return {
      eyebrow: "KNOWLEDGE BASE",
      title: "知识库管理",
      description: "上传、查看和删除课程资料，内容会立即参与检索与问答。",
    }
  }
  if (mode.value === "guide") {
    return {
      eyebrow: "LEARNING GUIDE",
      title: "学习说明",
      description: "了解如何整理资料、提出问题并判断回答是否可靠。",
    }
  }

  return {
    eyebrow: "SEMANTIC RETRIEVAL",
    title: "课程知识检索",
    description: "输入问题，查看语义最相关的课程原文。",
  }
})

const responseMetadata = computed(() => {
  if (
    mode.value === "search" &&
    (searchState.value.status === "success" ||
      searchState.value.status === "empty")
  ) {
    return {
      indexed_chunks: searchState.value.response.indexed_chunks,
      model: searchState.value.response.model,
      top_k: DEFAULT_INDEX_METADATA.top_k,
    }
  }
  if (mode.value === "ask" && askState.value.status === "success") {
    return {
      ...DEFAULT_INDEX_METADATA,
      model: askState.value.response.embedding_model,
    }
  }
  return DEFAULT_INDEX_METADATA
})

async function submitQuestion(question: string): Promise<void> {
  if (mode.value === "ask") {
    await ask(question)
    return
  }
  if (mode.value === "search") {
    await search(question)
  }
}
</script>

<template>
  <div class="app-shell">
    <AppSidebar v-model="mode" :disabled="isBusy" />

    <main>
      <header class="page-header">
        <p class="eyebrow">{{ pageCopy.eyebrow }}</p>
        <h1>{{ pageCopy.title }}</h1>
        <p>{{ pageCopy.description }}</p>
      </header>

      <QueryModeSwitch v-if="mode !== 'guide'" v-model="mode" :disabled="isBusy" />
      <SearchForm
        v-show="mode === 'ask' || mode === 'search'"
        :mode="mode"
        :loading="isBusy"
        @submit="submitQuestion"
      />

      <template v-if="mode === 'ask'">
        <AskStatus :state="askState" @retry="retryAsk" />
        <template v-if="askState.status === 'success'">
          <AnswerCard :response="askState.response" />
          <AnswerSources
            :sources="askState.response.sources"
            :answer-status="askState.response.answer_status"
          />
        </template>
      </template>
      <template v-else-if="mode === 'search'">
        <SearchStatus :state="searchState" @retry="retrySearch" />
        <SearchResults :state="searchState" />
      </template>
      <KnowledgeBasePanel v-else-if="mode === 'documents'" />
      <StudyGuide v-else />
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

@media (max-width: 560px) {
  main {
    padding-inline: var(--space-4);
  }

  .context-panel {
    padding-inline: var(--space-4);
  }
}
</style>

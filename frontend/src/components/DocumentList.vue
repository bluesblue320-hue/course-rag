<script setup lang="ts">
import type { Document } from "../types/documents"
import DocumentListItem from "./DocumentListItem.vue"

defineProps<{
  documents: Document[]
  loading: boolean
  error: string | null
  deletingDocumentId: string | null
  deleteError: string | null
}>()

const emit = defineEmits<{
  delete: [documentId: string]
  retry: []
}>()

const DELETE_CONFIRM_MESSAGE = "删除后，该文档将不再用于检索和问答。"

function confirmAndDelete(documentId: string): void {
  if (!window.confirm(DELETE_CONFIRM_MESSAGE)) {
    return
  }
  emit("delete", documentId)
}
</script>

<template>
  <section class="document-list" aria-labelledby="list-heading">
    <h2 id="list-heading">知识库文档</h2>

    <p v-if="deleteError" class="list-status error" role="alert">
      {{ deleteError }}
    </p>
    <p v-if="loading" class="list-status" role="status">正在加载文档列表…</p>
    <p v-else-if="error" class="list-status error" role="alert">
      {{ error }}
      <button type="button" class="retry-button" @click="emit('retry')">
        重试
      </button>
    </p>
    <p v-else-if="documents.length === 0" class="list-status">
      还没有上传文档。
    </p>

    <ul v-else class="document-items">
      <DocumentListItem
        v-for="document in documents"
        :key="document.document_id"
        :document="document"
        :deleting="deletingDocumentId === document.document_id"
        @delete="confirmAndDelete"
      />
    </ul>
  </section>
</template>

<style scoped>
.document-list {
  display: grid;
  gap: var(--space-3);
}

h2 {
  margin: 0;
  font-size: 1.2rem;
}

.document-items {
  display: grid;
  gap: var(--space-3);
  margin: 0;
  padding: 0;
  list-style: none;
}

.list-status {
  margin: 0;
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-muted);
  background: var(--color-surface);
  text-align: center;
}

.list-status.error {
  color: var(--color-danger);
}

.retry-button {
  margin-left: var(--space-2);
  padding: 6px 14px;
  border: 1px solid var(--color-primary);
  border-radius: var(--radius-sm);
  color: var(--color-primary);
  background: var(--color-surface);
  font-weight: 700;
}

.retry-button:hover:not(:disabled) {
  color: white;
  background: var(--color-primary);
}
</style>

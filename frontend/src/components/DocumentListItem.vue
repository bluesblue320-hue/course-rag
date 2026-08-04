<script setup lang="ts">
import type { Document } from "../types/documents"

defineProps<{
  document: Document
  deleting?: boolean
}>()

const emit = defineEmits<{
  delete: [documentId: string]
}>()

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }
  return `${Math.max(1, Math.round(bytes / 1024))} KB`
}

function formatCreatedAt(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString()
}
</script>

<template>
  <li class="document-item">
    <div class="document-info">
      <p class="document-name">{{ document.filename }}</p>
      <p class="document-meta">
        <span>{{ document.content_type }}</span>
        <span>{{ formatBytes(document.size_bytes) }}</span>
        <span>{{ document.chunk_count }} 个 Chunk</span>
        <span>{{ formatCreatedAt(document.created_at) }}</span>
      </p>
    </div>

    <span v-if="document.is_builtin" class="builtin-badge">内置</span>
    <button
      v-else
      type="button"
      class="delete-button"
      :disabled="deleting"
      @click="emit('delete', document.document_id)"
    >
      {{ deleting ? "删除中…" : "删除" }}
    </button>
  </li>
</template>

<style scoped>
.document-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-4) var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
}

.document-info {
  min-width: 0;
}

.document-name {
  margin: 0;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.document-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  margin: var(--space-2) 0 0;
  color: var(--color-muted);
  font-size: 0.85rem;
}

.builtin-badge {
  flex-shrink: 0;
  padding: 4px 10px;
  border-radius: 999px;
  color: var(--color-primary);
  background: #eef3ff;
  font-size: 0.85rem;
  font-weight: 700;
}

.delete-button {
  flex-shrink: 0;
  padding: 8px 16px;
  border: 1px solid var(--color-danger);
  border-radius: var(--radius-sm);
  color: var(--color-danger);
  background: var(--color-surface);
  font-weight: 700;
}

.delete-button:hover:not(:disabled) {
  color: white;
  background: var(--color-danger);
}

.delete-button:disabled {
  opacity: 0.65;
}

@media (max-width: 560px) {
  .document-item {
    align-items: stretch;
    flex-direction: column;
  }
}
</style>

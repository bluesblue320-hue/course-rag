<script setup lang="ts">
import { computed, inject, onMounted } from "vue"

import DocumentList from "./DocumentList.vue"
import DocumentUpload from "./DocumentUpload.vue"
import { useDocuments } from "../composables/useDocuments"
import { documentServiceKey } from "../services/documentService"

const documentService = inject(documentServiceKey)
if (!documentService) {
  throw new Error("DocumentService 未注入")
}

const {
  listState,
  uploadState,
  deleteState,
  isListLoading,
  isUploading,
  load,
  upload,
  deleteDocument,
  resetUpload,
} = useDocuments(documentService)

const listError = computed(() =>
  listState.value.status === "error" ? listState.value.message : null,
)
const deleteError = computed(() =>
  deleteState.value.status === "error" ? deleteState.value.message : null,
)
const deletingDocumentId = computed(() =>
  deleteState.value.status === "deleting" ? deleteState.value.documentId : null,
)
const uploadMessage = computed(() =>
  uploadState.value.status === "error"
    ? uploadState.value.message
    : uploadState.value.status === "success"
      ? "上传成功，内容已加入知识库。"
      : "",
)

onMounted(() => {
  load()
})
</script>

<template>
  <div class="knowledge-base-panel">
    <DocumentUpload
      :status="uploadState.status"
      :message="uploadMessage"
      :uploading="isUploading"
      @upload="upload"
      @reset-status="resetUpload"
    />
    <DocumentList
      :documents="
        listState.status === 'success' ? listState.documents : []
      "
      :loading="isListLoading"
      :error="listError"
      :deleting-document-id="deletingDocumentId"
      :delete-error="deleteError"
      @delete="deleteDocument"
      @retry="load"
    />
  </div>
</template>

<style scoped>
.knowledge-base-panel {
  display: grid;
  gap: var(--space-6);
}
</style>

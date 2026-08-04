<script setup lang="ts">
import { computed, ref, watch } from "vue"

import type { DocumentUploadState } from "../types/documents"

const props = withDefaults(
  defineProps<{
    status: DocumentUploadState["status"]
    message?: string
    uploading?: boolean
    maxUploadBytes?: number
  }>(),
  {
    status: "idle",
    message: "",
    uploading: false,
    maxUploadBytes: 10 * 1024 * 1024,
  },
)

const emit = defineEmits<{
  upload: [file: File]
  resetStatus: []
}>()

const selectedFile = ref<File | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

const allowedExtensions = ".txt,.md,.pdf"

function formatMaxBytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(0)} MB`
}

const sizeHint = computed(() => formatMaxBytes(props.maxUploadBytes))

function onFileSelected(event: Event): void {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  selectedFile.value = file ?? null
  if (props.status === "success" || props.status === "error") {
    emit("resetStatus")
  }
}

function submitUpload(): void {
  if (selectedFile.value && !props.uploading) {
    emit("upload", selectedFile.value)
  }
}

function clearSelection(): void {
  selectedFile.value = null
  if (fileInput.value) {
    fileInput.value.value = ""
  }
}

watch(
  () => props.status,
  (status) => {
    if (status === "success") {
      clearSelection()
    }
  },
)
</script>

<template>
  <section class="upload-panel" aria-labelledby="upload-heading">
    <h2 id="upload-heading">上传课程资料</h2>
    <p class="upload-hint">
      支持 .txt、.md 和文本型 .pdf，最大 {{ sizeHint }}。扫描 PDF 与图片识别暂不支持。
    </p>

    <form class="upload-form" @submit.prevent="submitUpload">
      <input
        ref="fileInput"
        id="document-file"
        class="file-input"
        type="file"
        :accept="allowedExtensions"
        :disabled="uploading"
        @change="onFileSelected"
      />
      <label class="file-label" for="document-file">
        {{ selectedFile ? selectedFile.name : "选择文件" }}
      </label>

      <button type="submit" :disabled="uploading || !selectedFile">
        {{ uploading ? "上传中…" : "上传" }}
      </button>
    </form>

    <p v-if="status === 'success'" class="upload-message success" role="status">
      上传成功，内容已加入知识库。
    </p>
    <p v-else-if="status === 'error' && message" class="upload-message error" role="alert">
      {{ message }}
    </p>
  </section>
</template>

<style scoped>
.upload-panel {
  display: grid;
  gap: var(--space-4);
  padding: var(--space-6);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  box-shadow: var(--shadow-card);
}

h2 {
  margin: 0;
  font-size: 1.2rem;
}

.upload-hint {
  margin: 0;
  color: var(--color-muted);
  font-size: 0.9rem;
}

.upload-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--space-3);
}

.file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  overflow: hidden;
}

.file-label {
  display: grid;
  min-height: 46px;
  place-items: center;
  padding: 0 var(--space-4);
  border: 1px solid #b9c2d3;
  border-radius: var(--radius-sm);
  color: var(--color-text);
  background: var(--color-surface-soft);
  cursor: pointer;
  overflow-wrap: anywhere;
}

button {
  min-width: 120px;
  padding: 12px 22px;
  border: 0;
  border-radius: var(--radius-sm);
  color: white;
  background: var(--color-primary);
  font-weight: 700;
}

button:hover:not(:disabled) {
  background: var(--color-primary-dark);
}

button:disabled {
  opacity: 0.65;
}

.upload-message {
  margin: 0;
  padding: var(--space-3);
  border-radius: var(--radius-sm);
}

.upload-message.success {
  color: var(--color-success);
  background: var(--color-success-soft);
}

.upload-message.error {
  color: var(--color-danger);
  background: var(--color-danger-soft);
}

@media (max-width: 560px) {
  .upload-form {
    grid-template-columns: minmax(0, 1fr);
  }

  button {
    width: 100%;
  }
}
</style>

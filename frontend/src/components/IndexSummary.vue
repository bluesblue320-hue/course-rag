<script setup lang="ts">
defineProps<{
  indexedChunks: number
  topK: number
  model: string
}>()

function shortModelName(model: string): string {
  return model.replace("paraphrase-multilingual-", "")
}
</script>

<template>
  <section class="panel" aria-labelledby="index-heading">
    <h2 id="index-heading">当前索引</h2>
    <dl>
      <div>
        <dt>文本块</dt>
        <dd>{{ indexedChunks }} 个文本块</dd>
      </div>
      <div>
        <dt>返回数量</dt>
        <dd>Top K · {{ topK }}</dd>
      </div>
      <div>
        <dt>向量模型</dt>
        <dd>{{ shortModelName(model) }}</dd>
      </div>
    </dl>
  </section>
</template>

<style scoped>
.panel {
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
}

h2 {
  margin: 0 0 var(--space-4);
  font-size: 1.2rem;
}

dl {
  display: grid;
  gap: var(--space-3);
  margin: 0;
}

dl div {
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface-soft);
}

dt {
  color: var(--color-muted);
  font-size: 0.8rem;
}

dd {
  margin: var(--space-1) 0 0;
  overflow-wrap: anywhere;
  font-weight: 700;
}
</style>

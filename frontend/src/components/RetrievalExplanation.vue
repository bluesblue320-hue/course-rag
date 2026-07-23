<template>
  <section class="panel" aria-labelledby="explanation-heading">
    <h2 id="explanation-heading">这一步发生了什么？</h2>
    <ol>
      <li><span>1</span>问题转为向量</li>
      <li><span>2</span>计算余弦相似度</li>
      <li><span>3</span>返回 Top 3 原文</li>
    </ol>
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
  margin: 0 0 var(--space-5);
  color: var(--color-success);
  font-size: 1.05rem;
}

ol {
  display: grid;
  gap: var(--space-5);
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  font-weight: 650;
}

li span {
  display: grid;
  width: 30px;
  height: 30px;
  place-items: center;
  border-radius: 999px;
  color: white;
  background: var(--color-success);
}
</style>

<template>
  <aside class="sidebar">
    <div class="brand">
      <svg viewBox="0 0 48 48" aria-hidden="true">
        <path d="M7 9h14c5 0 8 3 8 8v18c-2-3-5-4-9-4H7V9Z" />
        <circle cx="34" cy="31" r="7" />
        <path d="m39 36 5 5" />
      </svg>
      <strong>Course RAG</strong>
    </div>

    <nav aria-label="主导航">
      <span class="nav-item active" aria-current="page">语义检索</span>
      <span class="nav-item" aria-disabled="true">知识资料</span>
      <span class="nav-item" aria-disabled="true">学习说明</span>
    </nav>

    <p class="ready"><span aria-hidden="true" />本地知识库 · 已就绪</p>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  min-height: 100vh;
  padding: var(--space-8) var(--space-4);
  border-right: 1px solid var(--color-border);
  background: var(--color-surface);
  flex-direction: column;
}

.brand {
  display: grid;
  justify-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-8);
  color: var(--color-text);
  font-size: 1.2rem;
}

.brand svg {
  width: 52px;
  fill: none;
  stroke: var(--color-primary);
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 3;
}

nav {
  display: grid;
  gap: var(--space-2);
}

.nav-item {
  padding: 13px 15px;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  color: var(--color-muted);
}

.nav-item.active {
  color: var(--color-primary);
  border-color: #b9c7f0;
  background: #f5f7ff;
  font-weight: 700;
}

.nav-item[aria-disabled="true"] {
  opacity: 0.72;
}

.ready {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: auto 0 0;
  color: var(--color-muted);
  font-size: 0.85rem;
}

.ready span {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #2fb277;
}

@media (max-width: 900px) {
  .sidebar {
    min-height: auto;
    padding: var(--space-4);
    border-right: 0;
    border-bottom: 1px solid var(--color-border);
  }

  .brand {
    display: flex;
    margin-bottom: var(--space-3);
  }

  .brand svg {
    width: 36px;
  }

  nav {
    display: flex;
    flex-wrap: wrap;
  }

  .ready {
    margin-top: var(--space-3);
  }
}
</style>

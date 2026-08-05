<template>
  <div class="study-guide">
    <section class="guide-lead" aria-labelledby="guide-lead-title">
      <p class="guide-kicker">建议顺序</p>
      <h2 id="guide-lead-title">先准备资料，再提出问题，最后核对来源</h2>
      <p>
        Course RAG 会从当前知识库检索相关原文，并在资料足够时生成回答。回答是学习线索，引用来源才是核对依据。
      </p>
    </section>

    <section class="guide-section" aria-labelledby="workflow-title">
      <div class="section-heading">
        <span class="section-number" aria-hidden="true">01</span>
        <div>
          <p>START HERE</p>
          <h2 id="workflow-title">推荐学习流程</h2>
        </div>
      </div>

      <ol class="workflow-list">
        <li>
          <strong>整理并上传课程资料</strong>
          <span>在“知识资料”中上传 TXT、Markdown 或文本型 PDF，等待文档显示为已索引。</span>
        </li>
        <li>
          <strong>围绕一个知识点提问</strong>
          <span>在“知识问答”中说明课程概念、场景或疑问；需要查看原文时切换到“语义检索”。</span>
        </li>
        <li>
          <strong>阅读回答并核对来源</strong>
          <span>打开回答下方的引用，比较文件名、页码、Chunk 和相似度，再回到原始资料确认上下文。</span>
        </li>
      </ol>
    </section>

    <div class="guide-grid">
      <section class="guide-card" aria-labelledby="question-title">
        <p class="card-label">提问</p>
        <h2 id="question-title">怎样问得更有效</h2>
        <ul>
          <li>一次聚焦一个概念或一个具体任务。</li>
          <li>写出课程中的术语，并补充必要场景。</li>
          <li>需要逐段阅读时，先使用“语义检索”查看原文。</li>
        </ul>
        <div class="example">
          <span>示例</span>
          <p>“在本课程架构中，Service 层负责什么？请列出引用依据。”</p>
        </div>
      </section>

      <section class="guide-card" aria-labelledby="reliability-title">
        <p class="card-label">核对</p>
        <h2 id="reliability-title">怎样判断回答可靠</h2>
        <ul>
          <li>确认引用内容确实支持回答中的关键结论。</li>
          <li>相似度表示检索接近程度，不等同于事实正确率。</li>
          <li>资料不足时系统不会调用 LLM 生成答案。</li>
        </ul>
        <p class="threshold-note">
          当前回答相关度阈值为 <strong>0.35</strong>；该值由后端统一判断，前端不重新计算。
        </p>
      </section>
    </div>

    <section class="guide-section" aria-labelledby="material-title">
      <div class="section-heading">
        <span class="section-number" aria-hidden="true">02</span>
        <div>
          <p>KNOW YOUR MATERIALS</p>
          <h2 id="material-title">当前资料能力</h2>
        </div>
      </div>

      <dl class="capability-list">
        <div>
          <dt>支持格式</dt>
          <dd><code>.txt</code>、<code>.md</code>、文本型 <code>.pdf</code></dd>
        </div>
        <div>
          <dt>上传之后</dt>
          <dd>同步解析并建立索引，成功后立即参与检索和问答。</dd>
        </div>
        <div>
          <dt>暂不支持</dt>
          <dd>扫描 PDF 的 OCR、图片识别、Word、PowerPoint 和 Excel。</dd>
        </div>
        <div>
          <dt>资料维护</dt>
          <dd>上传资料可以删除；项目内置资料保留且不可删除。</dd>
        </div>
      </dl>
    </section>
  </div>
</template>

<style scoped>
.study-guide {
  display: grid;
  gap: var(--space-5);
}

.guide-lead,
.guide-section,
.guide-card {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  box-shadow: var(--shadow-card);
}

.guide-lead {
  padding: clamp(24px, 4vw, 36px);
  border-left: 5px solid var(--color-primary);
  background: linear-gradient(135deg, #f5f7ff 0%, var(--color-surface) 72%);
}

.guide-kicker,
.section-heading p,
.card-label {
  margin: 0;
  color: var(--color-primary);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.guide-lead h2 {
  margin: var(--space-2) 0 var(--space-3);
  font-size: clamp(1.25rem, 2.5vw, 1.65rem);
  line-height: 1.35;
}

.guide-lead > p:last-child {
  max-width: 720px;
  margin: 0;
  color: var(--color-muted);
  line-height: 1.75;
}

.guide-section,
.guide-card {
  padding: var(--space-6);
}

.section-heading {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.section-number {
  display: grid;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #eef2ff;
  color: var(--color-primary);
  font-weight: 800;
  place-items: center;
}

.section-heading h2,
.guide-card h2 {
  margin: var(--space-1) 0 0;
  font-size: 1.15rem;
}

.workflow-list {
  display: grid;
  margin: 0;
  padding: 0;
  counter-reset: guide-step;
  gap: var(--space-3);
  list-style: none;
}

.workflow-list li {
  display: grid;
  position: relative;
  padding: var(--space-4) var(--space-4) var(--space-4) 54px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface-soft);
  counter-increment: guide-step;
  gap: var(--space-1);
}

.workflow-list li::before {
  position: absolute;
  top: 18px;
  left: 18px;
  color: var(--color-primary);
  content: counter(guide-step, decimal-leading-zero);
  font-size: 0.78rem;
  font-weight: 800;
}

.workflow-list span,
.guide-card li,
.capability-list dd {
  color: var(--color-muted);
  line-height: 1.65;
}

.guide-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-5);
}

.guide-card ul {
  display: grid;
  margin: var(--space-4) 0 0;
  padding-left: var(--space-5);
  gap: var(--space-2);
}

.example,
.threshold-note {
  margin: var(--space-5) 0 0;
  padding: var(--space-4);
  border-radius: var(--radius-sm);
  background: var(--color-surface-soft);
}

.example span {
  color: var(--color-primary);
  font-size: 0.75rem;
  font-weight: 800;
}

.example p,
.threshold-note {
  line-height: 1.65;
}

.example p {
  margin: var(--space-2) 0 0;
}

.threshold-note strong {
  color: var(--color-primary);
}

.capability-list {
  display: grid;
  margin: 0;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
}

.capability-list div {
  padding: var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
}

.capability-list dt {
  margin-bottom: var(--space-2);
  font-weight: 800;
}

.capability-list dd {
  margin: 0;
}

.capability-list code {
  color: var(--color-primary-dark);
  font-weight: 700;
}

@media (max-width: 680px) {
  .guide-grid,
  .capability-list {
    grid-template-columns: minmax(0, 1fr);
  }

  .guide-section,
  .guide-card {
    padding: var(--space-5);
  }
}
</style>

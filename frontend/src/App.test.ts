import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { describe, expect, it, vi } from "vitest"

import App from "./App.vue"
import {
  AskServiceError,
  askServiceKey,
  type AskService,
} from "./services/askService"
import {
  documentServiceKey,
  type DocumentService,
} from "./services/documentService"
import type { SearchService } from "./services/searchService"
import { searchServiceKey } from "./services/searchService"
import type { AskResponse } from "./types/ask"
import type {
  DeleteDocumentResponse,
  Document,
} from "./types/documents"
import type { SearchResponse } from "./types/search"

const askResponse: AskResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责核心业务逻辑。[来源1]",
  answer_status: "answered",
  max_relevance_score: 0.8421,
  relevance_threshold: 0.35,
  retrieval_elapsed_ms: 12.4,
  generation_elapsed_ms: 680.7,
  total_elapsed_ms: 693.1,
  embedding_model: "embedding-model",
  llm_model: "llm-model",
  sources: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "Service 层负责核心业务逻辑。",
      document_id: "builtin-knowledge",
      filename: "knowledge.txt",
      page_number: null,
    },
  ],
}

const searchResponse: SearchResponse = {
  query: "业务逻辑应该写在哪一层？",
  elapsed_ms: 420,
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  results: [
    {
      rank: 1,
      score: 0.8421,
      chunk_index: 2,
      text: "service 层负责业务逻辑。",
      document_id: "builtin-knowledge",
      filename: "knowledge.txt",
      page_number: null,
    },
    {
      rank: 2,
      score: 0.7168,
      chunk_index: 1,
      text: "router 层接收 HTTP 请求。",
      document_id: "builtin-knowledge",
      filename: "knowledge.txt",
      page_number: null,
    },
    {
      rank: 3,
      score: 0.6234,
      chunk_index: 3,
      text: "repository 层负责数据访问。",
      document_id: "builtin-knowledge",
      filename: "knowledge.txt",
      page_number: null,
    },
  ],
}

function mountApp(
  askService: AskService = { ask: vi.fn().mockResolvedValue(askResponse) },
  searchService: SearchService = {
    search: vi.fn().mockResolvedValue(searchResponse),
  },
  documentService?: DocumentService,
): VueWrapper {
  return mount(App, {
    global: {
      provide: {
        [askServiceKey as symbol]: askService,
        [searchServiceKey as symbol]: searchService,
        ...(documentService
          ? { [documentServiceKey as symbol]: documentService }
          : {}),
      },
    },
  })
}

function modeButtons(wrapper: VueWrapper) {
  return wrapper.findAll(".mode-switch button")
}

function sidebarButtons(wrapper: VueWrapper) {
  return wrapper.findAll(".sidebar nav button")
}

describe("App", () => {
  it("defaults to intelligent Q&A and submits a trimmed top-3 ask request", async () => {
    const askMethod = vi.fn().mockResolvedValue(askResponse)
    const searchMethod = vi.fn().mockResolvedValue(searchResponse)
    const wrapper = mountApp({ ask: askMethod }, { search: searchMethod })

    expect(wrapper.text()).toContain("课程知识问答")
    expect(wrapper.text()).toContain("RAG QUESTION ANSWERING")
    expect(modeButtons(wrapper)[0].attributes("aria-pressed")).toBe("true")

    await wrapper.get("textarea").setValue("  Service 层负责什么？  ")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(askMethod).toHaveBeenCalledWith({
      question: "Service 层负责什么？",
      top_k: 3,
    })
    expect(searchMethod).not.toHaveBeenCalled()
  })

  it("shows ask loading, blocks switching, then renders answer details", async () => {
    let resolveAsk!: (value: AskResponse) => void
    const askMethod = vi.fn(
      () =>
        new Promise<AskResponse>((resolve) => {
          resolveAsk = resolve
        }),
    )
    const wrapper = mountApp({ ask: askMethod })

    await wrapper.get("textarea").setValue("Service 层负责什么？")
    await wrapper.get("form").trigger("submit")

    expect(wrapper.text()).toContain("正在检索课程资料并生成回答")
    expect(wrapper.get("textarea").attributes("disabled")).toBeDefined()
    expect(modeButtons(wrapper).every((button) => button.attributes("disabled") !== undefined)).toBe(
      true,
    )
    await modeButtons(wrapper)[1].trigger("click")
    expect(wrapper.text()).toContain("课程知识问答")

    resolveAsk(askResponse)
    await flushPromises()

    expect(wrapper.text()).toContain(askResponse.answer)
    expect(wrapper.text()).toContain("引用来源（1）")
    expect(wrapper.text()).toContain("Service 层负责核心业务逻辑。")
    expect(wrapper.text()).toContain("embedding-model")
    expect(wrapper.text()).toContain("llm-model")
  })

  it("switches to retrieval without requesting and keeps the shared input", async () => {
    const askMethod = vi.fn().mockResolvedValue(askResponse)
    const searchMethod = vi.fn().mockResolvedValue(searchResponse)
    const wrapper = mountApp({ ask: askMethod }, { search: searchMethod })

    await wrapper.get("textarea").setValue("业务逻辑应该写在哪一层？")
    await modeButtons(wrapper)[1].trigger("click")

    expect(wrapper.text()).toContain("课程知识检索")
    expect(wrapper.get("textarea").element.value).toBe("业务逻辑应该写在哪一层？")
    expect(askMethod).not.toHaveBeenCalled()
    expect(searchMethod).not.toHaveBeenCalled()

    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(searchMethod).toHaveBeenCalledWith({
      query: "业务逻辑应该写在哪一层？",
      top_k: 3,
    })
    expect(wrapper.findAll(".result-card")).toHaveLength(3)
    expect(wrapper.text()).toContain("找到 3 条相关内容")
  })

  it("preserves independent ask and search results across mode switches", async () => {
    const wrapper = mountApp()
    await wrapper.get("textarea").setValue("Service 层负责什么？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()
    expect(wrapper.text()).toContain(askResponse.answer)

    await modeButtons(wrapper)[1].trigger("click")
    await wrapper.get("textarea").setValue("业务逻辑应该写在哪一层？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()
    expect(wrapper.findAll(".result-card")).toHaveLength(3)
    expect(wrapper.text()).not.toContain(askResponse.answer)

    await modeButtons(wrapper)[0].trigger("click")
    expect(wrapper.text()).toContain(askResponse.answer)
    expect(wrapper.findAll(".result-card")).toHaveLength(0)
  })

  it("shows the 503 guidance and allows switching to retrieval", async () => {
    const askMethod = vi.fn().mockRejectedValue(
      new AskServiceError(
        "问答服务尚未完成配置，你仍可使用语义检索",
        "LLM_NOT_CONFIGURED",
        503,
      ),
    )
    const wrapper = mountApp({ ask: askMethod })

    await wrapper.get("textarea").setValue("问题")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(wrapper.get('[role="alert"]').text()).toContain(
      "仍可以切换到“语义检索”查看相关原文",
    )
    await modeButtons(wrapper)[1].trigger("click")
    expect(wrapper.text()).toContain("课程知识检索")
  })

  it("renders insufficient context as candidates rather than citations", async () => {
    const insufficientResponse: AskResponse = {
      ...askResponse,
      answer: "当前课程资料中没有足够信息回答这个问题。",
      answer_status: "insufficient_context",
      max_relevance_score: 0.12,
      generation_elapsed_ms: 0,
    }
    const wrapper = mountApp({
      ask: vi.fn().mockResolvedValue(insufficientResponse),
    })

    await wrapper.get("textarea").setValue("今天天气怎么样？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(wrapper.text()).toContain("资料不足")
    expect(wrapper.text()).toContain("检索候选（1）")
    expect(wrapper.text()).toContain("0.1200")
    expect(wrapper.text()).not.toContain("引用来源（1）")
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain("重试")

    await modeButtons(wrapper)[1].trigger("click")
    expect(wrapper.text()).toContain("课程知识检索")
    expect(wrapper.get("textarea").element.value).toBe("今天天气怎么样？")
  })

  it("shows safe RAG configuration guidance and keeps retrieval available", async () => {
    const wrapper = mountApp({
      ask: vi.fn().mockRejectedValue(
        new AskServiceError(
          "untrusted backend detail",
          "RAG_NOT_CONFIGURED",
          503,
        ),
      ),
    })

    await wrapper.get("textarea").setValue("问题")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(wrapper.get('[role="alert"]').text()).toContain(
      "语义检索仍可继续使用。",
    )
    expect(wrapper.text()).not.toContain("untrusted backend detail")
    await modeButtons(wrapper)[1].trigger("click")
    expect(wrapper.text()).toContain("课程知识检索")
  })


  it("renders the retrieval empty state without fake result cards", async () => {
    const searchService = {
      search: vi.fn().mockResolvedValue({
        ...searchResponse,
        query: "今天午饭吃什么？",
        results: [],
      }),
    }
    const wrapper = mountApp(undefined, searchService)

    await modeButtons(wrapper)[1].trigger("click")
    await wrapper.get("textarea").setValue("今天午饭吃什么？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(wrapper.text()).toContain("请换一种问法")
    expect(wrapper.findAll(".result-card")).toHaveLength(0)
  })

  it("enters knowledge base management and loads the document list", async () => {
    const listDocuments = vi.fn().mockResolvedValue({
      documents: [],
      document_count: 0,
      chunk_count: 0,
    })
    const documentService: DocumentService = {
      listDocuments,
      uploadDocument: vi.fn(),
      deleteDocument: vi.fn(),
    }
    const wrapper = mountApp(undefined, undefined, documentService)

    await sidebarButtons(wrapper)[1].trigger("click")

    expect(wrapper.text()).toContain("知识库管理")
    expect(wrapper.text()).toContain("上传课程资料")
    expect(wrapper.text()).toContain("知识库文档")
    expect(wrapper.get("textarea").isVisible()).toBe(false)
    await flushPromises()
    expect(listDocuments).toHaveBeenCalled()
  })

  it("opens the learning guide without making service requests", async () => {
    const askMethod = vi.fn().mockResolvedValue(askResponse)
    const searchMethod = vi.fn().mockResolvedValue(searchResponse)
    const wrapper = mountApp({ ask: askMethod }, { search: searchMethod })

    await sidebarButtons(wrapper)[2].trigger("click")

    expect(wrapper.text()).toContain("学习说明")
    expect(wrapper.text()).toContain("推荐学习流程")
    expect(wrapper.findAll(".workflow-list li")).toHaveLength(3)
    expect(modeButtons(wrapper)).toHaveLength(0)
    expect(wrapper.get("textarea").isVisible()).toBe(false)
    expect(askMethod).not.toHaveBeenCalled()
    expect(searchMethod).not.toHaveBeenCalled()
    expect(wrapper.get('[aria-current="page"]').text()).toBe("学习说明")

    await sidebarButtons(wrapper)[0].trigger("click")
    expect(wrapper.get(".page-header h1").text()).toBe("课程知识问答")
    expect(modeButtons(wrapper)).toHaveLength(3)
    expect(modeButtons(wrapper)[0].attributes("aria-pressed")).toBe("true")
  })

  it("refreshes the document list after a successful upload", async () => {
    const uploadedDocument: Document = {
      document_id: "doc-1",
      filename: "notes.txt",
      content_type: "text/plain",
      size_bytes: 1024,
      text_length: 512,
      chunk_count: 3,
      created_at: "2026-08-03T08:00:00Z",
      is_builtin: false,
    }
    const listDocuments = vi
      .fn()
      .mockResolvedValueOnce({ documents: [], document_count: 0, chunk_count: 0 })
      .mockResolvedValueOnce({
        documents: [uploadedDocument],
        document_count: 1,
        chunk_count: 3,
      })
    const documentService: DocumentService = {
      listDocuments,
      uploadDocument: vi.fn().mockResolvedValue(uploadedDocument),
      deleteDocument: vi.fn(),
    }
    const wrapper = mountApp(undefined, undefined, documentService)
    await modeButtons(wrapper)[2].trigger("click")
    await flushPromises()

    const file = new File(["alpha content"], "notes.txt", { type: "text/plain" })
    const input = wrapper.get('input[type="file"]').element as HTMLInputElement
    Object.defineProperty(input, "files", { value: [file] })
    await wrapper.get('input[type="file"]').trigger("change")
    await wrapper.get(".upload-form").trigger("submit")
    await flushPromises()

    expect(wrapper.text()).toContain("上传成功")
    expect(wrapper.text()).toContain("notes.txt")
  })

  it("removes a document from the list after a confirmed deletion", async () => {
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(true))
    const uploadedDocument: Document = {
      document_id: "doc-1",
      filename: "notes.txt",
      content_type: "text/plain",
      size_bytes: 1024,
      text_length: 512,
      chunk_count: 3,
      created_at: "2026-08-03T08:00:00Z",
      is_builtin: false,
    }
    const listDocuments = vi
      .fn()
      .mockResolvedValueOnce({
        documents: [uploadedDocument],
        document_count: 1,
        chunk_count: 3,
      })
      .mockResolvedValueOnce({ documents: [], document_count: 0, chunk_count: 0 })
    const documentService: DocumentService = {
      listDocuments,
      uploadDocument: vi.fn(),
      deleteDocument: vi.fn().mockResolvedValue({
        document_id: "doc-1",
        deleted: true,
        document_count: 0,
        chunk_count: 0,
      } satisfies DeleteDocumentResponse),
    }
    const wrapper = mountApp(undefined, undefined, documentService)
    await modeButtons(wrapper)[2].trigger("click")
    await flushPromises()
    expect(wrapper.text()).toContain("notes.txt")

    await wrapper.get("li button").trigger("click")
    await flushPromises()

    expect(wrapper.text()).not.toContain("notes.txt")
    expect(wrapper.text()).toContain("还没有上传文档")
    vi.unstubAllGlobals()
  })

  it("keeps ask and search fully working after visiting the knowledge base", async () => {
    const askMethod = vi.fn().mockResolvedValue(askResponse)
    const searchMethod = vi.fn().mockResolvedValue(searchResponse)
    const documentService: DocumentService = {
      listDocuments: vi.fn().mockResolvedValue({
        documents: [],
        document_count: 0,
        chunk_count: 0,
      }),
      uploadDocument: vi.fn(),
      deleteDocument: vi.fn(),
    }
    const wrapper = mountApp(
      { ask: askMethod },
      { search: searchMethod },
      documentService,
    )

    await wrapper.get("textarea").setValue("Service 层负责什么？")
    await modeButtons(wrapper)[2].trigger("click")
    await wrapper.get("textarea").setValue("业务逻辑应该写在哪一层？")
    await modeButtons(wrapper)[1].trigger("click")
    expect(wrapper.get("textarea").element.value).toBe("业务逻辑应该写在哪一层？")
    await wrapper.get("form").trigger("submit")
    await flushPromises()

    expect(searchMethod).toHaveBeenCalledWith({
      query: "业务逻辑应该写在哪一层？",
      top_k: 3,
    })
    expect(wrapper.findAll(".result-card")).toHaveLength(3)
  })
})

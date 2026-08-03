import type { SearchResult } from "../types/search"

export const DEFAULT_INDEX_METADATA = {
  indexed_chunks: 8,
  model: "paraphrase-multilingual-MiniLM-L12-v2",
  top_k: 3,
} as const

export const BUSINESS_RESULTS: SearchResult[] = [
  {
    rank: 1,
    score: 0.8421,
    chunk_index: 2,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "service 层负责业务逻辑，例如校验状态、计算价格和协调多个数据操作。",
  },
  {
    rank: 2,
    score: 0.7168,
    chunk_index: 1,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "router 层负责接收 HTTP 请求、读取参数，并调用 service 层。",
  },
  {
    rank: 3,
    score: 0.6234,
    chunk_index: 3,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "repository 层负责数据访问，使 service 层无需了解具体 SQL。",
  },
]

export const DATABASE_RESULTS: SearchResult[] = [
  {
    rank: 1,
    score: 0.8652,
    chunk_index: 3,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "repository 层负责查询、新增、修改和删除等数据访问操作。",
  },
  {
    rank: 2,
    score: 0.7521,
    chunk_index: 4,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "PostgreSQL 用于持久化用户、课程和订单等结构化数据。",
  },
  {
    rank: 3,
    score: 0.5987,
    chunk_index: 2,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "service 层通过 repository 协调数据操作，不直接编写 SQL。",
  },
]

export const EMBEDDING_RESULTS: SearchResult[] = [
  {
    rank: 1,
    score: 0.8913,
    chunk_index: 5,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "Embedding 是把文本转换成一组数字向量的过程。",
  },
  {
    rank: 2,
    score: 0.8144,
    chunk_index: 6,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "余弦相似度衡量两个向量方向有多接近。",
  },
  {
    rank: 3,
    score: 0.6742,
    chunk_index: 7,
    document_id: "builtin-knowledge",
    filename: "knowledge.txt",
    page_number: null,
    text: "RAG 会根据问题检索 Top K 相关 Chunk，再把原文交给生成模型。",
  },
]

export const UPLOADED_RESULTS: SearchResult[] = [
  {
    rank: 1,
    score: 0.8123,
    chunk_index: 0,
    document_id: "mock-document-1",
    filename: "课程讲义.pdf",
    page_number: 12,
    text: "上传的讲义内容：分层架构与业务逻辑。",
  },
]

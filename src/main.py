"""Run the minimal semantic retrieval command-line program."""

from pathlib import Path

from src.chunker import split_text
from src.embedding import EmbeddingService
from src.loader import load_text
from src.retriever import SearchResult, SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"


def _print_results(results: list[SearchResult]) -> None:
    """Print ranked retrieval results in a beginner-readable format."""
    for result in results:
        print(f"\nTop {result['rank']}")
        print(f"相似度：{float(result['score']):.4f}")
        print(f"Chunk编号：{result['chunk_index']}")
        print(f"原文：{result['text']}")


def main() -> None:
    """Build the document index once, then search until the user exits."""
    text = load_text(str(KNOWLEDGE_PATH))
    chunks = split_text(text)
    embedding_service = EmbeddingService()
    document_embeddings = embedding_service.encode_documents(chunks)
    retriever = SemanticRetriever(chunks, document_embeddings)

    print(f"成功加载 {len(chunks)} 个 Chunk。")

    while True:
        query = input("\n请输入问题（输入 exit/quit 或直接回车退出）：").strip()
        if not query or query.lower() in {"exit", "quit"}:
            print("程序已退出。")
            break

        query_embedding = embedding_service.encode_query(query)
        results = retriever.search(query_embedding, top_k=3)
        _print_results(results)


if __name__ == "__main__":
    main()

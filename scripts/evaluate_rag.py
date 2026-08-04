"""Run the offline RAG retrieval and refusal evaluation from the command line.

The command reuses the production ingestion and retrieval pipeline, never
calls a large language model, and never rewrites the production threshold.
Run it from the repository root::

    python -m scripts.evaluate_rag
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from src.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.evaluation.corpus import load_corpus, load_manifest
from src.evaluation.dataset import load_dataset
from src.evaluation.report import (
    CASES_FILENAME,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    write_reports,
)
from src.evaluation.runner import EmbeddingProtocol, EvaluationRun, run_evaluation
from src.evaluation.threshold import (
    DEFAULT_FALSE_ANSWER_WEIGHT,
    DEFAULT_FALSE_REFUSAL_WEIGHT,
    DEFAULT_THRESHOLD_END,
    DEFAULT_THRESHOLD_START,
    DEFAULT_THRESHOLD_STEP,
)
from src.exceptions import RagError
from src.rag_service import DEFAULT_MIN_RELEVANCE_SCORE

DEFAULT_MANIFEST = "eval/corpus_manifest.json"
DEFAULT_DATASET = "eval/dataset.jsonl"
DEFAULT_OUTPUT_DIR = "reports/generated"
DEFAULT_TOP_K = 5

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_EMBEDDING_UNAVAILABLE = 3

EmbeddingFactory = Callable[[str | None], tuple[EmbeddingProtocol, str]]


def repository_root() -> Path:
    """Return the repository root so the command works from any directory."""
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, with every default visible in ``--help``."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_rag",
        description=(
            "离线评估 RAG 检索质量与拒答决策，并在 calibration split 上校准"
            "相似度阈值。本命令不调用大模型，也不会修改生产阈值配置。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="评估语料清单路径，相对仓库根目录解析",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="评估数据集 JSONL 路径，相对仓库根目录解析",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="报告输出目录，相对仓库根目录解析",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="检索返回条数，实际取值会被抬到至少 5 以便计算 Hit@5",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=DEFAULT_THRESHOLD_START,
        help="阈值扫描起点",
    )
    parser.add_argument(
        "--threshold-end",
        type=float,
        default=DEFAULT_THRESHOLD_END,
        help="阈值扫描终点",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=DEFAULT_THRESHOLD_STEP,
        help="阈值扫描步长",
    )
    parser.add_argument(
        "--current-threshold",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help="当前生产阈值，只用于对比，不会被自动修改",
    )
    parser.add_argument(
        "--false-answer-weight",
        type=float,
        default=DEFAULT_FALSE_ANSWER_WEIGHT,
        help="错误放行的代价权重（业务选择，不是行业标准）",
    )
    parser.add_argument(
        "--false-refusal-weight",
        type=float,
        default=DEFAULT_FALSE_REFUSAL_WEIGHT,
        help="错误拒答的代价权重",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding 模型名称，缺省时使用 src.embedding 的生产默认模型",
    )
    return parser


def _resolve_path(raw_value: str, root: Path) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _display_path(path: Path, root: Path) -> str:
    """Return a repository-relative path so reports never leak local layout."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def default_embedding_factory(
    model_name: str | None,
) -> tuple[EmbeddingProtocol, str]:
    """Instantiate the production embedding service, importing it lazily.

    The import stays inside the function so ``--help`` never pays for loading
    the model runtime.
    """
    from src.embedding import DEFAULT_MODEL_NAME, EmbeddingService

    resolved = model_name or DEFAULT_MODEL_NAME
    return EmbeddingService(resolved), resolved


def _print_summary(run: EvaluationRun, output_dir: Path, root: Path) -> None:
    retrieval = run.retrieval_metrics
    test_recommended = run.test_metrics_recommended

    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        "离线 RAG 评估完成。",
        f"  语料文档数        : {run.chunk_configuration.document_count}",
        f"  语料 Chunk 数     : {run.chunk_configuration.chunk_count}",
        f"  题目总数          : {len(run.cases)}",
        f"  实际 Top-K        : {run.configuration.effective_top_k}",
        f"  Hit@1/3/5         : {fmt(retrieval.hit_at_1)} /"
        f" {fmt(retrieval.hit_at_3)} / {fmt(retrieval.hit_at_5)}",
        f"  MRR               : {fmt(retrieval.mean_reciprocal_rank)}",
        f"  当前阈值          : {run.configuration.current_threshold:.2f}",
        f"  推荐阈值          : {run.recommended_threshold:.2f}"
        "（仅由 calibration split 选出，不会自动写入生产配置）",
        f"  test 决策准确率   : {fmt(test_recommended.decision_accuracy)}",
        f"  test 错误放行率   : {fmt(test_recommended.false_answer_rate)}",
        f"  test 错误拒答率   : {fmt(test_recommended.false_refusal_rate)}",
        "报告已写入:",
    ]
    for filename in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
        lines.append(f"  {_display_path(output_dir / filename, root)}")
    print("\n".join(lines))


def main(
    argv: Sequence[str] | None = None,
    embedding_factory: EmbeddingFactory | None = None,
) -> int:
    """Run one evaluation and return a shell exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repository_root()
    factory = embedding_factory or default_embedding_factory

    manifest_path = _resolve_path(args.manifest, root)
    dataset_path = _resolve_path(args.dataset, root)
    output_dir = _resolve_path(args.output_dir, root)

    try:
        documents = load_manifest(manifest_path, root)
        corpus = load_corpus(documents)
        cases = load_dataset(dataset_path, corpus.document_ids)
    except RagError as exc:
        print(f"评估输入无效: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        embedding_service, embedding_model = factory(args.embedding_model)
    except Exception as exc:  # noqa: BLE001 - surface any model load failure
        print(
            "无法加载 Embedding 模型，评估未执行；"
            f"请确认模型已在本地缓存: {exc}",
            file=sys.stderr,
        )
        return EXIT_EMBEDDING_UNAVAILABLE

    try:
        run = run_evaluation(
            corpus=corpus,
            cases=cases,
            embedding_service=embedding_service,
            embedding_model=embedding_model,
            manifest_path=_display_path(manifest_path, root),
            dataset_path=_display_path(dataset_path, root),
            top_k=args.top_k,
            threshold_start=args.threshold_start,
            threshold_end=args.threshold_end,
            threshold_step=args.threshold_step,
            current_threshold=args.current_threshold,
            false_answer_weight=args.false_answer_weight,
            false_refusal_weight=args.false_refusal_weight,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
    except (RagError, ValueError) as exc:
        print(f"评估执行失败: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    write_reports(run, output_dir)
    _print_summary(run, output_dir, root)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

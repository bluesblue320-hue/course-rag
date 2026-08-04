"""Run the A/B reranking evaluation from the command line.

The command reuses the production retrieval pipeline, never calls a large
language model, and never modifies the production threshold.  Run it from
the repository root::

    python -m scripts.evaluate_reranker \\
      --manifest eval/corpus_manifest.json \\
      --dataset eval/dataset.jsonl \\
      --candidate-top-k 15 \\
      --final-top-k 5 \\
      --reranker-model "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from src.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.evaluation.corpus import load_corpus, load_manifest
from src.evaluation.dataset import load_dataset
from src.evaluation.report import SUMMARY_FILENAME as BASELINE_SUMMARY
from src.evaluation.reranking_report import (
    CASES_FILENAME,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
    write_reranking_reports,
)
from src.evaluation.reranking_runner import (
    FOCUS_CASE_IDS,
    run_reranking_evaluation,
)
from src.evaluation.runner import EmbeddingProtocol
from src.exceptions import RagError
from src.rag_service import DEFAULT_MIN_RELEVANCE_SCORE
from src.reranker import (
    DEFAULT_CANDIDATE_TOP_K,
    MAX_CANDIDATE_TOP_K,
    MIN_CANDIDATE_TOP_K,
)

DEFAULT_MANIFEST = "eval/corpus_manifest.json"
DEFAULT_DATASET = "eval/dataset.jsonl"
DEFAULT_OUTPUT_DIR = "reports/generated/reranking"
DEFAULT_FINAL_TOP_K = 5

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_EMBEDDING_UNAVAILABLE = 3
EXIT_RERANKER_UNAVAILABLE = 4

EmbeddingFactory = Callable[[str | None], tuple[EmbeddingProtocol, str]]
RerankerFactory = Callable[[str], object]


def repository_root() -> Path:
    """Return the repository root so the command works from any directory."""
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, with every default visible in ``--help``."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_reranker",
        description=(
            "对 vector-only 与 reranked 两种检索模式进行 A/B 对比评估。"
            "本命令不调用大模型，不修改生产配置，不写入运行时知识库。"
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
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
        help=f"向量检索候选池大小（最小 {MIN_CANDIDATE_TOP_K}，最大 {MAX_CANDIDATE_TOP_K}）",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=DEFAULT_FINAL_TOP_K,
        help="最终返回的 Top-K（最小 5）",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding 模型名称，缺省时使用生产默认模型",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="CrossEncoder Reranker 模型名称",
    )
    parser.add_argument(
        "--comparison-threshold",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help="对比拒答阈值（仓库默认值，不一定等于实际部署值）",
    )
    parser.add_argument(
        "--recommended-threshold",
        type=float,
        default=0.49,
        help="推荐拒答阈值（仅用于决策一致性检查）",
    )
    parser.add_argument(
        "--include-latency",
        action="store_true",
        default=False,
        help="是否测量并输出延迟数据（默认关闭，以保证确定性）",
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
    """Instantiate the production embedding service, importing it lazily."""
    from src.embedding import DEFAULT_MODEL_NAME, EmbeddingService

    resolved = model_name or DEFAULT_MODEL_NAME
    return EmbeddingService(resolved), resolved


def default_reranker_factory(model_name: str) -> object:
    """Instantiate the CrossEncoder reranker, importing it lazily."""
    from src.reranker import CrossEncoderReranker

    return CrossEncoderReranker(model_name)


def _validate_args(args: argparse.Namespace) -> list[str]:
    """Validate CLI arguments.  Returns a list of error messages."""
    errors: list[str] = []
    if args.candidate_top_k < MIN_CANDIDATE_TOP_K:
        errors.append(
            f"--candidate-top-k 不能小于 {MIN_CANDIDATE_TOP_K}"
        )
    if args.candidate_top_k > MAX_CANDIDATE_TOP_K:
        errors.append(
            f"--candidate-top-k 不能大于 {MAX_CANDIDATE_TOP_K}"
        )
    if args.final_top_k < 5:
        errors.append("--final-top-k 不能小于 5")
    if args.candidate_top_k < args.final_top_k:
        errors.append("--candidate-top-k 必须大于等于 --final-top-k")
    if not args.reranker_model:
        errors.append("--reranker-model 不能为空")
    return errors


def _print_summary(run, output_dir: Path, root: Path) -> None:
    vm = run.vector_metrics
    rm = run.reranked_metrics
    cm = run.candidate_metrics

    def fmt(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.4f}"

    lines = [
        "RAG Reranking A/B 评估完成。",
        f"  Embedding 模型   : {run.configuration.embedding_model}",
        f"  Reranker 模型    : {run.configuration.reranker_model}",
        f"  Candidate Top-K  : {run.configuration.candidate_top_k}",
        f"  Final Top-K      : {run.configuration.final_top_k}",
        f"  题目总数         : {len(run.cases)}",
        "",
        "  Candidate 指标:",
        f"    Hit@5/10/15    : {fmt(cm.hit_at_5)} / {fmt(cm.hit_at_10)} / {fmt(cm.hit_at_15)}",
        f"    MRR            : {fmt(cm.mean_reciprocal_rank)}",
        "",
        "  Vector-only 基线:",
        f"    Hit@1/3/5      : {fmt(vm.hit_at_1)} / {fmt(vm.hit_at_3)} / {fmt(vm.hit_at_5)}",
        f"    MRR            : {fmt(vm.mean_reciprocal_rank)}",
        "",
        "  Reranked 结果:",
        f"    Hit@1/3/5      : {fmt(rm.hit_at_1)} / {fmt(rm.hit_at_3)} / {fmt(rm.hit_at_5)}",
        f"    MRR            : {fmt(rm.mean_reciprocal_rank)}",
        "",
        f"  改善案例         : {len(run.improved_case_ids)}",
        f"  退化案例         : {len(run.regressed_case_ids)}",
        f"  决策一致性       : {'通过' if run.decision_invariance_passed else '未通过'}",
        f"  建议启用         : {'是' if run.recommend_enable else '否'}",
        "",
        "报告已写入:",
    ]
    for filename in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
        lines.append(f"  {_display_path(output_dir / filename, root)}")
    print("\n".join(lines))


def main(
    argv: Sequence[str] | None = None,
    embedding_factory: EmbeddingFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
) -> int:
    """Run one A/B evaluation and return a shell exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate args before loading any model
    errors = _validate_args(args)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return EXIT_INVALID_INPUT

    root = repository_root()
    emb_factory = embedding_factory or default_embedding_factory
    rnk_factory = reranker_factory or default_reranker_factory

    manifest_path = _resolve_path(args.manifest, root)
    dataset_path = _resolve_path(args.dataset, root)
    output_dir = _resolve_path(args.output_dir, root)

    # Load corpus and dataset
    try:
        documents = load_manifest(manifest_path, root)
        corpus = load_corpus(documents)
        cases = load_dataset(dataset_path, corpus.document_ids)
    except RagError as exc:
        print(f"评估输入无效: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    # Load embedding model
    try:
        embedding_service, embedding_model = emb_factory(args.embedding_model)
    except Exception as exc:  # noqa: BLE001
        print(
            f"无法加载 Embedding 模型: {exc}",
            file=sys.stderr,
        )
        return EXIT_EMBEDDING_UNAVAILABLE

    # Load reranker model
    try:
        reranker = rnk_factory(args.reranker_model)
    except Exception as exc:  # noqa: BLE001
        print(
            f"无法加载 Reranker 模型: {exc}",
            file=sys.stderr,
        )
        return EXIT_RERANKER_UNAVAILABLE

    # Run evaluation
    try:
        run = run_reranking_evaluation(
            cases=cases,
            corpus_chunks=corpus.chunks,
            embedding_service=embedding_service,
            embedding_model=embedding_model,
            reranker=reranker,
            reranker_model=args.reranker_model,
            manifest_path=_display_path(manifest_path, root),
            dataset_path=_display_path(dataset_path, root),
            candidate_top_k=args.candidate_top_k,
            final_top_k=args.final_top_k,
            comparison_threshold=args.comparison_threshold,
            recommended_threshold=args.recommended_threshold,
            include_latency=args.include_latency,
        )
    except (RagError, ValueError) as exc:
        print(f"评估执行失败: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    write_reranking_reports(run, output_dir)
    _print_summary(run, output_dir, root)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the offline RAG evaluation and threshold calibration CLI.

Example:
    python -m scripts.evaluate_rag \\
        --manifest eval/corpus_manifest.json \\
        --dataset eval/dataset.jsonl \\
        --top-k 5
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.embedding import EmbeddingService
from src.evaluation.models import EvaluationDataError, EvaluationError
from src.evaluation.runner import EvaluationRunner
from src.evaluation.threshold import ThresholdError
from src.rag_service import DEFAULT_MIN_RELEVANCE_SCORE

DEFAULT_MANIFEST = "eval/corpus_manifest.json"
DEFAULT_DATASET = "eval/dataset.jsonl"
DEFAULT_OUTPUT_DIR = "reports/generated/rag-evaluation"


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="离线评估 RAG 检索、拒答决策并校准相关性阈值。"
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="评估语料清单路径（默认 %(default)s）",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="评估数据集 JSONL 路径（默认 %(default)s）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索返回条数，至少 5 以便计算 Hit@5（默认 %(default)s）",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.20,
        help="阈值扫描起点（默认 %(default)s）",
    )
    parser.add_argument(
        "--threshold-end",
        type=float,
        default=0.60,
        help="阈值扫描终点（默认 %(default)s）",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="阈值扫描步长（默认 %(default)s）",
    )
    parser.add_argument(
        "--current-threshold",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help="当前生产阈值，用于对比（默认 %(default)s）",
    )
    parser.add_argument(
        "--false-answer-weight",
        type=float,
        default=3.0,
        help="错误放行（FP）权重（默认 %(default)s）",
    )
    parser.add_argument(
        "--false-refusal-weight",
        type=float,
        default=1.0,
        help="错误拒答（FN）权重（默认 %(default)s）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Chunk 大小（默认 %(default)s）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk 重叠（默认 %(default)s）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="报告输出目录（默认 %(default)s）",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if not 0 <= args.chunk_overlap < args.chunk_size:
        raise ValueError("chunk_overlap 必须介于 0 和 chunk_size 之间")
    from src.evaluation.threshold import validate_scan_parameters

    validate_scan_parameters(
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    for weight in (args.false_answer_weight, args.false_refusal_weight):
        if weight < 0:
            raise ValueError("权重不能为负数")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation and write reports, returning an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except (ValueError, ThresholdError) as exc:
        parser.error(str(exc))

    top_k = max(args.top_k, 5)
    if top_k != args.top_k:
        print(
            f"top_k 提升为 {top_k}，以确保能计算 Hit@5。",
            file=sys.stderr,
        )

    try:
        embedding_service = EmbeddingService()
        runner = EvaluationRunner(
            embedding_service,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        report_data = runner.run(
            manifest_path=Path(args.manifest),
            dataset_path=Path(args.dataset),
            top_k=top_k,
            threshold_start=args.threshold_start,
            threshold_end=args.threshold_end,
            threshold_step=args.threshold_step,
            current_threshold=args.current_threshold,
            false_answer_weight=args.false_answer_weight,
            false_refusal_weight=args.false_refusal_weight,
            output_dir=Path(args.output_dir),
        )
    except (EvaluationDataError, ThresholdError, EvaluationError) as exc:
        print(f"评估失败：{exc}", file=sys.stderr)
        return 1

    retrieval = report_data["retrieval_metrics"]["overall"]
    print("评估完成。")
    print(
        f"数据集：{report_data['dataset_counts']['total']} 题"
        f"（calibration {report_data['split_counts']['calibration']} / "
        f"test {report_data['split_counts']['test']}）"
    )
    print(
        f"Hit@1={retrieval['hit_at_1']} "
        f"Hit@3={retrieval['hit_at_3']} "
        f"Hit@5={retrieval['hit_at_5']} "
        f"MRR={retrieval['mrr']}"
    )
    print(
        f"当前阈值 {report_data['current_threshold']}，"
        f"推荐阈值 {report_data['recommended_threshold']}"
    )
    print(f"报告已写入：{args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

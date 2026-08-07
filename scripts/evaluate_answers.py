"""Score or generate final RAG answers and write deterministic reports.

Two mutually exclusive modes:

- offline (default): score an existing responses JSONL.  No model, database,
  or network is touched.
- ``--live``: explicitly run the production RagService over every case with a
  real embedding service and generation service.  This requires configured
  credentials and is never invoked implicitly.

Run from the repository root::

    python -m scripts.evaluate_answers --responses <path>
    python -m scripts.evaluate_answers --live

``--help`` never loads torch, sentence-transformers, or any model runtime.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from src.evaluation.answer_runner import (
    AnswerEvaluationRun,
    EmbeddingServiceProtocol,
    GenerationServiceProtocol,
    evaluate_answers_offline,
    write_responses_jsonl,
)
from src.evaluation.answer_report import (
    CASES_FILENAME,
    REPORT_FILENAME,
    SUMMARY_FILENAME,
)
from src.evaluation.dataset import load_dataset
from src.evaluation.answer_annotations import load_answer_annotations
from src.exceptions import (
    AnswerAnnotationValidationError,
    AnswerEvaluationError,
    AnswerResponseValidationError,
    GenerationConfigurationError,
    GenerationError,
    RagError,
)
from src.rag_service import DEFAULT_MIN_RELEVANCE_SCORE

DEFAULT_MANIFEST = "eval/corpus_manifest.json"
DEFAULT_DATASET = "eval/dataset.jsonl"
DEFAULT_ANNOTATIONS = "eval/answer_annotations.jsonl"
DEFAULT_OUTPUT_DIR = "reports/generated/answer-quality"
DEFAULT_TOP_K = 3
DEFAULT_PROMPT_VERSION = "prompt-builder-v1"

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_INVALID_RESPONSES = 3
EXIT_EMBEDDING_UNAVAILABLE = 4
EXIT_LLM_CONFIG_INVALID = 5
EXIT_LLM_CALL_FAILED = 6
EXIT_REPORT_WRITE_FAILED = 7

EmbeddingFactory = Callable[
    [str | None],
    tuple[EmbeddingServiceProtocol, str],
]
GenerationFactory = Callable[[], GenerationServiceProtocol]


def repository_root() -> Path:
    """Return the repository root so the command works from any directory."""
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, with every default visible in ``--help``."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_answers",
        description=(
            "评估最终回答质量：基于人工标注事实与引用的确定性检查。"
            "离线模式评分已有回答结果文件；--live 模式显式调用生产 RAG 管线"
            "生成新回答。本命令不使用 LLM Judge、不做语义相似度评分，"
            "--help 不会加载任何模型。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="评估语料清单路径，相对仓库根目录解析（仅 --live 使用）",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="评估数据集 JSONL 路径，相对仓库根目录解析",
    )
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS,
        help="回答质量标注 JSONL 路径，相对仓库根目录解析",
    )
    parser.add_argument(
        "--responses",
        default=None,
        help="回答结果 JSONL 路径，相对仓库根目录解析（离线模式）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="报告输出目录，相对仓库根目录解析",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="显式调用真实 Embedding 与 LLM 生成回答（与 --responses 互斥）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="检索返回条数（仅 --live 使用）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help="相关性拒答阈值（仅 --live 使用，不会写入任何生产配置）",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding 模型名称，缺省时使用 src.embedding 的生产默认模型",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="本次运行的名称，用于报告标识",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="模型标签（例如 local-configured-model），离线模式可为 null",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt 版本标签",
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
) -> tuple[EmbeddingServiceProtocol, str]:
    """Instantiate the production embedding service, importing it lazily."""
    from src.embedding import DEFAULT_MODEL_NAME, EmbeddingService

    resolved = model_name or DEFAULT_MODEL_NAME
    return EmbeddingService(resolved), resolved  # type: ignore[return-value]


def default_generation_factory() -> GenerationServiceProtocol:
    """Instantiate the production generation service from environment config."""
    from src.generation import GenerationService

    return GenerationService()


def _print_summary(run: AnswerEvaluationRun, output_dir: Path, root: Path) -> None:
    aggregate = run.aggregate_metrics

    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        "回答质量评估完成。",
        f"  运行模式          : {run.configuration.mode}",
        f"  题目总数          : {aggregate.case_count}",
        f"  决策准确率        : {fmt(aggregate.decision_accuracy)}",
        f"  错误放行率        : {fmt(aggregate.false_answer_rate)}",
        f"  错误拒答率        : {fmt(aggregate.false_refusal_rate)}",
        f"  平均标注事实覆盖率: {fmt(aggregate.average_fact_coverage)}",
        f"  平均来源支持覆盖率: {fmt(aggregate.average_grounded_fact_coverage)}",
        f"  引用合法率        : {fmt(aggregate.citation_validity_rate)}",
        f"  严格标注通过率    : {fmt(aggregate.strict_pass_rate)}",
        "  说明              : 标注级确定性指标，不是绝对正确率",
        "报告已写入:",
    ]
    for filename in (SUMMARY_FILENAME, CASES_FILENAME, REPORT_FILENAME):
        lines.append(f"  {_display_path(output_dir / filename, root)}")
    print("\n".join(lines))


def _run_live(
    args: argparse.Namespace,
    root: Path,
    embedding_factory: EmbeddingFactory,
    generation_factory: GenerationFactory,
) -> int:
    manifest_path = _resolve_path(args.manifest, root)
    dataset_path = _resolve_path(args.dataset, root)
    annotations_path = _resolve_path(args.annotations, root)
    output_dir = _resolve_path(args.output_dir, root)

    try:
        from src.evaluation.corpus import load_corpus, load_manifest

        documents = load_manifest(manifest_path, root)
        corpus = load_corpus(documents)
        cases = load_dataset(dataset_path, corpus.document_ids)
        annotations = load_answer_annotations(annotations_path, cases)
    except (RagError, AnswerAnnotationValidationError) as exc:
        print(f"评估输入无效: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        embedding_service, embedding_model = embedding_factory(
            args.embedding_model
        )
    except Exception as exc:  # noqa: BLE001 - surface any model load failure
        print(
            "无法加载 Embedding 模型，评估未执行；"
            f"请确认模型已在本地缓存: {exc}",
            file=sys.stderr,
        )
        return EXIT_EMBEDDING_UNAVAILABLE

    try:
        generation_service = generation_factory()
    except GenerationConfigurationError as exc:
        print(f"LLM 配置无效: {exc}", file=sys.stderr)
        return EXIT_LLM_CONFIG_INVALID

    from src.evaluation.answer_runner import run_live_answer_evaluation

    try:
        run = run_live_answer_evaluation(
            corpus=corpus,
            cases=cases,
            annotations=annotations,
            embedding_service=embedding_service,
            generation_service=generation_service,
            embedding_model=embedding_model,
            top_k=args.top_k,
            threshold=args.threshold,
            dataset_path=_display_path(dataset_path, root),
            annotations_path=_display_path(annotations_path, root),
            run_name=args.run_name or "live",
            model_label=args.model_label,
            prompt_version=args.prompt_version,
        )
    except GenerationError as exc:
        print(f"LLM 调用失败: {exc}", file=sys.stderr)
        return EXIT_LLM_CALL_FAILED
    except (RagError, ValueError) as exc:
        print(f"评估执行失败: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        write_responses_jsonl(run.responses, output_dir)
        from src.evaluation.answer_report import write_answer_reports

        write_answer_reports(run, output_dir)
    except OSError as exc:
        print(f"报告写入失败: {exc}", file=sys.stderr)
        return EXIT_REPORT_WRITE_FAILED

    _print_summary(run, output_dir, root)
    return EXIT_OK


def _run_offline(args: argparse.Namespace, root: Path) -> int:
    assert args.responses is not None
    dataset_path = _resolve_path(args.dataset, root)
    annotations_path = _resolve_path(args.annotations, root)
    responses_path = _resolve_path(args.responses, root)
    output_dir = _resolve_path(args.output_dir, root)

    # Display paths are computed once and reused: reports must never contain
    # absolute local paths, only repository-relative paths or bare filenames.
    dataset_display = _display_path(dataset_path, root)
    annotations_display = _display_path(annotations_path, root)
    responses_display = _display_path(responses_path, root)

    # Each input is loaded exactly once; the same parsed values feed the
    # runner so exit-code classification never re-reads the files.
    try:
        from src.evaluation.answer_responses import load_answer_responses

        cases = load_dataset(dataset_path)
        annotations = load_answer_annotations(annotations_path, cases)
        responses = load_answer_responses(responses_path, cases)
    except AnswerAnnotationValidationError as exc:
        print(f"标注无效: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except AnswerResponseValidationError as exc:
        print(f"回答结果无效: {exc}", file=sys.stderr)
        return EXIT_INVALID_RESPONSES
    except RagError as exc:
        print(f"评估输入无效: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT

    try:
        run = evaluate_answers_offline(
            cases=cases,
            annotations=annotations,
            responses=responses,
            dataset_path=dataset_display,
            annotations_path=annotations_display,
            responses_path=responses_display,
            output_dir=output_dir,
            run_name=args.run_name or "offline",
            model_label=args.model_label,
            prompt_version=args.prompt_version,
        )
    except (RagError, ValueError) as exc:
        print(f"评估执行失败: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except OSError as exc:
        print(f"报告写入失败: {exc}", file=sys.stderr)
        return EXIT_REPORT_WRITE_FAILED

    _print_summary(run, output_dir, root)
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    embedding_factory: EmbeddingFactory | None = None,
    generation_factory: GenerationFactory | None = None,
) -> int:
    """Run one answer-quality evaluation and return a shell exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repository_root()

    if args.live and args.responses is not None:
        print(
            "错误: --live 与 --responses 互斥，只能选择其中一种模式",
            file=sys.stderr,
        )
        return EXIT_INVALID_INPUT
    if not args.live and args.responses is None:
        print(
            "错误: 必须显式指定 --responses 或 --live 之一",
            file=sys.stderr,
        )
        return EXIT_INVALID_INPUT

    if args.live:
        embedding = embedding_factory or default_embedding_factory
        generation = generation_factory or default_generation_factory
        return _run_live(
            args,
            root,
            embedding,
            generation,
        )
    return _run_offline(args, root)


if __name__ == "__main__":
    raise SystemExit(main())

"""Static matplotlib charts summarizing an EvaluationReport, saved as PNGs.

This is a CLI tool (see evaluate.py), not a Streamlit page, so plots are
files on disk rather than interactive figures - matplotlib in headless
("Agg") mode, no display needed.
"""

import logging
from pathlib import Path
from typing import Dict, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluator.pipeline import EvaluationReport

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def _save(fig, output_dir: Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.debug("Saved plot to %s", path)
    return path


def _distribution_plot(values, title, xlabel, output_dir, filename) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(10, max(3, len(values))), edgecolor="black", alpha=0.75, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Question Count")
    return _save(fig, output_dir, filename)


def plot_precision_distribution(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    values = [r.precision_at_k for r in report.per_question]
    return _distribution_plot(values, "Precision@K Distribution", "Precision@K", output_dir, "precision_distribution.png")


def plot_recall_distribution(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    values = [r.recall_at_k for r in report.per_question]
    return _distribution_plot(values, "Recall@K Distribution", "Recall@K", output_dir, "recall_distribution.png")


def plot_f1_distribution(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    values = [r.f1 for r in report.per_question]
    return _distribution_plot(values, "F1 Score Distribution", "F1 Score", output_dir, "f1_distribution.png")


def plot_mrr_distribution(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    values = [r.reciprocal_rank for r in report.per_question]
    return _distribution_plot(
        values, "Reciprocal Rank Distribution (per-question MRR components)",
        "Reciprocal Rank", output_dir, "mrr_distribution.png",
    )


def plot_ndcg_distribution(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    values = [r.ndcg_at_k for r in report.per_question]
    return _distribution_plot(values, "nDCG@K Distribution", "nDCG@K", output_dir, "ndcg_distribution.png")


def plot_latency_distribution(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    retrieval = [r.retrieval_time_ms for r in report.per_question]
    generation = [r.generation_time_ms for r in report.per_question]
    ax.hist(retrieval, bins=10, alpha=0.6, label="Retrieval Time (ms)", edgecolor="black", color="steelblue")
    if any(generation):
        ax.hist(generation, bins=10, alpha=0.6, label="Generation Time (ms)", edgecolor="black", color="darkorange")
    ax.set_title("Latency Distribution")
    ax.set_xlabel("Milliseconds")
    ax.set_ylabel("Question Count")
    ax.legend()
    return _save(fig, output_dir, "latency_distribution.png")


def plot_retrieval_score_histogram(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    # FAISS (L2 distance, roughly 0-1) and BM25 (unbounded term-weight score)
    # live on very different scales - a shared x-axis would squash the FAISS
    # bars into a sliver, so each gets its own panel.
    faiss_scores = [
        t.faiss_score for r in report.per_question for t in r.chunk_traces if t.faiss_score is not None
    ]
    bm25_scores = [
        t.bm25_score for r in report.per_question for t in r.chunk_traces if t.bm25_score is not None
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(faiss_scores, bins=15, color="steelblue", edgecolor="black")
    axes[0].set_title("FAISS Scores (L2 distance, lower=closer)")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Chunk Count")

    axes[1].hist(bm25_scores, bins=15, color="darkorange", edgecolor="black")
    axes[1].set_title("BM25 Scores (higher=closer)")
    axes[1].set_xlabel("Score")

    fig.suptitle("Retrieval Score Histogram (across all retrieved chunks)")
    fig.subplots_adjust(wspace=0.3)
    return _save(fig, output_dir, "retrieval_score_histogram.png")


def plot_evaluation_summary_dashboard(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    rows = report.as_display_rows()
    # 0-1 quality metrics and millisecond timings are on very different
    # scales, so they get separate panels rather than one unreadable bar chart.
    quality_rows = [(label, float(value)) for label, value in rows if not value.endswith("ms")]
    timing_rows = [(label, float(value.replace(" ms", ""))) for label, value in rows if value.endswith("ms")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh([label for label, _ in quality_rows], [value for _, value in quality_rows], color="steelblue")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title("Quality Metrics (0-1)")
    axes[0].invert_yaxis()

    axes[1].barh([label for label, _ in timing_rows], [value for _, value in timing_rows], color="darkorange")
    axes[1].set_title("Timing Metrics (ms)")
    axes[1].invert_yaxis()

    fig.suptitle(f"Evaluation Summary (k={report.k}, n={len(report.per_question)} questions)")
    fig.subplots_adjust(wspace=0.5)
    return _save(fig, output_dir, "evaluation_summary_dashboard.png")


def plot_confusion_matrix(report: EvaluationReport, output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    """Retrieval-adapted confusion matrix, aggregated across the whole eval run.

    Standard TP/FP/FN/TN doesn't map cleanly onto open-domain retrieval: the
    corpus isn't exhaustively judged (see pipeline.py's module docstring), so
    "true negative" (a chunk correctly NOT retrieved, out of *every* chunk
    that wasn't) has no well-defined denominator here. TP/FP/FN are
    well-defined from what was actually retrieved vs. the relevance
    judgments and are shown; TN is marked N/A rather than fabricated.
    """
    tp = sum(sum(r.relevance_flags) for r in report.per_question)
    fp = sum(len(r.relevance_flags) - sum(r.relevance_flags) for r in report.per_question)
    fn = sum(max(r.total_relevant_estimate - sum(r.relevance_flags), 0) for r in report.per_question)

    display_matrix = np.array([[tp, fn], [fp, np.nan]], dtype=float)
    labels = [[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", "TN\nN/A"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="whitesmoke")
    ax.imshow(display_matrix, cmap=cmap)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Relevant", "Not Relevant"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Retrieved", "Not Retrieved"])
    ax.set_title("Retrieval Confusion Matrix (aggregated)")
    return _save(fig, output_dir, "confusion_matrix.png")


def generate_all_plots(
    report: EvaluationReport, output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR
) -> Dict[str, Path]:
    plot_functions = [
        plot_precision_distribution, plot_recall_distribution, plot_f1_distribution,
        plot_mrr_distribution, plot_ndcg_distribution, plot_latency_distribution,
        plot_retrieval_score_histogram, plot_evaluation_summary_dashboard, plot_confusion_matrix,
    ]
    paths = {fn.__name__: fn(report, output_dir) for fn in plot_functions}
    logger.info("Generated %d evaluation plot(s) in %s", len(paths), output_dir)
    return paths

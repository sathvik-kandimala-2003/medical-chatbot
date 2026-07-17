"""CLI entrypoint for the offline evaluation pipeline.

Usage:
    python evaluate.py
    python evaluate.py --dataset path/to/qa.json --k 5
    python evaluate.py --retrieval-only        # skip Gemini calls, just score retrieval
    python evaluate.py --output-dir my_run/

Runs a QA dataset through the exact same HybridRetriever (and, unless
--retrieval-only, the exact same Gemini generation path) the chatbot uses in
production, scores retrieval quality, and writes CSV/JSON/Markdown reports
plus PNG charts.
"""

import argparse
import logging
import os

from dotenv import find_dotenv, load_dotenv

from evaluation.datasets import load_dataset
from evaluation.datasets.loader import DEFAULT_DATASET_PATH
from evaluation.evaluator import Evaluator
from evaluation.plots import generate_all_plots
from evaluation.plots.generator import DEFAULT_OUTPUT_DIR as PLOTS_OUTPUT_DIR
from evaluation.reports import write_all_reports
from evaluation.reports.writer import DEFAULT_OUTPUT_DIR as REPORTS_OUTPUT_DIR
from retrieval.config import RetrievalConfig
from retrieval.loaders import build_hybrid_retriever

load_dotenv(find_dotenv())

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation of the chatbot's retrieval pipeline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to a QA dataset JSON file.")
    parser.add_argument("--k", type=int, default=5, help="Top-K cutoff for Precision@K/Recall@K/nDCG@K.")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Skip Gemini answer generation (faster, no API cost; Average Generation Time will read 0).",
    )
    parser.add_argument("--reports-dir", default=str(REPORTS_OUTPUT_DIR), help="Where to write CSV/JSON/Markdown reports.")
    parser.add_argument("--plots-dir", default=str(PLOTS_OUTPUT_DIR), help="Where to write PNG charts.")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading dataset from %s", args.dataset)
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("Dataset is empty - nothing to evaluate.")
        raise SystemExit(1)

    logger.info("Building the hybrid retrieval pipeline (same code path the chatbot uses)")
    hybrid_retriever = build_hybrid_retriever(config=RetrievalConfig(rerank_top_n=args.k))

    evaluator = Evaluator(hybrid_retriever, k=args.k, generate_answers=not args.retrieval_only)
    report = evaluator.evaluate_dataset(dataset)

    report_paths = write_all_reports(report, output_dir=args.reports_dir)
    plot_paths = generate_all_plots(report, output_dir=args.plots_dir)

    print("\n=== Offline Evaluation Summary ===")
    print(f"Questions evaluated: {len(report.per_question)} (k={report.k})")
    for label, value in report.as_display_rows():
        print(f"  {label}: {value}")

    print("\nReports:")
    for name, path in report_paths.items():
        print(f"  {name}: {path}")

    print(f"\nPlots ({len(plot_paths)} files): {args.plots_dir}")


if __name__ == "__main__":
    main()

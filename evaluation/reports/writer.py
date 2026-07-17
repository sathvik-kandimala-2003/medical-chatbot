"""CSV / JSON / Markdown report generation from an EvaluationReport.

Generated files go to output/ (git-ignored) next to this module - code and
generated artifacts stay in separate places on purpose.
"""

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Union

from evaluation.evaluator.pipeline import EvaluationReport

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

_CSV_FIELDS = [
    "question", "precision_at_k", "recall_at_k", "f1", "reciprocal_rank",
    "average_precision", "hit", "ndcg_at_k", "total_relevant_estimate",
    "retrieval_time_ms", "generation_time_ms", "retrieved_sources", "generation_error",
]


def write_csv_report(report: EvaluationReport, output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR) -> Path:
    """One row per question - the per-question evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "per_question_report.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for result in report.per_question:
            row = asdict(result)
            row["retrieved_sources"] = "; ".join(row["retrieved_sources"])
            writer.writerow({field: row[field] for field in _CSV_FIELDS})

    logger.info("Wrote CSV report to %s", path)
    return path


def write_json_report(report: EvaluationReport, output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR) -> Path:
    """Full report - overall statistics plus every per-question result, including chunk traces."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation_report.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    logger.info("Wrote JSON report to %s", path)
    return path


def write_markdown_report(report: EvaluationReport, output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.md"

    lines = [
        "# Offline Evaluation Summary",
        "",
        f"- Generated: {report.generated_at}",
        f"- k = {report.k}",
        f"- Questions evaluated: {len(report.per_question)}",
        "",
        "## Overall Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {label} | {value} |" for label, value in report.as_display_rows()]

    lines += [
        "",
        "## Per-Question Evaluation",
        "",
        "| Question | P@K | R@K | F1 | RR | AP | Hit | nDCG@K | Retrieval (ms) | Generation (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.per_question:
        lines.append(
            f"| {r.question} | {r.precision_at_k:.2f} | {r.recall_at_k:.2f} | {r.f1:.2f} | "
            f"{r.reciprocal_rank:.2f} | {r.average_precision:.2f} | {'yes' if r.hit else 'no'} | "
            f"{r.ndcg_at_k:.2f} | {r.retrieval_time_ms:.0f} | {r.generation_time_ms:.0f} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote Markdown summary to %s", path)
    return path


def write_all_reports(
    report: EvaluationReport, output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR
) -> Dict[str, Path]:
    return {
        "csv": write_csv_report(report, output_dir),
        "json": write_json_report(report, output_dir),
        "markdown": write_markdown_report(report, output_dir),
    }

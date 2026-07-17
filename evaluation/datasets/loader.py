"""Loads a QA dataset (JSON: a list of objects matching QAItem's fields) from disk."""

import json
import logging
from pathlib import Path
from typing import List, Union

from .schema import QAItem

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).parent / "sample_qa_dataset.json"


def load_dataset(path: Union[str, Path] = DEFAULT_DATASET_PATH) -> List[QAItem]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items = [
        QAItem(
            question=raw["question"],
            ground_truth_answer=raw.get("ground_truth_answer"),
            relevant_documents=raw.get("relevant_documents", []),
            relevant_chunks=raw.get("relevant_chunks", []),
            expected_sources=raw.get("expected_sources", []),
        )
        for raw in raw_items
    ]
    logger.info("Loaded %d question(s) from %s", len(items), path)
    return items

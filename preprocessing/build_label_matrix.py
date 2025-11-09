"""
Label matrix construction.

This module replaces the legacy script with a tidy workflow that:
1. Loads the processed session metadata JSON produced by `organize.py`.
2. Builds (or reuses) a symptom dictionary mapping labels to column indices.
3. Creates a full binary label matrix marking symptom/subject presence per session.
4. Optionally filters columns by minimum prevalence for downstream modelling.
5. Persists matrices and dictionaries in `data/processed/` for reproducibility.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

META_JSON_PATH = PROCESSED_DATA_DIR / "meta_cleaned.json"
SYMPTOM_DICTIONARY_PATH = PROCESSED_DATA_DIR / "symptom_dictionary.json"
FULL_LABEL_MATRIX_NPZ = PROCESSED_DATA_DIR / "label_matrix_full.npz"
FILTERED_LABEL_MATRIX_NPZ = PROCESSED_DATA_DIR / "label_matrix_filtered.npz"
FULL_LABEL_MATRIX_CSV = PROCESSED_DATA_DIR / "label_matrix_full.csv"
FILTERED_LABEL_MATRIX_CSV = PROCESSED_DATA_DIR / "label_matrix_filtered.csv"
FILTERED_DICTIONARY_PATH = PROCESSED_DATA_DIR / "symptom_dictionary_filtered.json"

# Metadata keys we inspect for symptom labels.
SYMPTOM_KEYS = ("Symptoms", "Psyc_Subjects")
DEFAULT_MIN_PREVALENCE = 0.02


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #


def configure_logging() -> None:
    """Initialise the module logger."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# --------------------------------------------------------------------------- #
# Session utilities
# --------------------------------------------------------------------------- #


def load_sessions(meta_path: Path = META_JSON_PATH) -> Dict[str, Dict[str, object]]:
    """Load the processed session metadata blob."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Processed metadata not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {meta_path}, found {type(data).__name__}")
    logging.info("Loaded %d sessions from %s", len(data), meta_path)
    return data


def iter_sessions(data: Dict[str, Dict[str, object]]) -> Iterator[Tuple[str, Dict[str, object]]]:
    """Iterate over sessions in numeric order where possible."""

    def sort_key(key: str) -> Tuple[int, str]:
        try:
            return int(key), key
        except ValueError:
            return math.inf, key

    for session_id in sorted(data.keys(), key=sort_key):
        yield session_id, data[session_id]


# --------------------------------------------------------------------------- #
# Symptom dictionary utilities
# --------------------------------------------------------------------------- #


def _clean_term(term: str) -> str:
    """Normalise a raw symptom/subject label string."""
    term = term.strip()
    if term.startswith("[^]"):
        term = term[4:]
    return term.strip()


def extract_labels(session: Dict[str, object]) -> List[str]:
    """Collect the set of labels (symptoms + psych subjects) for a session."""
    labels: List[str] = []
    for key in SYMPTOM_KEYS:
        raw_value = session.get(key)
        if not raw_value or not isinstance(raw_value, str):
            continue
        parts = [part for part in raw_value.split(";") if part]
        labels.extend(_clean_term(part) for part in parts if _clean_term(part))
    return labels


def build_symptom_dictionary(sessions: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    """Construct a dictionary mapping symptom labels to column indices."""
    label_to_index: Dict[str, int] = {}
    for _, session in iter_sessions(sessions):
        for label in extract_labels(session):
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
    logging.info("Built symptom dictionary with %d entries.", len(label_to_index))
    return label_to_index


def load_or_build_dictionary(sessions: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    """Load an existing symptom dictionary or create a new one."""
    if SYMPTOM_DICTIONARY_PATH.exists():
        logging.info("Loading existing symptom dictionary from %s", SYMPTOM_DICTIONARY_PATH)
        with SYMPTOM_DICTIONARY_PATH.open("r", encoding="utf-8") as handle:
            dictionary = json.load(handle)
        if not isinstance(dictionary, dict):
            raise TypeError(
                f"Expected dictionary in {SYMPTOM_DICTIONARY_PATH}, found {type(dictionary).__name__}"
            )
        return {str(label): int(index) for label, index in dictionary.items()}

    dictionary = build_symptom_dictionary(sessions)
    write_json(dictionary, SYMPTOM_DICTIONARY_PATH)
    return dictionary


# --------------------------------------------------------------------------- #
# Label matrix construction
# --------------------------------------------------------------------------- #


@dataclass
class LabelMatrix:
    matrix: np.ndarray
    labels: List[str]
    session_ids: List[str]


def build_label_matrix(
    sessions: Dict[str, Dict[str, object]],
    label_to_index: Dict[str, int],
) -> LabelMatrix:
    """Create a binary label matrix (sessions x labels)."""
    session_ids: List[str] = []
    labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    matrix = np.zeros((len(sessions), len(label_to_index)), dtype=np.int8)

    for row_idx, (session_id, session) in enumerate(iter_sessions(sessions)):
        session_ids.append(session_id)
        for label in extract_labels(session):
            col_idx = label_to_index.get(label)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1

    logging.info(
        "Constructed label matrix with shape %s (sessions x labels).",
        matrix.shape,
    )
    return LabelMatrix(matrix=matrix, labels=labels, session_ids=session_ids)


def filter_labels(
    label_matrix: LabelMatrix,
    min_prevalence: float = DEFAULT_MIN_PREVALENCE,
) -> LabelMatrix:
    """Filter label columns by minimum prevalence threshold."""
    matrix = label_matrix.matrix
    if matrix.size == 0:
        return LabelMatrix(matrix=np.empty((matrix.shape[0], 0)), labels=[], session_ids=label_matrix.session_ids)

    prevalence = matrix.sum(axis=0) / matrix.shape[0]
    mask = prevalence >= min_prevalence

    filtered_matrix = matrix[:, mask]
    filtered_labels = [label for label, keep in zip(label_matrix.labels, mask) if keep]

    dropped = [label for label, keep in zip(label_matrix.labels, mask) if not keep]
    logging.info(
        "Filtered label matrix to %d columns (dropped %d labels below %.2f prevalence).",
        filtered_matrix.shape[1],
        len(dropped),
        min_prevalence,
    )

    if dropped:
        logging.debug("Dropped labels: %s", ", ".join(dropped[:25]))

    return LabelMatrix(
        matrix=filtered_matrix,
        labels=filtered_labels,
        session_ids=label_matrix.session_ids,
    )


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def write_json(data: object, path: Path) -> None:
    """Serialize Python data to JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    logging.info("Wrote %s", path)


def write_matrix(label_matrix: LabelMatrix, csv_path: Path, npz_path: Path) -> None:
    """Persist the label matrix to CSV and compressed NumPy archive."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(label_matrix.matrix, index=label_matrix.session_ids, columns=label_matrix.labels)
    df.index.name = "session_id"
    df.to_csv(csv_path)
    logging.info("Wrote %s", csv_path)

    np.savez_compressed(
        npz_path,
        data=label_matrix.matrix.astype(np.int8),
        labels=np.array(label_matrix.labels, dtype=object),
        session_ids=np.array(label_matrix.session_ids, dtype=object),
    )
    logging.info("Wrote %s", npz_path)


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #


def main(min_prevalence: float = DEFAULT_MIN_PREVALENCE) -> None:
    """Build and persist the label matrices."""
    configure_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions()
    label_to_index = load_or_build_dictionary(sessions)

    full_matrix = build_label_matrix(sessions, label_to_index)
    write_matrix(full_matrix, FULL_LABEL_MATRIX_CSV, FULL_LABEL_MATRIX_NPZ)

    filtered_matrix = filter_labels(full_matrix, min_prevalence=min_prevalence)
    write_matrix(filtered_matrix, FILTERED_LABEL_MATRIX_CSV, FILTERED_LABEL_MATRIX_NPZ)

    filtered_dictionary = {label: idx for idx, label in enumerate(filtered_matrix.labels)}
    write_json(filtered_dictionary, FILTERED_DICTIONARY_PATH)


if __name__ == "__main__":
    main()
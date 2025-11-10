"""
Label matrix construction with label merging.

This module replaces the legacy script with a tidy workflow that:
1. Loads the processed session metadata JSON produced by `organize.py`.
2. Builds (or reuses) a symptom dictionary mapping labels to column indices.
3. Creates a full binary label matrix marking symptom/subject presence per session.
4. Optionally filters columns by minimum prevalence for downstream modeling.
5. Optionally merges groups of label names into broader categories, resolved
   via `data/processed/symptom_dictionary_filtered.json`.
6. Persists matrices and dictionaries in `data/processed/` for reproducibility.
Expected merge groups file:
- data/processed/label_merge_groups.json
  Format: { "anxiety": ["Anxiety", "Panic attacks"], "depression": ["Depression"] }
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
MERGE_GROUPS_PATH = PROCESSED_DATA_DIR / "label_merge_groups.json"
MERGED_LABEL_MATRIX_CSV = PROCESSED_DATA_DIR / "label_matrix_merged.csv"
MERGED_LABEL_MATRIX_NPZ = PROCESSED_DATA_DIR / "label_matrix_merged.npz"
MERGED_WITH_NONE_CSV = PROCESSED_DATA_DIR / "label_matrix_merged_with_none.csv"
MERGED_WITH_NONE_NPZ = PROCESSED_DATA_DIR / "label_matrix_merged_with_none.npz"
MERGED_FILTERED_LABEL_MATRIX_CSV = PROCESSED_DATA_DIR / "label_matrix_merged_filtered.csv"
MERGED_FILTERED_LABEL_MATRIX_NPZ = PROCESSED_DATA_DIR / "label_matrix_merged_filtered.npz"
MERGED_FILTERED_WITH_NONE_CSV = PROCESSED_DATA_DIR / "label_matrix_merged_filtered_with_none.csv"
MERGED_FILTERED_WITH_NONE_NPZ = PROCESSED_DATA_DIR / "label_matrix_merged_filtered_with_none.npz"

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
    """Construct a dictionary mapping symptom labels to column indices (FULL)."""
    label_to_index: Dict[str, int] = {}
    for _, session in iter_sessions(sessions):
        for label in extract_labels(session):
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
    logging.info("Built FULL symptom dictionary with %d entries.", len(label_to_index))
    return label_to_index


def load_or_build_dictionary(sessions: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    """Load an existing FULL symptom dictionary or create a new one."""
    if SYMPTOM_DICTIONARY_PATH.exists():
        logging.info("Loading FULL symptom dictionary from %s", SYMPTOM_DICTIONARY_PATH)
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
# Merge utilities (merge-first, using FULL dictionary)
# --------------------------------------------------------------------------- #

def _load_merge_spec() -> Dict[str, Sequence[str]]:
    """Load merge groups from MERGE_GROUPS_PATH."""
    if MERGE_GROUPS_PATH.exists():
        with MERGE_GROUPS_PATH.open("r", encoding="utf-8") as fh:
            spec = json.load(fh)
        if not isinstance(spec, dict):
            raise TypeError(f"Expected dict in {MERGE_GROUPS_PATH}, found {type(spec).__name__}")
        # Normalize to list[str]
        norm = {str(k): [str(vv) for vv in (v or [])] for k, v in spec.items()}
        logging.info("Loaded %d merge groups from %s", len(norm), MERGE_GROUPS_PATH)
        return norm
    logging.info("Merge groups file not found at %s; skipping merge stage.", MERGE_GROUPS_PATH)
    return {}


def _resolve_merge_indices_full(
    merge_spec: Dict[str, Sequence[str]],
    full_dictionary_path: Path = SYMPTOM_DICTIONARY_PATH,
) -> Dict[str, List[int]]:
    """Resolve merge groups against the FULL dictionary (label -> index)."""
    if not full_dictionary_path.exists():
        raise FileNotFoundError(f"Full dictionary not found: {full_dictionary_path}")
    with full_dictionary_path.open("r", encoding="utf-8") as fh:
        label_to_index = json.load(fh)
    label_to_index = {str(k): int(v) for k, v in label_to_index.items()}

    resolved: Dict[str, List[int]] = {}
    missing: List[Tuple[str, str]] = []

    for merged_name, members in merge_spec.items():
        idxs: List[int] = []
        for name in members:
            name_clean = _clean_term(str(name))
            if name_clean in label_to_index:
                idxs.append(label_to_index[name_clean])
            else:
                missing.append((merged_name, name_clean))
        if idxs:
            resolved[merged_name] = sorted(set(idxs))

    if missing:
        sample = "; ".join([f"{m}→{n}" for m, n in missing[:20]])
        logging.warning(
            "Some merge members not found in FULL dictionary (showing up to 20): %s", sample
        )

    if not resolved:
        raise ValueError("No merge groups could be resolved; check your merge_spec label names.")
    return resolved


def merge_labels_from_full(
    full_label_matrix: LabelMatrix,
    merge_spec: Dict[str, Sequence[str]],
    full_dictionary_path: Path = SYMPTOM_DICTIONARY_PATH,
) -> LabelMatrix:
    """Merge related labels by OR'ing selected columns (indices resolved using the FULL dictionary)."""
    if full_label_matrix.matrix.size == 0:
        return LabelMatrix(matrix=np.empty((len(full_label_matrix.session_ids), 0), dtype=np.int8),
                           labels=[], session_ids=full_label_matrix.session_ids)

    resolved = _resolve_merge_indices_full(merge_spec, full_dictionary_path)

    merged_cols: List[np.ndarray] = []
    merged_names: List[str] = []

    for merged_name, col_indices in resolved.items():
        valid_cols = [i for i in col_indices if 0 <= i < full_label_matrix.matrix.shape[1]]
        if not valid_cols:
            continue
        col = (full_label_matrix.matrix[:, valid_cols].sum(axis=1) > 0).astype(np.int8)
        merged_cols.append(col)
        merged_names.append(merged_name)

    if not merged_cols:
        raise ValueError("Merge produced no columns; check your merge_spec indices/names.")

    merged_matrix = np.column_stack(merged_cols).astype(np.int8)
    logging.info("Built MERGED (unfiltered) label matrix with shape %s over %d groups.",
                 merged_matrix.shape, len(merged_names))

    return LabelMatrix(matrix=merged_matrix,
                       labels=merged_names,
                       session_ids=full_label_matrix.session_ids)


def with_none_indicator(label_matrix: LabelMatrix) -> LabelMatrix:
    """Prepend a 'NONE' indicator column (1 if row has no positive labels, else 0)."""
    none_col = (label_matrix.matrix.sum(axis=1) == 0).astype(np.int8).reshape(-1, 1)
    mat = np.hstack([none_col, label_matrix.matrix])
    labels = ["NONE"] + label_matrix.labels
    logging.info("Added NONE indicator; %d rows have no labels after stage.",
                 int(none_col.sum()))
    return LabelMatrix(matrix=mat, labels=labels, session_ids=label_matrix.session_ids)


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

def main(
    min_prevalence: float = DEFAULT_MIN_PREVALENCE,
    do_merge: bool = True,
    add_none_column: bool = True
) -> None:
    """Build and persist the label matrices (merge-first, then filter for merged)."""
    configure_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load sessions + FULL dictionary (or build)
    sessions = load_sessions()
    full_label_to_index = load_or_build_dictionary(sessions)

    # 2) Build FULL matrix (unfiltered, unmerged)
    full_matrix = build_label_matrix(sessions, full_label_to_index)
    write_matrix(full_matrix, FULL_LABEL_MATRIX_CSV, FULL_LABEL_MATRIX_NPZ)

    # 3) Filter the FULL (unmerged) matrix by prevalence (legacy/useful baseline)
    filtered_matrix = filter_labels(full_matrix, min_prevalence=min_prevalence)
    write_matrix(filtered_matrix, FILTERED_LABEL_MATRIX_CSV, FILTERED_LABEL_MATRIX_NPZ)

    # Persist filtered dictionary so downstream users can align with the filtered unmerged matrix.
    filtered_dictionary = {label: idx for idx, label in enumerate(filtered_matrix.labels)}
    write_json(filtered_dictionary, FILTERED_DICTIONARY_PATH)

    # 4) MERGE FIRST (using FULL dictionary / FULL matrix), THEN FILTER the merged result
    if do_merge:
        merge_spec = _load_merge_spec()
        if merge_spec:
            # Merge on the FULL matrix (indices resolved from FULL dictionary)
            merged_full = merge_labels_from_full(
                full_label_matrix=full_matrix,
                merge_spec=merge_spec,
                full_dictionary_path=SYMPTOM_DICTIONARY_PATH,
            )
            write_matrix(merged_full, MERGED_LABEL_MATRIX_CSV, MERGED_LABEL_MATRIX_NPZ)

            if add_none_column:
                merged_full_with_none = with_none_indicator(merged_full)
                write_matrix(merged_full_with_none, MERGED_WITH_NONE_CSV, MERGED_WITH_NONE_NPZ)

            # Now filter the MERGED matrix by prevalence
            merged_filtered = filter_labels(merged_full, min_prevalence=min_prevalence)
            write_matrix(merged_filtered, MERGED_FILTERED_LABEL_MATRIX_CSV, MERGED_FILTERED_LABEL_MATRIX_NPZ)

            if add_none_column:
                merged_filtered_with_none = with_none_indicator(merged_filtered)
                write_matrix(merged_filtered_with_none, MERGED_FILTERED_WITH_NONE_CSV, MERGED_FILTERED_WITH_NONE_NPZ)
        else:
            logging.info("Merge stage skipped (no groups provided).")


if __name__ == "__main__":
    main()
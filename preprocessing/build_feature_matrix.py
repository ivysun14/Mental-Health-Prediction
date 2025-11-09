"""
Feature extraction pipeline for client transcripts.

This module consolidates the earlier `process_words`, `remove_cheating_words`, and
`calc_NRC` scripts into a single, testable workflow that produces:

* A symptom dictionary derived from session metadata.
* A naive (non-stemmed) vocabulary and TF-IDF feature matrix.
* A stemmed vocabulary and TF-IDF feature matrix.

Inputs are taken from `data/processed/meta_cleaned.json`. Outputs are written to
`data/processed/` for downstream modelling steps.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Union

import numpy as np
from nltk.stem import PorterStemmer

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

META_JSON_PATH = PROCESSED_DATA_DIR / "meta_cleaned.json"
SYMPTOM_DICTIONARY_PATH = PROCESSED_DATA_DIR / "symptom_dictionary.json"
NAIVE_VOCAB_PATH = PROCESSED_DATA_DIR / "vocabulary_naive.json"
STEM_VOCAB_PATH = PROCESSED_DATA_DIR / "vocabulary_stem.json"
SESSION_IDS_PATH = PROCESSED_DATA_DIR / "feature_matrix_session_ids.json"
NAIVE_TFIDF_PATH = PROCESSED_DATA_DIR / "feature_matrix_naive_tfidf.npz"
STEM_TFIDF_PATH = PROCESSED_DATA_DIR / "feature_matrix_stem_tfidf.npz"
DICTIONARY_FEATURES_PATH = PROCESSED_DATA_DIR / "nrc_moesm_features.npz"
NAIVE_COMBINED_FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_naive_plus_dict.npz"
STEM_COMBINED_FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_stem_plus_dict.npz"
FEATURE_INDEX_MAP_NAIVE_PATH = PROCESSED_DATA_DIR / "feature_index_map_naive.json"
FEATURE_INDEX_MAP_STEM_PATH = PROCESSED_DATA_DIR / "feature_index_map_stem.json"
FEATURE_INDEX_MAP_PATH = FEATURE_INDEX_MAP_NAIVE_PATH
LABEL_MATRIX_PATH = PROCESSED_DATA_DIR / "label_matrix_filtered.npz"
ALIGNED_LABEL_MATRIX_PATH = PROCESSED_DATA_DIR / "label_matrix_aligned_with_features.npz"

# Regex utilities reused from the earlier scripts.
WORD_PATTERN = re.compile(r"\b[\w']+\b")

# Fixed set of terms to mask from transcripts to avoid label leakage.
CHEATING_TERMS = ("anxiety", "depression")

# Speaker-separated JSON structure keys.
CLIENT_TEXT_KEY = "Client_Text"
SYMPTOMS_KEY = "Symptoms"
PSYC_SUBJECTS_KEY = "Psyc_Subjects"

# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #


def configure_logging() -> None:
    """Initialise the logger."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# --------------------------------------------------------------------------- #
# Data loading and iteration helpers
# --------------------------------------------------------------------------- #


def load_sessions(meta_path: Path = META_JSON_PATH) -> Dict[str, Dict[str, object]]:
    """Load the processed session metadata + transcripts blob."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Processed metadata not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary in {meta_path}, found {type(data).__name__}.")

    logging.info("Loaded %d sessions from %s", len(data), meta_path)
    return data


def iter_sessions(data: Dict[str, Dict[str, object]]) -> Iterator[Tuple[str, Dict[str, object]]]:
    """
    Yield `(session_id, session_record)` pairs in a deterministic order.

    Keys in the JSON dump are usually numeric strings; we sort numerically when possible.
    """
    def sort_key(key: str) -> Tuple[int, str]:
        try:
            return int(key), key
        except ValueError:
            return math.inf, key

    for session_id in sorted(data.keys(), key=sort_key):
        yield session_id, data[session_id]


# --------------------------------------------------------------------------- #
# Symptom dictionary and cheating word removal
# --------------------------------------------------------------------------- #


def _clean_symptom_term(term: str) -> str:
    """Normalise symptom labels exported from metadata."""
    if term.startswith("[^]"):
        term = term[4:]
    return term.strip()


def build_symptom_dictionary(sessions: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    """
    Construct a mapping from symptom/subject labels to integer indices.

    We aggregate entries from both the `Symptoms` field and the `Psyc_Subjects` field
    to match the historical pipeline.
    """
    symptom_to_index: Dict[str, int] = {}

    for _, record in iter_sessions(sessions):
        for key in (SYMPTOMS_KEY, PSYC_SUBJECTS_KEY):
            raw_value = record.get(key)
            if not raw_value or not isinstance(raw_value, str):
                continue
            for term in raw_value.split(";"):
                cleaned = _clean_symptom_term(term)
                if not cleaned:
                    continue
                if cleaned not in symptom_to_index:
                    symptom_to_index[cleaned] = len(symptom_to_index)

    logging.info("Built symptom dictionary with %d entries.", len(symptom_to_index))
    return symptom_to_index


def build_cheating_tokens(base_terms: Iterable[str] = CHEATING_TERMS) -> Tuple[set[str], set[str]]:
    """
    Build lookups for removing label leakage words from the transcripts.

    Returns
    -------
    token_set:
        lower-cased unigram tokens extracted from symptom labels.
    stem_set:
        Porter-stemmed versions of the same tokens for use with stemmed vocabularies.
    """
    porter = PorterStemmer()
    tokens: set[str] = set()
    stems: set[str] = set()

    for term in base_terms:
        for token in WORD_PATTERN.findall(term.lower()):
            tokens.add(token)
            stems.add(porter.stem(token))

    logging.info(
        "Identified %d cheating tokens (%d stems) derived from configured leakage terms.",
        len(tokens),
        len(stems),
    )
    return tokens, stems


# --------------------------------------------------------------------------- #
# Tokenisation helpers
# --------------------------------------------------------------------------- #


def tokenize_client_text(record: Dict[str, object], cheating_tokens: set[str]) -> List[str]:
    """Tokenise client text, removing punctuation and symptom leakage tokens."""
    lines = record.get(CLIENT_TEXT_KEY, [])
    if not isinstance(lines, Sequence):
        return []

    tokens: List[str] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        for token in WORD_PATTERN.findall(line.lower()):
            if token in cheating_tokens:
                continue
            tokens.append(token)
    return tokens


def tokenize_and_stem(record: Dict[str, object], cheating_tokens: set[str], cheating_stems: set[str]) -> List[str]:
    """Tokenise client text and apply Porter stemming, removing symptom leakage tokens/stems."""
    porter = PorterStemmer()
    raw_tokens = tokenize_client_text(record, cheating_tokens)
    stemmed_tokens: List[str] = []
    for token in raw_tokens:
        stem = porter.stem(token)
        if stem in cheating_stems:
            continue
        stemmed_tokens.append(stem)
    return stemmed_tokens


# --------------------------------------------------------------------------- #
# Vocabulary and feature matrix construction
# --------------------------------------------------------------------------- #


def build_vocabulary(documents: Iterable[List[str]]) -> Dict[str, int]:
    """Assign an index to each unique token encountered across all documents."""
    vocabulary: Dict[str, int] = {}
    for doc in documents:
        for token in doc:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


def build_feature_matrix(documents: List[List[str]], vocabulary: Dict[str, int]) -> np.ndarray:
    """Create a dense term-frequency matrix (documents x vocabulary)."""
    n_docs = len(documents)
    vocab_size = len(vocabulary)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for row_idx, doc in enumerate(documents):
        counts = Counter(doc)
        for token, frequency in counts.items():
            col_idx = vocabulary[token]
            matrix[row_idx, col_idx] = frequency
    return matrix


def compute_tfidf(tf_matrix: np.ndarray) -> np.ndarray:
    """Apply scikit-style TF-IDF weighting to a count matrix."""
    if tf_matrix.size == 0:
        return tf_matrix.copy()

    n_docs = tf_matrix.shape[0]
    # Document frequency: number of docs with the term.
    df = (tf_matrix > 0).sum(axis=0)
    idf = np.log((n_docs + 1) / (df + 1)) + 1  # Matches sklearn's smoothing.

    tfidf = tf_matrix * idf
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    tfidf /= row_norms
    return tfidf


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def write_json(data: object, path: Path) -> None:
    """Serialize a Python object to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def write_feature_matrix(matrix: np.ndarray, path: Path) -> None:
    """Save a TF-IDF matrix in compressed NumPy format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=matrix)


def write_feature_bundle(
    matrix: np.ndarray,
    path: Path,
    columns: Sequence[str],
    session_ids: Sequence[str],
) -> None:
    """Persist a feature matrix with its metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=matrix.astype(np.float32),
        columns=np.array(list(columns), dtype=object),
        session_ids=np.array(list(session_ids), dtype=object),
    )


def write_label_bundle(matrix: np.ndarray, path: Path, session_ids: Sequence[str], labels: Sequence[str]) -> None:
    """Persist aligned label matrix with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=matrix.astype(np.int8),
        labels=np.array(list(labels), dtype=object),
        session_ids=np.array(list(session_ids), dtype=object),
    )
    logging.info("Wrote %s", path)


def load_npz_features(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load a feature NPZ with data, columns, and session ids."""
    if not path.exists():
        raise FileNotFoundError(f"Feature archive not found: {path}")
    bundle = np.load(path, allow_pickle=True)
    matrix = bundle["data"]
    columns = bundle["columns"].astype(object).tolist()
    session_ids = bundle["session_ids"].astype(str).tolist()
    return matrix, columns, session_ids


def align_matrix_to_sessions(
    matrix: np.ndarray,
    matrix_session_ids: Sequence[str],
    target_session_ids: Sequence[str],
) -> np.ndarray:
    """Align rows of `matrix` to match the order of `target_session_ids`."""
    index_lookup = {str(session_id): idx for idx, session_id in enumerate(matrix_session_ids)}
    aligned = np.zeros((len(target_session_ids), matrix.shape[1]), dtype=matrix.dtype)
    missing: List[str] = []

    for row_idx, session_id in enumerate(target_session_ids):
        source_idx = index_lookup.get(str(session_id))
        if source_idx is None:
            missing.append(str(session_id))
            continue
        aligned[row_idx] = matrix[source_idx]

    if missing:
        logging.warning(
            "Missing %d sessions in dictionary features (examples: %s)",
            len(missing),
            ", ".join(missing[:5]),
        )

    return aligned


def build_feature_index_map(
    vocabulary: Dict[str, int],
    dictionary_columns: Sequence[str],
    tfidf_source: str,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Create a mapping from combined feature index to feature metadata."""
    index_map: Dict[str, Dict[str, str]] = {}
    column_labels: List[str] = []

    sorted_tokens = sorted(vocabulary.items(), key=lambda item: item[1])
    for combined_index, (token, _) in enumerate(sorted_tokens):
        index_map[str(combined_index)] = {
            "source": tfidf_source,
            "feature": token,
        }
        column_labels.append(token)

    offset = len(sorted_tokens)
    for position, column in enumerate(dictionary_columns):
        index_map[str(offset + position)] = {
            "source": "dictionary",
            "feature": column,
        }
        column_labels.append(column)

    return index_map, column_labels


def lookup_feature(
    index: Union[int, str],
    map_path: Path = FEATURE_INDEX_MAP_PATH,
) -> Dict[str, str]:
    """Return metadata for a combined feature index."""
    if not map_path.exists():
        raise FileNotFoundError(f"Feature index map not found: {map_path}")

    with map_path.open("r", encoding="utf-8") as handle:
        index_map: Dict[str, Dict[str, str]] = json.load(handle)

    key = str(index)
    if key not in index_map:
        raise KeyError(f"Feature index {index} not present in {map_path}")
    return index_map[key]


def compute_label_matrix(
    session_ids: Sequence[str],
    label_matrix_path: Path = LABEL_MATRIX_PATH,
) -> Tuple[np.ndarray, List[str]]:
    """Load the label matrix, align it to `session_ids`, and return it along with labels."""
    if not label_matrix_path.exists():
        logging.warning("Label matrix archive not found at %s", label_matrix_path)
        return np.empty((len(session_ids), 0), dtype=np.int8), []

    bundle = np.load(label_matrix_path, allow_pickle=True)
    label_matrix = bundle["data"]
    label_session_ids = bundle["session_ids"].astype(str).tolist()
    label_columns = bundle["labels"].astype(str).tolist() if "labels" in bundle else []

    aligned = align_matrix_to_sessions(label_matrix, label_session_ids, session_ids)
    return aligned, label_columns


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #


def main() -> None:
    """Execute the end-to-end feature extraction workflow."""
    configure_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Project root: %s", PROJECT_ROOT)

    sessions = load_sessions()
    symptom_dictionary = build_symptom_dictionary(sessions)
    cheating_tokens, cheating_stems = build_cheating_tokens()

    naive_documents: List[List[str]] = []
    stem_documents: List[List[str]] = []
    session_ids: List[str] = []

    for session_id, record in iter_sessions(sessions):
        session_ids.append(session_id)
        naive_tokens = tokenize_client_text(record, cheating_tokens)
        stem_tokens = tokenize_and_stem(record, cheating_tokens, cheating_stems)
        naive_documents.append(naive_tokens)
        stem_documents.append(stem_tokens)

    naive_vocabulary = build_vocabulary(naive_documents)
    stem_vocabulary = build_vocabulary(stem_documents)

    logging.info(
        "Constructed vocabularies | naive: %d terms | stemmed: %d terms",
        len(naive_vocabulary),
        len(stem_vocabulary),
    )

    naive_counts = build_feature_matrix(naive_documents, naive_vocabulary)
    stem_counts = build_feature_matrix(stem_documents, stem_vocabulary)

    naive_tfidf = compute_tfidf(naive_counts)
    stem_tfidf = compute_tfidf(stem_counts)

    write_json(symptom_dictionary, SYMPTOM_DICTIONARY_PATH)
    write_json(naive_vocabulary, NAIVE_VOCAB_PATH)
    write_json(stem_vocabulary, STEM_VOCAB_PATH)
    write_json(session_ids, SESSION_IDS_PATH)
    write_feature_matrix(naive_tfidf, NAIVE_TFIDF_PATH)
    write_feature_matrix(stem_tfidf, STEM_TFIDF_PATH)

    logging.info("Wrote symptom dictionary to %s", SYMPTOM_DICTIONARY_PATH)
    logging.info("Wrote vocabularies and TF-IDF matrices to %s", PROCESSED_DATA_DIR)

    label_matrix_aligned, label_columns = compute_label_matrix(session_ids)
    if label_matrix_aligned.size:
        write_label_bundle(label_matrix_aligned, ALIGNED_LABEL_MATRIX_PATH, session_ids, label_columns)
        logging.info(
            "Label matrix shape (aligned): %s (sessions x labels)",
            label_matrix_aligned.shape,
        )
    else:
        logging.warning(
            "Aligned label matrix is empty; verify label data is available at %s",
            LABEL_MATRIX_PATH,
        )

    if DICTIONARY_FEATURES_PATH.exists():
        dictionary_matrix, dictionary_columns, dictionary_session_ids = load_npz_features(
            DICTIONARY_FEATURES_PATH
        )
        dictionary_matrix = np.nan_to_num(dictionary_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        dictionary_matrix_aligned = align_matrix_to_sessions(
            dictionary_matrix, dictionary_session_ids, session_ids
        )

        naive_combined_matrix = np.hstack([naive_tfidf, dictionary_matrix_aligned])
        naive_feature_index_map, naive_column_labels = build_feature_index_map(
            naive_vocabulary, dictionary_columns, "tfidf_naive"
        )
        write_feature_bundle(
            naive_combined_matrix,
            NAIVE_COMBINED_FEATURE_MATRIX_PATH,
            naive_column_labels,
            session_ids,
        )
        write_json(naive_feature_index_map, FEATURE_INDEX_MAP_NAIVE_PATH)

        stem_combined_matrix = np.hstack([stem_tfidf, dictionary_matrix_aligned])
        stem_feature_index_map, stem_column_labels = build_feature_index_map(
            stem_vocabulary, dictionary_columns, "tfidf_stem"
        )
        write_feature_bundle(
            stem_combined_matrix,
            STEM_COMBINED_FEATURE_MATRIX_PATH,
            stem_column_labels,
            session_ids,
        )
        write_json(stem_feature_index_map, FEATURE_INDEX_MAP_STEM_PATH)

        logging.info(
            "Naive+dictionary feature matrix shape: %s (sessions x features)",
            naive_combined_matrix.shape,
        )
        logging.info(
            "Stem+dictionary feature matrix shape: %s (sessions x features)",
            stem_combined_matrix.shape,
        )
    else:
        logging.warning(
            "Dictionary feature archive not found at %s; skipping combined matrix generation.",
            DICTIONARY_FEATURES_PATH,
        )


if __name__ == "__main__":
    main()
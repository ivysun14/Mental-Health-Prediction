"""
NRC and concreteness feature engineering.

This module tidies the original NRC/MOESM scripts into a single workflow that:
1. Loads the processed session transcripts (`meta_cleaned.json`).
2. Computes NRC emotion scores and simple linguistic statistics per session.
3. Computes concreteness-style features using the MOESM lexicon.
4. Exports the results to `data/processed/` for downstream modelling.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LEXICON_DIR = PROJECT_ROOT / "data" / "dic_features"

META_JSON_PATH = PROCESSED_DATA_DIR / "meta_cleaned.json"
NRC_LEXICON_PATH = LEXICON_DIR / "NRC_dic.json"
MOESM_LEXICON_PATH = LEXICON_DIR / "13428_2013_403_MOESM1_ESM.csv"

NRC_FEATURES_CSV = PROCESSED_DATA_DIR / "nrc_features.csv"
NRC_FEATURES_NPZ = PROCESSED_DATA_DIR / "nrc_features.npz"
MOESM_FEATURES_CSV = PROCESSED_DATA_DIR / "moesm_features.csv"
MOESM_FEATURES_NPZ = PROCESSED_DATA_DIR / "moesm_features.npz"
COMBINED_FEATURES_CSV = PROCESSED_DATA_DIR / "nrc_moesm_features.csv"
COMBINED_FEATURES_NPZ = PROCESSED_DATA_DIR / "nrc_moesm_features.npz"

CLIENT_TEXT_KEY = "Client_Text"
WORD_PATTERN = re.compile(r"\b[\w']+\b")


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


def get_client_lines(session: Dict[str, object]) -> Sequence[str]:
    """Return the list of client utterances for a session."""
    lines = session.get(CLIENT_TEXT_KEY, [])
    return lines if isinstance(lines, Sequence) else []


# --------------------------------------------------------------------------- #
# Lexicon loaders
# --------------------------------------------------------------------------- #


def load_nrc_lexicon(path: Path = NRC_LEXICON_PATH) -> Dict[str, Dict[str, float]]:
    """Load the NRC emotion lexicon."""
    if not path.exists():
        raise FileNotFoundError(f"NRC lexicon not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        lexicon = json.load(handle)
    if not lexicon:
        raise ValueError(f"NRC lexicon at {path} is empty.")
    logging.info("Loaded NRC lexicon with %d entries.", len(lexicon))
    return {word.lower(): {emo: float(score) for emo, score in scores.items()} for word, scores in lexicon.items()}


def load_moesm_lexicon(path: Path = MOESM_LEXICON_PATH) -> Dict[str, float]:
    """Load the concreteness lexicon from the MOESM CSV."""
    if not path.exists():
        raise FileNotFoundError(f"MOESM lexicon not found: {path}")
    df = pd.read_csv(path)
    if "Word" not in df.columns or "Percent_known" not in df.columns:
        raise KeyError("MOESM lexicon must contain 'Word' and 'Percent_known' columns.")
    lexicon = {str(word).lower(): float(percent) for word, percent in zip(df["Word"], df["Percent_known"])}
    logging.info("Loaded MOESM lexicon with %d entries.", len(lexicon))
    return lexicon


# --------------------------------------------------------------------------- #
# Feature computation helpers
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> List[str]:
    """Lower-case tokenisation based on word boundaries."""
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def summarise(values: List[float]) -> Tuple[float, float]:
    """Return mean and standard deviation for a list of floats, handling empty input."""
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float32)
    return float(array.mean()), float(array.std(ddof=0))


def compute_nrc_features(
    lines: Sequence[str],
    emotions: Sequence[str],
    lexicon: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute NRC-based features for a single session."""
    emotion_scores: Dict[str, List[float]] = {emotion: [] for emotion in emotions}
    line_lengths: List[int] = []

    for line in lines:
        tokens = tokenize(line)
        if not tokens:
            continue
        line_lengths.append(len(tokens))
        for emotion in emotions:
            score = sum(lexicon.get(token, {}).get(emotion, 0.0) for token in tokens)
            score = score / len(tokens) if score else 0.0
            emotion_scores[emotion].append(score)

    features: Dict[str, float] = {}
    for emotion in emotions:
        mean, std = summarise(emotion_scores[emotion])
        features[f"{emotion}_mean"] = mean
        features[f"{emotion}_std"] = std

    mean_length, std_length = summarise([float(length) for length in line_lengths])
    features["avg_line_length"] = mean_length
    features["line_length_std"] = std_length
    features["line_count"] = float(len(line_lengths))
    features["token_count"] = float(sum(line_lengths))
    return features


def compute_moesm_features(lines: Sequence[str], lexicon: Dict[str, float]) -> Dict[str, float]:
    """Compute concreteness-style features for a session."""
    scores: List[float] = []
    for line in lines:
        tokens = tokenize(line)
        if not tokens:
            continue
        score = sum(lexicon.get(token, 0.0) for token in tokens)
        score = score / len(tokens) if score else 0.0
        scores.append(score)

    mean_score, std_score = summarise(scores)
    return {
        "concreteness_mean": mean_score,
        "concreteness_std": std_score,
    }


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def write_dataframe(df: pd.DataFrame, csv_path: Path, npz_path: Path) -> None:
    """Persist features to CSV and compressed NumPy archive."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)
    np.savez_compressed(
        npz_path,
        data=df.to_numpy(dtype=np.float32),
        columns=df.columns.to_numpy(),
        session_ids=df.index.to_numpy(),
    )
    logging.info("Wrote %s and %s", csv_path, npz_path)


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run the NRC + MOESM feature extraction workflow."""
    configure_logging()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions()
    nrc_lexicon = load_nrc_lexicon()
    emotions = sorted(next(iter(nrc_lexicon.values())).keys())
    moesm_lexicon = load_moesm_lexicon()

    nrc_records: List[Dict[str, float]] = []
    moesm_records: List[Dict[str, float]] = []
    session_ids: List[str] = []

    for session_id, session in iter_sessions(sessions):
        lines = get_client_lines(session)
        session_ids.append(session_id)
        nrc_records.append(compute_nrc_features(lines, emotions, nrc_lexicon))
        moesm_records.append(compute_moesm_features(lines, moesm_lexicon))

    nrc_df = pd.DataFrame(nrc_records, index=session_ids)
    nrc_df.index.name = "session_id"
    moesm_df = pd.DataFrame(moesm_records, index=session_ids)
    moesm_df.index.name = "session_id"

    combined_df = pd.concat([nrc_df, moesm_df], axis=1)

    write_dataframe(nrc_df, NRC_FEATURES_CSV, NRC_FEATURES_NPZ)
    write_dataframe(moesm_df, MOESM_FEATURES_CSV, MOESM_FEATURES_NPZ)
    write_dataframe(combined_df, COMBINED_FEATURES_CSV, COMBINED_FEATURES_NPZ)


if __name__ == "__main__":
    main()





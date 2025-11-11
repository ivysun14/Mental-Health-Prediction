"""
Train/test split utilities.

This script consumes the processed data artifacts and produces:
* 80/20 train-test split of the aligned feature and label matrices.
* 5-fold splits (indices only) within the training set for cross-validation.
* Persisted artifacts saved under `data/processed/splits/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

FEATURE_MATRIX_PATH = PROCESSED_DATA_DIR / "feature_matrix_stem_plus_dict.npz"
LABEL_MATRIX_PATH = PROCESSED_DATA_DIR / "label_matrix_merged_filtered_with_none.npz"
FEATURE_INDEX_MAP_PATH = PROCESSED_DATA_DIR / "feature_index_map_stem.json"

TRAIN_FEATURES_PATH = SPLIT_OUTPUT_DIR / "train_features.npz"
TRAIN_LABELS_PATH = SPLIT_OUTPUT_DIR / "train_labels.npz"
TRAIN_INDICES_PATH = SPLIT_OUTPUT_DIR / "train_indices.npy"
TEST_FEATURES_PATH = SPLIT_OUTPUT_DIR / "test_features.npz"
TEST_LABELS_PATH = SPLIT_OUTPUT_DIR / "test_labels.npz"
TEST_INDICES_PATH = SPLIT_OUTPUT_DIR / "test_indices.npy"
KFOLD_SPLITS_PATH = SPLIT_OUTPUT_DIR / "train_kfold_indices.json"

RANDOM_STATE_TRAIN_TEST = 42
RANDOM_STATE_KFOLD = 1234
KFOLD_FOLDS = 5
TEST_SIZE = 0.2


def load_feature_matrix(path: Path = FEATURE_MATRIX_PATH) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load the combined feature matrix NPZ bundle."""
    bundle = np.load(path, allow_pickle=True)
    matrix = bundle["data"]
    columns = bundle["columns"].astype(str).tolist()
    session_ids = bundle["session_ids"].astype(str).tolist()
    return matrix, columns, session_ids


def load_label_matrix(path: Path = LABEL_MATRIX_PATH) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load the merged+filtered label matrix bundle."""
    bundle = np.load(path, allow_pickle=True)
    matrix = bundle["data"]
    labels = bundle["labels"].astype(str).tolist()
    session_ids = bundle["session_ids"].astype(str).tolist()
    return matrix, labels, session_ids


def split_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    session_ids: List[str],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE_TRAIN_TEST,
) -> Dict[str, object]:
    """Perform an 80/20 train/test split and return the resulting slices."""
    indices = np.arange(features.shape[0])
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
        session_ids_train,
        session_ids_test,
    ) = train_test_split(
        features,
        labels,
        indices,
        session_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "indices_train": indices_train,
        "indices_test": indices_test,
        "session_ids_train": session_ids_train,
        "session_ids_test": session_ids_test,
    }


def generate_kfold_indices(
    train_indices: np.ndarray, folds: int = KFOLD_FOLDS, random_state: int = RANDOM_STATE_KFOLD
) -> List[Dict[str, List[int]]]:
    """Generate K-fold train/validation index sets within the training split."""
    kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits: List[Dict[str, List[int]]] = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices), start=1):
        splits.append(
            {
                "fold": fold,
                "train_indices": train_indices[train_idx].tolist(),
                "val_indices": train_indices[val_idx].tolist(),
            }
        )
    return splits


def save_npz(path: Path, array: np.ndarray) -> None:
    """Persist a NumPy array in compressed format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=array)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Persist JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    """Execute the train/test split workflow."""
    features, feature_columns, feature_session_ids = load_feature_matrix()
    labels, label_names, label_session_ids = load_label_matrix()

    if feature_session_ids != label_session_ids:
        raise ValueError("Feature and label session IDs are not aligned. Re-run preprocessing.")

    split_payload = split_train_test(features, labels, feature_session_ids)
    kfold_splits = generate_kfold_indices(split_payload["indices_train"])

    save_npz(TRAIN_FEATURES_PATH, split_payload["X_train"])
    save_npz(TRAIN_LABELS_PATH, split_payload["y_train"])
    np.save(TRAIN_INDICES_PATH, split_payload["indices_train"])

    save_npz(TEST_FEATURES_PATH, split_payload["X_test"])
    save_npz(TEST_LABELS_PATH, split_payload["y_test"])
    np.save(TEST_INDICES_PATH, split_payload["indices_test"])

    save_json(KFOLD_SPLITS_PATH, {"splits": kfold_splits})

    summary = {
        "feature_matrix_shape": features.shape,
        "label_matrix_shape": labels.shape,
        "train_shape": split_payload["X_train"].shape,
        "test_shape": split_payload["X_test"].shape,
        "num_features": features.shape[1],
        "num_labels": labels.shape[1] if labels.ndim > 1 else 1,
        "kfold_folds": KFOLD_FOLDS,
    }
    save_json(SPLIT_OUTPUT_DIR / "split_summary.json", summary)


if __name__ == "__main__":
    main()
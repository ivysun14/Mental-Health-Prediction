"""
Utilities for wrangling the metadata.

This script keeps only rows that have a matching transcript file on disk and
then combines the filtered Volume I metadata with the Volume II metadata so it
is easier to use downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Set

import pandas as pd

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

VOLUME1_FULL_METADATA_PATH = RAW_DATA_DIR / "publication_metadata_volumn1_full.csv"
VOLUME2_METADATA_PATH = RAW_DATA_DIR / "publication_metadata_volumn2.csv"
VOLUME1_FILTERED_METADATA_PATH = PROCESSED_DATA_DIR / "publication_metadata_volumn1_filtered.csv"
COMBINED_METADATA_PATH = PROCESSED_DATA_DIR / "publication_metadata_combined.csv"

# Columns that we want to keep aligned across both metadata files.
COLUMNS_TO_KEEP = [
    "file_name",
    "Entity_ID",
    "Abstract",
    "Client_Age",
    "Client_Gender",
    "Client_Marital_Status",
    "Client_Sexual_Orientation",
    "Psyc_Subjects",
    "Symptoms",
    "Therapies",
    "Therapist",
    "Real_Title",
]

# # Columns whose values may indicate multiple clients when delimited.
# MULTI_CLIENT_COLUMNS = [
#     "Client_Age",
#     "Client_Gender",
#     "Client_Marital_Status",
#     "Client_Sexual_Orientation",
# ]


def configure_logging() -> None:
    """Set up a simple logging configuration for the module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def get_transcript_filenames(transcripts_dir: Path) -> Set[str]:
    """
    Return all transcript filenames (basename only) within transcripts_dir.

    Parameters
    ----------
    transcripts_dir:
        Directory that contains transcript .txt files. The lookup is performed
        recursively to accommodate any nested structure.
    """
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcripts_dir}")

    transcript_files = {path.name for path in transcripts_dir.rglob("*.txt")}
    if not transcript_files:
        raise ValueError(f"No transcript files found in {transcripts_dir}.")

    return transcript_files


def filter_volume1_metadata(metadata_path: Path, allowed_filenames: Iterable[str]) -> pd.DataFrame:
    """
    Filter the Volume I metadata so that only rows with an existing transcript remain.
    """
    volume1_df = pd.read_csv(metadata_path)
    allowed_filenames = set(allowed_filenames)

    if "file_name" not in volume1_df.columns:
        raise KeyError("Expected a 'file_name' column in the Volume I metadata.")

    filtered_df = volume1_df.loc[volume1_df["file_name"].isin(allowed_filenames)].copy()

    num_dropped = len(volume1_df) - len(filtered_df)
    logging.info(
        "Filtered Volume I metadata: kept %d of %d rows (dropped %d).",
        len(filtered_df),
        len(volume1_df),
        num_dropped,
    )

    # Align nomenclature with Volume II.
    if "Client_Age_Range" in filtered_df.columns and "Client_Age" not in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={"Client_Age_Range": "Client_Age"})

    return filtered_df


def _validate_columns(df: pd.DataFrame, dataset_label: str) -> None:
    """Ensure that the requested columns are present to avoid KeyError later on."""
    missing_columns = [column for column in COLUMNS_TO_KEEP if column not in df.columns]
    if missing_columns:
        logging.warning(
            "%s metadata is missing columns: %s. They will be filled with NA.",
            dataset_label,
            ", ".join(missing_columns),
        )
        for column in missing_columns:
            df[column] = pd.NA


def combine_metadata(volume1_df: pd.DataFrame, volume2_path: Path) -> pd.DataFrame:
    """
    Select relevant columns and combine Volume I and Volume II metadata.
    """
    volume2_df = pd.read_csv(volume2_path)

    _validate_columns(volume1_df, "Volume I")
    _validate_columns(volume2_df, "Volume II")

    combined_df = pd.concat(
        [
            volume1_df[COLUMNS_TO_KEEP],
            volume2_df[COLUMNS_TO_KEEP],
        ],
        ignore_index=True,
    )
    logging.info("Combined metadata contains %d rows.", len(combined_df))
    # multi_mask = combined_df.apply(_has_multiple_clients, axis=1)
    # multi_count = int(multi_mask.sum())
    # if multi_count:
    #     combined_df = combined_df.loc[~multi_mask].reset_index(drop=True)
    #     logging.info(
    #         "Removed %d sessions involving multiple clients. Remaining rows: %d.",
    #         multi_count,
    #         len(combined_df),
    #     )
    return combined_df


def _has_multiple_clients(row: pd.Series) -> bool:
    """Heuristically detect sessions with multiple clients based on metadata fields."""
    for column in MULTI_CLIENT_COLUMNS:
        value = row.get(column)
        if not isinstance(value, str):
            continue
        normalised = value.strip().lower()
        if not normalised:
            continue
        if ";" in normalised or "&" in normalised or " and " in normalised:
            return True
    return False


def main(transcripts_dir: Path = TRANSCRIPTS_DIR) -> None:
    """Run the end-to-end Volume I preprocessing workflow."""
    configure_logging()
    logging.info("Using project root: %s", PROJECT_ROOT)
    logging.info("Looking for transcripts in: %s", transcripts_dir)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Processed data will be written to: %s", PROCESSED_DATA_DIR)

    transcript_filenames = get_transcript_filenames(transcripts_dir)
    logging.info("Found %d transcript files.", len(transcript_filenames))

    volume1_filtered = filter_volume1_metadata(
        VOLUME1_FULL_METADATA_PATH,
        allowed_filenames=transcript_filenames,
    )
    volume1_filtered.to_csv(VOLUME1_FILTERED_METADATA_PATH, index=False)
    logging.info("Saved filtered Volume I metadata to %s", VOLUME1_FILTERED_METADATA_PATH)

    combined_metadata = combine_metadata(volume1_filtered, VOLUME2_METADATA_PATH)
    combined_metadata.to_csv(COMBINED_METADATA_PATH, index=False)
    logging.info("Saved combined metadata to %s", COMBINED_METADATA_PATH)


if __name__ == "__main__":
    main()

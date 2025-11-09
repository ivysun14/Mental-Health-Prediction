"""
Utility for organizing transcript text alongside cleaned publication metadata.

The script reads the combined metadata produced during preprocessing, parses the
transcript files referenced in the metadata, and writes out a JSON payload keyed
by `Entity_ID`. Each entry contains the original metadata fields plus two lists:
`Client_Text` and `Therapist_Text`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"

METADATA_PATH = PROCESSED_DATA_DIR / "publication_metadata_combined.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "meta_cleaned.json"

# Columns we want to carry through to the final JSON document.
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

# Define the set of possible speakers we care about.
CLIENT_SPEAKERS = {"CLIENT", "PATIENT", "PT"}
THERAPIST_SPEAKERS = {"THERAPIST", "COUNSELOR", "DR", "ANALYST"}

# Pre-compiled regular expressions for performance and readability.
SPEAKER_PATTERN = re.compile(
    r"^\s*<p[^>]*>\s*(?:<b>)?\s*(?P<speaker>[A-Z\s]+?)\s*:\s*(?P<utterance>.*)$",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(r"\[[^\]]*\]")
HTML_TAG_PATTERN = re.compile(r"</?[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def configure_logging() -> None:
    """Configure the module-level logger."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata and ensure required columns are present."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    metadata = pd.read_csv(path)
    missing_columns = [column for column in COLUMNS_TO_KEEP if column not in metadata.columns]
    if missing_columns:
        raise KeyError(
            f"Missing expected columns in metadata: {', '.join(missing_columns)}. "
            "Make sure the preprocessing step has been run."
        )
    metadata = metadata[COLUMNS_TO_KEEP].copy()
    logging.info("Loaded metadata with %d rows from %s", len(metadata), path)
    return metadata


def normalise_text(raw_text: str) -> str:
    """Strip timestamps, HTML tags, and excessive whitespace from a line of text."""
    no_time = TIME_PATTERN.sub("", raw_text)
    no_html = HTML_TAG_PATTERN.sub("", no_time)
    normalised = WHITESPACE_PATTERN.sub(" ", no_html).strip()
    return normalised


def classify_and_extract(line: str) -> Tuple[str, str] | Tuple[None, None]:
    """
    Identify whether a line is spoken by the client or the therapist.

    Returns
    -------
    A tuple containing the speaker label ("client" or "therapist") and the cleaned text.
    If the speaker cannot be identified, returns (None, None).
    """
    match = SPEAKER_PATTERN.match(line)
    if not match:
        return None, None

    speaker = match.group("speaker").strip().upper()
    text = normalise_text(match.group("utterance"))
    if not text:
        return None, None

    if speaker in CLIENT_SPEAKERS:
        return "client", text
    if speaker in THERAPIST_SPEAKERS:
        return "therapist", text
    return None, None


def parse_transcript(transcript_path: Path) -> Tuple[List[str], List[str]]:
    """Extract client and therapist lines from a transcript file."""
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    client_lines: List[str] = []
    therapist_lines: List[str] = []

    with transcript_path.open("r", encoding="utf-8", errors="ignore") as transcript_file:
        for raw_line in transcript_file:
            role, text = classify_and_extract(raw_line)
            if role == "client":
                client_lines.append(text)
            elif role == "therapist":
                therapist_lines.append(text)

    logging.debug(
        "Parsed transcript %s | client lines: %d | therapist lines: %d",
        transcript_path.name,
        len(client_lines),
        len(therapist_lines),
    )
    return client_lines, therapist_lines


def build_sessions(metadata: pd.DataFrame, transcripts_dir: Path) -> Dict[str, Dict[str, object]]:
    """Combine metadata with parsed transcripts and return a dict keyed by Entity_ID."""
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcripts_dir}")

    sessions: Dict[str, Dict[str, object]] = {}
    missing_transcripts: List[str] = []

    for _, row in metadata.iterrows():
        entity_id = str(row["Entity_ID"])
        transcript_file = transcripts_dir / row["file_name"]

        try:
            client_lines, therapist_lines = parse_transcript(transcript_file)
        except FileNotFoundError:
            missing_transcripts.append(row["file_name"])
            continue

        session_data = row.to_dict()
        session_data["Client_Text"] = client_lines
        session_data["Therapist_Text"] = therapist_lines
        sessions[entity_id] = session_data

    if missing_transcripts:
        logging.warning(
            "Skipped %d sessions because transcript files were missing. Examples: %s",
            len(missing_transcripts),
            ", ".join(sorted(set(missing_transcripts))[:5]),
        )

    logging.info("Built %d session entries.", len(sessions))
    return sessions


def write_json(data: Dict[str, Dict[str, object]], output_path: Path) -> None:
    """Serialize the session dictionary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    logging.info("Wrote organized data to %s", output_path)


def main() -> None:
    """Run the organizing workflow end-to-end."""
    configure_logging()
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Reading metadata from: %s", METADATA_PATH)
    logging.info("Reading transcripts from: %s", TRANSCRIPTS_DIR)

    metadata = load_metadata(METADATA_PATH)
    sessions = build_sessions(metadata, TRANSCRIPTS_DIR)
    write_json(sessions, OUTPUT_PATH)


if __name__ == "__main__":
    main()










# Preprocessing Workflow

This directory contains the scripts that transform the raw transcript collection into model-ready matrices. Each script reads the artifacts produced by the previous step and writes its results to `data/processed/` (or `data/splits/` for train/test splits).

## 1. Prep metadata
- Run `python preprocessing/process_metadata.py`
- Loads `data/raw/publication_metadata_volumn1_full.csv` and `...volumn2.csv`
- Filters sessions to those with transcript files, removes multi-client entries, and harmonises columns, then writes:
  - `data/processed/publication_metadata_volumn1_filtered.csv`
  - `data/processed/publication_metadata_combined.csv`

## 2. Build session JSON
- Run `python preprocessing/organize.py`
- Reads the combined metadata and transcript files
- Produces `data/processed/meta_cleaned.json`, pairing each `Entity_ID` with
  metadata plus cleaned client/therapist text.

## 3. Dictionary features (NRC & MOESM)
- Run `python preprocessing/build_dictionary_feature.py`
- Uses `data/processed/meta_cleaned.json` and the lexicons in `dic_features/`
- Outputs emotion/concreteness features to `data/processed/nrc_moesm_features.(csv|npz)`

## 4. Label matrices
- Run `python preprocessing/build_label_matrix.py`
- Builds symptom dictionaries and binary label matrices from the session JSON
- Writes both full and filtered matrices (>=2% prevalence) to `data/processed/`
  - `label_matrix_full.npz`
  - `label_matrix_filtered.npz`
  - `label_matrix_aligned_with_features.npz` (aligned later in step 5)

## 5. Text feature matrices
- Run `python preprocessing/build_feature_matrix.py`
- Creates naive and stemmed vocabularies, removes leakage terms, computes TF-IDF
- Aligns dictionary features and labels with the TF-IDF matrices
- Key outputs:
  - `feature_matrix_naive_tfidf.npz`
  - `feature_matrix_stem_tfidf.npz`
  - `feature_matrix_naive_plus_dict.npz`
  - `feature_matrix_stem_plus_dict.npz`
  - `feature_index_map_naive.json` / `feature_index_map_stem.json`
  - `feature_matrix_session_ids.json`
  - `label_matrix_aligned_with_features.npz`

Typical shapes (using the current dataset):
- Naive + dictionary matrix: `(3667, 43113)`
- Stem + dictionary matrix: `(3667, 26312)`
- Filtered label matrix: `(3667, 49)`

## 6. Train/test splits
- Run `python preprocessing/train_test_split.py`
- Loads the stemmed TF-IDF + dictionary matrix and aligned labels
- Produces an 80/20 split and 5-fold CV indices for the training set
- Outputs are written to `data/splits/`:
  - `train_features.npz`, `train_labels.npz`, `train_indices.npy`
  - `test_features.npz`, `test_labels.npz`, `test_indices.npy`
  - `train_kfold_indices.json`
  - `split_summary.json`
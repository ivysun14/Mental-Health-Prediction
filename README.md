# Mental-Health-Prediction
This repository contains all scripts used in the manuscript: _Evaluating Large Language Models for Anxiety and Depression Classification using Counseling and Psychotherapy Transcripts_.

## utils
- `utils.py`: utility functions compatible to all models
    - `per_class_metrics`: collect all per-class metrics for predicted model outputs
    - `multilabel_metrics`: collect all multi-label metrics for predicted model outputs
    - `bootstrap_ci`: calculate confidence interval for a metric using stratified bootstrap
    - `load_fold_indices`: load pre-specified K-fold indices
    - `extract_multilabel_matrix`: Extract symptom columns from label matrix
_When use copy this file to the respective model folder._

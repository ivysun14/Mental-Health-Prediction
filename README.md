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

- `plot_finetune_curves.py`: utility script for visualizing training loss, evaluation loss, and validation metrics across epochs. _Requires a csv file that is produced automatically if your training script saves `trainer.state.log_hsitory`._
    - `plot_loss_curves`: plot training and evaluation loss across epochs
    - `plot_metric_curves`: plot validation metrics across epochs
    - How to use:
        - Create a directory under `models/your_model_name/`. 
        - Ensure it contains:
          - `your_model_name/full_train/log_history_fold_full.csv`
        - Add your model to the dictionary inside `plot_finetune_curves.py`:
          - `logs = {
              "Your Model": "models/your_model_name/full_train/log_history_fold_full.csv",
            }`
        - Re-run the plotting script.
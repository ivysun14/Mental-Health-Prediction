#!/usr/bin/env python

"""
plot_finetune_curves.py
=======================

Utility functions for visualizing fine-tuning behavior of Transformer models.

This module provides:

1) Loading HuggingFace Trainer log_history CSV files
2) Unified plotting functions
3) Multi-model comparison support

"""

import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ===============================
# Core loader for HF Trainer logs
# ===============================

def load_trainer_log(
    filename: str,
    metric_cols: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Load a HuggingFace Trainer log_history CSV and aggregate per epoch.

    Parameters
    ----------
    filename : str
        Path to CSV saved from `pd.DataFrame(trainer.state.log_history).to_csv(...)`.
    metric_cols : list of str, optional
        List of metric columns to extract (e.g. ['eval_subset_accuracy', 'eval_weighted_f1', 'eval_weighted_auroc']).
        If a column is missing, it will be skipped.

    Returns
    -------
    result : dict
        {
            'epoch_train': [...],
            'train_loss': [...],
            'epoch_eval': [...],
            'eval_loss': [...],
            '<metric_col>': [...],  # for each metric in metric_cols that exists
        }
    """
    df = pd.read_csv(filename)

    result = {}

    # --- Training loss per epoch ---
    if "loss" in df.columns:
        df_train = df[df["loss"].notna() & df["epoch"].notna()]
        if not df_train.empty:
            # last logged loss per epoch
            train_group = df_train.groupby("epoch").agg({"loss": "last"}).reset_index()
            result["epoch_train"] = train_group["epoch"].tolist()
            result["train_loss"] = train_group["loss"].tolist()
        else:
            result["epoch_train"] = []
            result["train_loss"] = []
    else:
        result["epoch_train"] = []
        result["train_loss"] = []

    # --- Eval loss + metrics per epoch ---
    if "eval_loss" in df.columns:
        df_eval = df[df["eval_loss"].notna() & df["epoch"].notna()]
        if not df_eval.empty:
            eval_group = df_eval.groupby("epoch").agg("last").reset_index()
            result["epoch_eval"] = eval_group["epoch"].tolist()
            result["eval_loss"] = eval_group["eval_loss"].tolist()

            if metric_cols is not None:
                for col in metric_cols:
                    if col in eval_group.columns:
                        result[col] = eval_group[col].tolist()
        else:
            result["epoch_eval"] = []
            result["eval_loss"] = []
    else:
        result["epoch_eval"] = []
        result["eval_loss"] = []

    return result


# ===================================
# Plot functions for multiple models
# ===================================

def plot_loss_curves(
    logs: Dict[str, str],
    out_path: str,
    metric_cols: Optional[List[str]] = None,
    title: str = "Training and Evaluation Loss"
):
    """
    Plot train/eval loss curves for multiple models.

    Parameters
    ----------
    logs : dict
        { model_label: log_csv_path, ... }
    out_path : str
        Path to save the PNG.
    metric_cols : list of str, optional
        Ignored here, but kept for API symmetry.
    title : str
        Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colormaps["tab10"]
    color_idx = 0

    for model_label, csv_path in logs.items():
        stats = load_trainer_log(csv_path, metric_cols=None)
        epoch_train = stats.get("epoch_train", [])
        train_loss = stats.get("train_loss", [])
        epoch_eval = stats.get("epoch_eval", [])
        eval_loss = stats.get("eval_loss", [])

        if epoch_train and train_loss:
            plt.plot(
                epoch_train,
                train_loss,
                label=f"{model_label} Train Loss",
                color=cmap.colors[color_idx % len(cmap.colors)],
                linestyle="-",
                marker="o",
            )

        if epoch_eval and eval_loss:
            plt.plot(
                epoch_eval,
                eval_loss,
                label=f"{model_label} Eval Loss",
                color=cmap.colors[color_idx % len(cmap.colors)],
                linestyle="--",
                marker="x",
            )

        color_idx += 1

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = PLOTS_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.clf()


def plot_metric_curves(
    logs: Dict[str, str],
    out_path: str,
    metric_cols: List[str],
    metric_labels: Optional[Dict[str, str]] = None,
    title: str = "Validation Metrics"
):
    """
    Plot one or more eval metrics for multiple models on the same figure.

    Parameters
    ----------
    logs : dict
        { model_label: log_csv_path, ... }
    out_path : str
        Path to save the PNG.
    metric_cols : list of str
        Metric column names from Trainer logs (e.g. ['eval_subset_accuracy', 'eval_weighted_f1', 'eval_weighted_auroc']).
    metric_labels : dict, optional
        Friendly display names for metrics. Keys = metric_cols; values = labels for legend.
    title : str
        Plot title.
    """
    if metric_labels is None:
        metric_labels = {m: m for m in metric_cols}

    plt.figure(figsize=(15, 9))
    cmap = matplotlib.colormaps["tab20c"]

    # Each model gets a "block" of colors
    for i, (model_label, csv_path) in enumerate(logs.items()):
        stats = load_trainer_log(csv_path, metric_cols=metric_cols)
        epoch_eval = stats.get("epoch_eval", [])

        if not epoch_eval:
            # no eval info
            continue

        for j, mcol in enumerate(metric_cols):
            if mcol not in stats:
                continue
            vals = stats[mcol]
            color = cmap.colors[i * 4 + j % 4]  # crude but effective color mapping
            linestyle = ["dotted", "dashed", "solid", "-."][j % 4]
            marker = ["o", "v", "h", "s"][j % 4]

            plt.plot(
                epoch_eval,
                vals,
                label=f"{model_label} {metric_labels.get(mcol, mcol)}",
                color=color,
                marker=marker,
                linestyle=linestyle,
            )

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Metric", fontsize=14)
    plt.title(title, fontsize=16)
    plt.ylim(0.0, 1.0)  # adjust as needed
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    out_path = PLOTS_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.clf()


# ===============================
# Example usages
# ===============================
if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[0]
    MODELS_DIR = PROJECT_ROOT / "models"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Example 1: Compare truncation vs chunking Longformer
    longformer_logs = {
        "Longformer Truncation": MODELS_DIR / "longformer_trunc" / "full_train" / "log_history_fold_full.csv",
        "Longformer Chunking": MODELS_DIR / "longformer_results_chunk" / "full_train" / "log_history_fold_full.csv",
    }

    # 1) Loss curves
    plot_loss_curves(
        logs=longformer_logs,
        out_path="longformer_loss_trunc_vs_chunk.png",
        title="Longformer Training & Eval Loss (Truncation vs Chunking)",
    )

    # 2) Validation metrics:
    # These should match what you return in utils.multilabel_metrics, prefixed by 'eval_' in logs
    metric_cols = [
        "eval_subset_accuracy",
        "eval_weighted_f1",
        "eval_weighted_auroc",
    ]
    metric_labels = {
        "eval_subset_accuracy": "Subset Accuracy",
        "eval_weighted_f1": "Weighted F1",
        "eval_weighted_auroc": "Weighted AUROC",
    }

    plot_metric_curves(
        logs=longformer_logs,
        out_path="longformer_metrics_trunc_vs_chunk.png",
        metric_cols=metric_cols,
        metric_labels=metric_labels,
        title="Longformer Validation Metrics (Truncation vs Chunking)",
    )

    # Example 2: Compare three models in truncation setting, e.g. BERT / RoBERTa / Longformer
    trunc_logs = {
        "BERT Truncation": MODELS_DIR / "bert_trunc" / "full_train" / "log_history_fold_full.csv",
        "RoBERTa Truncation": MODELS_DIR / "roberta_trunc" / "full_train" / "log_history_fold_full.csv",
        "Longformer Truncation": MODELS_DIR / "longformer_trunc" / "full_train" / "log_history_fold_full.csv",
    }

    plot_metric_curves(
        logs=trunc_logs,
        out_path="truncation_models_metrics.png",
        metric_cols=metric_cols,
        metric_labels=metric_labels,
        title="Performance for Fine-Tuned Truncation Models",
    )

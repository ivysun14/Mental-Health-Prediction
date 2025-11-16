#!/usr/bin/env python
"""
Longformer fine-tuning with truncation for multi-label classification.

Data:
- Text: data/processed/meta_cleaned.json
- Labels: data/processed/label_matrix_merged_filtered_with_none.npz
- Splits: data/splits/train_indices.npy, test_indices.npy, train_kfold_indices.json

Features:
- Optional K-fold CV on the training split.
- Full-train on all training indices.
- Evaluation on held-out test set.
- 95% bootstrap CIs for:
  - global multilabel metrics
  - per-class metrics
- Saves:
  - TensorBoard logs per fold
  - CSV log histories
  - test_metrics.json
  - test_predictions.csv (y_true, y_pred, prob per label)
  - multilabel_confusion_matrices.npy
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from sklearn import metrics as skm
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.utils import (
    multilabel_metrics,
    per_class_metrics,
    bootstrap_ci,
    load_fold_indices,
)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

META_JSON_PATH = PROCESSED_DATA_DIR / "meta_cleaned.json"
LABEL_MATRIX_PATH = PROCESSED_DATA_DIR / "label_matrix_merged_filtered_with_none.npz"

TRAIN_INDICES_PATH = SPLITS_DIR / "train_indices.npy"
TEST_INDICES_PATH = SPLITS_DIR / "test_indices.npy"
KFOLD_SPLITS_PATH = SPLITS_DIR / "train_kfold_indices.json"

SPLIT_RANDOM_SEED = 42
SPLIT_STRATIFICATION = "None (pre-defined split)"


# --------------------------------------------------------------------------- #
# Data loading & alignment
# --------------------------------------------------------------------------- #

def load_texts_and_labels(drop_none_label: bool = True, none_label_name: str = "NONE"):
    """Load client transcripts and label matrix, aligned by session ID."""
    bundle = np.load(LABEL_MATRIX_PATH, allow_pickle=True)
    Y = bundle["data"]  # (N, num_labels)
    label_names = bundle["labels"].astype(str).tolist()
    session_ids = bundle["session_ids"].astype(str).tolist()

    if drop_none_label and none_label_name in label_names:
        mask = np.array(label_names) != none_label_name
        Y = Y[:, mask]
        label_names = [lbl for lbl, keep in zip(label_names, mask) if keep]

    with META_JSON_PATH.open("r", encoding="utf-8") as f:
        sessions = json.load(f)

    texts = []
    for sid in session_ids:
        record = sessions[str(sid)]
        client_lines = record.get("Client_Text", [])
        if not isinstance(client_lines, list):
            client_lines = []
        text = " ".join(client_lines)
        texts.append(text)

    return texts, Y, label_names, session_ids


# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #

class LongformerDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
        }


# --------------------------------------------------------------------------- #
# GPU / logging helpers
# --------------------------------------------------------------------------- #

def print_gpu_utilization():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")
    except Exception as e:
        print(f"Could not query GPU memory: {e}")


def print_summary(result):
    print(f"Time: {result.metrics.get('train_runtime', float('nan')):.2f}")
    print(f"Samples/second: {result.metrics.get('train_samples_per_second', float('nan')):.2f}")
    print_gpu_utilization()


def get_precision_args():
    """Decide which mixed-precision flags to use for TrainingArguments based on the available device."""
    if torch.cuda.is_available():
        print("[INFO] Detected CUDA GPU – enabling fp16 mixed precision.")
        return dict(fp16=True, bf16=False)

    # Apple MPS backend (Mac) -> do NOT use fp16 with Accelerate
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("[INFO] Detected Apple MPS backend – running in full precision (no fp16).")
        return dict(fp16=False, bf16=False)

    # CPU fallback
    print("[INFO] No GPU detected – running in full precision on CPU.")
    return dict(fp16=False, bf16=False)


# --------------------------------------------------------------------------- #
# Metrics for Trainer
# --------------------------------------------------------------------------- #

def make_compute_metrics(threshold: float = 0.5):
    """Returns a compute_metrics function for HuggingFace Trainer that uses multilabel_metrics."""
    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()  # (n_samples, num_labels)
        y_pred_binary = (probs >= threshold).astype(int)
        y_true = p.label_ids

        result = multilabel_metrics(
            y_true=y_true,
            y_pred_proba=probs,
            y_pred_binary=y_pred_binary,
            prob=True,
        )
        return result

    return compute_metrics


# --------------------------------------------------------------------------- #
# K-fold training
# --------------------------------------------------------------------------- #

def train_one_fold(
    fold_id,
    train_indices,
    val_indices,
    full_encodings,
    label_names,
    device,
    args,
    output_root: Path,
):
    """Train and validate on a single fold."""
    fold_dir = output_root / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_inputs = {k: v[train_indices] for k, v in full_encodings.items()}
    val_inputs = {k: v[val_indices] for k, v in full_encodings.items()}

    train_dataset = LongformerDataset(train_inputs)
    val_dataset = LongformerDataset(val_inputs)

    num_labels = len(label_names)

    per_device_batch_size = args.per_device_train_batch_size
    grad_accum_steps = args.gradient_accumulation_steps
    num_epochs = args.num_train_epochs

    num_update_steps_per_epoch = math.ceil(
        len(train_dataset) / (per_device_batch_size * grad_accum_steps)
    )
    num_training_steps = int(num_epochs * num_update_steps_per_epoch)

    model = LongformerForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    model.to(device)

    precision_args = get_precision_args()
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        logging_dir=str(fold_dir / "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        eval_accumulation_steps=args.eval_accumulation_steps,
        save_total_limit=args.save_total_limit,
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
        **precision_args,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(args.threshold),
    )

    print(f"===== Training fold {fold_id} =====")
    result = trainer.train()
    print_summary(result)

    eval_metrics = trainer.evaluate()
    print(f"Fold {fold_id} eval metrics:", eval_metrics)

    log_history = trainer.state.log_history
    pd.DataFrame(log_history).to_csv(fold_dir / f"log_history_fold_{fold_id}.csv", index=False)

    best_ckpt = trainer.state.best_model_checkpoint or str(fold_dir)
    return eval_metrics, best_ckpt


def run_kfold_cv(full_encodings, label_names, device, args, output_root: Path):
    """Run K-fold CV using precomputed splits from train_kfold_indices.json."""
    raw_splits = load_fold_indices(str(KFOLD_SPLITS_PATH))
    splits = []
    for split in raw_splits:
        tr_idx = np.array(split["train_indices"], dtype=int)
        val_idx = np.array(split["val_indices"], dtype=int)
        splits.append(
            {
                "fold": split["fold"],
                "train_indices": tr_idx,
                "val_indices": val_idx,
            }
        )

    all_fold_metrics = []

    for split in splits:
        fold_id = split["fold"]
        tr_idx = split["train_indices"]
        val_idx = split["val_indices"]

        eval_metrics, best_ckpt = train_one_fold(
            fold_id=fold_id,
            train_indices=tr_idx,
            val_indices=val_idx,
            full_encodings=full_encodings,
            label_names=label_names,
            device=device,
            args=args,
            output_root=output_root,
        )
        eval_metrics["fold"] = fold_id
        eval_metrics["best_checkpoint"] = best_ckpt
        all_fold_metrics.append(eval_metrics)

    cv_df = pd.DataFrame(all_fold_metrics)
    cv_df.to_csv(output_root / "cv_results.csv", index=False)
    print("Cross-validation summary:\n", cv_df.describe())


# --------------------------------------------------------------------------- #
# Test evaluation using utils.py + bootstrap_ci
# --------------------------------------------------------------------------- #

def evaluate_on_test(
    best_model_path,
    full_encodings,
    test_indices,
    label_names,
    device,
    args,
    output_root: Path,
):
    """
    Evaluate best model on held-out test set.

    - Point estimates:
        * multilabel_metrics  (subset acc, F1s, hamming, weighted AUROC)
        * per_class_metrics   (per-class acc/precision/recall/specificity, AUROC/AUPRC)
    - Bootstrap 95% CIs:
        * global multilabel metrics
        * per-class metrics (per label, 1D)
    - Saves:
        * test_metrics.json   (point estimates + CIs, curves stripped)
        * test_predictions.csv (y_true, y_pred, prob per label)
        * multilabel_confusion_matrices.npy
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------ Build test dataset ------------------------ #
    test_inputs = {k: v[test_indices] for k, v in full_encodings.items()}
    test_dataset = LongformerDataset(test_inputs)

    model = LongformerForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)

    test_args = TrainingArguments(
        output_dir=str(output_root / "test_eval"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=False,
        do_eval=False,
        report_to=[],  # no TB logging for test-only pass
    )

    trainer = Trainer(model=model, args=test_args)

    # ------------------------ Raw predictions --------------------------- #
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    y_true = preds_output.label_ids  # shape: (n_samples, n_labels)

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred_binary = (probs >= args.threshold).astype(int)

    # ------------------------ Point estimates --------------------------- #
    multi_metrics = multilabel_metrics(
        y_true=y_true,
        y_pred_proba=probs,
        y_pred_binary=y_pred_binary,
        prob=True,
    )

    per_class = per_class_metrics(
        y_true=y_true,
        y_pred_proba=probs,
        y_pred_binary=y_pred_binary,
        class_names=label_names,
        prob=True,
    )

    # ------------------------ Helper for CIs ---------------------------- #

    def _ci_wrapper(y_true_arr, y_pred_arr, func):
        """
        Thin wrapper around bootstrap_ci returning a JSON-safe dict.
        `func` takes (y_true_boot, y_pred_boot) and returns a scalar.
        """
        lower, upper, mean = bootstrap_ci(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            metric_func=func,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
        )
        return {
            "lower": float(lower),
            "upper": float(upper),
            "mean": float(mean),
        }

    # ------------------------ Global (multilabel) CIs ------------------- #
    ci_global = {}

    # Metrics that use binary predictions
    global_binary_funcs = {
        "subset_accuracy": lambda yt, yp: skm.accuracy_score(yt, yp),
        "hamming_loss": lambda yt, yp: skm.hamming_loss(yt, yp),
        "macro_f1": lambda yt, yp: skm.f1_score(
            yt, yp, average="macro", zero_division=0
        ),
        "micro_f1": lambda yt, yp: skm.f1_score(
            yt, yp, average="micro", zero_division=0
        ),
        "weighted_f1": lambda yt, yp: skm.f1_score(
            yt, yp, average="weighted", zero_division=0
        ),
        "sample_f1": lambda yt, yp: skm.f1_score(
            yt, yp, average="samples", zero_division=0
        ),
    }

    for name, func in global_binary_funcs.items():
        ci_global[name] = _ci_wrapper(y_true, y_pred_binary, func)

    # Weighted AUROC uses probabilities
    ci_global["weighted_auroc"] = _ci_wrapper(
        y_true,
        probs,
        lambda yt, yp: skm.roc_auc_score(yt, yp, average="weighted"),
    )

    # ------------------------ Per-class CIs (1D) ------------------------ #
    per_class_ci = {}

    for j, class_name in enumerate(label_names):
        y_true_c = y_true[:, j]
        y_pred_c = y_pred_binary[:, j]
        prob_c = probs[:, j]

        def _specificity(yt, yp):
            # force 2x2 confusion matrix even if one class disappears occasionally
            cf = skm.confusion_matrix(yt, yp, labels=[0, 1])
            tn, fp, fn, tp = cf.ravel()
            denom = tn + fp
            return float(tn / denom) if denom > 0 else 0.0

        def _auroc(yt, yp):
            return skm.roc_auc_score(yt, yp)

        def _auprc(yt, yp):
            precision_curve, recall_curve, _ = skm.precision_recall_curve(yt, yp)
            return skm.auc(recall_curve, precision_curve)

        per_class_ci[class_name] = {
            "accuracy": _ci_wrapper(
                y_true_c, y_pred_c, lambda yt, yp: skm.accuracy_score(yt, yp)
            ),
            "precision": _ci_wrapper(
                y_true_c,
                y_pred_c,
                lambda yt, yp: skm.precision_score(
                    yt, yp, pos_label=1, zero_division=0
                ),
            ),
            "recall": _ci_wrapper(
                y_true_c,
                y_pred_c,
                lambda yt, yp: skm.recall_score(
                    yt, yp, pos_label=1, zero_division=0
                ),
            ),
            "specificity": _ci_wrapper(y_true_c, y_pred_c, _specificity),
            "auroc": _ci_wrapper(y_true_c, prob_c, _auroc),
            "auprc": _ci_wrapper(y_true_c, prob_c, _auprc),
        }

    # ------------------------ Make JSON-safe & save --------------------- #

    def _to_python_scalar(x):
        if isinstance(x, np.generic):
            return x.item()
        return x

    # Global multilabel metrics: ensure pure Python scalars
    point_estimates = {k: _to_python_scalar(v) for k, v in multi_metrics.items()}

    # Per-class metrics: drop ROC/PR curves and cast scalars
    per_class_slim = {}
    for class_name, stats in per_class.items():
        per_class_slim[class_name] = {}
        for k, v in stats.items():
            if k in ("roc_curve", "prc_curve"):
                continue  # skip huge arrays
            per_class_slim[class_name][k] = _to_python_scalar(v)

    test_summary = {
        "point_estimates": point_estimates,
        "per_class": per_class_slim,
        "ci_global": ci_global,
        "ci_per_class": per_class_ci,
        "label_names": label_names,
    }

    with (output_root / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    # ------------------------ Save predictions & confusion matrices ----- #

    pred_df = pd.DataFrame({"sample_idx": np.arange(len(y_true))})
    for j, name in enumerate(label_names):
        pred_df[f"y_true_{name}"] = y_true[:, j]
        pred_df[f"y_pred_{name}"] = y_pred_binary[:, j]
        pred_df[f"prob_{name}"] = probs[:, j]

    pred_df.to_csv(output_root / "test_predictions.csv", index=False)

    cm = skm.multilabel_confusion_matrix(y_true, y_pred_binary)
    np.save(output_root / "multilabel_confusion_matrices.npy", cm)

    print("Test evaluation complete. Metrics and predictions saved under:", output_root)


# --------------------------------------------------------------------------- #
# CLI & main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Longformer with truncation for multi-label classification."
    )
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--output_root", type=str, default="models/longformer_truncation")
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--drop_none_label", action="store_true", default=True)
    parser.add_argument("--none_label_name", type=str, default="NONE")

    parser.add_argument("--no_kfold", action="store_true", help="Skip K-fold CV if set.")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=95.0)

    return parser.parse_args()


def main():
    args = parse_args()

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Save config
    with (output_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Seeds
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    # Load texts + labels
    texts, Y_all, label_names, session_ids = load_texts_and_labels(
        drop_none_label=args.drop_none_label,
        none_label_name=args.none_label_name,
    )
    num_labels = len(label_names)
    print(f"Loaded {len(texts)} sessions with {num_labels} labels.")

    # Tokenizer + encodings with truncation
    tokenizer = LongformerTokenizer.from_pretrained(args.model_name)
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encodings["labels"] = torch.tensor(Y_all, dtype=torch.float)

    # Split indices
    train_indices = np.load(TRAIN_INDICES_PATH)
    test_indices = np.load(TEST_INDICES_PATH)

    # K-fold CV on the training set (optional)
    if not args.no_kfold:
        run_kfold_cv(encodings, label_names, device, args, output_root)

    # Full-train on all training indices (using train_indices for both train and val)
    full_eval_metrics, best_ckpt = train_one_fold(
        fold_id="full",
        train_indices=train_indices,
        val_indices=train_indices,
        full_encodings=encodings,
        label_names=label_names,
        device=device,
        args=args,
        output_root=output_root,
    )
    print("Full-train eval metrics:", full_eval_metrics)
    print("Using checkpoint for final test eval:", best_ckpt)

    # Final held-out test evaluation
    evaluate_on_test(
        best_model_path=best_ckpt,
        full_encodings=encodings,
        test_indices=test_indices,
        label_names=label_names,
        device=device,
        args=args,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()

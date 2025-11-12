"""
utils.py
========

Utility functions for:
1). Per-class metrics collections
2). Multilabel metrics collections
3). Bootstrap 95% CI calculation for a specified metric
4). Load K-fold indices JSON file
5). Extract labels for defined classes from the label matrix

Meant to be used by all models that output class probabilities.
"""


import json
import numpy as np
from sklearn import metrics


def per_class_metrics(y_true, y_pred_proba, y_pred_binary, class_names, prob=True):
    """
    Calculate per-class accuracy, precision, recall, specificity, AUROC, and AUPRC.
    Also record intermediates: TP, TN, FP, FN, and outputs necessary for ROC/PRC plots.
    
    Inputs:
    @y_true: true labels of shape (n, num_labels)
    @y_pred_proba: predicted probabilities of shape (n, num_labels)
    @y_pred_binary: predicted binary labels of shape (n, num_labels)
    @class_names: list of class names
    @prob: boolean value indicating whether the model outputs probabilities for classes,
            if False, y_pred_proba should be None and AUROC and AUPRC will not be calculated
    
    @return: dictionary with per-class metrics
    """

    if not prob and y_pred_proba is not None:
        raise ValueError("Provided y_pred_proba even though prob=False.")

    results = {}
    
    for i, class_name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred_binary[:, i]
        
        if prob: 
            y_prob_class = y_pred_proba[:, i]
        
        # Accuracy
        accuracy = metrics.accuracy_score(y_true_class, y_pred_class)

        # Confusion matrix
        cf_mat = metrics.confusion_matrix(y_true_class, y_pred_class)
        tn, fp, fn, tp = cf_mat.ravel().tolist()

        # Precision, recall, specificity
        precision = metrics.precision_score(y_true_class, y_pred_class, pos_label=1, zero_division='warn')
        recall = metrics.recall_score(y_true_class, y_pred_class, pos_label=1, zero_division='warn')
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        if specificity == 0 and tn != 0:
            print("WARNING: Potential zero-division occured in calculating specificity.")

        results[class_name] = {
            'accuracy': accuracy,
            'tp': tp,
            'tn': tn, 
            'fp': fp, 
            "fn": fn,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }

        # AUROC
        if prob:
            try:
                auroc = metrics.roc_auc_score(y_true_class, y_prob_class)
                fpr, tpr, thresholds_roc = metrics.roc_curve(y_true_class, y_prob_class, pos_label=1)
            except ValueError:
                auroc = np.nan
                print("ERROR: Got NaN for AUROC.")
            
            # AUPRC
            try:
                precision_curve, recall_curve, thresholds_prc = metrics.precision_recall_curve(y_true_class, y_prob_class, pos_label=1)
                auprc = metrics.auc(recall_curve, precision_curve)
            except ValueError:
                auprc = np.nan
                print("ERROR: Got NaN for AUPRC.")
        
            results[class_name].update({'auroc': auroc,
                'roc_curve': [fpr, tpr, thresholds_roc],
                'auprc': auprc,
                'prc_curve': [precision_curve, recall_curve, thresholds_prc]})
    
    return results


def multilabel_metrics(y_true, y_pred_proba, y_pred_binary, prob=True):
    """
    Calculate multi-label classification metrics, these includes subset accuracy,
    hamming loss, macro-F1, micro-F1, weighted-F1, sample-based-F1, and weighted AUROC.
    
    Inputs:
    @y_true: true labels of shape (n, num_labels)
    @y_pred_proba: predicted probabilities of shape (n, num_labels)
    @y_pred_binary: predicted binary labels of shape (n, num_labels)
    @prob: boolean value indicating whether the model outputs probabilities for classes,
            if False, y_pred_proba should be None and AUROC will not be calculated
    
    @return: dictionary with multilabel metrics
    """
    
    if not prob and y_pred_proba is not None:
        raise ValueError("Provided y_pred_proba even though prob=False.")
    
    results = {}

    # Subset accuracy (all labels must be correct)
    accuracy = metrics.accuracy_score(y_true, y_pred_binary)

    # Hamming loss (unpack class and treat each label individually)
    ham_loss = metrics.hamming_loss(y_true, y_pred_binary)
    
    # Macro-F1
    f1_macro = metrics.f1_score(y_true, y_pred_binary, average='macro', zero_division='warn')
    
    # Micro-F1
    f1_micro = metrics.f1_score(y_true, y_pred_binary, average='micro', zero_division='warn')

    # Weighted-F1
    f1_weighted = metrics.f1_score(y_true, y_pred_binary, average='weighted', zero_division='warn')

    # Sample-based-F1
    f1_sample = metrics.f1_score(y_true, y_pred_binary, average='samples', zero_division='warn')

    results = {
        'subset_accuracy': accuracy,
        'hamming_loss': ham_loss,
        'macro_f1': f1_macro,
        'micro_f1': f1_micro,
        'weighted_f1': f1_weighted,
        'sample_f1': f1_sample
    }

    if prob:
        # AUROC (weighted)
        try:
            auroc = metrics.roc_auc_score(y_true, y_pred_proba, average='weighted')
        except ValueError:
            auroc = np.nan
            print("ERROR: Got NaN for weighted AUROC.")

        results.update({'weighted_auroc': auroc})
    
    return results


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, ci=95):
    """
    Calculate confidence interval for a metric using stratified bootstrap.
    
    Inputs:
    @y_true: true labels
    @y_pred: predicted labels or probabilities
    @metric_func: function to calculate metric (takes y_true, y_pred)
    @n_bootstrap: number of bootstrap samples
    @ci: confidence interval percentage
    
    @return: (lower_bound, upper_bound, mean)
    """
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        # Stratified bootstrap: sample with replacement while preserving class distribution
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        if len(np.unique(y_true[indices])) < 2:
            # Skip if bootstrap sample doesn't contain both classes
            continue
            
        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)
    
    scores = np.array(scores)
    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)
    mean = np.mean(scores)
    
    return lower, upper, mean


def load_fold_indices(fold_file):
    """
    Load pre-defined fold indices from JSON file.
    
    Inputs:
    @fold_file: path to JSON file containing fold indices
    
    @return: list of dictionaries with 'train_indices' and 'val_indices'
    """
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)
    return fold_data['splits']


def extract_multilabel_matrix(y, symptom_dict, classes, negative_label = -1):
    """
    Extract multiple symptom columns from label matrix and convert to {negative_label, 1}.
    
    Inputs:
    @y: label matrix with labels {0, 1}
    @symptom_dict: dictionary mapping symptom names to column indices
    @classes: list of symptom names to extract
    @negative_label: an integer in {-1, 0} indicating what value negative samples should be coded to, default -1
    
    @return: label matrix of shape (n, len(classes)) with values {negative_label, 1}
    """
    indices = [symptom_dict[s] for s in classes]
    extracted = y[:, indices]
    extracted[extracted == 0] = negative_label
    return extracted
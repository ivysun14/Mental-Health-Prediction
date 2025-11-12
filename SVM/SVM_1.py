"""
SVM_1.py
========

Training of Support Vector Machine classifiers for anxiety and depression classification.
Involves hyperparameter tuning of a linear SVM and a RBF SVM with multi-label metrics.
The pipeline internally trains a SVM model for each symptom label, and calculate per-class
metrics as well as combining models for multiple symptom labels for multilabel evaluation
metrics.

TODO:
1. Double check label instances coded as {0, 1} or {-1, 1}
2. Check if the set of per-class metrics and multi-label metrics selected to include are justified
3. Finish correcting the workflow
"""


#################### VARIABLES ####################

SYMPTOM_DICT = "SYMPTOM_DICT_PATH"
TRAIN_MATRIX = "TRAIN_MATRIX_PATH"
TRAIN_LABEL = "TRAIN_LABEL_PATH"
KFOLD_INDICES_FILE = "KFOLD_INDICES_FILE_PATH"

RANDOM_SEED = 1234                              # seed for reproducibility
TARGET_SYMPTOMS = ["anxiety", "depression"]     # symptom labels to build prediction model for

# define per-class metrics
PER_CLASS_METRIC = ["accuracy", "tp", "tn", "fp", "fn", "precision", "recall", "specificity", 
                    "auroc", "roc_curve", "auprc", "prc_curve"]

# define multi-label metrics
MULTILABEL_METRICS = ["subset_accuracy", "hamming_loss", "macro_f1", "micro_f1",
                      "weighted_f1", "sample_f1", "weighted_auroc"]

# define result dir structure
dirs = ["results",
        "results/linear",
        "results/linear/figures",
        "results/linear/metrics",
        "results/linear/test_prediction",
        "results/RBF",
        "results/RBF/figures",
        "results/RBF/metrics",
        "results/RBF/test_prediction"]


#################### SET UP ####################

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import collections

from string import punctuation
from sklearn.svm import SVC
from sklearn import metrics

from utils import per_class_metrics, multilabel_metrics, load_fold_indices, extract_multilabel_matrix

# set up directories
for dir_path in dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# set random seed
np.random.seed(RANDOM_SEED)


#################### SVM FUNCTIONS ####################

def cv_performance(clf_dict, X, Y, fold_splits, class_metrics, multi_metrics, class_names):
    """
    Perform cross-validation using pre-defined folds.
    
    Inputs:
    @clf_dict: dictionary of classifiers, one per label
    @X: feature matrix of shape (n, d)
    @Y: label matrix of shape (n, num_labels) with values {-1, 1}
    @fold_splits: list of dictionaries with 'train_indices' and 'val_indices'
    @class_metrics: list of per-class metric names
    @multi_metrics: list of multilabel metric names
    @class_names: list of class names
    
    @return: dictionaries with mean and std for per-class metrics and multilabel metrics
    """

    n_splits = len(fold_splits)
    num_labels = len(class_names)
    
    # Store scores for each fold
    per_class_metric_scores = {cn: {l: [] for l in class_metrics} for cn in class_names}
    multilabel_metric_scores = {m: [] for m in multi_metrics}
    
    for fold_idx, fold in enumerate(fold_splits):
        
        train_indices = np.array(fold['train_indices'])
        val_indices = np.array(fold['val_indices'])
        
        X_train, X_val = X[train_indices], X[val_indices]
        Y_train, Y_val = Y[train_indices], Y[val_indices]
        
        # Convert from {-1, 1} to {0, 1} for compatibility
        Y_train_binary = (Y_train + 1) / 2
        Y_val_binary = (Y_val + 1) / 2
        
        # Train each classifier and collect predictions
        #Y_pred_proba = np.zeros((len(val_indices), num_labels))
        Y_pred_binary = np.zeros((len(val_indices), num_labels))
        
        for label_idx, class_name in enumerate(class_names):
            
            clf = clf_dict[class_name]
            
            # Train on this label
            clf.fit(X_train, Y_train[:, label_idx])
            
            # Get decision function (signed distance to hyperplane)
            y_pred = clf.decision_function(X_val)
            
            # Binary predictions
            Y_pred_binary[:, label_idx] = np.sign(y_pred)
            Y_pred_binary[y_pred == 0, label_idx] = 1       # Points on boundary -> positive

        # Calculate per-class metrics for this fold
        per_class = per_class_metrics(Y_val_binary, None, Y_pred_binary, class_names, prob=False)
        for class_name in class_names:
            for metric_name in class_metrics:
                per_class_metric_scores[class_name][metric_name].append(per_class[class_name][metric_name])
        
        # Calculate multilabel metrics for this fold
        multi_scores = multilabel_metrics(Y_val_binary, None, Y_pred_binary, prob=False)
        for metric in multi_metrics:
            multilabel_metric_scores[metric].append(multi_scores[metric])
    
    # Calculate mean and std across folds (per-class metrics)
    results_class = {}
    for class_name in class_names:
        results_class[class_name] = {}
        for metric_name in class_metrics:
            scores = per_class_metric_scores[class_name][metric_name]
            results_class[class_name][metric_name] = {
                'mean': np.nanmean(scores),
                'std': np.nanstd(scores),
                'scores': scores
            }

    # Calculate mean and std across folds (multilabel metrics)
    results_multi = {}
    for metric in multi_metrics:
        results_multi[metric] = {
            'mean': np.mean(multi_scores[metric]),
            'std': np.std(multi_scores[metric]),
            'scores': multi_scores[metric]
        }
    
    return results_class, results_multi


'''
# This function can calculate six different performance metrics for the predicted output. These are
# accuracy, F1-Score, AUROC, precision, sensitivity, and specificity.
#
def performance(y_true, y_pred, metric="accuracy"):
    """
    Inputs:
    @y_true: true labels of each example, of shape (n, )
    @y_pred: (continuous-valued) predicted labels of each example, of shape (n, )
    @metric: a string specifying one of the six performance measures.
             'accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity'

    @return: a float representing performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    points_on_boundary,  = np.where(y_label == 0)
    points_classified_pos, = np.where(y_label == 1)
    points_classified_neg, = np.where(y_label == -1)
    print(f"For this run there are {points_on_boundary.shape} examples being predicted to lie right on the separating booundary.")
    print(f"For this run there are {points_classified_pos.shape} examples being predicted to have the symptom.")
    print(f"For this run there are {points_classified_neg.shape} examples being predicted to not have the symptom.")
    print("Note that right after this messsage all points lying right on the separating plane are classified as having the symptom.")

    # if a prediction is 0, treat that as 1
    y_label[y_label == 0] = 1

    # compute performance
    if metric == "accuracy":  # fraction of correctly classified samples
      score = metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":  # harmonic mean of the precision and recall
      score = metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
      score = metrics.roc_auc_score(y_true, y_label)
    elif metric == "precision":  # precision aka. of all we predicted to have the symptom, what fraction actually has the symptom
      score = metrics.precision_score(y_true, y_label)
    else:
      mcm = metrics.confusion_matrix(y_true, y_label)
      tn, fp, fn, tp = mcm.ravel()
      if metric == "sensitivity":  # recall aka. of all who actually have the symptom, what fraction did we correctly predict as having it
        score = tp / (tp + fn)
      if metric == "specificity":  # of all who don't have the symptom, what fraction did we correctly predict as not having it
        score = tn / (tn + fp)

    return score


# This function takes in a classifier, splits the data X and labels y into k-folds, perform k-fold cross validations,
# and calculates all specified performance metrics for the classifier by averaging the performance scores across folds.
#
def cv_performance(clf, X, y, kf, metric):
    """
    Inputs:
    @clf: a SVM classifier, aka. an instance of SVC
    @X: the feature matrix we constructed with shape (n, d)
    @y: the labels of each data point with shape (n,), note this is binary labels {1,-1}
    @kf: an instance of cross_validation.KFold or cross_validation.StratifiedKFold
    @metric: a list of strings specifying the performance metrics to calculate for

    @return: a numpy array of floats representing the average CV performance across k folds for all metrics
    """

    metric_score = np.zeros((len(metric), kf.get_n_splits(X, y)))
    counter = 0

    # split data based on cross validation kf and loop for k times (aka k folds)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # train SVM
        clf.fit(X_train, y_train)
        # predict using trained classifier, output the signed distance of a sample to the hyperplane
        y_pred = clf.decision_function(X_test)
        # metric score
        for j in range(len(metric)):
          metric_score[j][counter] = performance(y_test, y_pred, metric[j])
        counter += 1

    score = np.average(metric_score, axis=1)

    return score


# This function calls cv_performance and performs hyperparameter selection for the linear-kernel SVM
# by selecting the hyperparameter that maximizes each metric's average performance score across k-fold CV.
#
def select_param_linear(X, y, kf, metric, symptom):
    """
    Inputs:
    @X: the feature matrix we constructed with shape (n, d)
    @y: the labels of each data point with shape (n,), note this is binary labels {1,-1}
    @kf: an instance of cross_validation.KFold or cross_validation.StratifiedKFold
    @metric: a list of strings specifying the performance metrics to calculate for
    @symptom: the name of the symptom trying to classify, for output file naming purpose only

    @return: a list of floats representing the optimal hyperparameter values for linear-kernel SVM based on each metric
    """

    print('Linear SVM Hyperparameter Selection based on ' + (', '.join(metric)) + ':')

    # pre-define a range of C values, C here is the hyperparameter used in linear-kernel SVM
    C_range = 10.0 ** np.arange(-3, 3)

    # train linear-kernel SVM using different C values and calculate average k-fold cross validation score
    c_score_T = np.zeros((len(C_range), len(metric)))

    for i in range(len(C_range)):
      clf = SVC(kernel='linear', C=C_range[i])  # define SVM instance
      c_score_T[i] = cv_performance(clf, X, y, kf, metric)
  
    # transpose the matrix
    c_score = c_score_T.T

    # obtain best score across c values for each metric
    best_index = np.argmax(c_score, axis=1)
    best_C = np.zeros(len(metric))
    for i in range(len(best_index)):
      best_C[i] = C_range[best_index[i]]
      print(f"For {metric[i]}, cv scores across different parameters are {c_score[i]}")

    np.savetxt(f"results/linear/linear_SVM_c_score_matrix_{symptom}.txt", c_score)
    np.savetxt(f"results/linear/linear_SVM_optimal_params_{symptom}.txt", best_C)

    return best_C
'''

def select_param_linear(X, Y, fold_splits, class_metrics, multi_metrics, class_names, symptom_str):
    """
    Hyperparameter selection for linear SVM.
    
    Inputs:
    @X: feature matrix of shape (n, d)
    @Y: label matrix of shape (n, num_labels) with values {-1, 1}
    @fold_splits: list of dictionaries with 'train_indices' and 'val_indices'
    @class_metrics: list of per-class metric names
    @multi_metrics: list of multilabel metric names
    @class_names: list of class names
    @symptom_str: string for output file naming
    
    @return: dictionary with best C for each metric
    """

    print('\nLinear SVM Hyperparameter Selection:')
    
    C_range = 10.0 ** np.arange(-3, 3)
    num_labels = len(class_names)
    
    results_by_C = {}  # Store results for each C value
    
    for C in C_range:
        
        print(f"\nTesting C={C}")
        
        clf_dict = {cn: SVC(kernel='linear', C=C) for cn in class_names}        # Create classifier for each symptom label
        cv_results = cv_performance(clf_dict, X, y, fold_splits,                # Evaluate with cross-validation
                                    class_metrics, multi_metrics, class_names)
        results_by_C[C] = cv_results
        
        # Print results
        for metric in metrics_list:
            print(f"  {metric}: {cv_results[metric]['mean']:.4f} ± {cv_results[metric]['std']:.4f}")
    
    # Select best C for each metric
    best_C = {}
    for metric in metrics_list:
        scores = [(C, results_by_C[C][metric]['mean']) for C in C_range]
        
        # For hamming_loss, lower is better
        if metric == 'hamming_loss':
            best_C[metric] = min(scores, key=lambda x: x[1])[0]
        else:
            best_C[metric] = max(scores, key=lambda x: x[1])[0]
        
        print(f"\nBest C for {metric}: {best_C[metric]}")
    
    # Save results
    with open(f"results/linear/linear_SVM_cv_results_{symptom_str}.json", 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for C in C_range:
            json_results[str(C)] = {}
            for metric in metrics_list:
                json_results[str(C)][metric] = {
                    'mean': float(results_by_C[C][metric]['mean']),
                    'std': float(results_by_C[C][metric]['std'])
                }
        json.dump(json_results, f, indent=2)
    
    with open(f"results/linear/linear_SVM_optimal_params_{symptom_str}.json", 'w') as f:
        json.dump({k: float(v) for k, v in best_C.items()}, f, indent=2)
    
    return best_C, results_by_C


'''
# Similar to above, this function calls cv_performance and performs hyperparameter selection for the RBF-kernel SVM
# by selecting the hyperparameter that maximizes each pairwise metric's average performance score across k-fold CV.
#
def select_param_rbf(X, y, kf, metric, symptom):
    """
    Inputs:
    @X: the feature matrix we constructed with shape (n, d)
    @y: the labels of each data point with shape (n,), note this is binary labels {1,-1}
    @kf: an instance of cross_validation.KFold or cross_validation.StratifiedKFold
    @metric: a list of strings specifying the performance metrics to calculate for
    @symptom: the name of the symptom trying to classify, for output file naming purpose only
    
    @returns: a numpy array of shape (len(metric), 2) with each row represents a tuple of floats (C, gamma)
              which are the optimal hyperparameters for RBF-kernel SVM for each metric
    """

    print('\nRBF SVM Hyperparameter Selection based on ' + (', '.join(metric)) + ':')

    # pre-define a range of gamma and C values, which are both hyperparameters used in RBF-kernel SVM
    # construct a grid to make sure we test every single possible combinations of the two hyperparameters
    C_range = 10.0 ** np.arange(-3, 4)
    gamma_range = 10.0 ** np.arange(-5, 2)
    tuple_score_T = np.zeros((len(C_range)*len(gamma_range), len(metric)))
    tuple_dict = {}

    counter = 0
    # train a SVM classifier using some values of the hyperparameters and calculate average performance score
    for i in range(len(C_range)):
      for j in range(len(gamma_range)):
        clf = SVC(kernel='rbf', C=C_range[i], gamma=gamma_range[j])  # define SVM instance
        evaluate_row_num = i+j+counter*(len(gamma_range)-1)
        tuple_score_T[evaluate_row_num] = cv_performance(clf, X, y, kf, metric)
        tuple_dict[str(evaluate_row_num)] = np.array([C_range[i], gamma_range[j]])
      counter += 1

    # transpose the matrix
    tuple_score = tuple_score_T.T
    np.savetxt(f"results/RBF/RBF_SVM_tuple_score_matrix_{symptom}.txt", tuple_score)

    # obtain best score across all pairwise (c, gamma) values for each metric
    best_index = np.argmax(tuple_score, axis=1)
    best_tuple = np.zeros((len(metric), 2))
    for z in range(len(best_index)):
      best_tuple[z] = tuple_dict[str(best_index[z])]
      print(f"For {metric[z]}, the best cv scores across different parameters is {tuple_score[z][best_index[z]]}")

    np.savetxt(f"results/RBF/RBF_SVM_optimal_params_{symptom}.txt", best_tuple)

    return best_tuple
'''

def select_param_rbf_multilabel(X, y, fold_splits, metrics_list, class_names, symptom_str):
    """
    Hyperparameter selection for RBF SVM in multi-label setting.
    
    Inputs:
    @X: feature matrix of shape (n, d)
    @y: label matrix of shape (n, num_labels) with values {-1, 1}
    @fold_splits: list of dictionaries with 'train_indices' and 'val_indices'
    @metrics_list: list of metric names
    @class_names: list of class names
    @symptom_str: string for output file naming
    
    @return: dictionary with best (C, gamma) for each metric
    """
    print('\nRBF SVM Hyperparameter Selection (Multi-label):')
    
    C_range = 10.0 ** np.arange(-3, 4)
    gamma_range = 10.0 ** np.arange(-5, 2)
    num_labels = len(class_names)
    
    # Store results for each (C, gamma) combination
    results_by_params = {}
    
    for C in C_range:
        for gamma in gamma_range:
            print(f"\nTesting C={C}, gamma={gamma}")
            
            # Create classifier for each label
            clf_dict = {cn: SVC(kernel='rbf', C=C, gamma=gamma) for cn in class_names}
            
            # Evaluate with cross-validation
            cv_results = cv_performance_multilabel(clf_dict, X, y, fold_splits,
                                                   metrics_list, class_names)
            results_by_params[(C, gamma)] = cv_results
            
            # Print results
            for metric in metrics_list:
                print(f"  {metric}: {cv_results[metric]['mean']:.4f} ± {cv_results[metric]['std']:.4f}")
    
    # Select best (C, gamma) for each metric
    best_params = {}
    for metric in metrics_list:
        scores = [(params, results_by_params[params][metric]['mean']) 
                  for params in results_by_params.keys()]
        
        # For hamming_loss, lower is better
        if metric == 'hamming_loss':
            best_params[metric] = min(scores, key=lambda x: x[1])[0]
        else:
            best_params[metric] = max(scores, key=lambda x: x[1])[0]
        
        print(f"\nBest (C, gamma) for {metric}: {best_params[metric]}")
    
    # Save results
    with open(f"results/RBF/RBF_SVM_cv_results_{symptom_str}.json", 'w') as f:
        json_results = {}
        for params in results_by_params.keys():
            key = f"C={params[0]}_gamma={params[1]}"
            json_results[key] = {}
            for metric in metrics_list:
                json_results[key][metric] = {
                    'mean': float(results_by_params[params][metric]['mean']),
                    'std': float(results_by_params[params][metric]['std'])
                }
        json.dump(json_results, f, indent=2)
    
    with open(f"results/RBF/RBF_SVM_optimal_params_{symptom_str}.json", 'w') as f:
        json_params = {k: {'C': float(v[0]), 'gamma': float(v[1])} 
                       for k, v in best_params.items()}
        json.dump(json_params, f, indent=2)
    
    return best_params, results_by_params

'''
# Finally, this is rather a trivial function that outputs the performance score of the final chosen models.
#
def performance_test(clf, X, y, symptom, metric="accuracy", model="linear"):
    """
    Inputs:
    @clf: a TRAINED SVM classifier that has already been fitted to the data.
    @X: the feature matrix we constructed with shape (n, d)
    @y: the labels of each data point with shape (n,), note this is binary labels {1,-1}
    @symptom: the name of the symptom trying to classify, for output file naming purpose only
    @metric: a string specifying the performance metric to calculate for
    @model: the type of kernel being used, for output naming purpose only

    @return: a float representing the performance score of the classifier
    """

    y_pred = clf.decision_function(X)
    np.savetxt(f"results/test_prediction/predicted_distance_to_hyperplane_{model}_{symptom}.txt", y_pred)
    
    score = performance(y, y_pred, metric)

    return score
'''


'''
#################### SVM HYPERPARAMETER SELECTION ####################

# read symptom dictionary, make sure it is in the right dir so can be found
f = open('symptom_dictionary_merged_with_none_new')
symptoms = json.load(f)
f.close()

# split the data to 80/20, save the 20% as final test data, make sure these files are in the correct dir so can be found
X_training = np.loadtxt("training_examples.txt")
y_training = np.loadtxt("training_labels_new.txt")
X_testing = np.loadtxt("testing_examples.txt")
y_testing = np.loadtxt("testing_labels_new.txt")

print("Training set shape:", X_training.shape, y_training.shape)
print("Test set shape:", X_testing.shape, y_testing.shape)

# perform stratified k-fold, in which the folds are made by preserving the percentage of samples for each class
kf = StratifiedKFold(n_splits=5)

# since we are focusing on anxiety, extract y as the anxiety labels and process it into {-1, 1} labels
y_training = extract_symptom_labels(y_training, symptoms, symptom=target_symptom)
y_testing = extract_symptom_labels(y_testing, symptoms, symptom=target_symptom)

print(f"A single (neg) training example looks like: {X_training[0]}")
print(f"The corresponding label for that example looks like: {y_training[0]}")
print(f"A single (pos) training example looks like: {X_training[4]}")
print(f"The corresponding label for that example looks like: {y_training[4]}")

# for each metric, select optimal hyperparameter for linear-kernel SVM
optimalC_each_metric = select_param_linear(X_training, y_training, kf, metric=metric_list, symptom=target_symptom)
print(f"Optimal C for each metric is {optimalC_each_metric}")

# for each metric, select optimal hyperparameter for RBF-kernel SVM
optimalTuple_each_metric = select_param_rbf(X_training, y_training, kf, metric=metric_list, symptom=target_symptom)
print(f"Optimal C and gamma for each metric is {optimalTuple_each_metric}")
'''

#################### MAIN EXECUTION ####################
if __name__ == "__main__":
    
    # Load symptom dictionary
    with open(SYMPTOM_DICT, 'r') as f:
      symptoms = json.load(f)
    
    # Load data
    with np.load(TRAIN_MATRIX) as data:
        X_training = data
    with np.load(TRAIN_LABEL) as data:
        Y_training = data
    
    # Load fold indices
    fold_splits = load_fold_indices(KFOLD_INDICES_FILE)
    
    print(f"Training set shape: {X_training.shape}, {Y_training.shape}")
    print(f"Number of folds: {len(fold_splits)}")
    
    # Extract multi-label matrix
    Y_training_multilabel = extract_multilabel_matrix(Y_training, symptoms, TARGET_SYMPTOMS, negative_label=-1)
    print(f"Multi-label matrix shape: {Y_training_multilabel.shape} and labels {np.unique(Y_training_multilabel)}")
    
    # Create symptom string for file naming
    symptom_str = "_".join(TARGET_SYMPTOMS)
    
    # Hyperparameter selection for Linear SVM
    print("\n" + "="*60)
    print("LINEAR SVM HYPERPARAMETER SELECTION")
    print("="*60)
    best_C_linear, linear_results = select_param_linear_multilabel(
        X_training, Y_training_multilabel, fold_splits, 
        metric_list, TARGET_SYMPTOMS, symptom_str
    )
    
    # Hyperparameter selection for RBF SVM
    print("\n" + "="*60)
    print("RBF SVM HYPERPARAMETER SELECTION")
    print("="*60)
    best_params_rbf, rbf_results = select_param_rbf_multilabel(
        X_training, y_training_multilabel, fold_splits,
        metric_list, target_symptoms, symptom_str
    )
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SELECTION COMPLETE")
    print("="*60)
    print(f"\nResults saved to results/ directory")
    print(f"Linear SVM optimal parameters: results/linear/linear_SVM_optimal_params_{symptom_str}.json")
    print(f"RBF SVM optimal parameters: results/RBF/RBF_SVM_optimal_params_{symptom_str}.json")
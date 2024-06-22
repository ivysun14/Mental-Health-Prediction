#################### VARIABLES ####################
# change this variable to build prediction model for desired symptom label
target_symptom = "anxiety"

# change this to selected hyperparameters
optimal_c_linear = 100
optimal_c_rbf = 1000
optimal_gamma_rbf = 10

# define metric list: total 6 metrics we will check
metric_list = ["accuracy", "f1_score", "auroc",
               "precision", "sensitivity", "specificity"]

# define result dir structure
dirs = ["results",
        "results/linear",
        "results/RBF",
        "results/figures",
        "results/test_prediction"]

#################### SET UP ####################
# import libraries
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import collections

from string import punctuation
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
#import utility  # add diagnostic and graphing functions into this file


#################### CLUSTER RUN ####################
#target_symptom = sys.argv[1]
#outdir = sys.argv[2]

#dirs = [outdir+"results",
#        outdir+"results/linear",
#        outdir+"results/RBF",
#        outdir+"results/figures"]
#################### FUNCTIONS ####################

# set up directories
for dir_path in dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# set random seed
np.random.seed(1234)

#################### FUNCTIONS ####################
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


# This function extracts the corresponding label column from the label matrix provided the symptom name and
# the symptom dictionary, and convert the labels from {0,1} to {-1,1}.
#
def extract_symptom_labels(y, symptom_dict, symptom="anxiety"):
    """
    Inputs:
    @y: the label matrix with labels {0, 1}
    @symptom_dict: the symptom dictionary
    @symptom: the class / symptom we wish to build one-to-rest SVM classifier on, default "Anxiety"

    @return: a float representing the performance score of the classifier
    """
    
    index = symptom_dict[symptom]
    extract_label = y[:, index]
    extract_label[extract_label == 0] = -1
    return extract_label


#################### SVM PREDICTION ####################
# read symptom dictionary, make sure it is in the right dir so can be found
f = open('symptom_dictionary_merged_with_none_new')
symptoms = json.load(f)
f.close()

# split the data to 80/20, save the 20% as final test data, make sure these files are in the correct dir so can be found
X_training = np.loadtxt("training_examples.txt")
y_training = np.loadtxt("training_labels_new.txt")
X_testing = np.loadtxt("testing_examples.txt")
y_testing = np.loadtxt("testing_labels_new.txt")

y_training = extract_symptom_labels(y_training, symptoms, symptom=target_symptom)
y_testing = extract_symptom_labels(y_testing, symptoms, symptom=target_symptom)

print("Training set shape for this particular symptom:", X_training.shape, y_training.shape)
print("Test set shape for this particular symptom:", X_testing.shape, y_testing.shape)

# train linear-kernel SVM and RBF-kernel SVM with selected, optimal hyperparameters
linear_clf = SVC(kernel='linear', C=optimal_c_linear)
linear_clf = linear_clf.fit(X_training, y_training)
rbf_clf = SVC(kernel='rbf', C=optimal_c_rbf, gamma=optimal_gamma_rbf)
rbf_clf = rbf_clf.fit(X_training, y_training)

# Save the entire model (including support vectors)
joblib.dump(linear_clf, "results/linear/svm_model_linear.joblib")
joblib.dump(rbf_clf, "results/RBF/svm_model_rbf.joblib")

# test the performance of these two classifiers on the 100 subsetted transcripts
linear_metric_score = np.zeros(len(metric_list))
rbf_metric_score = np.zeros(len(metric_list))

# calculate bootstrap metric scores
for i in range(len(metric_list)):
    linear_metric_score[i] = performance_test(
        linear_clf, X_testing, y_testing, target_symptom, metric_list[i], model="linear")
    rbf_metric_score[i] = performance_test(
        rbf_clf, X_testing, y_testing, target_symptom, metric_list[i], model="RBF")

print(f"For the linear-kernel SVM, final metric scores are: {linear_metric_score}")
print(f"For the RBF-kernel SVM, final metric scores are: {rbf_metric_score}\n")
print("End of project.")
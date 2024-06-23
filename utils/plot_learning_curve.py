#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:22:30 2023

@author: maguo
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import joblib
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import json

def distance_to_hyperplane(model, X, y):
    # Get the decision function values
    decision_function = model.decision_function(X)
    
    # Take the absolute distance and return the mean
    distances = np.abs(decision_function)
    return np.mean(distances)

# Learning curve function
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    sizes = [0.1, 0.325, 0.55, 0.775, 0.1]
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Distance to hyperplane")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=distance_to_hyperplane)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def extract_symptom_labels(y, symptom_dict, symptom):
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


def load_data(target_symptom):
    # Load your pre-trained SVM model and dictionary
    svm_model = joblib.load(f'results_{target_symptom}/RBF/svm_model_rbf.joblib')
    # Define the percentage of the training data to use for each iteration

    f = open('symptom_dictionary_merged_with_none')
    symptoms = json.load(f)
    f.close()
    
    X_train = np.loadtxt("training_examples.txt")
    y_train = np.loadtxt("training_labels.txt")
    #X_test = np.loadtxt("testing_examples.txt")
    #y_test = np.loadtxt("testing_labels.txt")

    y_train = extract_symptom_labels(y_train, symptoms, symptom=target_symptom)
    #y_test = extract_symptom_labels(y_test, symptoms, symptom=target_symptom)

    plot_learning_curve(svm_model, f"Learning Curve for {target_symptom}", X_train, y_train, cv=5, n_jobs=-1)
    plt.title(f'Learning Curve for {target_symptom}')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Average distance to Separating Hyper-plane')
    plt.savefig(f'results_{target_symptom}/figures/svm_rbf_learning_curve.png',bbox_inches='tight')
    
#print("processing: frustration")
#load_data("frustration")
#print("processing: anger")
#load_data("anger")
print("processing: anxiety")
load_data("anxiety")
print("processing: depression")
load_data("depression")

"""
Created by Junwei (Ivy) Sun
Date: 12/02/2023

This file splits the total data set into 80% for training (training + dev) and
20% for testing. The input data are:

- feature matrix: feature_matrix_total.txt
- label matrix: label_matrix_merged_with_none.txt
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split

# set random seed for result reproducibility
np.random.seed(1234)

# read in feature and label matrices
feature_matrix = np.loadtxt("feature_matrix_total.txt")
print(feature_matrix)

label_matrix = np.loadtxt("label_matrix_merge_with_none.txt")
print(label_matrix)

# observe correspondance between feature matrix and label matrix
print(f"The shape of the feature matrix is {feature_matrix.shape}")
print(f"The shape of the label matrix is {label_matrix.shape}")

# observe label dictionary
f = open('symptom_dictionary_merged_with_none')
symptoms = json.load(f)
f.close()
print(symptoms)

indices = np.arange(feature_matrix.shape[0])
print(indices)
print(indices.shape)

# randomly select 20% of the total data to serve as test set
(
    X_training,
    X_testing,
    y_training,
    y_testing,
    indices_training,
    indices_testing
) = train_test_split(feature_matrix, label_matrix, indices, test_size=0.2, random_state=42)

print("Training set shape:", X_training.shape, y_training.shape)
print("Test set shape:", X_testing.shape, y_testing.shape)
print("Indices shape:", indices_training.shape, indices_testing.shape)

# save for later use
np.savetxt("train_test_data/training_examples.txt", X_training)
np.savetxt("train_test_data/training_labels.txt", y_training)
np.savetxt("train_test_data/testing_examples.txt",  X_testing)
np.savetxt("train_test_data/testing_labels.txt", y_testing)
np.savetxt("train_test_data/training_example_indices.txt",  indices_training)
np.savetxt("train_test_data/testing_example_indices.txt", indices_testing)
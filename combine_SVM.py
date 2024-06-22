import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

anxiety_dist = np.loadtxt("results/test_prediction/predicted_distance_to_hyperplane_RBF_anxiety.txt")
depression_dist = np.loadtxt("results/test_prediction/predicted_distance_to_hyperplane_RBF_depression.txt")

anxiety_pred = np.sign(anxiety_dist)
anxiety_pred[anxiety_pred == 0] = 1
anxiety_pred[anxiety_pred == -1] = 0
depression_pred  = np.loadtxt("results/test_prediction/predicted_labels_RBF_depression.txt")

anxiety_pred = anxiety_pred.reshape(-1, 1)
depression_pred = depression_pred.reshape(-1, 1)
y_preds = np.concatenate((anxiety_pred, depression_pred), axis=1)

label_mat = np.loadtxt("testing_labels_new.txt")
anxiety_label = label_mat[:, 1]
depression_label = label_mat[:, 2]
anxiety_label = anxiety_label.reshape(-1, 1)
depression_label = depression_label.reshape(-1, 1)
y_true = np.concatenate((anxiety_label,depression_label),axis=1)

anxiety_dist = anxiety_dist.reshape(-1, 1)
depression_dist = depression_dist.reshape(-1, 1)
y_prob = np.concatenate((anxiety_dist,depression_dist),axis=1)

print(y_preds.shape)
print(y_true.shape)
print(y_prob.shape)

accuracy = accuracy_score(y_true, y_preds)
f1_weighted = f1_score(y_true, y_preds, average='weighted')
auroc_weighted = roc_auc_score(y_true,  y_prob, average='weighted')

print(accuracy)
print(f1_weighted)
print(auroc_weighted)
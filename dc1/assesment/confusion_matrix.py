from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/TruePred_.csv')

class_labels = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']

true = data['true']
pred = data['pred']

cm = confusion_matrix(true, pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, xticklabels=class_labels,
            yticklabels=class_labels, cbar_kws={"label": "Scale"})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix', weight='bold', size=16)
plt.show();

# Cost Matrix


severity_weights = np.array([
    [0, 0.85, 0.55, 0.8, 0.9, 0.6],
    [3.55, 0, 3.65, 7, 4, 3.7],
    [1.75, 2.15, 0, 3.4, 2.2, 1.9],
    [0.05, 0.45, 0.15, 0, 0.5, 0.2],
    [3.85, 4.25, 3.95, 7.6, 0, 4],
    [4.65, 5.05, 4.75, 9.2, 5.1, 0]
])


def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def sum_normalize(matrix):
    row_sums = np.sum(matrix, axis=1)
    normalized_matrix = matrix / row_sums[:, np.newaxis]
    return normalized_matrix


def matrix_sum_normalize(matrix):
    norm_matrix = matrix / matrix.sum()
    return norm_matrix


# Computing the normalized matrix
normalized_matrix = matrix_sum_normalize(cm)
# Computing normalised weights
normalised_weights = matrix_sum_normalize(severity_weights)

# Computing the finalized matrix where we multiply it with the weights
weighted_conf_matrix = normalized_matrix * severity_weights
print(weighted_conf_matrix)



# Computing precision and recall from cm
precision = cm.diagonal() / cm.sum(axis=0)
recall = cm.diagonal() / cm.sum(axis=1)

# Handling potential division by zero issues
precision = np.nan_to_num(precision)
recall = np.nan_to_num(recall)

# Compute F1 score for each class
f1 = 2 * (precision * recall) / (precision + recall)

# Compute micro-average F1 score
micro_f1 = f1_score(true, pred, average='micro')

# Compute macro-average F1 score
macro_f1 = f1_score(true, pred, average='macro')


# Compute the custom score
score = weighted_conf_matrix.sum()


print("Class-wise F1 Score:", f1)
print("Micro-average F1 Score:", micro_f1)
print("Macro-average F1 Score:", macro_f1)
print("Custom score:",score)
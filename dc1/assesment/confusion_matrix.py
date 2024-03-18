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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, xticklabels=class_labels, yticklabels=class_labels, cbar_kws={"label": "Scale"})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix', weight='bold', size=16)
plt.show();


# Cost Matrix

severity_weights = np.array([
    [1, 1.1, 1.2, 1.6, 1.4, 1.3],
    [1.7, 1, 1.1, 2, 1.3, 1.2],
    [1.9, 1.8, 1, 1.8, 1.2, 1.1],
    [1.1, 1.2, 1.3, 1, 1.5, 1.4],
    [2.3, 2.2, 2.1, 2.4, 1, 2],
    [2.1, 2, 1.9, 2.2, 1.1, 1]
])
# Computing the normalized matrix
normalized_matrix = cm / cm.sum(axis=1, keepdims=True)

# Cmoputing the finalized matrix where we multiply it with the weights
weighted_conf_matrix = normalized_matrix * severity_weights


# Computing precision and recall from cm
precision = cm.diagonal() / cm.sum(axis=0)
recall = cm.diagonal() / cm.sum(axis=1)

# Computing precision and recall from weighted_conf_matrix
precision_weight = weighted_conf_matrix.diagonal() / weighted_conf_matrix.sum(axis=0)
recall_weight = weighted_conf_matrix.diagonal() / weighted_conf_matrix.sum(axis=1)

# Handling potential division by zero issues
precision = np.nan_to_num(precision)
recall = np.nan_to_num(recall)

precision_weight = np.nan_to_num(precision_weight)
recall_weight = np.nan_to_num(recall_weight)

# Compute F1 score for each class
f1 = 2 * (precision * recall) / (precision + recall)

# Compute F1 score for each class after adding weights
f1_weight = 2 * (precision_weight * recall_weight) / (precision_weight + recall_weight)


# Compute micro-average F1 score
micro_f1 = f1_score(true, pred, average='micro')

# Compute micro-average F1 score with weights
micro_f1_weight = np.nan_to_num(f1_weight).mean()

# Compute macro-average F1 score

macro_f1 = f1_score(true, pred, average='macro')

# Compute macro-average F1 score with weights
macro_f1_weight = np.average(f1_weight, weights=np.unique(true, return_counts=True)[1])

print("Class-wise F1 Score:", f1)
print("Micro-average F1 Score:", micro_f1)
print("Macro-average F1 Score:", macro_f1)


print("Class-wise F1 Score with weights added:", f1_weight)
print("Micro-average F1 Score with weights added:", micro_f1_weight)
print("Macro-average F1 Score with weights added:", macro_f1_weight)

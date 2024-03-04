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



# Assuming you already have the confusion matrix 'cm'
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

print("Class-wise F1 Score:", f1)
print("Micro-average F1 Score:", micro_f1)
print("Macro-average F1 Score:", macro_f1)

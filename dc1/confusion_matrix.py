from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../my_data/TruePred_.csv')

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
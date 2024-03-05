import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Change to parent directory
path = Path.cwd()
os.chdir(path.parent)

# Import the data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/Y_train.npy')
y_test = np.load('data/Y_test.npy')


# Distribution of image labels

def dist_image_labels():
    """Creates a stacked bar chart for the distribution of image labels for the training and test data"""

    # Data overview: counts, unique values statistics
    unique_vals_train, counts_train = np.unique(y_train, return_counts=True)
    unique_vals_test, counts_test = np.unique(y_test, return_counts=True)

    # Create a stacked barchart
    labels = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']
    plt.bar(labels, counts_train)
    plt.bar(labels, counts_test, bottom=counts_train)
    plt.ylabel("Count")
    plt.legend(["Training", "Test"])
    plt.title("Distribution of image labels")
    # plt.savefig('Distribution of image labels')
    plt.show()


# dist_image_labels()

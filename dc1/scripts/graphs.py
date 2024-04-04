import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def train_plot():
    """Creates a stacked bar chart for the distribution of image labels for the training data"""
    # Data overview: counts, unique values statistics
    unique_vals_train, counts_train = np.unique(y_train, return_counts=True)
    total = np.sum(counts_train)
    percentages = counts_train / total * 100
    labels = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']
    plt.bar(labels, counts_train)
    for i, percentage in enumerate(percentages):
        plt.text(i, counts_train[i] + 10, f"{percentage:.2f}%", ha='center')

    plt.ylabel("Count")
    plt.legend("Training")
    plt.title("Training Set class imbalance")
    plt.show()


# Combined confusion matrices
def combined_confusion_matrices():
    """Creates a visualization of three selected confusion matrices."""

    # Confusion matrices
    cm1 = np.array([
        [43, 69, 8, 15, 20, 0],
        [28, 91, 5, 5, 7, 3],
        [46, 80, 16, 34, 12, 1],
        [95, 81, 34, 92, 45, 6],
        [13, 26, 9, 25, 14, 3],
        [15, 29, 12, 10, 3, 5]
    ])
    cm2 = np.array([
        [25, 16, 5, 104, 1, 4],
        [14, 57, 15, 48, 0, 5],
        [12, 21, 14, 138, 1, 3],
        [26, 24, 9, 287, 2, 5],
        [4, 2, 2, 78, 0, 4],
        [1, 9, 3, 43, 0, 18]
    ])
    cm3 = np.array([
        [34, 25, 11, 54, 19, 12],
        [14, 65, 16, 23, 13, 8],
        [18, 35, 43, 66, 20, 7],
        [48, 35, 46, 142, 60, 22],
        [5, 6, 10, 44, 20, 5],
        [6, 9, 8, 20, 5, 26]
    ])

    class_labels = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']

    # Determine the common color limits for all heatmaps
    vmin = min(cm1.min(), cm2.min(), cm3.min())
    vmax = max(cm1.max(), cm2.max(), cm3.max())

    # Set up the figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Plot the first heatmap
    sns.heatmap(cm1, annot=True, ax=axs[0], cbar=False, fmt="d", cmap="Greens", annot_kws={"size": 16},
                xticklabels=class_labels, yticklabels=class_labels, vmin=vmin, vmax=vmax)
    axs[0].set_title('M1')
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')

    # Plot the second heatmap
    sns.heatmap(cm2, annot=True, ax=axs[1], cbar=False, fmt="d", cmap="Greens", annot_kws={"size": 16},
                xticklabels=class_labels, yticklabels=class_labels, vmin=vmin, vmax=vmax)
    axs[1].set_title('M4')
    axs[1].set_xlabel('Predicted Label')

    # Plot the third heatmap
    sns.heatmap(cm3, annot=True, ax=axs[2], cbar=False, fmt="d", cmap="Greens", annot_kws={"size": 16},
                xticklabels=class_labels, yticklabels=class_labels, vmin=vmin, vmax=vmax)
    axs[2].set_title('M5')
    axs[2].set_xlabel('Predicted Label')

    for ax in axs:
        ax.set_ylim(len(class_labels), 0)  # Reverse y-axis to match the default heatmap order

    # Add colorbar next to the heatmaps
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(axs[0].collections[0], cax=cax)
    cbar.set_label('Scale')

    # Add main title to the figure
    plt.subplots_adjust(top=0.85)  # Adjust the top margin to accommodate the main title
    fig.suptitle(
        'Confusion Matrices for Template Model (M1), ResNet with SMOTE (M4), and ResNet with Oversampling (M5)',
        weight='bold')

    plt.tight_layout()
    # plt.savefig('Confusion matrices for selected models')
    plt.show()

# combined_confusion_matrices()

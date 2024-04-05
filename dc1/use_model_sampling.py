import torch
import numpy as np
from net import Net
from image_dataset import ImageDataset, Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn as nn
import os
from model import getModel
from assesment.custom_metric import custom_metric, matrix_sum_normalize
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dc1.configurations import sampling_method

# initialize seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(mode=True, warn_only=True)

# ----------------------------------------------------------
model_file_name = "model_04_01_15_50.txt"
# ----------------------------------------------------------

def load_model_from_path(path_to_model: str):
    """
    Loads the model from the path.
    :param path_to_model: path to the file in which the weights are saves
    :param model: model for which the weights are supposed to be loaded
    :return: the model with saved weights
    """
    model = getModel()
    model.load_state_dict(torch.load(path_to_model))
    return model


def prepare_dataset_for_forward_pass(path_to_data: str, test_data: bool = True):
    """
    Prepares the data for the forward pass
    :param path_to_data: Your path to data
    :param test_data: Whether to prepare test data (True; default) or train data (False)
    :return: Data that you can pass forward in the model
    """
    train_dataset = ImageDataset(
        Path(path_to_data + r"/preprocessed/remove_outliers/X_train.npy"),
        Path(path_to_data + r"/preprocessed/remove_outliers/Y_train.npy"), sampling_method=sampling_method, train=True)
    test_dataset = ImageDataset(
        Path(path_to_data + r"/preprocessed/remove_outliers/X_test.npy"),
        Path(path_to_data + r"/preprocessed/remove_outliers/Y_test.npy"), sampling_method=sampling_method, train=False)
    if not test_data:
        X = train_dataset[:][0]
        Y = train_dataset[:][1]
    else:
        X = test_dataset[:][0]
        Y = test_dataset[:][1]
    lst_X = []
    lst_Y = []
    for i in range(1000):
        lst_X.append(X[i])
        lst_Y.append(Y[i])

    return torch.stack(lst_X).float(), torch.tensor(lst_Y).long()


def use_model(path_to_model: str, path_to_data: str, test_data: bool = True):
    processed_data = prepare_dataset_for_forward_pass(path_to_data, test_data)
    model = load_model_from_path(path_to_model)
    model.eval()
    with torch.no_grad():
        pred = model(processed_data[0])
    return pred


def calculate_metrics():
    true = []
    predicted = []
    count_correct = 0
    for batch in range(8):
        print(f'Batch number: {batch}/8')
        true_vals = \
            prepare_dataset_for_forward_pass(
                r"data",
                (batch * 1000, (batch + 1) * 1000)
            )[1]
        predictions = use_model(
            r"model_weights/{}".format(model_file_name),
            r"data",
            (batch * 1000, (batch + 1) * 1000)

        )
        for i in range(1000):
            true.append(true_vals[i])
            predicted.append(predictions[i])
            if true_vals[i] == np.argmax(predictions[i]):
                count_correct += 1
    print('Batch number: 8/8')
    true_vals = \
        prepare_dataset_for_forward_pass(
            r"data",
            (8000, 8420)
        )[1]
    predictions = use_model(
        r"model_weights/{}".format(model_file_name),
        r"data",
        (8000, 8420)
    )
    for i in range(420):
        true.append(true_vals[i])
        predicted.append(predictions[i])
        if true_vals[i] == np.argmax(predictions[i]):
            count_correct += 1

    accuracy = count_correct / 8420
    # print(f'Accuracy: {count_correct / 8420}')

    overall_f1 = f1_score(true_vals, np.argmax(predictions, axis=1), average='weighted')
    # print(f"Overall F1 Score: {overall_f1}")

    # Calculate precision and recall form confusion matrix
    cm = confusion_matrix(true_vals, np.argmax(predictions, axis=1))
    print(f"Confusion matrix: {cm}")
    precision = (cm.diagonal() / cm.sum(axis=0)).tolist()
    recall = (cm.diagonal() / cm.sum(axis=1)).tolist()
    print(f"Accuracy: {accuracy}, f1: {overall_f1}, precision: {precision}, recall: {recall}")

    # Compute custom metric
    custom_score = custom_metric(cm)
    print(f"Custom score: {custom_score}")

    # Save confusion matrix
    class_labels = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, xticklabels=class_labels,
                yticklabels=class_labels, cbar_kws={"label": "Scale"})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix', weight='bold', size=16)
    plt.savefig('Confusion matrix of best resnet with oversampling with balanced datasets = FALSE - CORRECT')
    plt.show();

    return accuracy, overall_f1, precision, recall, custom_score


def save_results_to_json(model_file_name_: str):
    accuracy, f1, precision, recall, custom_score = calculate_metrics()

    data = {"model": model_file_name_, "accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "custom score": custom_score}

    path = "results/resnet/experiment_results_metrics"
    path_file = f"{path}/experiment_results.json"
    json_data = []

    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

        with open(path_file, "w") as write_file:
            json_data.append(data)
            json.dump(json_data, write_file, indent=4)

    else:
        with open(path_file, 'r') as file:
            json_data = json.load(file)
            json_data.append(data)

    # Save the modified JSON back to the file
    with open(path_file, 'w') as f:
        json.dump(json_data, f, indent=4)


save_results_to_json(model_file_name)
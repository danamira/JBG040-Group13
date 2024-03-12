import torch
import pandas as pd
from dc1.net import Net
from dc1.image_dataset import ImageDataset, Path
from typing import Tuple
import re
import torch.nn as nn


def load_model_from_path(path_to_model: str, model: nn.Module = Net(6)):
    """
    Loads the model from the path.
    :param path_to_model: path to the file in which the weights are saves
    :param model: model for which the weights are supposed to be loaded
    :return: the model with saved weights
    """
    model.load_state_dict(torch.load(path_to_model))
    return model


def prepare_dataset_for_forward_pass(path_to_data: str, indices: Tuple = (0, 1000), test_data: bool = True):
    """
    Prepares the data for the forward pass
    :param path_to_data: Your path to data
    :param indices: Indices (default: all of them)
    :param test_data: Whether to prepare test data (True; default) or train data (False)
    :return: Data that you can pass forward in the model
    """
    train_dataset = ImageDataset(
        Path(path_to_data + r"\X_train.npy"),
        Path(path_to_data + r"\Y_train.npy"))
    test_dataset = ImageDataset(
        Path(path_to_data + r"\X_test.npy"),
        Path(path_to_data + r"\Y_test.npy"))
    if not test_data:
        X = train_dataset[:][0]
        Y = train_dataset[:][1]
    else:
        X = test_dataset[:][0]
        Y = test_dataset[:][1]
    lst_X = []
    lst_Y = []

    for i in range(indices[0], indices[1]):
        lst_X.append(X[i])
        lst_Y.append(Y[i])

    return torch.stack(lst_X).float(), torch.tensor(lst_Y).long()


def use_model(path_to_model: str, path_to_data: str, indices: Tuple = (0, 1000), test_data: bool = True, model: nn.Module = Net(6)):
    processed_data = prepare_dataset_for_forward_pass(path_to_data, indices, test_data)
    model = load_model_from_path(path_to_model, model)
    model.eval()
    with torch.no_grad():
        pred = model(processed_data[0])
    return pred


def save_predictions_in_csv(weights_path:str, path_to_data: str, model: nn.Module = Net(6)):
    true = []
    predicted = []
    for batch in range(8):
        print(batch)
        true_vals = \
            prepare_dataset_for_forward_pass(
                path_to_data,
                (batch * 1000, (batch + 1) * 1000)
            )[1]
        predictions = use_model(
            weights_path,
            path_to_data,
            (batch * 1000, (batch + 1) * 1000),
            model=model
        )
        for i in range(1000):
            true.append(true_vals[i])
            predicted.append(predictions[i].argmax())
    true_vals = \
        prepare_dataset_for_forward_pass(
            path_to_data,
            (8000, 8420)
        )[1]
    predictions = use_model(
        weights_path,
        path_to_data,
        (8000, 8420),
        model=model

    )
    for i in range(420):
        true.append(true_vals[i])
        predicted.append(predictions[i].argmax())

    pred_dataframe = pd.DataFrame({'true': true, 'pred': predicted})

    def clean_csv(x):
        return int(re.search(r'\d+', x).group())

    pred_dataframe = pred_dataframe[['true', 'pred']].map(clean_csv)
    pred_dataframe.to_csv('TruePred.csv')


def save_results_in_csv():
    true = []
    predicted = []
    for batch in range(8):
        print(batch)
        true_vals = \
            prepare_dataset_for_forward_pass(
                r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data",
                (batch * 1000, (batch + 1) * 1000)
            )[1]
        predictions = use_model(
            r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge "
            r"1\JBG040-Group13\dc1\model_weights\model_02_28_22_55.txt",
            r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data",
            (batch * 1000, (batch + 1) * 1000)

        )
        for i in range(1000):
            true.append(true_vals[i])
            predicted.append(predictions[i])
    print(8)
    true_vals = \
        prepare_dataset_for_forward_pass(
            r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data",
            (8000, 8420)
        )[1]
    predictions = use_model(
        r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge "
        r"1\JBG040-Group13\dc1\model_weights\model_02_28_22_55.txt",
        r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data",
        (8000, 8420)

    )
    for i in range(420):
        true.append(true_vals[i])
        predicted.append(predictions[i])

    pred_dataframe = pd.DataFrame({'true': true, 'logits': predicted})

    def clean_csv(x):
        return int(re.search(r'\d+', x).group())

    pred_dataframe.to_csv('TrueLogit_.csv')




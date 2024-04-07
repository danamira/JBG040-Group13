import torch
import matplotlib.pyplot as plt
import numpy as np
from dc1.image_dataset import ImageDataset
from pathlib import Path


def prepare_data_for_ViT_training(data, zeros: bool = False):
    lst_X = []
    lst_Y = []
    X = data[:][0]
    Y = data[:][1]
    for i in range(len(data)):
        if not zeros:
            X_tens = X[i].unsqueeze_(-1)
            X_tens = X_tens.expand(1, 128, 128, 3)
            X_tens = X_tens.view(128, 128, 3)
            X_tens = X_tens.permute(2, 0, 1)
            lst_X.append(X_tens)
        else:
            X_tens = X[i]
            zer = torch.zeros_like(X_tens)
            lst = [X_tens, zer, zer]
            torch.stack(lst)
        lst_Y.append(Y[i])
    return torch.stack(lst_X).float(), torch.tensor(lst_Y).long()


def save_prepared_data(zeros: bool = False):
    train_dataset = ImageDataset(
        Path(r"/dc1/data/X_train.npy"),
        Path(r"/dc1/data/Y_train.npy"))
    test_dataset = ImageDataset(
        Path(r"/dc1/data/X_test.npy"),
        Path(r"/dc1/data/Y_test.npy"))
    if not zeros:
        processed_train = prepare_data_for_ViT_training(train_dataset)
        processed_test = prepare_data_for_ViT_training(test_dataset)

    else:
        processed_train = prepare_data_for_ViT_training(train_dataset, True)
        processed_test = prepare_data_for_ViT_training(test_dataset, True)

    X_train = processed_train[0].numpy()
    Y_train = processed_train[1].numpy()
    X_test = processed_test[0].numpy()
    Y_test = processed_test[1].numpy()

    np.save("../../scripts/X_train_p.npy", X_train)
    np.save("../../scripts/Y_train_p.npy", Y_train)
    np.save("../../scripts/X_test_p.npy", X_test)
    np.save("../../scripts/Y_test_p.npy", Y_test)

save_prepared_data()



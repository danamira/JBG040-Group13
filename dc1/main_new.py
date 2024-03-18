# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net, Net_experiments, EarlyStopper
from dc1.train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import json

# retrieve current time to label artifacts
now = datetime.now()
model_file_name = f"model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}"
model_file_r_path = f"model_weights/{model_file_name}.txt"

# Experiment Description
experiment_type = 'optimizers'
optimizer_used = 'Adam'
description = '"early_stopping": "TRUE", "epochs": 10, "layers": 3, "convolutional_kernel": 5, "max_pooling_kernel": ' \
              '[4, 3, 2], "num_features_in_first_level": 64, "feature_decrement_step": "Half", "linear_layer": [16, ' \
              '2, 2]'

# Aggregation of description
data = {"model": model_file_name, "optimizer": optimizer_used, "description": description}


# ------------------------------------------------------------

def saving_model_info(experiment_type_, data_):
    path = f"../results/CNN-template/{experiment_type_}"
    path_file = f'{path}/experiment_results.json'

    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")

        with open(path_file, "w") as write_file:
            json.dump([data_], write_file, indent=4)

    else:
        with open(path_file, 'r') as file:
            json_data = json.load(file)
            json_data.append(data_)

    # Save the modified JSON back to the file
    with open(path_file, 'w') as f:
        json.dump(json_data, f, indent=4)


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load the train and test data set
    train_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
    test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    if optimizer_used == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    elif optimizer_used == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # Initialize early stopping
    early_stopper = EarlyStopper(patience=3, min_delta=10)

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
            torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    for e in range(n_epochs):
        if activeloop:
            # Training:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            print(f'Train loss {losses}')
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            print(f'Mean train loss {mean_loss}')
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            losses = test_model(model, test_sampler, loss_function, device)
            print(f'Test loss {losses}')
            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            print(f'Mean test loss {mean_loss}')
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            # Early stopping
            if early_stopper.early_stop(mean_loss):
                print('Early stopping')
                break

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    # check if model_weights/ subdir exists
    if not Path("../model_weights/").exists():
        os.mkdir(Path("../model_weights/"))

    # Saving the model
    torch.save(model.state_dict(), f'../{model_file_r_path}')
    saving_model_info(experiment_type, data)

    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)
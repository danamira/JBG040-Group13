import os
# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from net import EarlyStopper
from dc1.train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import json
from model import getModel
import random
from dc1.configurations import sampling_method
import numpy as np

dirOfThisFile=(os.path.dirname(os.path.realpath(__file__)))
os.chdir(dirOfThisFile)
cwd = os.getcwd()
print(cwd)

# initialize seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(mode=True, warn_only=True)

# retrieve current time to label artifacts
now = datetime.now()
model_file_name = f"model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}"
model_file_r_path = f"model_weights/{model_file_name}.txt"

# Experiment Description
architecture = "resnet"
experiment_type = "seeds"
optimizer_used = "SGD"  # Change depending on model used
early_stopping = "false"
epochs = "10"
# layers = "4"
# convolutional_kernel = "3"
# max_pooling_kernel = "[2,2,2,2]"
# num_features_in_first_level = "128"
# feature_decrement_factor = "0.5"
# linear_layer = "576"
comment = "layer_list=[1,3,4,2,1], model trained with seed 42"

# Aggregation of description
data = {"model": model_file_name,
        "optimizer": optimizer_used,
        "early_stopping": early_stopping,
        "epochs": epochs,
        # "layers": layers,
        # "convolutional_kernel": convolutional_kernel,
        # "max_pooling_kernel": max_pooling_kernel,
        # "num_features_in_first_level": num_features_in_first_level,
        # "feature_decrement_factor": feature_decrement_factor,
        # "linear_layer": linear_layer,
        "comments": comment}


# ------------------------------------------------------------

def save_experiment(experiment_type_, data_, architecture_):
    path = f"results/{architecture_}/{experiment_type_}"
    path_file = f'{path}/experiment_results_seeds.json'
    json_data = []
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")

        with open(path_file, "w") as write_file:
            json_data.append(data_)
            json.dump(json_data, write_file, indent=4)

    else:
        with open(path_file, 'r') as file:
            json_data = json.load(file)
            json_data.append(data_)

    # Save the modified JSON back to the file
    with open(path_file, 'w') as f:
        json.dump(json_data, f, indent=4)


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load the train and test data set
    train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"), sampling_method=sampling_method,
                                 train=True)
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"), sampling_method=sampling_method,
                                train=False)
    # Get the value of lambda from arguments
    weight_decay = args.weight_decay

    model = getModel()

    # Initialize optimizer(s) and loss function(s)
    if optimizer_used == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1, weight_decay=weight_decay)
    elif optimizer_used == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # Initialize early stopping
    early_stopper = EarlyStopper(patience=3, min_delta=10, manual=False)

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # Set to false in order to enable GPU training acceleration
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # initialize seeds for CUDA
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    elif (torch.backends.mps.is_available() and not DEBUG):  # PyTorch supports Apple Silicon GPUs from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Let's now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    earlyStopped = False

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
                earlyStopped = True
                print('Early stopping')
                break

            #  Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(os.path.join(Path.cwd(),'model_weights/'))

    # Saving the model
    torch.save(model.state_dict(), f'{model_file_r_path}')
    try:
        save_experiment(experiment_type, data, architecture)
    except:
        print('!! Experiment not saved. Please record your experiment manually. !!')

    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    if not earlyStopped:
        ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
        ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
        fig.legend()
        # Check if /artifacts/ subdir exists
        if not Path("artifacts/").exists():
            os.mkdir(os.path.join(Path.cwd(),"artifacts/"))

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
    parser.add_argument("--weight_decay", help="specifies the lambda value for L2 regularisation", default=0.1,
                        type=float)
    args = parser.parse_args()

    main(args)

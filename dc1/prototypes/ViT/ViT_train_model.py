from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.train_test import train_model, test_model
from ViT import IEViT, FullImageEmbedder, ModifiedViT
from dc1.use_model import save_predictions_in_csv
# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
from ViT import ModifiedViT
import timeit


# TrainerViT adapted from main

class TrainerViT:
    """
    Trainer for ViT.
    """
    TRAIN_DATA = ImageDataset(
        Path(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data\X_train.npy"),
        Path(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data\Y_train.npy"))
    TEST_DATA = ImageDataset(
        Path(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data\X_test.npy"),
        Path(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data\Y_test.npy"))

    def __init__(self, model: nn.Module, epochs: int, batch_size: int, optimizer: optim.Optimizer,
                 criterion=nn.CrossEntropyLoss, debug: bool = False, balanced_batches: bool = True):
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.balanced_batches = balanced_batches
        self.model = model
        self.debug = debug

    def train(self, activeloop: bool = True):
        """
        Trains the model.
        """
        if torch.cuda.is_available() and not self.debug:
            print("@@@ CUDA device found, enabling CUDA training...")
            device = "cuda"
            self.model.to(device)
        elif (
                torch.backends.mps.is_available() and not self.debug
        ):  # PyTorch supports Apple Silicon GPU's from version 1.12
            print("@@@ Apple silicon device enabled, training with Metal backend...")
            device = "mps"
            self.model.to(device)

        else:
            print("@@@ No GPU boosting device found, training on CPU...")
            device = "cpu"

        train_sampler = BatchSampler(
            batch_size=self.batch_size, dataset=TrainerViT.TRAIN_DATA, balanced=self.balanced_batches
        )
        test_sampler = BatchSampler(
            batch_size=100, dataset=TrainerViT.TEST_DATA, balanced=self.balanced_batches
        )

        mean_losses_train: List[torch.Tensor] = []
        mean_losses_test: List[torch.Tensor] = []

        start = timeit.default_timer()
        for e in range(self.epochs):
            if activeloop:
                # Training:
                losses = train_model(self.model, train_sampler, self.optimizer, self.criterion, device)
                # Calculating and printing statistics:
                mean_loss = sum(losses) / len(losses)
                mean_losses_train.append(mean_loss)
                print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

                # Testing:
                losses = test_model(self.model, test_sampler, self.criterion, device)

                # # Calculating and printing statistics:
                mean_loss = sum(losses) / len(losses)
                mean_losses_test.append(mean_loss)
                print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

                ### Plotting during training
                plotext.clf()
                plotext.scatter(mean_losses_train, label="train")
                plotext.scatter(mean_losses_test, label="test")
                plotext.title("Train and test loss")

                plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

                plotext.show()

        # retrieve current time to label artifacts
        now = datetime.now()
        # check if model_weights/ subdir exists
        if not Path("model_weights/").exists():
            os.mkdir(Path("model_weights/"))

        # Saving the model
        torch.save(self.model.state_dict(),
                   f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

        # Create plot of losses
        figure(figsize=(9, 10), dpi=80)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)

        ax1.plot(range(1, 1 + self.epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
        ax2.plot(range(1, 1 + self.epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
        fig.legend()

        # Check if /artifacts/ subdir exists
        if not Path("artifacts/").exists():
            os.mkdir(Path("artifacts/"))

        # save plot of losses
        fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

        stop = timeit.default_timer()
        print(f"Training Time: {stop - start:.2f}s")


encoder = FullImageEmbedder(768)
model = IEViT(image_size=128, patch_size=16, num_layers=12, hidden_dim=768, mlp_dim=3072, num_heads=12,
              full_embedding=encoder, num_classes=6)
trainer = TrainerViT(model=model,
                     epochs=10,
                     batch_size=16,
                     optimizer=optim.Adam(params=model.parameters(), lr=0.0001))

trainer.train()
# save_predictions_in_csv(
#     r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\prototypes\ViT\model_weights\model_03_10_18_04.txt",
#     r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data",
#     model=ModifiedViT(
#         image_size=128,
#         patch_size=16,
#         num_layers=12,
#         hidden_dim=768,
#         mlp_dim=3072,
#         num_heads=12,
#         num_classes=6
#     )
# )

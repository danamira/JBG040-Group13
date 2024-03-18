import numpy as np
import pandas as pd
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
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
import plotext  # type: igno
import re
from datetime import datetime
from pathlib import Path
from typing import List
from net import Net


# x_dataset = np.load(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data\X_train.npy")
# y_dataset = np.load(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data\Y_train.npy")

# print(x_dataset[0])

# print(torch.zeros(1, 1, 5))
# print(torch.cuda.is_available())

T_data = [[[1., 2.,3.], [3., 4.,4.]],
          [[5., 6.,0.], [7., 8.,6.]]]
T = torch.tensor(T_data)
print(T.shape)
for i in T_data:
    print(max(i))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path

from model import getModel
from image_dataset import ImageDataset


def visualize_cam(img_tensor, model, class_idx):
    # Assuming model.forward(img_tensor) returns logits and last layer feature maps
    logits, feature_maps = model(img_tensor)

    # Convert logits to probabilities and select the class of interest
    probs = torch.nn.functional.softmax(logits, dim=1)
    class_prob = probs[0, class_idx]

    # Use feature maps and class weight to generate CAM
    # This is a simplified approach; adjust based on your actual model architecture
    fc_weights = model.fc.weight.data  # Assuming model.fc is the final FC layer
    cam = torch.matmul(fc_weights[class_idx], feature_maps.squeeze(0))

    # Normalize the CAM for visualization
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Resize CAM and overlay on the original image for visualization
    # Note: You may need additional steps here based on your image and CAM sizes

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor.squeeze(0).permute(1, 2, 0))  # Assuming img_tensor is (B, C, H, W)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(cam.cpu().numpy(), cmap='jet', alpha=0.5)
    plt.title("Class Activation Map")
    plt.show()
# Usage example
# Assuming `img_tensor` is your preprocessed image tensor and `class_idx` is the class you're interested in
# visualize_cam(model, img_tensor, class_idx, device)



trainData = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"))

img = trainData.__getitem__(0)
# print(trainData)
visualize_cam(img[0].unsqueeze(0),getModel(),img[1])


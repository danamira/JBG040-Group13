import torch
import numpy as np
import random

# Set a seed value at the very beginning of your script
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# Continue with your script as before
from dc1.image_dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from dc1.resnet import ResNet, Bottleneck
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM

image_name = input("Enter the name of your image:")
model_name = input("Enter the full name of the model weights:")

trainData = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"))
modelPath = f"dc1/model_weights/{model_name}"

def main():
    imagesGroups = {i: [] for i in range(6)}

    for x in trainData:
        imagesGroups[x[1]] = x[0]

    image_path = 'input/' + image_name
    image = Image.open(image_path)

    # Define the transformations: resize, convert to grayscale, and then convert to a tensor
    transformations = transforms.Compose([
        transforms.Resize((128, 128)),  # Replace desired_height and desired_width with your values
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.ToTensor(),  # Convert to tensor
        # new stuff=========================
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        # ====================================
    ])

    model = ResNet(Bottleneck, layer_list=[1, 3, 4, 2, 1, 1], num_classes=6, num_channels=1)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    image_transformed = transformations(image)

    model.eval()
    with torch.no_grad():
        outputs = (model(image_transformed.unsqueeze(0)))
    outputs = F.softmax(outputs, dim=-1).tolist()[0]

    diseases = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']

    result = {diseases[i]: round(outputs[i], 3) for i in range(6)}

    probabilities_str = "Probabilities: "
    for key, value in result.items():
        probabilities_str += key + ": " + str(value) + " "

    # ----------------- CAM CODE -----------------
    model = ResNet(Bottleneck, layer_list=[1, 3, 4, 2, 1], num_classes=6, num_channels=1)

    lastConvLayer = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=lastConvLayer)

    img = image_transformed

    targets = None

    grayscale_cam = cam(input_tensor=img.unsqueeze(0), targets=targets)

    grayscale_cam = grayscale_cam[0]

    npimage = img[0].numpy()

    zeroes = np.zeros_like(npimage)
    x = np.stack((npimage, zeroes, zeroes), axis=-1)

    visualization = show_cam_on_image(x, grayscale_cam, use_rgb=True, image_weight=opacity)

    model_outputs = cam.outputs

    # ----------------- END CAM CODE -----------------

    # ----------------- DISPLAY THE IMAGES -----------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))

    # Display the first image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', weight='bold', fontsize=16)

    # Display the second image
    axes[1].imshow(visualization)
    axes[1].set_title('CAM', weight='bold', fontsize=16)

    # Hide the axes
    for ax in axes:
        ax.axis('off')

    fig.text(0.5, 0.1, probabilities_str, horizontalalignment='center', fontsize=11)
    fig.text(0.5, 0.05, "Diagnosis: " + max(result, key=result.get), horizontalalignment='center', fontsize=11)

    # plt.tight_layout()
    plt.show()
    # ----------------- END DISPLAY THE IMAGES -----------------



while not input('continue? ') in ['n', 'no']:
    opacity = input("Enter the number for the opacity or type default:")
    try:
        opacity = float(opacity)  # Try converting input to a float
    except ValueError:
        opacity = 0.5
    main()

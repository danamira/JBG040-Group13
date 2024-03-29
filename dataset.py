from dc1.image_dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
from dc1.resnet import ResNet, Bottleneck
import torch.nn.functional as F

trainData = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"))

transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

diseases = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']
imagesGroups = {disease: [] for disease in diseases}

for image_tensor, label_idx in trainData:
    image = transforms.ToPILImage()(image_tensor)
    disease = diseases[label_idx]
    image_transformed = transformations(image)
    imagesGroups[disease].append(image_transformed)


output_dir = Path("data_for_generation")
output_dir.mkdir(exist_ok=True)

for disease, images in imagesGroups.items():
    if images:
        all_images = torch.cat(images, dim=0)
        save_path = output_dir / f"{disease}.pt"
        torch.save(all_images, save_path)
        print(f"Saved {len(images)} images for {disease} at {save_path}")

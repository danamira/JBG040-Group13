from dc1.image_dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch
# from dc1.model import getModel,getModelFileName
from dc1.resnet import ResNet,Bottleneck
import torch.nn.functional as F


trainData = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"))

modelPath = "dc1/model_weights/best.txt"



i = 1


# while not input('continue? ') in ['n''no']:
#     x=(trainData.__getitem__(i))
#     print(x)
#     plt.imshow(x[0].permute(1,2,0))
#     plt.show()
#     i+=1

image_path = 'input/image.png'
image = Image.open(image_path)

# Define the transformations: resize, convert to grayscale, and then convert to a tensor
transformations = transforms.Compose([
    transforms.Resize((128, 128)),  # Replace desired_height and desired_width with your values
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor(),  # Convert to tensor
])

model = ResNet(Bottleneck,layer_list=[1,3,4,2,1],num_classes=6,num_channels=1)
model.load_state_dict(torch.load(modelPath))
image_transformed = transformations(image)


model.eval()
with torch.no_grad():
    outputs = (model(image_transformed.unsqueeze(0)))

outputs = F.softmax(outputs, dim=-1).tolist()[0]

diseases = ['Atelectasis','Effusion','Infiltration','No finding','Nodule','Pneumothorax']


result ={diseases[i]:round(outputs[i],3) for i in range(6)}

print("Probabilities:",result)

print("Diagnosis:", max(result, key=result.get))
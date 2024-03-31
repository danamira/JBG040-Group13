import matplotlib.pyplot as plt
import torch
import torchvision.transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import cv2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from resnet import ResNet,Bottleneck
from image_dataset import ImageDataset
from pathlib import Path
import numpy as np



model = ResNet(Bottleneck,layer_list=[1,3,4,2,1],num_classes=6,num_channels=1)

lastConvLayer= [model.layer4[-1]]

# print(lastConvLayer)

cam = GradCAM(model=model,target_layers=lastConvLayer)

trainData = ImageDataset(Path("dc1/data/X_train.npy"), Path("dc1/data/Y_train.npy"))

img = trainData.__getitem__(0)




targets = None

grayscale_cam = cam(input_tensor=img[0].unsqueeze(0), targets=targets)

# print(grayscale_cam[0].shape)
# quit()
grayscale_cam = grayscale_cam[0]





npimage = img[0][0].numpy()

# print(npimage.cat([torch.zeros((128,128,1)),torch.zeros((128,128,1))]))
zeroes = np.zeros_like(npimage)
x = np.stack((npimage,zeroes,zeroes),axis=-1)
# print(x)


# quit()


visualization = show_cam_on_image(x, grayscale_cam, use_rgb=True)



model_outputs = cam.outputs

plt.imshow(x)
plt.show()

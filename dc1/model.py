from net import Net
# from resnetCAM import ResNetCAM, Bottleneck
from resnet import ResNet,Bottleneck


# Determines if you want to train the original legacy model
USE_LEGACY_MODEL = False


# ================================================================== #
# Specify the model to be trained as the return value of getModel()  #
# ================================================================== #
def getModel():
    if USE_LEGACY_MODEL:
        return Net(n_classes=6)
    return ResNet(Bottleneck,layer_list=[1,3,4,2,1],num_classes=6,num_channels=1)
    # return ResNetCAM(Bottleneck, num_classes=6, num_channels=1)


def getModelFileName():
    return "model_03_31_16_02.txt"

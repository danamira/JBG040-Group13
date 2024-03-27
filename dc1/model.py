from net import Net
from resnetCAM import ResNetCAM, Bottleneck


# Determines if you want to train the original legacy model
USE_LEGACY_MODEL = False


# ================================================================== #
# Specify the model to be trained as the return value of getModel()  #
# ================================================================== #
def getModel():
    if USE_LEGACY_MODEL:
        return Net(n_classes=6)
    return ResNetCAM(Bottleneck, num_classes=6, num_channels=1)


def getModelFileName():
    return "model_03_18_14_39.txt"

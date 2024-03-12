import torch
import numpy as np
from net import Net
from image_dataset import ImageDataset, Path
from resnet import ResNet
from resnet import Bottleneck

def load_model_from_path(path_to_model):
    """
    Loads the model from the path.
    :param path_to_model: path to the file in which the weights are saves
    :return: the model with saved weights
    """
    model = ResNet(Bottleneck,layer_list=[3,4,6,3],num_classes=6,num_channels=1)
    model.load_state_dict(torch.load(path_to_model))
    return model


def prepare_dataset_for_forward_pass(path_to_data: str, test_data: bool = True):
    """
    Prepares the data for the forward pass
    :param path_to_data: Your path to data
    :param test_data: Whether to prepare test data (True; default) or train data (False)
    :return: Data that you can pass forward in the model
    """
    train_dataset = ImageDataset(
        Path(path_to_data + r"/X_train.npy"),
        Path(path_to_data + r"/Y_train.npy"))
    test_dataset = ImageDataset(
        Path(path_to_data + r"/X_test.npy"),
        Path(path_to_data + r"/Y_test.npy"))
    if not test_data:
        X = train_dataset[:][0]
        Y = train_dataset[:][1]
    else:
        X = test_dataset[:][0]
        Y = test_dataset[:][1]
    lst_X = []
    lst_Y = []
    for i in range(1000):
        lst_X.append(X[i])
        lst_Y.append(Y[i])

    return torch.stack(lst_X).float(), torch.tensor(lst_Y).long()


def use_model(path_to_model: str, path_to_data: str, test_data: bool = True):
    processed_data = prepare_dataset_for_forward_pass(path_to_data, test_data)
    model = load_model_from_path(path_to_model)
    model.eval()
    with torch.no_grad():
        pred = model(processed_data[0])
    return pred


predictions = use_model(
    r"model_weights/model_03_11_21_37.txt",
    r"data",
    True
)
true_vals = prepare_dataset_for_forward_pass(r"data")[1]
count_correct=0
for i in range(len(predictions)):
    print(f"True: {true_vals[i]}. Predicted: {np.argmax(predictions[i])}")
    if(true_vals[i]==np.argmax(predictions[i])):
        count_correct+=1
print(count_correct)
print(count_correct/len(predictions))






# prepare_dataset_for_forward_pass(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data")

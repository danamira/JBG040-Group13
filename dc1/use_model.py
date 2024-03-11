import torch
import numpy as np
from net import Net
from dc1.GoogLeNet import GoogLeNet
from image_dataset import ImageDataset, Path
from sklearn.metrics import accuracy_score
from torch.nn.functional import cross_entropy


def load_model_from_path(path_to_model):
    """
    Loads the model from the path.
    :param path_to_model: path to the file in which the weights are saves
    :return: the model with saved weights
    """
    # model = Net(6)
    model = GoogLeNet(6)
    model.load_state_dict(torch.load(path_to_model))
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for (x, y) in test_loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            predictions = model(x)

            # Record predictions and labels
            all_predictions.extend(predictions.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    # Calculate loss (you might need to modify this depending on your loss function)
    loss = cross_entropy(predictions, y)

    return loss.item(), accuracy


def prepare_dataset_for_forward_pass(path_to_data: str, test_data: bool = True):
    """
    Prepares the data for the forward pass
    :param path_to_data: Your path to data
    :param test_data: Whether to prepare test data (True; default) or train data (False)
    :return: Data that you can pass forward in the model
    """
    train_dataset = ImageDataset(
        Path(path_to_data + r"\X_train.npy"),
        Path(path_to_data + r"\Y_train.npy"))
    test_dataset = ImageDataset(
        Path(path_to_data + r"\X_test.npy"),
        Path(path_to_data + r"\Y_test.npy"))
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


# predictions = use_model(
#     r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\model_weights\model_02_28_22_55.txt",
#     r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data",
#     True
# )
# true_vals = prepare_dataset_for_forward_pass(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data")[1]
# for i in range(len(predictions)):
#     print(f"True: {true_vals[i]}. Predicted: {predictions[i]}")


def main():
    # Paths and settings
    path_to_model = r"C:\Users\Askeniia\Desktop\dc\dc1\model_weights\model_03_01_16_23.txt"
    path_to_data = r"C:\Users\Askeniia\Desktop\dc\dc1\data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available

    # Prepare test dataset
    test_dataset = ImageDataset(Path(path_to_data + r"\X_test.npy"), Path(path_to_data + r"\Y_test.npy"))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    model = load_model_from_path(path_to_model)
    model.to(device)  # Make sure the model is on the correct device

    # Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)

    # Report metrics
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()





# prepare_dataset_for_forward_pass(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\data")

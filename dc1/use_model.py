import torch
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from prototypes.ViT.ViT import ModifiedViT
from dc1.net import Net
from dc1.image_dataset import ImageDataset, Path
from typing import Tuple
import re
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer


def load_model_from_path(path_to_model: str, model: nn.Module = Net(6)):
    """
    Loads the model from the path.
    :param path_to_model: path to the file in which the weights are saves
    :param model: model for which the weights are supposed to be loaded
    :return: the model with saved weights
    """
    model.load_state_dict(torch.load(path_to_model))
    return model


def prepare_dataset_for_forward_pass(path_to_data: str, indices: Tuple = (0, 1000), test_data: bool = True):
    """
    Prepares the data for the forward pass
    :param path_to_data: Your path to data
    :param indices: Indices (default: all of them)
    :param test_data: Whether to prepare test data (True; default) or train data (False)
    :return: Data that you can pass forward in the model
    """
    train_dataset = ImageDataset(
        Path(path_to_data + r"\X_train_p.npy"),
        Path(path_to_data + r"\Y_train_p.npy"))
    test_dataset = ImageDataset(
        Path(path_to_data + r"\X_test_p.npy"),
        Path(path_to_data + r"\Y_test_p.npy"))
    if not test_data:
        X = train_dataset[:][0]
        Y = train_dataset[:][1]
    else:
        X = test_dataset[:][0]
        Y = test_dataset[:][1]
    lst_X = []
    lst_Y = []
    for i in range(indices[0], indices[1]):
        lst_X.append(X[i])
        lst_Y.append(Y[i])

    return torch.stack(lst_X).float(), torch.tensor(lst_Y).long()


def use_model(path_to_model: str, path_to_data: str, indices: Tuple = (0, 1000), test_data: bool = True,
              model: nn.Module = Net(6)):
    processed_data = prepare_dataset_for_forward_pass(path_to_data, indices, test_data)
    model = load_model_from_path(path_to_model, model)
    model.eval()
    with torch.no_grad():
        pred = model(processed_data[0])
    return pred


def save_predictions_in_csv(path_to_model: str, path_to_data: str, model: nn.Module = Net(6)):
    true = []
    predicted = []
    for batch in range(8):
        print(batch)
        true_vals = \
            prepare_dataset_for_forward_pass(
                path_to_data,
                (batch * 1000, (batch + 1) * 1000)
            )[1]
        predictions = use_model(
            path_to_model,
            path_to_data,
            (batch * 1000, (batch + 1) * 1000),
            model=model
        )
        #print(predictions)
        for i in range(1000):
            true.append(true_vals[i])
            predicted.append(predictions[i].argmax())
    true_vals = \
        prepare_dataset_for_forward_pass(
            path_to_data,
            (8000, 8420)
        )[1]
    predictions = use_model(
        path_to_model,
        path_to_data,
        (8000, 8420),
        model=model

    )
    for i in range(420):
        true.append(true_vals[i])
        predicted.append(predictions[i].argmax())

    pred_dataframe = pd.DataFrame({'true': true, 'pred': predicted})

    # pred_dataframe = pred_dataframe[['true', 'pred']].map(clean_csv)
    pred_dataframe.to_csv('TruePred.csv')


# save_predictions_in_csv(
#     weights_path=r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\prototypes\ViT\model_weights\model_03_10_18_04.txt",
#     path_to_data=r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data",
#     model=VisionTransformer(image_size=128, patch_size=16, num_layers=12, hidden_dim=768, mlp_dim=3072, num_heads=12,
#                             num_classes=6)
# )


def clean_csv(x):
    return int(re.search(r'\d+', x).group())


# def save_results_in_csv():
#     true = []
#     predicted = []
#     for batch in range(8):
#         print(batch)
#         true_vals = \
#             prepare_dataset_for_forward_pass(
#                 r"/dc1/data",
#                 (batch * 1000, (batch + 1) * 1000)
#             )[1]
#         predictions = use_model(
#             r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge "
#             r"1\JBG040-Group13\dc1\model_weights\model_02_28_22_55.txt",
#             r"/dc1/data",
#             (batch * 1000, (batch + 1) * 1000)
#
#         )
#         for i in range(1000):
#             true.append(true_vals[i])
#             predicted.append(predictions[i])
#     print(8)
#     true_vals = \
#         prepare_dataset_for_forward_pass(
#             r"/dc1/data",
#             (8000, 8420)
#         )[1]
#     predictions = use_model(
#         r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge "
#         r"1\JBG040-Group13\dc1\model_weights\model_02_28_22_55.txt",
#         r"/dc1/data",
#         (8000, 8420)
#
#     )
#     for i in range(420):
#         true.append(true_vals[i])
#         predicted.append(predictions[i])
#
#     pred_dataframe = pd.DataFrame({'true': true, 'logits': predicted})
#
#     pred_dataframe.to_csv('TrueLogit_.csv')


def evaluate_from_csv(path):
    file = pd.read_csv(path)
    file = file[["true", "pred"]]
    clean_file = file.map(clean_csv)
    accuracy = accuracy_score(clean_file['true'], clean_file['pred'])
    # print(f'Accuracy: {count_correct / 8420}')

    overall_f1 = f1_score(clean_file['true'], clean_file['pred'], average='weighted')
    # print(f"Overall F1 Score: {overall_f1}")

    # Calculate precision and recall form confusion matrix
    cm = confusion_matrix(clean_file['true'], clean_file['pred'])
    precision = (cm.diagonal() / cm.sum(axis=0)).tolist()
    recall = (cm.diagonal() / cm.sum(axis=1)).tolist()
    print(f"Accuracy: {accuracy}, f1: {overall_f1}, precision: {precision}, recall: {recall}")

    return accuracy, overall_f1, precision, recall


save_predictions_in_csv(
    path_to_model=r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge "
                  r"1\JBG040-Group13\dc1\model_weights\model_03_29_20_58.txt",
    path_to_data=r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\scripts",
    model=VisionTransformer(image_size=128, patch_size=16, num_layers=12, hidden_dim=768, mlp_dim=3072, num_heads=12,
                            num_classes=6)
)

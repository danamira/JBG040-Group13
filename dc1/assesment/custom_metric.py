import numpy as np


cm = np.array([[46, 25, 11, 9, 44, 20],
               [21, 65, 7, 6, 21, 19],
               [29, 38, 29, 12, 46, 35],
               [54, 40, 36, 34, 127, 62],
               [11, 9, 6, 9, 40, 15],
               [7, 13, 3, 4, 17, 30]])


def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def sum_normalize(matrix):
    row_sums = np.sum(matrix, axis=1)
    normalized_matrix = matrix / row_sums[:, np.newaxis]
    return normalized_matrix


def matrix_sum_normalize(matrix):
    norm_matrix = matrix / matrix.sum()
    return norm_matrix


def custom_metric(confusion_matrix_):
    # Define cost matrix
    severity_weights = np.array([
        [0, 0.85, 0.55, 0.8, 0.9, 0.6],
        [3.55, 0, 3.65, 7, 4, 3.7],
        [1.75, 2.15, 0, 3.4, 2.2, 1.9],
        [0.05, 0.45, 0.15, 0, 0.5, 0.2],
        [3.85, 4.25, 3.95, 7.6, 0, 4],
        [4.65, 5.05, 4.75, 9.2, 5.1, 0]
    ])

    # Computing the normalized matrix
    normalized_matrix = matrix_sum_normalize(confusion_matrix_)
    # Computing normalised weights
    normalised_weights = matrix_sum_normalize(severity_weights)

    # Computing the finalized matrix where we multiply it with the weights
    weighted_conf_matrix = normalized_matrix * severity_weights
    # print(weighted_conf_matrix)

    # Compute the custom score
    score = weighted_conf_matrix.sum()

    return score

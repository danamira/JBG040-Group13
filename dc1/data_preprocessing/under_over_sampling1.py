from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import numpy as np


def no_sampling(X_train, y_train):
    return X_train, y_train


def undersampling(y_train, X_train):
    """Randomly removes elements from the majority classes."""
    rus = RandomUnderSampler(random_state=42, sampling_strategy='not minority')
    rus.fit_resample(X_train[:, :, :, 0][:, :, 0], y_train)
    X_resampled = X_train[rus.sample_indices_]
    y_resampled = y_train[rus.sample_indices_]
    return X_resampled, y_resampled


def oversampling(y_train, X_train):
    """Randomly removes elements from the majority classes."""
    rus = RandomOverSampler(random_state=42, sampling_strategy='not majority')
    rus.fit_resample(X_train[:, :, :, 0][:, :, 0], y_train)
    X_resampled = X_train[rus.sample_indices_]
    y_resampled = y_train[rus.sample_indices_]
    return X_resampled, y_resampled


def smote_sampling(y_train, X_train):
    """Over-samples the data using Synthetic Minority Oversampling Technique (SMOTE)."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
    # Reshape the resampled data back to the original format
    X_resampled = X_resampled.reshape(-1, *X_train.shape[1:])
    return X_resampled, y_resampled


def smoteenn_sampling(y_train, X_train):
    """Combines over- and under-sampling using SMOTE."""
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
    # Reshape the resampled data back to the original format
    X_resampled = X_resampled.reshape(-1, *X_train.shape[1:])
    return X_resampled, y_resampled


def smoteomek_sampling(y_train, X_train):
    """Combines over- and under-sampling using SMOTE."""
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
    # Reshape the resampled data back to the original format
    X_resampled = X_resampled.reshape(-1, *X_train.shape[1:])
    return X_resampled, y_resampled



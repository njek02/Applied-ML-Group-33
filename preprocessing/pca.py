from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def to_pca(norm_spectrogram: list[(np.ndarray)]) -> np.ndarray:
    """
    Applies PCA to a list of normalized spectrograms.

    Each spectrogram is flattened into a 1D feature vector,
    standardized, and then reduced to 2 principal components.

    Parameters:
    norm_spectrogram : list of np.ndarray
        A list of 2D normalized spectrograms (e.g., shape (128, 173) each).

    Returns:
    np.ndarray
        A 2D array of shape (n_samples, 2), where each row is the 
        PCA-transformed representation of the corresponding input spectrogram.
    """
    features = np.array([spec.flatten() for spec in norm_spectrogram])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(scaled_features)

    return features_pca

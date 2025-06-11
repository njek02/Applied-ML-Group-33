import sys
import os
import numpy as np
import pandas as pd
from whale_call_project.preprocessing.normalization import spectrogram_normalization
import librosa
from whale_call_project.preprocessing.wave_to_image import wave_to_spec
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from whale_call_project.models.SVM import SVCModel 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



class Preprocess():
    """
    Preprocess audio files for SVM input.
    """    
    def __init__(self) -> None:
        """
        Initialize the preprocessing pipeline for SVM.
        """        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50)),
            ('svm', SVCModel())
        ])


    def preprocess_training_files(self, file_location: str) -> None:
        """
        Preprocess training audio files and fit the SVM model.

        Args:
            file_location (str): Path to the training data folder.
        """
        spec_list = []
        dataset = pd.read_csv(f"data/data_labels/{file_location}.csv")
        for file_name in dataset["clip_name"]:
            spec_list.append(create_spectrogram(f"data/{file_location}/{file_name}"))

        flat_spec_list = np.mean(spec_list, axis=1).reshape(len(spec_list),-1)
        x_train = np.array(flat_spec_list)
        self.pipeline.fit(x_train, dataset["label"])


    def preprocess_validation_test_files(self, file_location: str) -> np.ndarray:
        """
        Preprocess validation/test audio files for SVM input.

        Args:
            file_location (str): Path to the validation/test data folder.
        
        Returns:
            np.ndarray: Predicted labels for the validation/test data.
        """
        spec_list = []
        dataset = pd.read_csv(f"data/data_labels/{file_location}.csv")
        for file_name in dataset["clip_name"]:
            spec_list.append(create_spectrogram(f"data/{file_location}/{file_name}"))

        flat_spec_list = np.mean(spec_list, axis=1).reshape(len(spec_list),-1)
        x_predict = np.array(flat_spec_list)
        return self.pipeline.predict(x_predict)


def create_spectrogram(file_path: str) -> np.ndarray:
    """
    Create a spectrogram from an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        norm_spec_image (np.ndarray): Normalized spectrogram of the audio file.
    """    
    audio_path = os.path.abspath(file_path)

    raw_audio, sr = librosa.load(audio_path, sr=None)
    spec_image = wave_to_spec(y=raw_audio, sr=sr)

    norm_spec_image = spectrogram_normalization(spec_image)
    return norm_spec_image

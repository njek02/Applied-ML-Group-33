import sys
import os
import numpy as np
import pandas as pd
from normalization import peak_normalization, spectrogram_normalization
from pca import to_pca
# from split_data import datasplit
import librosa
from wave_to_spec import wave_to_spec
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_name.models.SVM import SVCModel 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



class Preprocess():
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50)),
            ('svm', SVCModel())
        ])


    def preprocess_training_files(self, file_location):
        spec_list = []
        dataset = pd.read_csv(f"{file_location}.csv")
        for file_name in dataset["clip_name"]:
            spec_list.append(create_spectrogram(f"{file_location}/{file_name}"))

        flat_spec_list = np.mean(spec_list, axis=1).reshape(len(spec_list),-1)
        x_train = np.array(flat_spec_list)
        self.pipeline.fit(x_train, dataset["label"])


    def preprocess_validation_test_files(self, file_location):
        spec_list = []
        dataset = pd.read_csv(f"{file_location}.csv")
        for file_name in dataset["clip_name"]:
            spec_list.append(create_spectrogram(f"{file_location}/{file_name}"))

        flat_spec_list = np.mean(spec_list, axis=1).reshape(len(spec_list),-1)
        x_predict = np.array(flat_spec_list)
        return self.pipeline.predict(x_predict)


def create_spectrogram(file_path: str):
    audio_path = os.path.abspath(file_path)

    raw_audio, sr = librosa.load(audio_path, sr=None)
    spec_image = wave_to_spec(y=raw_audio, sr=sr, n_fft=256, hop_len=64, n_mels=64)

    norm_spec_image = spectrogram_normalization(spec_image)
    return norm_spec_image


if __name__ == "__main__":
    # Example usage
    # preprocessor = Preprocess()
    # preprocessor.preprocess_training_files("data/training_data")
    # val_list = preprocessor.preprocess_validation_test_files("data/validation_data")
    # test_list = preprocessor.preprocess_validation_test_files("data/test_data")

    # np.save("data/val_list.npy", val_list)
    # np.save("data/test_list.npy", test_list)

    # print(len(val_list), np.count_nonzero(val_list))
    # print(len(test_list), np.count_nonzero(test_list))

    val_list = np.load("data/val_list.npy")
    test_list = np.load("data/test_list.npy")
    val_data = pd.read_csv("data/validation_data.csv")
    test_data = pd.read_csv("data/test_data.csv")
    y_val = np.array(val_data["label"])
    y_test = np.array(test_data["label"])
    print(np.count_nonzero(y_val))
    print(np.count_nonzero(y_test))
    print(np.count_nonzero(val_list - y_val))
    print(np.count_nonzero(test_list- y_test))

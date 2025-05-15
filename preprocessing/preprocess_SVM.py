import sys
import os
import numpy as np
import pandas as pd
from normalization import peak_normalization, spectrogram_normalization
# from pca import to_pca
# from split_data import datasplit
import librosa
from wave_to_spec import wave_to_spec
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics.visualization import Visualizer
from metrics.evaluation import evaluate_model
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
        # np.save("data/x_train.npy", x_train)
        self.pipeline.fit(x_train, dataset["label"])


    def preprocess_validation_test_files(self, file_location):
        spec_list = []
        dataset = pd.read_csv(f"{file_location}.csv")
        for file_name in dataset["clip_name"]:
            spec_list.append(create_spectrogram(f"{file_location}/{file_name}"))

        flat_spec_list = np.mean(spec_list, axis=1).reshape(len(spec_list),-1)
        x_predict = np.array(flat_spec_list)
        # np.save(f"{file_location}_list.npy", x_predict)
        return self.pipeline.predict(x_predict)


def create_spectrogram(file_path: str):
    audio_path = os.path.abspath(file_path)

    raw_audio, sr = librosa.load(audio_path, sr=None)
    spec_image = wave_to_spec(y=raw_audio, sr=sr, n_fft=256, hop_len=64, n_mels=64)

    norm_spec_image = spectrogram_normalization(spec_image)
    return norm_spec_image


if __name__ == "__main__":
    # Example usage
    preprocessor = Preprocess()
    # preprocessor.preprocess_training_files("data/training_data")
    # val_predict = preprocessor.preprocess_validation_test_files("data/validation_data")
    # test_predict = preprocessor.preprocess_validation_test_files("data/test_data")
    
    training_data = np.load("data/x_train.npy")
    train_dataset = pd.read_csv("data/training_data.csv")
    preprocessor.pipeline.fit(training_data, train_dataset["label"])

    val_list = np.load("data/validation_data_list.npy")
    val_dataset = pd.read_csv("data/validation_data.csv")
    val_predict = preprocessor.pipeline.predict(val_list)

    test_list = np.load("data/test_data_list.npy")
    test_dataset = pd.read_csv("data/test_data.csv")
    test_predict = preprocessor.pipeline.predict(test_list)

    print(evaluate_model("classification_report", val_dataset["label"], val_predict))
    print(evaluate_model("classification_report", test_dataset["label"], test_predict))

    Visualizer().plot_confusion_matrix(val_predict, val_dataset["label"], "Validation")
    Visualizer().plot_confusion_matrix(test_predict, test_dataset["label"], "Test")


    # val_list = np.load("data/val_list.npy")
    # test_list = np.load("data/test_list.npy")
    # val_data = pd.read_csv("data/validation_data.csv")
    # test_data = pd.read_csv("data/test_data.csv")
    # y_val = np.array(val_data["label"])
    # y_test = np.array(test_data["label"])
    # print(np.count_nonzero(y_val))
    # print(np.count_nonzero(y_test))
    # print(np.count_nonzero(val_list - y_val))
    # print(np.count_nonzero(test_list- y_test))

from whale_call_project.preprocessing.preprocess_SVM import Preprocess, evaluate_model
from whale_call_project.metrics.visualization import Visualizer
import numpy as np
import pandas as pd


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

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whale_call_project.preprocessing.preprocess_SVM import Preprocess
from whale_call_project.metrics.evaluation import evaluate_model
from whale_call_project.metrics.visualization import Visualizer
import pandas as pd

if __name__ == "__main__":

    preprocessor = Preprocess()
    training_data = preprocessor.preprocess_training_files("training_data")
    val_predict = preprocessor.preprocess_validation_test_files("validation_data")
    test_predict = preprocessor.preprocess_validation_test_files("test_data")

    val_dataset = pd.read_csv("data/data_labels/validation_data.csv")
    test_dataset = pd.read_csv("data/data_labels/test_data.csv")


    print("Evaluation on Validation Set:")
    print("-" * 50)
    print("Precision: ", evaluate_model("precision", val_dataset["label"], val_predict))
    print("Recall: ", evaluate_model("recall", val_dataset["label"], val_predict))
    print("F1_score: ", evaluate_model("f1_score", val_dataset["label"], val_predict))

    Visualizer().plot_confusion_matrix(val_predict, val_dataset["label"], "Validation")

    print()

    print("Evaluation on Test Set:")
    print("-" * 50)
    print("Precision: ", evaluate_model("precision", test_dataset["label"], test_predict))
    print("Recall: ", evaluate_model("recall", test_dataset["label"], test_predict))
    print("F1_score: ", evaluate_model("f1_score", test_dataset["label"], test_predict))

    Visualizer().plot_confusion_matrix(test_predict, test_dataset["label"], "Test")

    print("-" * 50)


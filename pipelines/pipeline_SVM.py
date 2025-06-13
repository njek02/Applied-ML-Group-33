from whale_call_project.preprocessing.preprocess_SVM import Preprocess
from whale_call_project.metrics.visualization import Visualizer
from whale_call_project.metrics.evaluation import evaluate_model
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, ConfusionMatrixDisplay, PrecisionRecallDisplay


if __name__ == "__main__":

    preprocessor = Preprocess()
    training_data = preprocessor.preprocess_training_files("training_data")
    val_predict = preprocessor.preprocess_validation_test_files("validation_data")
    test_predict = preprocessor.preprocess_validation_test_files("test_data")

    val_dataset = pd.read_csv("data/data_labels/validation_data.csv")
    test_dataset = pd.read_csv("data/data_labels/test_data.csv")

    # test_predict = np.ones(len(val_dataset["label"]))


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
    conf_matrix = confusion_matrix(y_true=test_dataset["label"], y_pred=test_predict)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(test_dataset["label"]))
    disp.plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    print("-" * 50)

    def bootstrap_ci(y_true, y_pred, metric=f1_score, n_bootstrap=5, alpha=0.05):
        n = len(y_true)
        scores = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            scores.append(metric(np.array(y_true)[idx], np.array(y_pred)[idx]))
        lower = np.percentile(scores, 100 * (alpha / 2))
        upper = np.percentile(scores, 100 * (1 - alpha / 2))
        return np.mean(scores), (lower, upper)

    # Example usage:


    mean_acc, (ci_lower, ci_upper) = bootstrap_ci(test_predict, test_dataset["label"])
    print(f"F-1: {mean_acc:.4f} (95% CI: {ci_lower:.4f}–{ci_upper:.4f})")


import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    confusion_matrix,
)

def evaluate_model(name, y_true, y_pred):
    results = {
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

    if name in results:
        return results[name]
    else:
        raise ValueError(f"Unrecognized metric name: {name}")


def save_evaluation_results(report, name, path):
    df = pd.DataFrame(report).transpose()
    full_path = f"{path}/{name}_evaluation_report.csv"
    df.to_csv(full_path, index=True)
    print(f"Evaluation report saved to {full_path}")
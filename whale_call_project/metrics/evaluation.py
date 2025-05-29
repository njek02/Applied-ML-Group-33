import pandas as pd
from typing import Any, Dict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def evaluate_model(name, y_true, y_pred) -> Any:
    """_
    This function is used to perfrom the metrics.

    Args:
        name (str): The name of the metric desired to be performed.
        y_true (np.ndarray): Correct target values.
        y_pred (np.ndarray): Estimated target values predicted by the model.

    Raises:
        ValueError: Desired metric not found in dictionary

    Returns:
        Any : The requested metric
    """    
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


def save_evaluation_results(report: Dict[str, Dict[str, float]], name: str, path: str) -> None:

    """Saves a classification report as a CSV file.

    Args:
        report (dict): The classification report of the model's performance.
        name (str): The name to include in the output file.
        path (str): The directory path where the CSV file will be saved
    """    
    df = pd.DataFrame(report).transpose()
    full_path = f"{path}/{name}_evaluation_report.csv"
    df.to_csv(full_path, index=True)
    print(f"Evaluation report saved to {full_path}")
import matplotlib.pyplot as mp
import seaborn as sb
from metrics.evaluation import evaluate_model
from sklearn.metrics import roc_curve, auc

class Visualizer():

    def plot_confusion_matrix(self, y_pred, y_true, classes, model_name):
        """
        Plot the confusion matrix for the given predictions and true labels.
        """
        cm = evaluate_model("confusion_matrix", y_true, y_pred)
        mp.figure(figsize=(10, 7))
        sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        mp.title(f"Confusion Matrix for {model_name}")
        mp.xlabel("Predicted")
        mp.ylabel("True")
        mp.savefig(f"confusion_matrix_{model_name}.png")
        mp.show()

    def plot_binary_ROC_AUC(self, y_true, y_pred_prob, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        mp.figure(figsize=(8, 6))
        mp.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        mp.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.50)")
        mp.title(f"ROC Curve: {model_name}")
        mp.xlabel("False Positive Rate")
        mp.ylabel("True Positive Rate")
        mp.legend(loc="best")
        mp.tight_layout()
        mp.savefig(f"results/visualisations/{model_name}_roc_auc.png")
        mp.show()
    
    def plot_loss(self, history, model_name):
        """
        Plot the training and validation loss over epochs.
        """
        mp.figure(figsize=(10, 6))
        mp.plot(history.history['loss'], label='Training Loss')
        mp.plot(history.history['val_loss'], label='Validation Loss')
        mp.title(f"Loss Curve for {model_name}")
        mp.xlabel("Epochs")
        mp.ylabel("Loss")
        mp.legend()
        mp.tight_layout()
        mp.savefig(f"results/visualisations/{model_name}_loss_curve.png")
        mp.show()
import matplotlib.pyplot as mp
import seaborn as sb
from metrics.evaluation import evaluate_model
from sklearn.metrics import roc_curve, auc, confusion_matrix

class Visualizer():

    def plot_confusion_matrix(self, y_pred, y_true, model_name):
        """
        Plot the confusion matrix for the given predictions and true labels.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print("Confusion Matrix (Actual vs Predicted):")
        print("               Pred 0    Pred 1")
        print(f"Actual 0     {cm[0][0]:>8} {cm[0][1]:>8}")
        print(f"Actual 1     {cm[1][0]:>8} {cm[1][1]:>8}")

        return cm

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
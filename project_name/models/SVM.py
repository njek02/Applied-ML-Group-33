from sklearn.svm import SVC
import numpy as np

class SVCModel():
    def __init__(self) -> None:
        """
        Initializes an SVC model with a radial kernel of a specific degree.
        """        
        self.svm = SVC(kernel="rbf")

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits the SVC model to the training data.

        Args:
            x_train (np.ndarray): Training features
            y_train (np.ndarray): Target labels

        """        
        self.svm.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the given input data.
        Args:
            x (np.ndarray): Input features

        Returns:
            np.ndarray: The predicted class labels for each input data point.
        """             
        return self.svm.predict(x)

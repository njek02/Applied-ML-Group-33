from sklearn.svm import SVC
import numpy as np

class SVCModel():
    def __init__(self):
        """
        Initializes an SVC model with a polynomial kernel of a specific degree.
        """        
        self.svm = SVC(kernel="poly", degree = 2)

    def fit(self, x_train, y_train) -> None:
        """
        Fits the SVC model to the training data.

        Args:
            x_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Target labels

        Returns:
            SVC: Trained SVC model
        """        
        self.svm.fit(x_train, y_train)

    def predict(self, x) -> np.ndarray:
        """
        Predicts the class labels for the given input data.


        Args:
            x (numpy.ndarray): Predicted class labels for each input sample
        """        
        return self.svm.predict(x)

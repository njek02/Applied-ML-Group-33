from sklearn.svm import SVC


class SVCModel():
    def __init__(self):
        self.svm = SVC(kernel="poly", degree = 2)

    def fit(self, x_train, y_train):
        return self.svm.fit(x_train, y_train)

    def predict(self, x):
        self.svm.predict(x)

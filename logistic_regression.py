import numpy as np

class BinaryLogisticRegressionBase:
    def __init__(self, eta, iterations=20):
        self.eta = eta
        self.iters = iterations
    
    def __str__(self):
        return "Base Binary Logistic Regression Object, NOT TRAINABLE"
    
    @staticmethod
    def _add_bias(X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
    
    @staticmethod
    def _sigmoid(theta):
        return 1 / (1 + np.exp(-theta))
    
    @staticmethod
    def _ReLu(theta):
        return np.maximum(theta, 0)
    
    def predict_proba(self, X, add_bias=True):
        Xb = self._add_bias(X) if add_bias else X
        return self._sigmoid(Xb @ self.w_)
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5)

class BinaryLogisticRegression(BinaryLogisticRegressionBase):
    def __str__(self):
        if(hasattr(self, 'w_')):
            return "binary logistic regression object with coefficients:\n" + str(self.w_)
        else:
            return "untrained binary logistic regression object"
    
    def _get_gradient(self, X, y):
        gradient = np.zeros(self.w_.shape)
        for (xi, yi) in zip(X, y):
            gradi = (yi - self.predict_proba(xi, add_bias=False)) * xi
            gradient += gradi.reshape(self.w_.shape)
        return gradient/float(len(y))

    def fit(self, X, y):
        Xb = self._add_bias(X)
        num_samples, num_features = Xb.shape
        self.w_ = np.zeros((num_features, 1))

        for _ in range(self.iters):
            gradient = self._get_gradient(Xb, y)
            self.w_  += gradient * self.eta

# # TESTING CODE
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import numpy as np
# import plotly

# ds = load_iris()
# X = ds.data
# y = (ds.target > 1).astype(np.int)

# blr = BinaryLogisticRegression(eta=0.1, iterations=12)
# blr.fit(X, y)
# yhat = blr.predict(X)
# yhat = yhat.reshape(yhat.shape[0],)
# print("accuracy score: ", accuracy_score(y, yhat))

# lr = LogisticRegression()
# lr.fit(X, y)
# yhat_lr = lr.predict(X)
# print("accuracy score: ", accuracy_score(y, yhat_lr))
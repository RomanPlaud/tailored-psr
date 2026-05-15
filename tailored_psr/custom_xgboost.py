import numpy as np
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import custom_mapping, custom_mapping_derivative

class CustomXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self.model = None

    def _objective(self, y_true, y_pred):
        z = y_pred
        y = y_true
        
        p = custom_mapping(z)
        grad = p - y
        hess = custom_mapping_derivative(z)
        
        return grad, hess

    def fit(self, X, y):
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self._objective,
            eval_metric=None, 
            **self.kwargs
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        z = self.model.predict(X, output_margin=True)
        p = custom_mapping(z)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
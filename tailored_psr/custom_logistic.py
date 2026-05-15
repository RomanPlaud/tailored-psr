import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import custom_mapping, custom_loss

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1e15, fit_intercept=True, max_iter=1000, tol=1e-6):
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def _loss_and_grad(self, w, X, y):
        # Linear Predictor
        z = X @ w
        p = custom_mapping(z)
        
        # Loss
        loss = np.sum(custom_loss(p, y))
        
        # Gradient
        grad = X.T @ (p - y)
        
        # Regularization
        reg_strength = 1.0 / self.C if self.C > 0 else 0.0
        
        if self.fit_intercept:
            w_reg = w[:-1]
            loss += 0.5 * reg_strength * np.sum(w_reg**2)
            grad[:-1] += reg_strength * w_reg
        else:
            loss += 0.5 * reg_strength * np.sum(w**2)
            grad += reg_strength * w
            
        return loss, grad

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X_fit = np.hstack([X, np.ones((n_samples, 1))])
            w0 = np.zeros(n_features + 1)
        else:
            X_fit = X
            w0 = np.zeros(n_features)
            
        res = minimize(
            fun=self._loss_and_grad,
            x0=w0,
            args=(X_fit, y),
            method='L-BFGS-B',
            jac=True,
            tol=self.tol,
            options={'maxiter': self.max_iter}
        )
        
        if self.fit_intercept:
            self.coef_ = res.x[:-1]
            self.intercept_ = res.x[-1]
        else:
            self.coef_ = res.x
            self.intercept_ = 0.0
            
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        z = X @ self.coef_ + self.intercept_
        p = custom_mapping(z)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from .utils import custom_mapping_torch, custom_loss_torch

class CustomCanonicalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, y):
        p = custom_mapping_torch(z)
        loss = custom_loss_torch(p, y)
        ctx.save_for_backward(p, y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        p, y = ctx.saved_tensors
        # Gradient dL/dz = (p - y) / N
        N = p.shape[0]
        grad_z = (p - y) / N
        return grad_z * grad_output, None

def custom_canonical_loss(z, y):
    return CustomCanonicalLoss.apply(z, y)


class CustomMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_sizes=(64, 32), lr=0.001, max_iter=200, batch_size=2048, device='cpu'):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.device = device
        self.model_ = None
        self.scaler_ = None

    def _build_model(self, input_dim):
        layers = []
        in_dim = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_ = self._build_model(X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        
        self.model_.train()
        for epoch in range(self.max_iter):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                z = self.model_(batch_X).squeeze()
                loss = custom_canonical_loss(z, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            z = self.model_(X_tensor).squeeze()
            p = custom_mapping_torch(z).cpu().numpy()
            
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

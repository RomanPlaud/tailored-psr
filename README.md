# 🎯 Tailoring Strictly Proper Scoring Rules for Downstream Tasks

*An Application to Causal Inference*

**Accepted to ICML 2026** 🏆

👤 **Authors:** Roman Plaud*, Alexandre Perez-Lebel*, Antoine Saillenfest, Thomas Bonald, Marine Le Morvan, Gaël Varoquaux, Matthieu Labeau

---

### 💡 What is this?

This repository contains the Python implementation of the propensity score estimators introduced in our paper. 

Standard log-loss training ignores downstream causal tasks, leading to bias and variance explosions in Inverse Probability Weighting (IPW) estimators. To fix this, we derive a **task-specific proper scoring rule** and its associated **canonical link function (quartic link)** to minimize the theoretical upper bound on the IPW Mean Squared Error (MSE). 

The result? Stable, robust, and mathematically sound optimization that heavily penalizes errors in extreme probability regions! 🚀

### 🛠️ Quick Start

We provide plug-and-play estimators for three major paradigms (SciPy, PyTorch, and XGBoost). 

```python
from src_supp.custom_logistic import CustomLogisticRegression
from src_supp.custom_mlp import CustomMLP
from src_supp.custom_xgboost import CustomXGBoost

# 📈 1. Logistic Regression (SciPy L-BFGS-B)
model_lr = CustomLogisticRegression()
model_lr.fit(X_train, y_train)
probs_lr = model_lr.predict_proba(X_test)

# 🧠 2. Multi-Layer Perceptron (PyTorch Autograd)
model_mlp = CustomMLP(hidden_sizes=(64, 32), device='cpu')
model_mlp.fit(X_train, y_train)
probs_mlp = model_mlp.predict_proba(X_test)

# 🌳 3. Gradient Boosting (XGBoost Native)
model_xgb = CustomXGBoost(n_estimators=100)
model_xgb.fit(X_train, y_train)
probs_xgb = model_xgb.predict_proba(X_test)

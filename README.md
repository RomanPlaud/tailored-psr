# Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference

This directory contains the implementation of the propensity score estimators described in the paper:
**"Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference"**
*Roman Plaud^\**, Alexandre Perez-Lebel^\**, Antoine Saillenfest\*, Thomas Bonald\*, Marine Le Morvan\*, Gaël Varoquaux\*, Matthieu Labeau^\**

**(Accepted to ICML 2026)**

These models use the proposed custom **Canonical Link Function** (Quartic Link) and its associated **Proper Scoring Rule** to minimize the theoretical upper bound on the Mean Squared Error of IPW estimators.

## Mathematical Foundation

We utilize a custom link function $g(\cdot)$ such that the link is **canonical** with respect to the loss function $\mathcal{L}$. This property simplifies the gradient of the loss with respect to the linear predictor $z$ (logits) to:
$$
\frac{\partial \mathcal{L}}{\partial z} = p - y
$$
where $p = g^{-1}(z)$ is the predicted probability and $y \in \{0, 1\}$ is the observed outcome.

This property is leveraged across all three implementations to avoid complex differentiation through the quartic root-finding algorithm used in $g^{-1}(z)$.

## Components

### 1. `utils.py`
Contains the core mathematical kernels:
* **`custom_mapping(z)`**: Computes $p = g^{-1}(z)$ by solving the quartic equation associated with the link.
* **`custom_loss(p, y)`**: Computes the proper scoring rule loss.
* **`custom_mapping_derivative(z)`**: Analytical derivative of the mapping (Hessian).
* **PyTorch Adapters**: `custom_mapping_torch` and `custom_loss_torch` for tensor operations.
* **Numerical Stability**: Implements safe internal upcasting to `float64` for polynomial root-finding and dynamic boundary clamping to prevent catastrophic cancellation and division-by-zero during tensor evaluations.

### 2. `custom_logistic.py`
**`CustomLogisticRegression`**
* Optimizes the custom loss using `scipy.optimize.minimize` (L-BFGS-B).
* **Gradient**: Explicitly computed as $X^T (p - y)$, enabling efficient and stable convergence without automatic differentiation.

### 3. `custom_mlp.py`
**`CustomMLP`**
* A PyTorch-based Multi-Layer Perceptron.
* **Custom Autograd Function**: `CustomCanonicalLoss` implements a manual backward pass.
    * **Forward**: Computes loss normally.
    * **Backward**: Returns the gradient $(p - y) / N$ directly to the logits $z$ (this assumes the loss is reduced via the mean).
    * **Benefit**: Skips backpropagation through the mathematically expensive and potentially unstable quartic root-finding operations in the forward pass mapping.

### 4. `custom_xgboost.py`
**`CustomXGBoost`**
* Wrapper around `XGBClassifier`.
* **Custom Objective**: Supplies the gradient ($p - y$) and Hessian ($1 / w_\ell(p)$) directly to XGBoost, allowing it to optimize the non-standard loss function efficiently using second-order approximation.

## Installation

You can install this package locally using `pip`:
```bash
pip install -e .
```
This will automatically install necessary dependencies (`numpy`, `scipy`, `scikit-learn`, `torch`, `xgboost`).

## Usage Examples

```python
from tailored_psr.custom_logistic import CustomLogisticRegression
from tailored_psr.custom_mlp import CustomMLP
from tailored_psr.custom_xgboost import CustomXGBoost

# Logistic Regression
model_lr = CustomLogisticRegression()
model_lr.fit(X_train, y_train)
probs = model_lr.predict_proba(X_test)

# MLP
model_mlp = CustomMLP(hidden_sizes=(64, 32), device='cpu')
model_mlp.fit(X_train, y_train)
probs = model_mlp.predict_proba(X_test)

# XGBoost
model_xgb = CustomXGBoost(n_estimators=100)
model_xgb.fit(X_train, y_train)
probs = model_xgb.predict_proba(X_test)
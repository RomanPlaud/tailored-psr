import numpy as np
import torch
import warnings

_EPS = 1e-12

# ==============================================================================
# NUMPY IMPLEMENTATIONS
# ==============================================================================

def custom_mapping(z):
    z_arr = np.asarray(z, dtype=np.float64)
    theta = z_arr
    theta2 = theta**2

    A = theta2 * 0.25 - 3.0
    B = theta2 * 0.5 - 2.0

    half_B = B * 0.5
    third_A = A / 3.0
    delta = half_B**2 + third_A**3

    v = np.zeros_like(theta)
    
    mask_pos = delta > 1e-12
    if np.any(mask_pos):
        sqrt_delta = np.sqrt(delta[mask_pos])
        hb = half_B[mask_pos]
        term1 = -hb + sqrt_delta
        term2 = -hb - sqrt_delta
        
        cbrt1 = np.sign(term1) * np.abs(term1)**(1.0/3.0)
        cbrt2 = np.sign(term2) * np.abs(term2)**(1.0/3.0)
        v[mask_pos] = cbrt1 + cbrt2

    mask_neg = ~mask_pos
    if np.any(mask_neg):
        neg_third_A = -third_A[mask_neg]
        k = np.sqrt(np.maximum(neg_third_A, 1e-12))
        arg = -half_B[mask_neg] / (k**3)
        arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)
        phi = np.arccos(arg)
        v[mask_neg] = 2.0 * k * np.cos(phi / 3.0)

    z_aux = 4.0 * (v + 2.0)
    z_aux = np.maximum(z_aux, 1e-12) 
    
    sqrt_z = np.sqrt(z_aux)
    inner_term = 24.0 - z_aux + 32.0/sqrt_z
    u = 0.5 * (sqrt_z + np.sqrt(np.maximum(inner_term, 0.0)))
    
    sign = np.sign(theta)
    term_p = np.maximum(1.0 - 4.0/u, 0.0)
    p = 0.5 * (1.0 + sign * np.sqrt(term_p))
    
    return p

def custom_loss(q, y):
    q = np.clip(q, _EPS, 1.0 - _EPS)
    entropy_term = 2.0 * np.log(q * (1.0 - q))
    
    loss_y1 = (1.0 / q**2) - (2.0 / (1.0 - q)) + entropy_term
    loss_y0 = (1.0 / (1.0 - q)**2) - (2.0 / q) + entropy_term
    
    return np.where(y == 1, loss_y1, loss_y0)

def custom_mapping_derivative(z):
    q = custom_mapping(z)
    q = np.clip(q, _EPS, 1.0 - _EPS)
    hessian = (2.0/q**2) + (2.0/(1.0-q)**2) + (2.0/q**3) + (2.0/(1.0-q)**3)
    return 1.0 / hessian

# ==============================================================================
# PYTORCH IMPLEMENTATIONS
# ==============================================================================

class ExactCanonicalInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta):
        # Cast to float64 for stable root finding
        theta_f64 = theta.to(torch.float64)
        theta2 = theta_f64**2
        
        A = theta2 * 0.25 - 3.0
        B = theta2 * 0.5 - 2.0
        
        half_B = B * 0.5
        third_A = A / 3.0
        delta = half_B**2 + third_A**3
        
        v = torch.zeros_like(theta_f64)
        
        mask_pos = delta > 1e-12
        if mask_pos.any():
            sqrt_delta = torch.sqrt(delta[mask_pos])
            hb = half_B[mask_pos]
            term1 = -hb + sqrt_delta
            term2 = -hb - sqrt_delta
            cbrt1 = torch.sign(term1) * torch.pow(torch.abs(term1), 1.0/3.0)
            cbrt2 = torch.sign(term2) * torch.pow(torch.abs(term2), 1.0/3.0)
            v[mask_pos] = cbrt1 + cbrt2

        mask_neg = ~mask_pos
        if mask_neg.any():
            neg_third_A = -third_A[mask_neg]
            k = torch.sqrt(torch.clamp(neg_third_A, min=1e-12))
            arg = -half_B[mask_neg] / (k**3)
            arg = torch.clamp(arg, -1.0 + 1e-12, 1.0 - 1e-12)
            phi = torch.acos(arg)
            v[mask_neg] = 2.0 * k * torch.cos(phi / 3.0)

        z = 4.0 * (v + 2.0)
        z = torch.clamp(z, min=1e-12)
        sqrt_z = torch.sqrt(z)
        inner_term = 24.0 - z + 32.0/sqrt_z
        u = 0.5 * (sqrt_z + torch.sqrt(torch.clamp(inner_term, min=0.0)))
        
        sign = torch.sign(theta_f64)
        term_p = torch.clamp(1.0 - 4.0/u, min=0.0)
        p = 0.5 * (1.0 + sign * torch.sqrt(term_p))
        
        # Cast back to original dtype before returning
        p = p.to(theta.dtype)
        
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        p, = ctx.saved_tensors
        # Dynamically set eps based on dtype to avoid float32 rounding to 1.0
        eps = 1e-6 if p.dtype == torch.float32 else 1e-12
        p = torch.clamp(p, min=eps, max=1.0 - eps)
        
        hessian = 2.0/(p**2) + 2.0/((1.0-p)**2) + 2.0/(p**3) + 2.0/((1.0-p)**3)
        grad_theta = 1.0 / hessian
        return grad_output * grad_theta

def custom_mapping_torch(z):
    return ExactCanonicalInverse.apply(z)

def custom_loss_torch(p, target):
    eps = 1e-6 if p.dtype == torch.float32 else 1e-12
    p = torch.clamp(p, min=eps, max=1.0 - eps)
    entropy_term = 2.0 * torch.log(p * (1.0 - p))
    loss_y1 = (1.0 / (p**2)) - (2.0 / (1.0 - p)) + entropy_term
    loss_y0 = (1.0 / ((1.0 - p)**2)) - (2.0 / p) + entropy_term
    loss = torch.where(target == 1, loss_y1, loss_y0)
    # NOTE: If this is changed to .sum() or reduction='none', the
    # CustomCanonicalLoss.backward() pass dividing by N must be updated accordingly!
    return loss.mean()
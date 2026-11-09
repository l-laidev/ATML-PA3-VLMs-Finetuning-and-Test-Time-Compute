import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

def compute_weight_space_diagnostics(W_0: torch.Tensor, W_t: torch.Tensor, U_k: torch.Tensor) -> Dict[str, float]:
    W_0_f = W_0.float()
    W_t_f = W_t.float()
    Delta_W = W_t_f - W_0_f
    
    norm_delta_w = torch.linalg.norm(Delta_W, ord="fro").item()
    norm_w0 = torch.linalg.norm(W_0_f, ord="fro").item()
    eta = norm_delta_w / (norm_w0 + 1e-8)
    
    projected_delta = U_k @ (U_k.T @ Delta_W)
    norm_projected = torch.linalg.norm(projected_delta, ord="fro").item()
    rho_k = (norm_projected ** 2) / ((norm_delta_w ** 2) + 1e-12)
    
    return {"eta": eta, "rho_k": rho_k, "norm_delta_w": norm_delta_w}

def compute_model_displacement(initial_params: Dict[str, torch.Tensor], current_model: nn.Module) -> float:
    total_sq_diff = 0.0
    for name, param in current_model.named_parameters():
        if name in initial_params:
            theta_0 = initial_params[name].to(param.device).float()
            diff = param.float() - theta_0
            total_sq_diff += torch.sum(diff ** 2).item()
    return np.sqrt(total_sq_diff)

def compute_relative_activation_change(z, W_0, W_t):
    num = torch.linalg.norm(W_t @ z.T, ord=2).item()
    den = torch.linalg.norm(W_0 @ z.T, ord=2).item()
    r_z = num / (den + 1e-8)
    return r_z
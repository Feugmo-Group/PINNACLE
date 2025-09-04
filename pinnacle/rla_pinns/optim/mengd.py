from math import sqrt
from torch import Tensor, zeros_like, cat
from typing import List, Dict, Tuple
from argparse import ArgumentParser, Namespace
from torch.nn import Module
from torch.optim import Optimizer
from torch.autograd import grad as grad
import torch 
from rla_pinns.optim.line_search import grid_line_search

class MENGD(Optimizer):
    def __init__(self, params, lr=1e-3, damping=1):
            defaults = dict(lr=lr, damping=damping)
            super().__init__(params, defaults)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def step(self, closure=None):
        " closure() should return residuals tensor of shape [N_residuals,] " 
        
        group = self.param_groups[0]
        params = group["params"]
        damping = group["damping"]

        residuals = closure()

        J = self._compute_jacobian(residuals, params) # J is of shape [N_residuals, N_params]

        # Apply Woodbury identity
        # Solve (JJ^T + λI)x = r 
        kernel = J @ J.T  # [N_residuals, N_residuals]
        regularized_kernel = kernel + damping * torch.eye(kernel.shape[0], device=kernel.device)

        # Solve for x in (JJ^T + λI)x = r
        L = torch.linalg.cholesky(regularized_kernel)
        x = torch.cholesky_solve(residuals.unsqueeze(-1), L).squeeze(-1)

        natural_gradient = (1.0 / damping) * J.T @ x #shape [N_params,]

        directions = natural_gradient.split([p.numel() for p in params])
        directions = [d.view(p.shape) for d, p in zip(directions, params)]
        self._update_parameters(directions, closure)

    def _compute_jacobian(self, residuals, params):
        " Compute the Jacobian matrix of residuals with respect to parameters"

        output = residuals.reshape(-1)
        grads = grad(
                output,
                list(params),
                (torch.eye(output.shape[0]).to(self.device),),
                is_grads_batched=True, retain_graph=True,allow_unused=True
        )

        valid_grads = [grad.flatten().reshape(len(output), -1) 
                for grad in grads if grad is not None]
            
        return torch.cat(valid_grads, 1)
    
    def _validate_jacobian(self, residuals, params, J):
        """Quick sanity check for Jacobian computation"""
        expected_shape = (len(residuals), sum(p.numel() for p in params))
        actual_shape = J.shape
        if expected_shape != actual_shape:
            raise ValueError(f"Jacobian shape mismatch: expected {expected_shape}, got {actual_shape}")
        print(f"✅ Jacobian validated: {actual_shape}")
    
    def _update_parameters(
            self,
            directions,
            closure,
        ) -> None:
            (group,) = self.param_groups
            lr = group["lr"]
            params = group["params"]

            if isinstance(lr, float):
                for p, d in zip(params, directions): #directions is update directions and p is each parameter. So we update p by adding d scaled by lr
                    p.data.add_(d, alpha=-lr)

            else:
                if lr[0] == "grid_line_search":
                    # need a closure that returns the loss here, so this is incorrect for now
                    def f():
                        residuals = closure() 
                        return torch.mean(residuals**2)

                    grid = lr[1]
                    grid_line_search(f, params, directions, grid)

                else:
                    raise ValueError(f"Unsupported line search: {lr[0]}.")
         



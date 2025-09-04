from torch import Tensor, cholesky_solve, einsum, arange, randn, diag
from typing import List, Dict, Callable, Tuple
from hessianfree.cg import cg
from functools import partial
from torch.linalg import qr, cholesky, solve_triangular, svd
from rla_pinns.optim.utils import (
    apply_joint_JJT,
    compute_joint_JJT,
)


# NOTE: create test that increases the number of iterations to ascertain we recover the exact solve
def pcg_nystrom(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    g: Tensor,
    dt: str,
    dev: str,
    damping: float,
    l: int,
):
    N = len(g)
    fn = partial(
        apply_joint_JJT,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    )

    B = nystrom_stable_fast(fn, N, l, dt, dev)

    precond = partial(apply_B, B, damping)
    x_iters, m_iters, reason = cg(A=fn, b=g, M=precond, max_iter=5)
    step = x_iters[-1].unsqueeze(-1)

    return step


def sketch_and_project(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    g: Tensor,
    dt: str,
    dev: str,
    damping: float,
    l: int,
):
    N = len(g)
    Omega = randn(l, N, device=dev, dtype=dt)
    JJTOT = apply_joint_JJT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
        Omega.T,
    )

    Og = einsum("i j, j -> i", Omega, g)

    JJT_h = Omega @ JJTOT
    idx = arange(l, device=dev)
    JJT_h[idx, idx] = JJT_h.diag() + damping
    L = cholesky(JJT_h)

    invg = cholesky_solve(Og.unsqueeze(-1), L).squeeze(-1)
    step = Omega.T @ invg
    return step.unsqueeze(-1)


# NOTE: create tests for these function
def apply_inv_sketch(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    g: Tensor,
    dt: str,
    dev: str,
    damping: float,
    l: int,
):
    """Apply the inverse of the GPU-efficient Nyström approximation to g."""
    fn = partial(
        apply_joint_JJT,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    )
    B = nystrom_stable_fast(fn, len(g), l, dt, dev)
    out = apply_B(B, damping, g)
    return out.unsqueeze(-1)

# NOTE: create tests for these function
def apply_inv_sketch_naive(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    g: Tensor,
    dt: str,
    dev: str,
    damping: float,
    l: int,
):
    """Apply the inverse of the stable Nyström approximation to g."""
    fn = partial(
        apply_joint_JJT,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    )
    U, Lambda = nystrom_stable(fn, len(g), l, dt, dev)
    
    UTg = U.T @ g
    rhs = (g - U @ UTg) / damping
    lhs = U @ (diag(1 / (Lambda + damping)) @ UTg)

    return (rhs + lhs).unsqueeze(-1)


# NOTE: create tests for these function
def apply_inv_exact(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    g: Tensor,
    dt: str,
    dev: str,
    damping: float,
    l: int,
):
    """Apply the inverse of the exact joint JJT to g."""
    JJT = compute_joint_JJT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ).detach()

    idx = arange(JJT.shape[0], device=dev)
    JJT[idx, idx] = JJT.diag() + damping


    L = cholesky(JJT)

    out = cholesky_solve(g.unsqueeze(1), L)

    return out


def apply_B(B: Tensor, damping: float, g: Tensor) -> Tensor:
    """Apply the inverse of B to g."""
    BTB = B.T @ B
    idx = arange(BTB.shape[0], device=B.device)
    BTB[idx, idx] = BTB.diag() + damping

    L = cholesky(BTB)
    BTg = B.T @ g

    invBTg = cholesky_solve(BTg.unsqueeze(-1), L).squeeze(-1)
    P_inv = B @ invBTg
    out = (g - P_inv) / damping
    return out


def nystrom_naive(
    apply_A: Callable[[Tensor], Tensor], dim: int, sketch_size: int, dt: str, dev: str
) -> Tensor:
    """Compute a naive Nyström approximation."""

    Omega = randn(dim, sketch_size, dtype=dt, device=dev)
    AO = apply_A(Omega)
    OAO = Omega.T @ AO
    B = cholesky_solve(AO.T, cholesky(OAO))
    A_hat = AO @ B

    return A_hat


def nystrom_stable(
    A: Callable[[Tensor], Tensor], dim: int, sketch_size: int, dt: str, dev: str
) -> Tuple[Tensor, Tensor]:
    """Compute a stable Nyström approximation."""

    O = randn(dim, sketch_size, device=dev, dtype=dt)
    O, _ = qr(O)
    Y = A(O).detach()
    nu = 1e-7
    Y.add_(O, alpha=nu)
    C = cholesky(O.T @ Y, upper=True)
    B = solve_triangular(C, Y, upper=True, left=False)
    U, Sigma, _ = svd(B, full_matrices=False)
    Lambda = (Sigma**2 - nu).clamp(min=0.0)
    return U, Lambda


def nystrom_stable_fast(
    A: Callable[[Tensor], Tensor], dim: int, sketch_size: int, dt: str, dev: str
) -> Tuple[Tensor, Tensor]:
    """Parital Nyström approximation that avoids the last SVD as it's highly inefficient in GPU"""

    O = randn(dim, sketch_size, device=dev, dtype=dt)
    Y = A(O).detach()
    nu = 1e-7
    Y.add_(O, alpha=nu)
    C = cholesky(O.T @ Y, upper=True)
    B = solve_triangular(C, Y, upper=True, left=False)

    return B

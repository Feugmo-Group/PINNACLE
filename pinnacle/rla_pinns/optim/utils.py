from math import sqrt
from typing import Dict, List, Tuple
from torch import Tensor, cat, zeros, cat
from einops import einsum
from torch.nn import Module
from rla_pinns import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from rla_pinns.pinn_utils import (
    evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
)
import torch


INTERIOR_LOSS_EVALUATORS = {
    "poisson": poisson_equation.evaluate_interior_loss,
    "heat": heat_equation.evaluate_interior_loss,
    "fokker-planck-isotropic": fokker_planck_isotropic_equation.evaluate_interior_loss,
    "log-fokker-planck-isotropic": log_fokker_planck_isotropic_equation.evaluate_interior_loss,  # noqa: B950
}

EVAL_FNS = {
    "poisson": {
        "interior": poisson_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
    "heat": {
        "interior": heat_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
    "fokker-planck-isotropic": {
        "interior": fokker_planck_isotropic_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
    "log-fokker-planck-isotropic": {
        "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
}


def evaluate_losses_with_layer_inputs_and_grad_outputs(
    layers: List[Module],
    X_Omega: Tensor,
    y_Omega: Tensor,
    X_dOmega: Tensor,
    y_dOmega: Tensor,
    equation: str,
    ggn_type: str = "type-2",
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
]:
    """Evaluate interior and boundary losses/residuals & layer inputs/grad outputs.

    Args:
        layers: The layers that form the neural network.
        X_Omega: The input data for the interior loss.
        y_Omega: The target data for the interior loss.
        X_dOmega: The input data for the boundary loss.
        y_dOmega: The target data for the boundary loss.
        ggn_type: The GGN type.
        equation: The PDE to solve.

    Returns:
        The differentiable interior loss, differentiable boundary loss, differentiable
        interior residual, differentiable boundary residual,
        layer inputs of the interior loss, layer gradient outputs of the interior loss,
        layer inputs of the boundary loss, layer gradient outputs of the boundary loss.
    """
    assert ggn_type == "type-2"
    interior_evaluator = EVAL_FNS[equation]["interior"]
    boundary_evaluator = EVAL_FNS[equation]["boundary"]

    interior_loss, interior_res, interior_inputs, interior_grad_outputs = (
        interior_evaluator(layers, X_Omega, y_Omega, ggn_type)
    )
    boundary_loss, boundary_res, boundary_inputs, boundary_grad_outputs = (
        boundary_evaluator(layers, X_dOmega, y_dOmega, ggn_type)
    )

    return (
        interior_loss,
        boundary_loss,
        interior_res,
        boundary_res,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    )


def compute_individual_JJT(
    inputs: Dict[int, Tensor], grad_outputs: Dict[int, Tensor]
) -> Tensor:
    """Compute the Jacobian outer product of an individual (boundary/interior) residual.

    Args:
        inputs: The layer inputs for the interior or boundary loss.
        grad_outputs: The layer gradient outputs for the interior or boundary loss.

    Returns:
        The Jacobian outer product. Has shape `(N, N)` where `N` is the batch size used
        for the boundary/interior loss.
    """
    (N,) = {t.shape[0] for t in list(inputs.values()) + list(grad_outputs.values())}
    ((dev, dt),) = {
        (t.device, t.dtype) for t in list(inputs.values()) + list(grad_outputs.values())
    }
    JJT = zeros((N, N), device=dev, dtype=dt)

    for idx in inputs:
        J = einsum(
            # gradients are scaled by 1/N, but we need 1/√N for the outer product
            grad_outputs[idx] * sqrt(N),
            inputs[idx],
            "n ... d_out, n ... d_in -> n d_out d_in",
        )
        J = J.flatten(start_dim=1)
        JJT.add_(J @ J.T)

    return JJT


def compute_joint_JJT(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
) -> Tensor:
    """Compute the Jacobian outer product of the joint residual.

    Args:
        interior_inputs: The layer inputs for the interior loss.
        interior_grad_outputs: The layer gradient outputs for the interior loss.
        boundary_inputs: The layer inputs for the boundary loss.
        boundary_grad_outputs: The layer gradient outputs for the boundary loss.

    Returns:
        The Jacobian outer product. Has shape `(N, N)` where `N = N_Omega + N_dOmega`
        is the sum of the interior and boundary loss batch sizes.
    """
    (N_Omega,) = {
        t.shape[0]
        for t in list(interior_inputs.values()) + list(interior_grad_outputs.values())
    }
    (N_dOmega,) = {
        t.shape[0]
        for t in list(boundary_inputs.values()) + list(boundary_grad_outputs.values())
    }
    ((dev, dt),) = {
        (t.device, t.dtype)
        for t in list(interior_inputs.values())
        + list(interior_grad_outputs.values())
        + list(boundary_inputs.values())
        + list(boundary_grad_outputs.values())
    }

    JJT = zeros((N_Omega + N_dOmega, N_Omega + N_dOmega), device=dev, dtype=dt)

    for idx in interior_inputs:

        J_boundary = einsum(
            # gradients are scaled by 1/N, but we need 1/√N for the outer product
            boundary_grad_outputs[idx] * sqrt(N_dOmega),
            boundary_inputs[idx],
            "n d_out, n d_in -> n d_out d_in",
        )
        J_interior = einsum(
            # gradients are scaled by 1/N, but we need 1/√N for the outer product
            interior_grad_outputs[idx] * sqrt(N_Omega),
            interior_inputs[idx],
            "n ... d_out, n ... d_in -> n d_out d_in",
        )
        J = cat([J_interior, J_boundary], dim=0).flatten(start_dim=1)

        JJT.add_(J @ J.T)

    return JJT


def apply_joint_JJT(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    M: Tensor,
) -> Tensor:
    """Multiply the kernel onto a matrix in data space.

    Considers both the interior and the boundary loss.

    Args:
        interior_inputs: The layer inputs for the interior loss.
        interior_grad_outputs: The layer gradient outputs for the interior loss.
        boundary_inputs: The layer inputs for the boundary loss.
        boundary_grad_outputs: The layer gradient outputs for the boundary loss.
        M: The matrix to multiply the kernel with. Has shape
        `(N_Omega + N_dOmega, K)` where `N_Omega` is the batch size for the
        evaluation of the interior loss, `N_dOmega` is the batch size for the
        evaluation of the boundary loss and `K` is the number of columns.

    Returns:
        The result of multiplying the kernel with the matrix. Has the same shape as M
    """
    (N_Omega,) = {
        t.shape[0]
        for t in list(interior_inputs.values()) + list(interior_grad_outputs.values())
    }
    (N_dOmega,) = {
        t.shape[0]
        for t in list(boundary_inputs.values()) + list(boundary_grad_outputs.values())
    }

    M_out = torch.zeros_like(M)

    for idx in interior_inputs:

        J_boundary = einsum(
            # gradients are scaled by 1/N, but we need 1/√N for the outer product
            boundary_grad_outputs[idx] * sqrt(N_dOmega),
            boundary_inputs[idx],
            "n d_out, n d_in -> n d_out d_in",
        )
        J_interior = einsum(
            # gradients are scaled by 1/N, but we need 1/√N for the outer product
            interior_grad_outputs[idx] * sqrt(N_Omega),
            interior_inputs[idx],
            "n ... d_out, n ... d_in -> n d_out d_in",
        )
        J = cat([J_interior, J_boundary], dim=0).flatten(start_dim=1)

        M_out.add_(J @ (J.T @ M))

    return M_out


def apply_joint_J(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    M: List[Tensor],
) -> Tensor:
    """Multiply the Jacobian onto a matrix in parameter space.

    Considers both the interior and the boundary loss.

    Args:
        interior_inputs: The layer inputs for the interior loss.
        interior_grad_outputs: The layer gradient outputs for the interior loss.
        boundary_inputs: The layer inputs for the boundary loss.
        boundary_grad_outputs: The layer gradient outputs for the boundary loss.
        M: The matrix to multiply the Jacobian with. Is a list where each entry
        has the same leading shape as a parameter and the trailing dimension is K.

    Returns:
        The result of multiplying the Jacobian with the matrix. Has shape `(N_Omega +
        N_dOmega,K)` where `N_Omega` and `N_dOmega` are the interior and boundary loss
        batch sizes.
    """
    J_interior_M = _apply_individual_J(interior_inputs, interior_grad_outputs, M)
    J_boundary_M = _apply_individual_J(boundary_inputs, boundary_grad_outputs, M)
    return cat([J_interior_M, J_boundary_M])


def _apply_individual_J(
    inputs: Dict[int, Tensor], grad_outputs: Dict[int, Tensor], M: List[Tensor]
) -> Tensor:
    """Multiply the Jacobian onto a matrix in parameter space (tensor list format).

    Considers only a single loss, i.e. either the interior or the boundary loss.

    Args:
        inputs: A dictionary containing the inputs to layers with parameters.
        grad_outputs: A dictionary containing the gradient outputs of layers with
            parameters.
        M: The matrix to multiply the Jacobian with. It's a list where each entry
        is a tensor whose leading dimmensions match the parameter and the
        trailing dimension is K.

    Returns:
        The result of multiplying the Jacobian with the matrix. Has shape `(N,K)` where
        `N` is the batch size.
    """
    assert 2 * len(inputs) == 2 * len(grad_outputs) == len(M)

    ((N, dev, dt),) = {
        (t.shape[0], t.device, t.dtype)
        for t in list(inputs.values()) + list(grad_outputs.values())
    }

    (K,) = {t.shape[-1] for t in M}

    JM = zeros((N, K), device=dev, dtype=dt)

    for idx, layer_idx in enumerate(inputs):
        M_weight, M_bias = M[2 * idx], M[2 * idx + 1]
        M_joint = cat([M_weight, M_bias.unsqueeze(-2)], dim=1)

        # NOTE: Doing this in a single einsum is much slower
        temp = einsum(
            grad_outputs[layer_idx],
            M_joint,
            "n ... d_out, d_out d_in k -> n ... d_in k",
        )
        JM.add_(
            einsum(
                temp,
                inputs[layer_idx],
                "n ... d_in k, n ... d_in -> n k ",
            )
        )

    # grad_outputs are scaled by 1/N, but we need 1/√N for the Jacobian
    return JM.mul_(sqrt(N))


def apply_joint_JT(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    M: Tensor,
) -> List[Tensor]:
    """Multiply the transpose Jacobian onto a matrix in data space.

    Considers both the interior and the boundary loss.

    Args:
        interior_inputs: The layer inputs for the interior loss.
        interior_grad_outputs: The layer gradient outputs for the interior loss.
        boundary_inputs: The layer inputs for the boundary loss.
        boundary_grad_outputs: The layer gradient outputs for the boundary loss.
        M: The matrix to multiply the transpose Jacobian with. Has shape
        `(N_Omega + N_dOmega, K)` where `N_Omega` is the batch size for the
        evaluation of the interior loss, `N_dOmega` is the batch size for the
        evaluation of the boundary loss and `K` is the number of columns.

    Returns:
        The result of multiplying the transpose Jacobian with the matrix. Has same
        format as the parameter space, i.e. is a tensor list, but with a trailing
        dimension of size K.
    """
    # split into interior and boundary terms
    (N_Omega,) = {
        t.shape[0]
        for t in list(interior_inputs.values()) + list(interior_grad_outputs.values())
    }
    (N_dOmega,) = {
        t.shape[0]
        for t in list(boundary_inputs.values()) + list(boundary_grad_outputs.values())
    }
    M_interior, M_boundary = M.split([N_Omega, N_dOmega])

    return [
        JTM_interior.add_(JTM_boundary)
        for JTM_interior, JTM_boundary in zip(
            _apply_individual_JT(interior_inputs, interior_grad_outputs, M_interior),
            _apply_individual_JT(boundary_inputs, boundary_grad_outputs, M_boundary),
        )
    ]


def _apply_individual_JT(
    inputs: Dict[int, Tensor], grad_outputs: Dict[int, Tensor], M: Tensor
) -> List[Tensor]:
    """Multiply the transpose Jacobian onto a matrix in data space.

    Considers only a single loss, i.e. either the interior or the boundary loss.

    Args:
        inputs: A dictionary containing the inputs to layers with parameters.
        grad_outputs: A dictionary containing the gradient outputs of layers with
            parameters.
        M: The matrix to multiply the transpose Jacobian with. Has shape `(N, K)`
        where N is the batch size and K is the number of columns.

    Returns:
        The result of multiplying the transpose Jacobian with the matrix. Has same
        format as the parameter space, i.e. is a tensor list, but with a trailing
        dimension of size K.
    """
    assert len(inputs) == len(grad_outputs)

    JTM = []

    for layer_idx in inputs:
        # NOTE: Doing this in a single einsum is much slower
        JTM_joint = einsum(
            grad_outputs[layer_idx],
            M,
            "n ... d_out, n k -> n ... d_out k",
        )
        JTM_joint = einsum(
            JTM_joint,
            inputs[layer_idx],
            "n ... d_out k, n ... d_in -> d_out d_in k",
        )
        JTM_weight, JTM_bias = JTM_joint.split(
            [inputs[layer_idx].shape[-1] - 1, 1], dim=1
        )
        JTM.extend([JTM_weight, JTM_bias.squeeze(1)])

    # grad_outputs are scaled by 1/N, but we need 1/√N for the transposed Jacobian
    (N,) = {t.shape[0] for t in list(inputs.values()) + list(grad_outputs.values())}
    sqrt_N = sqrt(N)

    for M in JTM:
        M.mul_(sqrt_N)

    return JTM

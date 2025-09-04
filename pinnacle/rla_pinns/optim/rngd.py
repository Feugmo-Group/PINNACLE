from math import sqrt
from torch import Tensor, zeros_like, cat
from typing import List, Dict, Tuple
from argparse import ArgumentParser, Namespace
from torch.nn import Module
from torch.optim import Optimizer
from rla_pinns import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from rla_pinns.optim.utils import (
    evaluate_losses_with_layer_inputs_and_grad_outputs,
    apply_joint_J,
    apply_joint_JT,
)
from rla_pinns.optim.rand_utils import (
    pcg_nystrom,
    apply_inv_exact,
    apply_inv_sketch,
    sketch_and_project,
    apply_inv_sketch_naive,
)
from rla_pinns.pinn_utils import evaluate_boundary_loss
from rla_pinns.parse_utils import parse_known_args_and_remove_from_argv
from rla_pinns.optim.line_search import grid_line_search, parse_grid_line_search_args


def parse_randomized_args(verbose: bool = False, prefix="RNGD_") -> Namespace:
    parser = ArgumentParser(
        description="Parse arguments for setting up the randomized optimizer."
    )
    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=RNGD.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}rank_val",
        type=int,
        help="Low-rank approximation parameter.",
        default=0,
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping parameter.",
        default=1e-8,
    )
    parser.add_argument(
        f"--{prefix}approximation",
        type=str,
        choices=["nystrom", "exact", "sap", "pcg", "naive"],
        help="Randomized method to approximate the range of a low-rank matrix.",
        default="exact",
    )
    parser.add_argument(
        f"--{prefix}momentum",
        type=float,
        default=0.0,
        help="Momentum parameter for the optimizer.",
    )

    args = parse_known_args_and_remove_from_argv(parser)

    # overwrite the lr value
    lr = f"{prefix}lr"
    if any(char.isdigit() for char in getattr(args, lr)):
        setattr(args, lr, float(getattr(args, lr)))

    if getattr(args, lr) == "grid_line_search":
        grid = parse_grid_line_search_args(verbose=verbose)
        setattr(args, lr, (getattr(args, lr), grid))

    if verbose:
        print("Parsed arguments for randomized optimizer: ", args)

    return args


class RNGD(Optimizer):

    LOSS_EVALUATORS = {
        "poisson": {
            "interior": poisson_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "heat": {
            "interior": heat_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "fokker-planck-isotropic": {
            "interior": fokker_planck_isotropic_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "log-fokker-planck-isotropic": {
            "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
    }
    SUPPORTED_EQUATIONS = list(LOSS_EVALUATORS.keys())

    STEP_FUNCTIONS = {
        "nystrom": apply_inv_sketch,
        "naive": apply_inv_sketch_naive,
        "exact": apply_inv_exact,
        "sap": sketch_and_project,
        "pcg": pcg_nystrom,
    }

    def __init__(
        self,
        layers: List[Module],
        lr: float = 1e-3,
        equation: str = "poisson",
        damping: float = 0.0,
        approximation: str = "exact",
        rank_val: int = 0,
        momentum: float = 0.0,
        *,
        maximize: bool = False,
    ):

        params = sum((list(layer.parameters()) for layer in layers), [])
        defaults = dict(
            lr=lr,
            damping=damping,
            maximize=maximize,
            momentum=momentum,
            approximation=approximation,
            rank_val=rank_val,
            equation=equation,
            beta=0.0,
            alpha=0.0,
        )
        super().__init__(params, defaults)

        if equation not in self.LOSS_EVALUATORS:
            raise ValueError(
                f"Equation {equation} not supported. "
                f"Supported equations are: {list(self.LOSS_EVALUATORS.keys())}."
            )

        self.equation = equation
        self.layers = layers
        self.layers_idx = [
            idx for idx, layer in enumerate(self.layers) if list(layer.parameters())
        ]
        self.steps = 0

        self.l = rank_val
        self._approximation = approximation

        if approximation == "exact":
            assert (
                rank_val == 0 or rank_val is None
            ), "Rank value is not used with exact approximation. Has to be zero or None."
        elif approximation in ["nystrom", "sap", "pcg", "naive"]:
            assert rank_val > 0, "Rank value must be a positive integer."
        else:
            raise ValueError(f"Randomization method {approximation} not supported.")

        if momentum != 0.0:
            (group,) = self.param_groups
            for p in group["params"]:
                self.state[p]["phi"] = zeros_like(p)


    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        (
            interior_loss,
            boundary_loss,
            interior_residual,
            boundary_residual,
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
            self.layers, X_Omega, y_Omega, X_dOmega, y_dOmega, self.equation
        )

        N_dOmega = X_dOmega.shape[0]
        boundary_residual = boundary_residual.detach() / sqrt(N_dOmega)

        N_Omega = X_Omega.shape[0]
        interior_residual = interior_residual.detach() / sqrt(N_Omega)

        epsilon = -cat([interior_residual, boundary_residual]).flatten()

        directions = self._get_directions(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            epsilon,
        )

        self._update_parameters(directions, X_Omega, y_Omega, X_dOmega, y_dOmega)
        self.steps += 1
        return interior_loss, boundary_loss

    def _update_parameters(
        self,
        directions: Tensor,
        X_Omega: Tensor,
        y_Omega: Tensor,
        X_dOmega: Tensor,
        y_dOmega: Tensor,
    ) -> None:
        (group,) = self.param_groups
        lr = group["lr"]
        params = group["params"]

        if isinstance(lr, float):
            for p, d in zip(params, directions):
                p.data.add_(d, alpha=lr)

        else:
            if lr[0] == "grid_line_search":

                def f() -> Tensor:
                    """Closure to evaluate the loss.

                    Returns:
                        Loss value.
                    """
                    interior_loss = self._eval_loss(X_Omega, y_Omega, "interior")
                    boundary_loss = self._eval_loss(X_dOmega, y_dOmega, "boundary")
                    return interior_loss + boundary_loss

                grid = lr[1]
                grid_line_search(f, params, directions, grid)

            else:
                raise ValueError(f"Unsupported line search: {lr[0]}.")

    def _eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        """Evaluate the loss.

        Args:
            X: Input data.
            y: Target data.
            loss_type: Type of the loss function. Can be `'interior'` or `'boundary'`.

        Returns:
            The differentiable loss.
        """
        loss_evaluator = self.LOSS_EVALUATORS[self.equation][loss_type]
        loss, _, _ = loss_evaluator(self.layers, X, y)
        return loss

    def _get_directions(
        self,
        interior_inputs: Dict[int, Tensor],
        interior_grad_outputs: Dict[int, Tensor],
        boundary_inputs: Dict[int, Tensor],
        boundary_grad_outputs: Dict[int, Tensor],
        residuals: Tensor,
    ):
        (group,) = self.param_groups
        params = group["params"]
        damping = group["damping"]
        momentum = group["momentum"]
        (dev,) = {p.device for p in params}
        (dt,) = {p.dtype for p in params}

        if momentum != 0.0:
            J_phi = apply_joint_J(
                interior_inputs,
                interior_grad_outputs,
                boundary_inputs,
                boundary_grad_outputs,
                [self.state[p]["phi"].unsqueeze(-1) for p in params],
            ).squeeze(-1)

            zeta = residuals - J_phi.mul_(momentum)
        else:
            zeta = residuals

        step = self.STEP_FUNCTIONS[self._approximation](
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            zeta,
            dt,
            dev,
            damping,
            self.l,
        )

        step = [
            s.squeeze(-1)
            for s in apply_joint_JT(
                interior_inputs,
                interior_grad_outputs,
                boundary_inputs,
                boundary_grad_outputs,
                step,
            )
        ]

        if momentum != 0.0:
            for p, s in zip(params, step):
                self.state[p]["phi"].mul_(momentum).add_(s)

            step = [
                self.state[p]["phi"] / sqrt(1 - momentum ** (2 * (self.steps + 1)))
                for p in params
            ]
        else:
            step = [s.view(p.shape) for s, p in zip(step, params)]
        return step

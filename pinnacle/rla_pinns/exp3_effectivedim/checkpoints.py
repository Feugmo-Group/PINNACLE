from os import path
from glob import glob
from torch import load
from typing import Tuple
from tueplots import bundles
from torch.nn import Sequential
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torch.linalg import eigvalsh, matrix_rank
from rla_pinns.train import set_up_layers
from palettable.colorbrewer import sequential
from rla_pinns.optim.utils import (
    evaluate_losses_with_layer_inputs_and_grad_outputs,
    compute_joint_JJT,
)

# color options: https://jiffyclub.github.io/palettable/colorbrewer/
COLORS = {
    "SGD": sequential.Reds_4.mpl_colors[-2],
    "Adam": sequential.Reds_4.mpl_colors[-1],
    "ENGD": sequential.Blues_5.mpl_colors[-3],
    "ENGD (Woodbury)": sequential.Blues_5.mpl_colors[-2],
    "ENGD (Nystrom)": sequential.Blues_5.mpl_colors[-1],
    "SPRING": sequential.Greens_4.mpl_colors[-2],
    "SPRING (Nystrom)": sequential.Greens_4.mpl_colors[-1],
    "HessianFree": "black",
}

LINESTYLE = {
    "SGD": "-",
    "Adam": "-",
    "ENGD": "-",
    "ENGD (Woodbury)": "-",
    "ENGD (Nystrom)": "-",
    "SPRING": "-",
    "SPRING (Nystrom)": "-",
    "HessianFree": "-",
}


def get_effective_dim(kernel, damping):
    """Compute the effective dimension."""
    eigvals = eigvalsh(kernel).clamp(min=0)
    d_eff = (eigvals / (eigvals + damping)).sum()
    return d_eff


def evaluate_checkpoint(checkpoint: str) -> Tuple[float, int]:
    """Evaluate a single checkpoint and return its eigenvalues."""
    checkpoint_name = path.splitext(path.basename(checkpoint))[0]
    print(f"Processing checkpoint {checkpoint_name}.")
    data = load(checkpoint)

    config = data["config"]
    equation = config["equation"]
    dim_Omega = config["dim_Omega"]
    architecture = config["model"]

    X_Omega = data["X_Omega"]
    y_Omega = data["y_Omega"]
    X_dOmega = data["X_dOmega"]
    y_dOmega = data["y_dOmega"]

    damping = config["RNGD_damping"]

    layers = set_up_layers(architecture, equation, dim_Omega)
    layers = [layer.to(X_Omega.device, X_Omega.dtype) for layer in layers]
    model = Sequential(*layers).to(X_Omega.device)
    model.load_state_dict(data["model"])

    (
        _,
        _,
        _,
        _,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )

    JJT = compute_joint_JJT(
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    )

    d_eff = get_effective_dim(JJT, damping)
    num_params = sum(p.numel() for layer in layers for p in layer.parameters())
    d_eff = d_eff.item() / (X_Omega.shape[0] + X_dOmega.shape[0])
    return d_eff, num_params


def process_checkpoints(checkpoint_dir, optimizer):
    d_effs = {}
    steps = set()
    dim_Omega = set()
    equation = set()
    num_params = set()

    # Filter checkpoints based on equation
    for i, checkpoint in enumerate(sorted(glob(path.join(checkpoint_dir, "*.pt")))):
        checkpoint_name = path.splitext(path.basename(checkpoint))[0]
        info = checkpoint_name.split("_")
        opt = info[-2]

        if opt != optimizer:
            continue

        step = int(info[-1][-7:])  # Extract the last word
        N_Omega = int(info[1][:-1])  # Extract the third last word

        d, params = evaluate_checkpoint(checkpoint)

        if opt not in d_effs.keys():
            d_effs[opt] = []
        d_effs[opt].append(d)

        steps = steps | {step}
        dim_Omega = dim_Omega | {N_Omega}
        equation = equation | {info[0]}
        num_params = num_params | {params}

    # Retrieve data
    (dim_Omega,) = dim_Omega
    (equation,) = equation
    (num_params,) = num_params
    steps = sorted(list(steps))
    return steps, d_effs


def main():
    """Visualize eigenvalues for each checkpoint."""
    parser = ArgumentParser(description="Plot solutions of checkpoints.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="sweeps/checkpoints",
        help="Directory containing checkpoints that should be visualized.",
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="ENGDw",
        help="Optimizer used in the experiment.",
    )
    args = parser.parse_args()
    checkpoint_dir = path.abspath(args.checkpoint_dir)

    steps, d_effs = process_checkpoints(
        checkpoint_dir, args.optimizer
    )

    # Plot all effective dimensions for a given experiment
    HEREDIR = path.dirname(path.abspath(__file__))
    with plt.rc_context(
        bundles.neurips2023(rel_width=0.5, usetex=not args.disable_tex)
    ):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Steps")
        ax.set_xscale("log")
        ax.set_ylabel(r"$d_{\mathrm{eff}} / N$")

        # ax.set_title(
        #     f"{dim_Omega}d Poisson (D = {num_params})"
        #     if args.disable_tex
        #     else rf"${dim_Omega}d Poisson (D = {num_params})$"
        # )
        ax.grid(True, alpha=0.5)

        for opt_name, d_vals in d_effs.items():
            name = "ENGD (Woodbury)" if opt_name == "ENGDw" else opt_name
            ax.plot(
                sorted(steps),
                d_vals,
                label=name,
                color=COLORS[name],
                linestyle=LINESTYLE[name],
            )

        ax.legend()

        plt.savefig(
            path.join(
                HEREDIR,
                f"effective_dim_over_step_{'engd' if args.optimizer == 'ENGDw' else 'spring'}.pdf",
            ),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()

"""Create a pretty plot that groups together the results for 1d heat."""

from argparse import ArgumentParser
from itertools import product
from os import path

from matplotlib import pyplot as plt
from tueplots import bundles

from rla_pinns.exp11_poisson5d_large import plot_1k as SMALL
from rla_pinns.exp11_poisson5d_large.plot_1k import colors, linestyles
from rla_pinns.exp11_poisson5d_large import plot_5k as MEDIUM
from rla_pinns.exp11_poisson5d_large import plot_10k as BIG
from rla_pinns.wandb_utils import load_best_run

BATCH_SIZES = [1000, 5000, 10000]

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Summarize the experiments on heat 1d in one figure."
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    args = parser.parse_args()

    # Experiments as columns
    COLUMNS = [SMALL, MEDIUM, BIG]
    IGNORE = {"ENGD (diagonal)"}

    # Define axes labels
    y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}
    # y_to_ylabel = {"l2_error": "$L_2$ error"}
    x_to_xlabel = {"step": "Iteration", "time": "Time [s]"}
    # x_to_xlabel = {"time": "Time [s]"}

    # Build 4x3 grid
    with plt.rc_context(
        bundles.neurips2023(
            rel_width=1.0,
            nrows=6,
            ncols=4,
            usetex=not args.disable_tex,
        ),
    ):
        # enable siunitx grouping for D titles
        plt.rcParams[
            "text.latex.preamble"
        ] += r"\usepackage[group-separator={,}, group-minimum-digits={3}]{siunitx}"

        fig, axs = plt.subplots(4, len(COLUMNS), sharey="row")
        # Loop through each combination to fill each row
        for row_index, ((x, xlabel), (y, ylabel)) in enumerate(
            product(x_to_xlabel.items(), y_to_ylabel.items())
        ):
            i = 0
            for col_index, exp in enumerate(COLUMNS):
                ax = axs[row_index, col_index]

                # Axis formatting
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.5)
                ax.set_xlabel(xlabel)
                if col_index == 0:
                    ax.set_ylabel(ylabel)

                if args.disable_tex:
                    title = f"N={BATCH_SIZES[i]}"
                else:
                    title = rf"$N=\num{BATCH_SIZES[i]}$"
                ax.set_title(title)
                i += 1

                # Plot each optimizer run
                for sweep_id, name in exp.sweep_ids.items():
                    if name in IGNORE:
                        continue
                    df_history, _ = load_best_run(
                        exp.entity,
                        exp.project,
                        sweep_id,
                        save=False,
                        update=False,
                        savedir=exp.DATADIR,
                    )
                    x_data = {
                        "step": df_history["step"] + 1,
                        "time": df_history["time"] - df_history["time"].min(),
                    }[x]
                    # Only label once (first row)
                    label = (
                        name
                        if row_index == 0 and col_index == 0 and "*" not in name
                        else None
                    )
                    ax.plot(
                        x_data,
                        df_history[y],
                        label=label,
                        color=colors[name],
                        linestyle=linestyles[name],
                    )

                # For time-based plots, ensure positive x-axis
                if x == "time":
                    ax.set_xlim(left=1)

        # Shared legend at bottom
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(labels),
            handlelength=1.35,
            columnspacing=0.9,
            frameon=True,
        )

        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = path.join(path.dirname(path.abspath(__file__)), "l2_line_search.pdf")
        plt.savefig(out_file, bbox_inches="tight")
        print(f"Saved combined figure to {out_file}")

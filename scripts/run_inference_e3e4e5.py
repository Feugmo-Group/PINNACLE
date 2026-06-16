#!/usr/bin/env python3
"""
Inference pass over E3a, E3b, E4, E5 checkpoints.
For each best_model.pt:
  - Loads the network + config
  - Evaluates film thickness L(t, E) against FEM reference
  - Records final relative error at each of the 5 test voltages
Produces:
  - outputs/inference/e3a_errors.csv
  - outputs/inference/e3b_errors.csv
  - outputs/inference/e4_errors.csv
  - outputs/inference/e5_errors.csv
  - outputs/inference/figures/  (E3a heatmap, E4 curve, E5 curve)
"""
import sys, re, json, pathlib
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path("/app")
EXP  = REPO / "outputs" / "experiments"
OUT  = REPO / "outputs" / "inference"
FIG  = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "pinnacle"))

# FEM reference: final film thickness (nm) at t=400000s per voltage
FEM_FINAL_NM = {0.1: 1.27, 0.4: 3.67, 1.0: 8.60, 1.6: 14.0, 1.8: 16.1}
VOLTAGES = sorted(FEM_FINAL_NM)
FEM_DIR  = REPO / "pinnacle" / "FEM"

# ── helpers ─────────────────────────────────────────────────────────────────

def load_fem_curves():
    """Return dict voltage -> (t_s array, L_nm array) from FEM txt files."""
    curves = {}
    for v in VOLTAGES:
        fpath = FEM_DIR / f"{v} V.txt"
        if not fpath.exists():
            continue
        data = np.genfromtxt(fpath, delimiter='\t', names=True)
        t_col = 'Times' if 'Times' in data.dtype.names else data.dtype.names[0]
        L_col = 'Filmthicknessm' if 'Filmthicknessm' in data.dtype.names else data.dtype.names[2]
        valid = ~(np.isnan(data[t_col]) | np.isnan(data[L_col]))
        curves[v] = (data[t_col][valid], data[L_col][valid] * 1e9)  # t in s, L in nm
    return curves

FEM_CURVES = load_fem_curves()

def infer_errors(exp_dir: pathlib.Path) -> dict | None:
    """
    Load latest best_model.pt from exp_dir, run inference, return
    dict with keys: per-voltage RelError_final[%] and mean_error.
    Returns None on failure.
    """
    ckpts = sorted(exp_dir.glob("*/checkpoints/best_model.pt"))
    if not ckpts:
        return None
    ckpt_path = ckpts[-1]
    run_dir   = ckpt_path.parents[1]
    cfg_path  = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None

    try:
        from omegaconf import OmegaConf
        from networks.networks import NetworkManager
        from physics.physics import ElectrochemicalPhysics

        config = OmegaConf.load(cfg_path)
        device = torch.device("cpu")

        # Build network and physics from config
        networks = NetworkManager(config, device)
        physics  = ElectrochemicalPhysics(config, device)

        # Load weights
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck.get("state_dict", {}))
        networks.load_state_dict(sd)
        networks.eval()

        # Detect model dtype from first parameter
        model_dtype = next(iter(networks.get_all_parameters())).dtype

        scales = physics.scales
        result = {"best_loss": float(ck.get("best_loss", float("nan")))}

        with torch.no_grad():
            for v in VOLTAGES:
                if v not in FEM_CURVES:
                    continue
                t_s, L_fem_nm = FEM_CURVES[v]
                t_hat = torch.tensor(t_s / scales.tc,
                                     dtype=model_dtype).unsqueeze(1)
                E_hat = torch.full((len(t_s), 1), v / scales.phic,
                                   dtype=model_dtype)
                L_hat = networks['film_thickness'](
                    torch.cat([t_hat, E_hat], dim=1)
                ).squeeze().cpu().numpy()
                L_pinn_nm = L_hat * scales.lc * 1e9

                # Final relative error (last time point)
                L_pinn_f = float(L_pinn_nm[-1])
                L_fem_f  = float(L_fem_nm[-1])
                rel_err  = abs(L_pinn_f - L_fem_f) / (abs(L_fem_f) + 1e-12) * 100.0
                result[f"RelErr_{v}V"] = rel_err

        vol_errs = [result[f"RelErr_{v}V"] for v in VOLTAGES if f"RelErr_{v}V" in result]
        result["mean_err"] = float(np.mean(vol_errs)) if vol_errs else float("nan")
        return result

    except Exception as e:
        print(f"  WARN {exp_dir.name}: {e}")
        return None

# ── E3a: 30 random anchor seeds ─────────────────────────────────────────────

print("\n=== E3a: 30-seed anchor robustness ===")
e3a_rows = []
for d in sorted(EXP.glob("e3a_seed*_ntk_l2")):
    m = re.match(r"e3a_seed(\d+)_ntk_l2", d.name)
    if not m: continue
    seed = int(m.group(1))
    r = infer_errors(d)
    if r:
        row = {"seed": seed, **r}
        e3a_rows.append(row)
        print(f"  seed{seed:2d}: mean_err={r['mean_err']:.2f}%")

if e3a_rows:
    df3a = pd.DataFrame(e3a_rows)
    df3a.to_csv(OUT / "e3a_errors.csv", index=False)
    mn  = df3a["mean_err"].mean()
    std = df3a["mean_err"].std(ddof=1)
    mx  = df3a["mean_err"].max()
    print(f"E3a summary: mean={mn:.2f}% ± {std:.2f}%, max={mx:.2f}%  (N={len(df3a)})")

    # Heat-map: seed vs voltage
    fig, ax = plt.subplots(figsize=(8, 5))
    vol_cols = [f"RelErr_{v}V" for v in VOLTAGES]
    mat = df3a[vol_cols].values
    im = ax.imshow(mat.T, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=min(10, np.nanpercentile(mat, 95)))
    ax.set_xticks(range(len(df3a)))
    ax.set_xticklabels([f"{r['seed']}" for r in e3a_rows], fontsize=7, rotation=90)
    ax.set_yticks(range(len(VOLTAGES)))
    ax.set_yticklabels([f"{v} V" for v in VOLTAGES], fontsize=10)
    ax.set_xlabel("Anchor seed", fontsize=12)
    ax.set_ylabel("Test voltage", fontsize=12)
    ax.set_title(f"E3a anchor robustness: final relative error (%) — mean={mn:.2f}%±{std:.2f}%", fontsize=11)
    plt.colorbar(im, ax=ax, label="Rel. error (%)")
    plt.tight_layout()
    fig.savefig(FIG / "e3a_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIG}/e3a_heatmap.png")

# ── E3b: systematic 5-position sweep ────────────────────────────────────────

print("\n=== E3b: systematic anchor sweep ===")
e3b_rows = []
for d in sorted(EXP.glob("e3b_*_ntk_l2")):
    r = infer_errors(d)
    if r:
        row = {"config": d.name, **r}
        e3b_rows.append(row)
        print(f"  {d.name}: mean_err={r['mean_err']:.2f}%")

if e3b_rows:
    df3b = pd.DataFrame(e3b_rows)
    df3b.to_csv(OUT / "e3b_errors.csv", index=False)
    print(f"E3b worst-case: {df3b['mean_err'].max():.2f}%")

# ── E4: data-efficiency sweep ────────────────────────────────────────────────

print("\n=== E4: data-efficiency sweep ===")
e4_rows = []
for d in sorted(EXP.glob("e4_N*_seed*_ntk_l2")):
    m = re.match(r"e4_N(\d+)_seed(\d+)_ntk_l2", d.name)
    if not m: continue
    N, seed = int(m.group(1)), int(m.group(2))
    r = infer_errors(d)
    if r:
        row = {"N_data": N, "seed": seed, **r}
        e4_rows.append(row)

if e4_rows:
    df4 = pd.DataFrame(e4_rows)
    df4.to_csv(OUT / "e4_errors.csv", index=False)

    # Aggregate: mean ± std over seeds per N_data
    agg = df4.groupby("N_data")["mean_err"].agg(["mean", "std", "count"]).reset_index()
    print("E4 data-efficiency (mean rel err % over seeds):")
    for _, row in agg.iterrows():
        print(f"  N={int(row['N_data']):3d}: mean={row['mean']:.2f}% ± {row['std']:.2f}%  (n={int(row['count'])})")

    # Log-scale figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(agg["N_data"], agg["mean"], yerr=agg["std"],
                marker='o', capsize=4, linewidth=2, color='steelblue')
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("Number of FEM anchor points $N_\\mathrm{data}$", fontsize=12)
    ax.set_ylabel("Mean relative film-thickness error (%)", fontsize=12)
    ax.set_title("E4: Data efficiency — error vs. anchor count", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "e4_data_efficiency.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIG}/e4_data_efficiency.png")

# ── E5: noise robustness sweep ───────────────────────────────────────────────

print("\n=== E5: noise robustness sweep ===")
e5_rows = []
SIGMA_MAP = {"00": 0, "001": 1, "005": 5, "010": 10, "020": 20, "050": 50}
for d in sorted(EXP.glob("e5_s*_seed*_ntk_l2")):
    m = re.match(r"e5_s(\w+)_seed(\d+)_ntk_l2", d.name)
    if not m: continue
    sigma_tag, seed = m.group(1), int(m.group(2))
    sigma_pct = SIGMA_MAP.get(sigma_tag, float(sigma_tag))
    r = infer_errors(d)
    if r:
        row = {"sigma_pct": sigma_pct, "seed": seed, **r}
        e5_rows.append(row)

if e5_rows:
    df5 = pd.DataFrame(e5_rows)
    df5.to_csv(OUT / "e5_errors.csv", index=False)

    agg5 = df5.groupby("sigma_pct")["mean_err"].agg(["mean", "std", "count"]).reset_index()
    print("E5 noise robustness (mean rel err % over seeds):")
    for _, row in agg5.iterrows():
        print(f"  sigma={int(row['sigma_pct']):3d}%: mean={row['mean']:.2f}% ± {row['std']:.2f}%  (n={int(row['count'])})")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(agg5["sigma_pct"], agg5["mean"], yerr=agg5["std"],
                marker='s', capsize=4, linewidth=2, color='darkorange')
    ax.set_xlabel("Anchor noise level $\\sigma$ (%)", fontsize=12)
    ax.set_ylabel("Mean relative film-thickness error (%)", fontsize=12)
    ax.set_title("E5: Noise robustness — error vs. anchor noise", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG / "e5_noise_robustness.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIG}/e5_noise_robustness.png")

# ── summary JSON ─────────────────────────────────────────────────────────────

summary = {}
if e3a_rows:
    df3a_s = pd.DataFrame(e3a_rows)
    summary["e3a"] = {
        "n_seeds": len(df3a_s),
        "mean_err_pct": round(float(df3a_s["mean_err"].mean()), 2),
        "std_err_pct":  round(float(df3a_s["mean_err"].std(ddof=1)), 2),
        "max_err_pct":  round(float(df3a_s["mean_err"].max()), 2),
    }
if e3b_rows:
    df3b_s = pd.DataFrame(e3b_rows)
    summary["e3b"] = {
        "worst_case_err_pct": round(float(df3b_s["mean_err"].max()), 2),
        "configs": df3b_s[["config", "mean_err"]].to_dict("records"),
    }
if e4_rows:
    df4_s = pd.DataFrame(e4_rows)
    agg4 = df4_s.groupby("N_data")["mean_err"].mean().to_dict()
    summary["e4"] = {str(int(k)): round(float(v), 2) for k, v in sorted(agg4.items())}
if e5_rows:
    df5_s = pd.DataFrame(e5_rows)
    agg5 = df5_s.groupby("sigma_pct")["mean_err"].mean().to_dict()
    summary["e5"] = {str(int(k)): round(float(v), 2) for k, v in sorted(agg5.items())}

(OUT / "inference_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSummary written to {OUT}/inference_summary.json")
print("\n=== DONE ===")

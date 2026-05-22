"""
Gradient-based yaw optimization using PyTorch autograd through the FLORAF
C++ sequential solver. Exact analytical gradients of AEP with respect to
per-turbine, per-wind-condition yaw angles are back-propagated via ``autograd``
and consumed by an Adam optimizer.

Variable shape
--------------
The optimization variable ``yaw`` has shape ``[n_findex, n_turbines]``:
one yaw angle per turbine per wind condition. This is the physically correct
parameterization — the optimal yaw for a given turbine depends on wind
direction and speed, so no reduction across conditions is applied.

Usage
-----
    python gradient_yaw_optimization.py
    python gradient_yaw_optimization.py --iters 300 --lr 1.0 --n-turbs 9
    python gradient_yaw_optimization.py --no-plot
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from floris import FlorisModel, WindRose
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


# ---------------------------------------------------------------------------
# Model / wind-rose helpers (shared with gradient_layout_optimization.py)
# ---------------------------------------------------------------------------

def _get_fmodel(device: str = "cpu") -> FlorisModel:
    """Build a FLORIS model using the FLORAF C++ sequential solver."""
    fdefaults = FlorisModel.get_defaults()
    fdefaults["logging"]["console"]["enable"] = False
    fdefaults["logging"]["file"]["enable"] = False
    fdefaults["solver"]["backend"] = "cpp"
    fdefaults["solver"]["device"] = device
    fdefaults["solver"]["cpp_solver"] = "sequential"
    fdefaults["wake"]["enable_secondary_steering"] = False
    fdefaults["wake"]["enable_yaw_added_recovery"] = False
    fdefaults["wake"]["enable_transverse_velocities"] = False
    fdefaults["wake"]["enable_active_wake_mixing"] = False
    return FlorisModel(fdefaults)


def _wind_rose(n_wdirs: int = 36, n_wspeeds: int = 5, ti: float = 0.06) -> WindRose:
    """Synthetic North Sea wind rose with a dominant SW peak."""
    wd = np.linspace(0.0, 360.0, n_wdirs, endpoint=False)
    ws = np.linspace(4.0, 18.0, n_wspeeds)

    dir_freq = 0.45 * np.exp(-0.5 * ((wd - 240) % 360 / 38) ** 2) + 0.04
    dir_freq /= dir_freq.sum()

    k, c = 2.0, 11.0
    spd_freq = (k / c) * (ws / c) ** (k - 1) * np.exp(-(ws / c) ** k)
    spd_freq /= spd_freq.sum()

    freq_table = np.outer(dir_freq, spd_freq)
    freq_table /= freq_table.sum()

    return WindRose(
        wind_directions=wd,
        wind_speeds=ws,
        freq_table=freq_table,
        ti_table=ti,
    )


def _row_layout(n_turbs: int, spacing_D: float = 5.0, D: float = 126.0):
    """Place turbines in a single east-west row."""
    lx = np.arange(n_turbs, dtype=float) * spacing_D * D
    ly = np.zeros(n_turbs)
    return lx, ly


def _grid_layout(
    n_turbs: int,
    spacing_x_D: float = 5.0,
    spacing_y_D: float = 3.0,
    stagger_D: float = 1.5,
    D: float = 126.0,
):
    """
    Place turbines on a staggered grid.

    Turbines are arranged in rows running east-west.  Odd-numbered rows are
    offset by *stagger_D* diameters in x to break the regular grid pattern,
    which creates a more realistic farm layout where not all turbines are in
    direct wake of upstream neighbours.

    Parameters
    ----------
    n_turbs:
        Total number of turbines.
    spacing_x_D:
        Down-wind spacing between turbines in the same row, in rotor diameters.
    spacing_y_D:
        Cross-wind row spacing, in rotor diameters.
    stagger_D:
        x-offset applied to every other row, in rotor diameters.
    D:
        Rotor diameter (m).
    """
    cols = int(np.ceil(np.sqrt(n_turbs)))
    rows = int(np.ceil(n_turbs / cols))

    lx_list, ly_list = [], []
    for row in range(rows):
        x_offset = (stagger_D * D) if (row % 2 == 1) else 0.0
        for col in range(cols):
            if len(lx_list) >= n_turbs:
                break
            lx_list.append(col * spacing_x_D * D + x_offset)
            ly_list.append(row * spacing_y_D * D)

    return np.array(lx_list), np.array(ly_list)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _aep_tensor(fmodel: FlorisModel, freq_tensor: torch.Tensor) -> torch.Tensor:
    """
    Return a scalar AEP tensor (Wh/yr) with ``grad_fn`` connected to
    ``fmodel.core._cpp_farm.yaw_angles``.

    The caller must have already injected the ``requires_grad=True`` yaw
    tensor into ``fmodel.core._cpp_farm.yaw_angles`` before calling this.
    """
    fmodel.run()
    power = fmodel.core.get_farm_power_tensor()       # [F, T] W
    farm_power_per_findex = power.sum(dim=1)           # [F]
    return (freq_tensor * farm_power_per_findex).sum() * 8760.0


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run_yaw_optimization(
    n_turbs: int,
    wind_rose: WindRose,
    n_iters: int,
    lr: float,
    yaw_max_deg: float = 30.0,
    layout: str = "row",
    live_plot: bool = True,
    device: str = "cpu",
) -> dict:
    """
    Optimize yaw angles to maximise AEP and return a results dict.

    Parameters
    ----------
    n_turbs:
        Number of turbines.
    wind_rose:
        Wind resource description.
    n_iters:
        Number of Adam gradient steps.
    lr:
        Adam learning rate in degrees.
    yaw_max_deg:
        Hard constraint: yaw angles are clamped to ``[-yaw_max_deg, yaw_max_deg]``
        after every optimizer step.
    layout:
        ``"row"`` — single east-west row (default).
        ``"grid"`` — staggered multi-row grid.
    live_plot:
        Show a live convergence plot.
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """
    D = 126.0
    if layout == "grid":
        lx, ly = _grid_layout(n_turbs, D=D)
    else:
        lx, ly = _row_layout(n_turbs, spacing_D=5.0, D=D)
    n_turbs = len(lx)  # _grid_layout may round up to fill the grid

    freq_np = wind_rose.unpack_freq()                         # [F]
    freq_t  = torch.tensor(freq_np, dtype=torch.float32, device=device)

    n_findex = len(freq_np)

    # Build model and set layout + wind conditions.
    fmodel = _get_fmodel(device=device)
    fmodel.set(layout_x=lx.tolist(), layout_y=ly.tolist(), wind_data=wind_rose)

    # ------------------------------------------------------------------
    # Baseline AEP (zero yaw)
    # ------------------------------------------------------------------
    with torch.no_grad():
        base_aep_t = _aep_tensor(fmodel, freq_t)
    base_aep = base_aep_t.item()
    print(f"\nInitial AEP (zero yaw): {base_aep / 1e9:.4f} GWh/yr")

    # ------------------------------------------------------------------
    # Optimization variable: yaw [n_findex, n_turbines], degrees
    # ------------------------------------------------------------------
    yaw = torch.zeros(
        n_findex, n_turbs,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    # Inject the leaf tensor into the C++ farm struct.  It will be preserved
    # across calls to initialize_domain() because _preserve_if_grad detects
    # requires_grad=True.  Do NOT call fmodel.set(yaw_angles=...) inside the
    # optimization loop — that writes to _py_core only, bypassing _cpp_farm.
    fmodel.core._cpp_farm.yaw_angles = yaw

    optimizer = torch.optim.Adam([yaw], lr=lr)

    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("AEP improvement (%)")
        ax.set_title("Gradient-based yaw optimization (PyTorch autograd + Adam)")
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.grid(True, alpha=0.4)
        (aep_line,) = ax.plot([], [], "o-", color="#2c7fb8", ms=3, lw=1.5)
        plt.tight_layout()
        plt.pause(0.05)

    aep_history = [base_aep]
    t0 = time.perf_counter()

    for step in range(1, n_iters + 1):
        optimizer.zero_grad()

        aep = _aep_tensor(fmodel, freq_t)
        loss = -aep
        loss.backward()
        optimizer.step()

        # Hard constraint: clamp yaw angles to physical range.
        with torch.no_grad():
            yaw.clamp_(-yaw_max_deg, yaw_max_deg)
            yaw.grad.zero_()

        current_aep = aep.item()
        aep_history.append(current_aep)
        improvement = 100.0 * (current_aep / base_aep - 1.0)
        print(
            f"  Step {step:4d}/{n_iters}"
            f" | AEP = {current_aep / 1e9:.4f} GWh/yr"
            f" | Δ = {improvement:+.3f}%"
        )

        if live_plot:
            iters = np.arange(len(aep_history))
            improv = 100.0 * (np.array(aep_history) / base_aep - 1.0)
            aep_line.set_data(iters, improv)
            ax.set_xlim(-1, max(len(aep_history) + 1, 10))
            pad = max(abs(improv).max() * 0.15, 0.05)
            ax.set_ylim(improv.min() - pad, improv.max() + pad)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    elapsed = time.perf_counter() - t0
    final_aep = aep_history[-1]
    final_yaw = yaw.detach().cpu().numpy().copy()   # [n_findex, n_turbines], degrees

    print(f"\nYaw opt finished in {elapsed:.1f} s  ({n_iters} iterations)")
    print(
        f"Final AEP: {final_aep / 1e9:.4f} GWh/yr"
        f"  ({100 * (final_aep / base_aep - 1):+.3f}% vs zero-yaw)\n"
    )

    return dict(
        base_aep=base_aep,
        final_aep=final_aep,
        aep_history=aep_history,
        final_yaw=final_yaw,
        elapsed=elapsed,
        n_iters=n_iters,
        n_findex=n_findex,
        n_turbs=n_turbs,
        yaw_max_deg=yaw_max_deg,
    )


# ---------------------------------------------------------------------------
# Serial Refine (SR) yaw optimization for comparison
# ---------------------------------------------------------------------------

def _get_fmodel_python() -> FlorisModel:
    """Build a FLORIS model using the standard Python backend (GCH-like defaults)."""
    fdefaults = FlorisModel.get_defaults()
    fdefaults["logging"]["console"]["enable"] = False
    fdefaults["logging"]["file"]["enable"] = False
    fdefaults["wake"]["enable_secondary_steering"] = False
    fdefaults["wake"]["enable_yaw_added_recovery"] = False
    fdefaults["wake"]["enable_transverse_velocities"] = False
    fdefaults["wake"]["enable_active_wake_mixing"] = False
    return FlorisModel(fdefaults)


def run_sr_optimization(
    lx: np.ndarray,
    ly: np.ndarray,
    wind_rose: WindRose,
    base_aep: float,
    yaw_max_deg: float = 30.0,
) -> dict:
    """
    Run YawOptimizationSR on the same layout and wind rose and return a results dict.

    Parameters
    ----------
    lx, ly:
        Turbine positions (metres), matching the gradient optimizer layout.
    wind_rose:
        Wind resource description.
    base_aep:
        Baseline AEP (Wh/yr) computed at zero yaw, for relative improvement reporting.
    yaw_max_deg:
        Symmetric yaw angle bound applied to both minimum and maximum.
    """
    fmodel = _get_fmodel_python()
    fmodel.set(layout_x=lx.tolist(), layout_y=ly.tolist(), wind_data=wind_rose)

    yaw_opt = YawOptimizationSR(
        fmodel=fmodel,
        minimum_yaw_angle=-yaw_max_deg,
        maximum_yaw_angle=yaw_max_deg,
        Ny_passes=[5, 4],
        exclude_downstream_turbines=True,
    )

    print("Running Serial Refine yaw optimization...")
    t0 = time.perf_counter()
    df_opt = yaw_opt.optimize()
    elapsed = time.perf_counter() - t0

    # Apply optimal yaw angles to evaluate AEP over the full wind rose.
    yaw_angles_opt = np.vstack(df_opt["yaw_angles_opt"])  # [n_findex, n_turbs]
    fmodel_eval = _get_fmodel_python()
    fmodel_eval.set(
        layout_x=lx.tolist(),
        layout_y=ly.tolist(),
        wind_data=wind_rose,
        yaw_angles=yaw_angles_opt,
    )
    fmodel_eval.run()
    final_aep = fmodel_eval.get_farm_AEP()

    print(f"SR opt finished in {elapsed:.1f} s")
    print(
        f"Final AEP (SR): {final_aep / 1e9:.4f} GWh/yr"
        f"  ({100 * (final_aep / base_aep - 1):+.3f}% vs zero-yaw)\n"
    )

    return dict(
        base_aep=base_aep,
        final_aep=final_aep,
        elapsed=elapsed,
        final_yaw=yaw_angles_opt,
        n_findex=yaw_angles_opt.shape[0],
        n_turbs=yaw_angles_opt.shape[1],
        yaw_max_deg=yaw_max_deg,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _print_comparison_table(grad_res: dict, sr_res: dict) -> None:
    """Print a formatted side-by-side results table."""
    g = grad_res
    s = sr_res

    print("\n" + "=" * 62)
    print(f"{'':30s}  {'Gradient (Adam)':>15}  {'SR (FLORIS)':>11}")
    print("-" * 62)
    print(f"{'Iterations / passes':30s}  {g['n_iters']:>15d}  {'–':>11}")
    print(f"{'Wall time (s)':30s}  {g['elapsed']:>15.1f}  {s['elapsed']:>11.1f}")
    print(f"{'Initial AEP (GWh/yr)':30s}  {g['base_aep'] / 1e9:>15.4f}  {'(same)':>11}")
    print(f"{'Final AEP (GWh/yr)':30s}  {g['final_aep'] / 1e9:>15.4f}  {s['final_aep'] / 1e9:>11.4f}")
    pct_g = 100 * (g['final_aep'] / g['base_aep'] - 1)
    pct_s = 100 * (s['final_aep'] / s['base_aep'] - 1)
    print(f"{'AEP improvement (%)':30s}  {pct_g:>+15.3f}  {pct_s:>+11.3f}")
    print("=" * 62 + "\n")


def _plot_comparison(grad_res: dict, sr_res: dict) -> None:
    """Plot AEP convergence (gradient) and final-AEP bar comparison side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Yaw optimization comparison: autograd (Adam) vs Serial Refine (FLORIS)",
        fontsize=13,
        fontweight="bold",
    )

    colours = {"grad": "#2c7fb8", "sr": "#d7191c"}

    # Left panel — gradient convergence curve
    ax = axes[0]
    hist = np.array(grad_res["aep_history"])
    improvement = 100.0 * (hist / grad_res["base_aep"] - 1.0)
    ax.plot(
        np.arange(len(improvement)),
        improvement,
        "o-",
        color=colours["grad"],
        ms=3,
        lw=1.5,
        label=f"Gradient (Adam)  final {improvement[-1]:+.3f}%",
    )
    pct_sr = 100.0 * (sr_res["final_aep"] / sr_res["base_aep"] - 1.0)
    ax.axhline(pct_sr, color=colours["sr"], ls="--", lw=1.5, label=f"SR (FLORIS)  {pct_sr:+.3f}%")
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set_xlabel("Adam iteration", fontsize=11)
    ax.set_ylabel("AEP improvement (%)", fontsize=11)
    ax.set_title("Gradient convergence", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # Right panel — per-turbine yaw angle comparison for the highest-frequency condition
    ax = axes[1]
    # Find the wind condition with the highest frequency weight
    freq_np = sr_res["final_yaw"].shape[0]  # just n_findex; we pick peak condition
    # Use argmax of absolute mean yaw as a representative active condition
    mean_abs_yaw_grad = np.abs(grad_res["final_yaw"]).mean(axis=1)
    peak_idx = int(np.argmax(mean_abs_yaw_grad))

    n_turbs = grad_res["n_turbs"]
    turb_ids = np.arange(n_turbs)
    width = 0.35

    ax.bar(
        turb_ids - width / 2,
        grad_res["final_yaw"][peak_idx],
        width,
        color=colours["grad"],
        label="Gradient (Adam)",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
    )
    sr_yaw_peak = sr_res["final_yaw"][peak_idx] if peak_idx < sr_res["final_yaw"].shape[0] else np.zeros(n_turbs)
    ax.bar(
        turb_ids + width / 2,
        sr_yaw_peak,
        width,
        color=colours["sr"],
        label="SR (FLORIS)",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set_xlabel("Turbine index", fontsize=11)
    ax.set_ylabel("Yaw angle (°)", fontsize=11)
    ax.set_title(f"Optimal yaw angles at peak condition (findex {peak_idx})", fontsize=11)
    ax.set_xticks(turb_ids)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def _print_results(res: dict) -> None:
    wd_idx_peak = int(np.argmax(res["final_yaw"].max(axis=1)))
    print("Optimal yaw angles at peak-frequency wind condition "
          f"(findex {wd_idx_peak}), degrees:")
    print("  " + "  ".join(
        f"T{i}: {res['final_yaw'][wd_idx_peak, i]:+.1f}"
        for i in range(res["n_turbs"])
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",   type=int,   default=200,  help="Adam steps (default: 200)")
    parser.add_argument("--lr",      type=float, default=0.5,  help="Adam lr in degrees (default: 0.5)")
    parser.add_argument("--n-turbs", type=int,   default=6,    help="Turbines in row (default: 6)")
    parser.add_argument("--n-wdirs", type=int,   default=36,   help="Wind directions (default: 36)")
    parser.add_argument("--n-wspeeds", type=int, default=5,    help="Wind speed bins (default: 5)")
    parser.add_argument("--yaw-max", type=float, default=25.0, help="Max |yaw| degrees (default: 30)")
    parser.add_argument(
        "--layout",
        choices=["row", "grid"],
        default="row",
        help="Farm layout: 'row' (single row, default) or 'grid' (staggered multi-row)",
    )
    parser.add_argument("--compare", action="store_true", help="Run SR yaw optimization and compare results.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable live plot")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gradient-based wind farm yaw optimization")
    print("=" * 60)
    print(f"  Turbines    : {args.n_turbs}")
    print(f"  Wind dirs   : {args.n_wdirs}")
    print(f"  Wind speeds : {args.n_wspeeds}")
    print(f"  Adam steps  : {args.iters}")
    print(f"  Adam lr     : {args.lr} deg")
    print(f"  |yaw| max   : {args.yaw_max} deg")
    print(f"  Layout      : {args.layout}")
    print(f"  SR compare  : {'yes' if args.compare else 'no'}")
    print()

    wind_rose = _wind_rose(args.n_wdirs, args.n_wspeeds)

    results = run_yaw_optimization(
        n_turbs=args.n_turbs,
        wind_rose=wind_rose,
        n_iters=args.iters,
        lr=args.lr,
        yaw_max_deg=args.yaw_max,
        layout=args.layout,
        live_plot=args.plot,
        device="mps"
    )

    _print_results(results)

    if args.compare:
        # Reconstruct the layout arrays used by the gradient optimizer.
        D = 126.0
        if args.layout == "grid":
            lx, ly = _grid_layout(args.n_turbs, D=D)
        else:
            lx, ly = _row_layout(args.n_turbs, spacing_D=5.0, D=D)
        lx = lx[: results["n_turbs"]]
        ly = ly[: results["n_turbs"]]

        sr_results = run_sr_optimization(
            lx=lx,
            ly=ly,
            wind_rose=wind_rose,
            base_aep=results["base_aep"],
            yaw_max_deg=args.yaw_max,
        )
        _print_comparison_table(results, sr_results)
        if args.plot:
            _plot_comparison(results, sr_results)

    if args.plot:
        plt.ioff()
        plt.show()

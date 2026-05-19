"""
Gradient-based layout optimization using PyTorch autograd through the FLORAF
C++ sequential solver. Exact analytical gradients of AEP with respect to
turbine x/y positions are back-propagated via ``autograd`` and consumed by
an Adam optimizer.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)


### Helpers for setting up the model and gradient optimization

def _get_fmodel(cpp=False, device="cpu") -> FlorisModel:
    """Build a FLORIS model from defaults."""
    fdefaults = FlorisModel.get_defaults()
    fdefaults["logging"]["console"]["enable"] = False
    fdefaults["logging"]["file"]["enable"] = False
    if cpp:
        fdefaults["solver"]["backend"] = "cpp"
        fdefaults["solver"]["device"] = device
        fdefaults["solver"]["cpp_solver"] = "sequential"
    fdefaults["wake"]["enable_secondary_steering"] = False
    fdefaults["wake"]["enable_yaw_added_recovery"] = False
    fdefaults["wake"]["enable_transverse_velocities"] = False
    fdefaults["wake"]["enable_active_wake_mixing"] = False
    fmodel = FlorisModel(fdefaults)
    return fmodel


def _aep_tensor(fmodel, freq_tensor):
    """Return a scalar AEP tensor with grad_fn attached to the layout inputs.

    Expects `fmodel.core._cpp_farm.layout_x` / `fmodel.core._cpp_farm.layout_y` to already be
    set to the `requires_grad=True` tensors before calling this function.
    `fmodel.run()` is called internally.
    """
    fmodel.run()
    power = fmodel.core.get_farm_power_tensor()      # [F, T]
    farm_power_per_findex = power.sum(dim=1)          # [F]
    return (freq_tensor * farm_power_per_findex).sum() * 8760.0


def _wind_rose(n_wdirs=36, n_wspeeds=5, ti=0.06):
    """Synthetic North Sea wind rose.

    Directional distribution has a dominant SW peak (~240°) and a secondary
    NNW peak (~340°).
    Wind speeds follow a Weibull distribution (k=2, c=11 m/s, mean ≈ 9.8 m/s).
    TI is applied uniformly for all bins.
    """
    wd = np.linspace(0.0, 360.0, n_wdirs, endpoint=False)
    ws = np.linspace(4.0, 18.0, n_wspeeds)

    # Directional PDF: dominant SW (~240°), secondary NNW (~340°), broad S shoulder
    dir_freq = (
        0.45 * np.exp(-0.5 * ((wd - 240) % 360 / 38) ** 2)
        # + 0.25 * np.exp(-0.5 * ((wd - 340) % 360 / 30) ** 2)
        # + 0.10 * np.exp(-0.5 * ((wd - 200) % 360 / 25) ** 2)
        + 0.04                                    # uniform background
    )
    dir_freq /= dir_freq.sum()

    # Speed PDF: Weibull (k=2, c=11 m/s) evaluated at bin centres
    k, c = 2.0, 11.0
    spd_freq = (k / c) * (ws / c) ** (k - 1) * np.exp(-(ws / c) ** k)
    spd_freq /= spd_freq.sum()

    # Joint frequency table [n_wdirs, n_wspeeds] (directions independent of speed)
    freq_table = np.outer(dir_freq, spd_freq)
    freq_table /= freq_table.sum()

    return WindRose(
        wind_directions=wd,
        wind_speeds=ws,
        freq_table=freq_table,
        ti_table=ti,
    )


def _grid_layout(n_turbs, xmin, xmax, ymin, ymax, rng):
    """Place turbines on a rough grid and add a jitter."""
    cols = int(np.ceil(np.sqrt(n_turbs)))
    rows = int(np.ceil(n_turbs / cols))
    xs = np.linspace(xmin + 0.1 * (xmax - xmin), xmax - 0.1 * (xmax - xmin), cols)
    ys = np.linspace(ymin + 0.1 * (ymax - ymin), ymax - 0.1 * (ymax - ymin), rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.ravel()[:n_turbs]
    grid_y = grid_y.ravel()[:n_turbs]
    jitter = 0.05 * min(xmax - xmin, ymax - ymin)
    grid_x += rng.uniform(-jitter, jitter, n_turbs)
    grid_y += rng.uniform(-jitter, jitter, n_turbs)
    grid_x = np.clip(grid_x, xmin, xmax)
    grid_y = np.clip(grid_y, ymin, ymax)
    return grid_x, grid_y


def _spacing_penalty(lx, ly, min_dist):
    """Return a non-negative penalty that is zero when all turbines satisfy the
    minimum separation *min_dist* (m).  Uses a smooth quadratic ramp so that
    gradients always exist.
    """
    pos = torch.stack([lx, ly], dim=1)          # [T, 2]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # [T, T, 2]
    dist = diff.pow(2).sum(dim=2).clamp(min=1.0).sqrt()  # [T, T]
    T = lx.shape[0]
    idx_i, idx_j = torch.triu_indices(T, T, offset=1)
    pair_dist = dist[idx_i, idx_j]
    return torch.relu(min_dist - pair_dist).pow(2).sum()


### Live-plot setup

def _setup_figure(xmin, xmax, ymin, ymax, wind_rose, n_turbs, D):
    """Build the three-panel figure and return handles for live updates."""
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        "Gradient-based layout optimization (PyTorch autograd + Adam)",
        fontsize=13,
        fontweight="bold"
    )

    # Three columns: wind rose (narrow) | layout | AEP (equal, wide)
    gs = fig.add_gridspec(1, 3, wspace=0.4, width_ratios=[2, 5, 5])
    ax_rose   = fig.add_subplot(gs[0, 0], projection="polar")
    ax_layout = fig.add_subplot(gs[0, 1])
    ax_aep    = fig.add_subplot(gs[0, 2])

    # Layout panel
    ax_layout.set_aspect("equal")
    ax_layout.set_xlim(xmin - 50, xmax + 50)
    ax_layout.set_ylim(ymin - 50, ymax + 50)
    ax_layout.set_xlabel("x (m)", fontsize=11)
    ax_layout.set_ylabel("y (m)", fontsize=11)
    ax_layout.set_title("Farm layout", fontsize=11)

    # Farm boundary (rectangle)
    rect = plt.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor="#2c7fb8", facecolor="#edf8fb", zorder=0,
    )
    ax_layout.add_patch(rect)

    # Wind rose panel
    # wd = wind_rose.wind_directions
    # freq = wind_rose.freq_table[:, 0]
    # theta = np.radians(90.0 - wd)               # met → math angle
    # ax_rose.bar(
    #     theta,
    #     freq,
    #     width=np.radians(360.0 / len(wd)),
    #     color="#2c7fb8",
    #     alpha=0.7,
    #     edgecolor="none",
    # )
    # ax_rose.set_theta_zero_location("N")
    # ax_rose.set_theta_direction(-1)
    # ax_rose.set_xticks([])
    # ax_rose.set_yticks([])
    # ax_rose.set_title("Wind rose", fontsize=10, pad=8)
    wind_rose.plot(ax=ax_rose) #, color="#2c7fb8", alpha=0.7, edgecolor="none")


    # Trail lines (one per turbine, initially empty)
    trail_colours = plt.cm.tab10(np.linspace(0, 1, n_turbs))
    trail_lines = []
    for k in range(n_turbs):
        (ln,) = ax_layout.plot([], [], "-", color=trail_colours[k], alpha=0.35, lw=1.2)
        trail_lines.append(ln)

    # Turbine scatter (initial position grey, current position coloured)
    init_sc = ax_layout.scatter([], [], s=120, c="grey", zorder=4, label="Initial", marker="o", alpha=0.5)
    curr_sc = ax_layout.scatter([], [], s=160, zorder=5, label="Current", marker="^", edgecolors="k", linewidths=0.8)

    # Rotor diameter reference circle drawn on first turbine
    rotor_patch = plt.Circle((0, 0), D / 2, fill=False, edgecolor="grey", lw=0.8, ls="--", zorder=3)
    ax_layout.add_patch(rotor_patch)
    ax_layout.text(0.02, 0.02, f"D = {D:.0f} m", transform=ax_layout.transAxes, fontsize=8, color="grey")
    ax_layout.legend(loc="lower right", fontsize=9, framealpha=0.8)

    # AEP panel
    ax_aep.set_xlabel("Iteration", fontsize=11)
    ax_aep.set_ylabel("AEP improvement (%)", fontsize=11)
    ax_aep.set_title("optimization progress", fontsize=11)
    ax_aep.grid(True, alpha=0.4)
    (aep_line,) = ax_aep.plot([], [], "o-", color="#2c7fb8", ms=3, lw=1.5, label="Gradient (Adam)")
    ax_aep.axhline(0, color="grey", ls="--", lw=1)
    ax_aep.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.pause(0.05)

    handles = dict(
        fig=fig,
        ax_layout=ax_layout,
        ax_aep=ax_aep,
        trail_lines=trail_lines,
        init_sc=init_sc,
        curr_sc=curr_sc,
        rotor_patch=rotor_patch,
        aep_line=aep_line,
        trail_hist=[[] for _ in range(n_turbs)],   # list of (x, y) per turbine
        trail_colours=trail_colours,
    )
    return handles


def _update_figure(handles, lx, ly, lx_init, ly_init, aep_history, base_aep, D):
    """Refresh the live figure in-place."""
    lx_np = lx.detach().numpy()
    ly_np = ly.detach().numpy()

    # Update trails
    for k in range(len(lx_np)):
        handles["trail_hist"][k].append((lx_np[k], ly_np[k]))
        xs, ys = zip(*handles["trail_hist"][k])
        handles["trail_lines"][k].set_data(xs, ys)

    # Update turbine positions
    handles["init_sc"].set_offsets(np.column_stack([lx_init, ly_init]))
    handles["curr_sc"].set_offsets(np.column_stack([lx_np, ly_np]))
    handles["curr_sc"].set_facecolor(handles["trail_colours"])

    # Update rotor reference patch position (first turbine)
    handles["rotor_patch"].center = (lx_np[0], ly_np[0])

    # Update AEP curve
    iters = np.arange(len(aep_history))
    improvement = 100.0 * (np.array(aep_history) / base_aep - 1.0)
    handles["aep_line"].set_data(iters, improvement)
    ax = handles["ax_aep"]
    ax.set_xlim(-1, max(len(aep_history) + 1, 10))
    pad = max(abs(improvement).max() * 0.15, 0.5)
    ax.set_ylim(min(improvement.min() - pad, -0.5), improvement.max() + pad)

    handles["fig"].canvas.draw()
    handles["fig"].canvas.flush_events()
    plt.pause(0.001)


### Gradient descent

def run_gradient_optimization(
    n_turbs: int,
    wind_rose: WindRose,
    n_iters: int,
    lr: float,
    live_plot: bool = True,
    seed: int = 42,
    device: str = "cpu",
):
    """Run gradient-based layout optimization and return results dict."""

    rng = np.random.default_rng(seed)
    D = 126.0           # NREL 5 MW rotor diameter (m)
    min_dist = 2 * D    # minimum turbine spacing

    # Farm boundary
    xmin, xmax = 0.0, 2000.0
    ymin, ymax = 0.0, 1500.0

    # Wind resource
    freq_np = wind_rose.unpack_freq()                 # [F], sums to ≈1
    freq_t = torch.tensor(freq_np, dtype=torch.float32, device=device)  # [F]

    # Initial layout: grid + jitter (stays within boundaries)
    lx_init_np, ly_init_np = _grid_layout(n_turbs, xmin, xmax, ymin, ymax, rng)

    # FLORIS model with FLORAF backend
    fmodel = _get_fmodel(cpp=True, device=device)
    fmodel.set(
        layout_x=lx_init_np.tolist(),
        layout_y=ly_init_np.tolist(),
        wind_data=wind_rose,
    )

    # Optimization variables - leaf tensors for layout positions
    lx = torch.tensor(lx_init_np, dtype=torch.float32, device=device, requires_grad=True)
    ly = torch.tensor(ly_init_np, dtype=torch.float32, device=device, requires_grad=True)

    # Plant the grad-tracked tensors into the FLORAF farm once.
    fmodel.core._cpp_farm.layout_x = lx
    fmodel.core._cpp_farm.layout_y = ly

    # Baseline AEP
    with torch.no_grad():
        base_aep_t = _aep_tensor(fmodel, freq_t)
    base_aep = base_aep_t.item()
    print(f"\nInitial AEP : {base_aep/1e9:8.4f} GWh/yr")

    # Penalty scale, this weights the spacing penalty based on the severity of the incursion
    # into the min_dist constraint.
    # The penalty is zero when all pairs are above the minimum distance.
    # It grows quadratically as pairs get closer than min_dist.
    # The coefficient is chosen so that a ~1 D spacing violation produces a penalty comparable
    # to a ~0.1% AEP step.
    spacing_coeff = 1e4

    # Adam optimizer
    optimizer = torch.optim.Adam([lx, ly], lr=lr)

    if live_plot:
        plt.ion()
        fig_handles = _setup_figure(
            xmin, xmax, ymin, ymax, wind_rose, n_turbs, D
        )
        _update_figure(fig_handles, lx, ly, lx_init_np, ly_init_np, [base_aep], base_aep, D)

    aep_history  = [base_aep]
    t0 = time.perf_counter()

    for step in range(1, n_iters + 1):
        optimizer.zero_grad()

        # Forward: AEP through the C++ solver (grad_fn preserved)
        aep = _aep_tensor(fmodel, freq_t)
        
        # Penalty: minimum-spacing constraint
        penalty = spacing_coeff * _spacing_penalty(lx, ly, min_dist)

        # Maximise AEP via minimise negative AEP + penalty
        loss = -aep + penalty
        loss.backward()
        optimizer.step()

        # Hard boundary projection: clamp positions back into the domain
        with torch.no_grad():
            lx.clamp_(xmin, xmax)
            ly.clamp_(ymin, ymax)
            # Zero out gradients for clamped dims to avoid bias buildup
            lx.grad.zero_()
            ly.grad.zero_()

        current_aep = aep.item()
        aep_history.append(current_aep)

        improvement = 100.0 * (current_aep / base_aep - 1.0)
        print(f"  Step {step:4d}/{n_iters} | AEP = {current_aep/1e9:.4f} GWh/yr | Δ = {improvement:+.2f}%")

        if live_plot:
            _update_figure(fig_handles, lx, ly, lx_init_np, ly_init_np, aep_history, base_aep, D)

    elapsed = time.perf_counter() - t0

    # Final positions and AEP
    final_lx = lx.detach().cpu().numpy().copy()
    final_ly = ly.detach().cpu().numpy().copy()
    final_aep = aep_history[-1]

    print(f"\nGradient opt finished in {elapsed:.1f} s  ({n_iters} iterations)")
    print(f"Final AEP   : {final_aep/1e9:8.4f} GWh/yr ({100*(final_aep/base_aep - 1):+.2f}% vs initial)\n")

    results = dict(
        init_lx=lx_init_np,
        init_ly=ly_init_np,
        final_lx=final_lx,
        final_ly=final_ly,
        base_aep=base_aep,
        final_aep=final_aep,
        aep_history=aep_history,
        elapsed=elapsed,
        n_iters=n_iters,
        live_plot=live_plot,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
    )

    if live_plot:
        results["fig_handles"] = fig_handles

    return results


### Scipy SLSQP

def run_scipy_optimization(
    init_lx,
    init_ly,
    wind_rose,
    base_aep: float,
    xmin: float = 0.0,
    xmax: float = 2000.0,
    ymin: float = 0.0,
    ymax: float = 1500.0,
):
    """Run LayoutOptimizationScipy on the same problem and return results."""
    D   = 126.0
    boundaries = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]

    # Use the standard Python FLORIS backend (GCH model, no C++ extension)
    fmodel = _get_fmodel(cpp=False)
    fmodel.set(
        layout_x=init_lx.tolist(),
        layout_y=init_ly.tolist(),
        wind_data=wind_rose,
    )

    opt_options = {
        "maxiter": 100,     # same iteration budget as a generous cap
        "disp": True,
        "iprint": 1,
        "ftol": 1e-9,
        "eps": 0.1 * D,     # FD step ≈ 10 % of rotor diameter
    }

    layout_opt = LayoutOptimizationScipy(
        fmodel,
        boundaries,
        min_dist=2 * D,
        optOptions=opt_options,
    )

    print("Running scipy SLSQP optimization")

    t0 = time.perf_counter()
    sol = layout_opt.optimize()
    elapsed = time.perf_counter() - t0

    # Evaluate the discovered layout
    fmodel.set(layout_x=sol[0], layout_y=sol[1])
    fmodel.run()
    final_aep_scipy = fmodel.get_farm_AEP()

    # _aep_record stores -farm_aep/initial_aep at every objective evaluation.
    # Convert to improvement % for plotting: pct = (-val - 1) * 100
    aep_record_raw = np.array(layout_opt._aep_record)
    scipy_aep_improvement = (-aep_record_raw - 1.0) * 100.0

    print(f"Scipy opt finished in {elapsed:.1f} s")
    print(f"Final AEP (scipy): {final_aep_scipy/1e9:.4f} GWh/yr ({100*(final_aep_scipy/base_aep - 1):+.2f}% vs initial)\n")

    return dict(
        final_lx=np.array(sol[0]),
        final_ly=np.array(sol[1]),
        base_aep=base_aep,
        final_aep=final_aep_scipy,
        elapsed=elapsed,
        n_func_evals=layout_opt.residual_plant.nfev,
        aep_improvement=scipy_aep_improvement,
    )


### Results comparison

def _print_comparison_table(grad_res, scipy_res):
    """
    Print a formatted side-by-side results table.
    """
    g = grad_res
    s = scipy_res

    print("\n" + "=" * 62)
    print(f"{'':30s}  {'Gradient (Adam)':>15}  {'Scipy (SLSQP)':>13}")
    print("-" * 62)
    print(f"{'Iterations':30s}  {g['n_iters']:>15d}  {'–':>13}")
    print(f"{'Function evaluations':30s}  {g['n_iters']:>15d}  {s['n_func_evals']:>13d}")
    print(f"{'Wall time (s)':30s}  {g['elapsed']:>15.1f}  {s['elapsed']:>13.1f}")
    print(f"{'Initial AEP':30s}  {g['base_aep']/1e9:>15.4f}  {'(same)':>13}")
    print(f"{'Final AEP':30s}  {g['final_aep']/1e9:>15.4f}  {s['final_aep']/1e9:>13.4f}  GWh/yr")
    pct_g = 100 * (g['final_aep'] / g['base_aep'] - 1)
    pct_s = 100 * (s['final_aep'] / s['base_aep'] - 1)
    print(f"{'AEP improvement (%)':30s}  {pct_g:>+15.2f}  {pct_s:>+13.2f}  %")
    print("=" * 62)


def _plot_comparison(grad_res, scipy_res, wind_rose, xmin, xmax, ymin, ymax):
    """Side-by-side final layout + AEP progress comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Layout optimization comparison: autograd vs finite-difference", fontsize=13, fontweight="bold")

    colours = {"grad": "#2c7fb8", "scipy": "#d7191c", "init": "grey"}

    for ax, res, label, col in [
        (axes[0], grad_res, "Gradient (Adam)", colours["grad"]),
        (axes[1], scipy_res, "Scipy (SLSQP)", colours["scipy"]),
    ]:
        ax.set_aspect("equal")
        ax.set_xlim(xmin - 80, xmax + 80)
        ax.set_ylim(ymin - 80, ymax + 80)
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="#2c7fb8", facecolor="#edf8fb", zorder=0,
        )
        ax.add_patch(rect)
        ax.scatter(grad_res["init_lx"], grad_res["init_ly"], s=90, c=colours["init"], label="Initial", zorder=3, alpha=0.6)
        ax.scatter(res["final_lx"], res["final_ly"], s=140, c=col, marker="^", edgecolors="k", lw=0.8, label="Optimal", zorder=4)
        # Arrows from initial → final
        for x0, y0, xf, yf in zip(
            grad_res["init_lx"], grad_res["init_ly"],
            res["final_lx"], res["final_ly"],
        ):
            ax.annotate("", xy=(xf, yf), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.6))
        pct = 100 * (res["final_aep"] / res["base_aep"] - 1)
        ax.set_title(f"{label}\n+{pct:.2f}% AEP  |  {res['elapsed']:.0f} s", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlabel("x (m)", fontsize=10)
        ax.set_ylabel("y (m)", fontsize=10)
        ax.grid(True, alpha=0.3)

    # AEP-progress panel — gradient per-step + scipy per-evaluation curves
    ax = axes[2]
    hist = np.array(grad_res["aep_history"])
    improvement = 100.0 * (hist / grad_res["base_aep"] - 1.0)
    ax.plot(
        np.arange(len(improvement)),
        improvement,
        "o-",
        color=colours["grad"],
        ms=2,
        lw=1.5,
        label="Gradient (Adam)"
    )

    # Scipy convergence: x = function evaluation index, y = AEP improvement %
    scipy_impr = scipy_res["aep_improvement"]
    ax.plot(
        np.arange(len(scipy_impr)),
        scipy_impr,
        "-",
        color=colours["scipy"], lw=1.5, alpha=0.85, label="Scipy (SLSQP)"
    )

    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set_xlabel("Evaluations (grad steps / scipy func calls)", fontsize=10)
    ax.set_ylabel("AEP improvement (%)", fontsize=11)
    ax.set_title("AEP convergence", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    # plt.savefig("gradient_layout_opt_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run scipy SLSQP and compare results.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable live plotting.")
    parser.add_argument("--iters", type=int, default=200, help="Number of Adam gradient steps (default: 200).")
    parser.add_argument("--lr", type=float, default=25.0, help="Adam learning rate in metres (default: 25.0).")
    parser.add_argument("--n-turbs", type=int, default=6, help="Number of turbines (default: 6).")
    parser.add_argument("--n-wdirs", type=int, default=36, help="Number of wind directions in wind rose (default: 36).")
    parser.add_argument("--n-wspeeds", type=int, default=5, help="Number of wind speed bins in wind rose (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initial layout jitter (default: 42).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device for C++ solver (default: cpu). Use 'cuda' if torch.cuda.is_available().")
    args = parser.parse_args()

    print("=" * 62)
    print("  Gradient-based wind farm layout optimization")
    print("=" * 62)
    print(f"  Turbines      : {args.n_turbs}")
    print(f"  Wind dirs     : {args.n_wdirs}")
    print(f"  Wind speeds   : {args.n_wspeeds}")
    print(f"  Adam steps    : {args.iters}")
    print(f"  Adam lr       : {args.lr} m")
    print(f"  Live plot     : {'yes' if args.plot else 'no'}")
    print(f"  Scipy compare : {'yes' if args.compare else 'no'}")
    print()

    wind_rose = _wind_rose(args.n_wdirs, args.n_wspeeds)

    # Gradient optimization
    grad_res = run_gradient_optimization(
        n_turbs=args.n_turbs,
        wind_rose=wind_rose,
        n_iters=args.iters,
        lr=args.lr,
        live_plot=args.plot,
        seed=args.seed,
        device=args.device,
    )

    pct = 100 * (grad_res["final_aep"] / grad_res["base_aep"] - 1)
    print(f"AEP improvement: {pct:+.2f}%")

    # Scipy optimization
    if args.compare:
        scipy_res = run_scipy_optimization(
            init_lx=grad_res["init_lx"],
            init_ly=grad_res["init_ly"],
            wind_rose=wind_rose,
            base_aep=grad_res["base_aep"],
            xmin=grad_res["xmin"], xmax=grad_res["xmax"],
            ymin=grad_res["ymin"], ymax=grad_res["ymax"],
        )
        _print_comparison_table(grad_res, scipy_res)
        _plot_comparison(
            grad_res,
            scipy_res,
            wind_rose,
            xmin=grad_res["xmin"], xmax=grad_res["xmax"],
            ymin=grad_res["ymin"], ymax=grad_res["ymax"],
        )

    if args.plot:
        # handles = grad_res["fig_handles"]
        # pct = 100 * (grad_res["final_aep"] / grad_res["base_aep"] - 1)
        # handles["fig"].suptitle(
        #     f"Gradient-based layout optimization  |  AEP {pct:+.2f}%  |  {grad_res['elapsed']:.0f} s",
        #     fontsize=12,
        #     fontweight="bold",
        # )
        # handles["fig"].savefig("gradient_layout_opt.png", dpi=150, bbox_inches="tight")
        plt.ioff()
        plt.show()

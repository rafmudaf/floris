# FLORAF

FLORAF is a compiled backend alternative that plugs into FLORIS to enable automatic gradients
and GPU acceleration with Torch.

**Installation**

Using Python 3.10 - 3.13, install "floraf": `pip install floraf`.

This will install the appropriate binary wheel which includes:
- Compiled C++ backend
- FLORAF Python code that interfaces the C++ backend to FLORIS
- Fork of FLORIS that includes an interface to the FLORAF backend

The FLORIS fork is still installed as "floris" and it is forked from v4.6.4.
You can import FLORIS and use it the same as NREL/FLORIS v4.6.4.

**License**

Add your license string to one of the following options.

Environment variable:
`export FLORAF_LICENSE_KEY=license_key`

File:
Either set the `FLORAF_LICENSE_FILE` environment variable to the path to the license file,
or create set the license key in `~/.floraf/license.key`.
The license file should contain only the license key.

**Usage**

A comparison with the default SLSQP-based layout optimization is available at
`examples/gradient_layout_optimization.py`.
Below is a reduced form of that example.

```python

import time

import numpy as np
import torch

from floris import FlorisModel, WindRose


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

    Requires `fmodel.core._cpp_farm.layout_x` / `fmodel.core._cpp_farm.layout_y` to already be
    set to the `requires_grad=True` tensors before calling this function.
    """
    fmodel.run()
    power = fmodel.core.get_farm_power_tensor()      # [F, T]
    farm_power_per_findex = power.sum(dim=1)          # [F]
    return (freq_tensor * farm_power_per_findex).sum() * 8760.0


def _wind_rose(n_wdirs, n_wspeeds, ti):
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


### Gradient descent

def run_gradient_optimization(
    n_turbs: int,
    wind_rose: WindRose,
    n_iters: int,
    lr: float,
    seed: int = 42,
):
    """Run gradient-based layout optimization."""
    rng = np.random.default_rng(seed)
    D = 126.0           # NREL 5 MW rotor diameter (m)
    min_dist = 2 * D    # minimum turbine spacing

    # Farm boundary
    xmin, xmax = 0.0, 2000.0
    ymin, ymax = 0.0, 1500.0

    # Wind resource
    freq_np = wind_rose.unpack_freq()                 # [F], sums to ≈1
    freq_t = torch.tensor(freq_np, dtype=torch.float32)

    # Initial layout: grid + jitter (stays within boundaries)
    lx_init_np, ly_init_np = _grid_layout(n_turbs, xmin, xmax, ymin, ymax, rng)

    # FLORIS model with FLORAF backend
    fmodel = _get_fmodel(cpp=True, device="cpu")
    fmodel.set(
        layout_x=lx_init_np.tolist(),
        layout_y=ly_init_np.tolist(),
        wind_data=wind_rose,
    )

    # Optimization variables - leaf tensors for layout positions
    lx = torch.tensor(lx_init_np, dtype=torch.float32, requires_grad=True)
    ly = torch.tensor(ly_init_np, dtype=torch.float32, requires_grad=True)

    # Plant the grad-tracked tensors into the FLORAF farm once
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

    elapsed = time.perf_counter() - t0
    final_aep = aep_history[-1]
    print(f"\nGradient opt finished in {elapsed:.1f} s  ({n_iters} iterations)")
    print(f"Final AEP   : {final_aep/1e9:8.4f} GWh/yr ({100*(final_aep/base_aep - 1):+.2f}% vs initial)\n")

    return aep_history

if __name__ == "__main__":

    N_TURBS = 6
    N_WDIRS = 36
    N_WSPEEDS = 5
    N_ITERATIONS = 100
    LEARNING_RATE = 25.0    # meters; this is the step size for each iteration
    SEED = 42

    wind_rose = _wind_rose(N_WDIRS, N_WSPEEDS, ti=0.06)

    aep_history = run_gradient_optimization(
        n_turbs=N_TURBS,
        wind_rose=wind_rose,
        n_iters=N_ITERATIONS,
        lr=LEARNING_RATE,
        seed=SEED,
    )

    pct = 100 * (aep_history[-1] / aep_history[0] - 1)
    print(f"AEP improvement: {pct:+.2f}%")
```

# FLORIS Wake Modeling and Wind Farm Controls Software

FLORIS is a controls-focused wind farm simulation software incorporating
steady-state engineering wake models into a performance-focused Python
framework. It has been in active development at NLR since 2013 and the latest
release is [FLORIS v.4.6.4](https://github.com/NatLabRockies/floris/releases/latest).
Online documentation is available at https://natlabrockies.github.io/floris.

The software is in active development and engagement with the development team
is highly encouraged. If you are interested in using FLORIS to conduct studies
of a wind farm or extending FLORIS to include your own wake model, please join
the conversation in [GitHub Discussions](https://github.com/NatLabRockies/floris/discussions/)!

## WETO software

FLORIS is primarily developed with the support from the U.S. Department of Energy and
is part of the [WETO Software Stack](https://natlabrockies.github.io/WETOStack).
For more information and other integrated modeling software, see:

- [Portfolio Overview](https://natlabrockies.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://natlabrockies.github.io/WETOStack/_static/entry_guide/index.html)
- [Wind Farm Controls Workshop](https://www.youtube.com/watch?v=f-w6whxIBrA&list=PL6ksUtsZI1dwRXeWFCmJT6cEN1xijsHJz)

## Installation

**If upgrading from a previous version, it is recommended to install FLORIS v4 into a new virtual environment**.
If you intend to use [pyOptSparse](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/) with FLORIS,
it is recommended to install that package first before installing FLORIS.

FLORIS can be installed by downloading the source code or via the PyPI
package manager with `pip`.

The simplest method is with `pip` by using this command:

```bash
pip install floris
```

Developers and anyone who intends to inspect the source code
can install FLORIS by downloading the git repository
from GitHub with ``git`` and use ``pip`` to locally install it.
It is highly recommended to use a Python virtual environment manager
such as [conda](https://docs.conda.io/en/latest/miniconda.html)
in order to maintain a clean and sandboxed environment. The following
commands in a terminal or shell will download and install FLORIS.

```bash
    # Download the source code from the `main` branch
    git clone -b main https://github.com/NatLabRockies/floris.git

    # If using conda, be sure to activate your environment prior to installing
    # conda activate <env name>

    # If using pyOptSpare, install it first
    conda install -c conda-forge pyoptsparse

    # Install FLORIS
    pip install -e floris
```

With both methods, the installation can be verified by opening a Python interpreter
and importing FLORIS:

```python
>>> import floris
>>> help(floris)

Help on package floris:

NAME
    floris

PACKAGE CONTENTS
    convert_floris_input_v3_to_v4
    convert_turbine_v3_to_v4
    core (package)
    cut_plane
    floris_model
    flow_visualization
    layout_visualization
    logging_manager
    optimization (package)
    parallel_floris_model
    turbine_library (package)
    type_dec
    uncertain_floris_model
    utilities
    version
    wind_data

VERSION
    4.6.4

FILE
    ~/floris/floris/__init__.py
```

It is important to regularly check for new updates and releases as new
features, improvements, and bug fixes will be issued on an ongoing basis.

## Quick Start

FLORIS is a Python package run on the command line typically by providing
an input file with an initial configuration. It can be installed with
```pip install floris``` (see [installation](https://natlabrockies.github.io/floris/installation.html)).
The typical entry point is
[FlorisModel](https://natlabrockies.github.io/floris/_autosummary/floris.floris_model.html)
which accepts the path to the input file as an argument. From there,
changes can be made to the initial configuration through the
[FlorisModel.set](https://natlabrockies.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.set)
routine, and the simulation is executed with
[FlorisModel.run](https://natlabrockies.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.run).

```python
from floris import FlorisModel
fmodel = FlorisModel("path/to/input.yaml")
fmodel.set(
    wind_directions=[i for i in range(10)],
    wind_speeds=[8.0]*10,
    turbulence_intensities=[0.06]*10
)
fmodel.run()
```

Finally, results can be analyzed via post-processing functions available within
[FlorisModel](https://natlabrockies.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel)
such as
- [FlorisModel.get_turbine_layout](https://natlabrockies.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.get_turbine_layout)
- [FlorisModel.get_turbine_powers](https://natlabrockies.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.get_turbine_powers)
- [FlorisModel.get_farm_AEP](https://natlabrockies.github.io/floris/_autosummary/floris.floris_model.html#floris.floris_model.FlorisModel.get_farm_AEP)

and in two visualization packages: [layoutviz](https://natlabrockies.github.io/floris/_autosummary/floris.layout_visualization.html) and [flowviz](https://natlabrockies.github.io/floris/_autosummary/floris.flow_visualization.html).
A collection of examples describing the creation of simulations as well as
analysis and post processing are included in the
[repository](https://github.com/NatLabRockies/floris/tree/main/examples). Examples are also listed
in the [online documentation](https://natlabrockies.github.io/floris/examples/001_opening_floris_computing_power.html).

## Engaging on GitHub

FLORIS leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NatLabRockies/floris/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NatLabRockies/floris/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/96): Include current and future work on a timeline and assign a person to "own" it

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NatLabRockies/floris/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NatLabRockies/floris/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NatLabRockies/floris/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.


# License

BSD 3-Clause License

Copyright (c) 2025, Alliance for Energy Innovation LLC, All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

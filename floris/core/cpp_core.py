"""
cpp_core.py — Phase 8: CppCore Python wrapper
==============================================

Drop-in replacement for ``floris.core.core.Core`` backed by the C++ libtorch
solver extension.  ``CppCore`` implements the exact Core interface surface that
``FlorisModel`` uses so that swapping backends requires no changes to
``FlorisModel`` beyond passing ``backend="cpp"`` at construction.

Architecture
------------
``CppCore`` maintains two parallel representations:

* ``_py_core``  — a full Python ``Core`` instance, used for all attribute access
  (power/thrust functions, turbine type maps, ``as_dict()`` / ``from_dict()``
  round-trips, AWC/operating-setpoint storage, etc.).

* ``_cpp_farm``, ``_cpp_flow_field``, ``_cpp_grid`` — C++ struct instances
  (from ``floris_cpp``) that hold the tensors used for computation.

Calling ``initialize_domain()`` syncs mutable state (yaw angles, layout, wind
conditions) from the Python Core into the C++ structs before running the C++
initialization kernels.  After ``finalize()``, results (``u``,
``turbulence_intensity_field``) are synced back so that ``FlorisModel``'s
power-computation path — which reads from ``self.core.flow_field.u`` — works
unchanged.

Supported
---------
``run()``, ``run_no_wake()``, ``get_turbine_powers()`` and all gradient-based
optimisation inputs (layout_x, yaw_angles fully differentiable through the C++
wavefront solver).

Not supported
-------------
``solve_for_viz``, ``solve_for_points``,
``solve_for_velocity_deficit_profiles`` — these raise ``NotImplementedError``.
Use ``FlorisModel(backend='python')`` for flow-field visualisation.

Phase 8 limitations
-------------------
* Heterogeneous turbine farms: the C++ solver uses the first power/thrust table
  for **all** turbines.  Full per-turbine table dispatch is a Phase 11 task.
* Floating / tilt effects: not forwarded to the C++ backend.
* GPU device support: ``device`` is forwarded to all tensor allocations.
  Pass ``device="cuda"`` or ``device="mps"`` to run computation on GPU.
  All wake kernels use ``CompositeImplicitAutograd`` and dispatch automatically.
  Note: ``compute_wave_layers()`` (wavefront solver only) always uses CPU
  for its sort-index accessors; wave-layer computation itself is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from floris.core.base import State
from floris.core.core import Core
from floris.utilities import load_yaml


# ---------------------------------------------------------------------------
# Module-level import of the C++ extension (deferred to avoid circular imports
# at the top of floris/__init__.py; floris_cpp is large and only needed when
# the cpp backend is actually requested).
# ---------------------------------------------------------------------------

def _to_f32(x, device="cpu"):
    """
    Convert *x* to a float32 ``torch.Tensor`` on *device*.

    Accepts both ``numpy.ndarray`` (v4.6.4 Python core) and
    ``torch.Tensor``.  The autograd graph is preserved so that inputs
    carrying ``requires_grad=True`` remain connected to the solver output,
    enabling gradient-based layout and yaw optimisation.
    """
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(np.asarray(x, dtype=np.float32)).to(device)


def _get_tilt_angles(py_farm):
    """
    Return the tilt_angles array from *py_farm*, or a ref_tilt-based fallback.

    ``tilt_angles`` is populated by ``set_tilt_to_ref_tilt()`` which is called
    from ``Core.initialize_domain()`` just before the wake solve.  If it has
    not been called yet (e.g. during CppCore construction), we fall back to
    broadcasting ``ref_tilts`` so that cos(tilt)/cos(ref_tilt) = 1.

    Returns None only if neither attribute is available (unlikely in practice).
    """
    tilt = getattr(py_farm, 'tilt_angles', None)
    tilt_arr = np.asarray(tilt) if tilt is not None else np.array([])
    if tilt_arr.ndim == 2 and tilt_arr.shape[1] > 0:
        return tilt_arr

    # Fallback: broadcast ref_tilts to [n_findex, n_turbines]
    ref = getattr(py_farm, 'ref_tilts', None)
    if ref is not None:
        ref_arr = np.asarray(ref)
        if ref_arr.size > 0:
            yaw_arr = np.asarray(getattr(py_farm, 'yaw_angles', np.zeros((1, 1))))
            n_fi = yaw_arr.shape[0] if yaw_arr.ndim >= 2 else 1
            return np.broadcast_to(ref_arr, (n_fi, ref_arr.size)).copy()
    return None


def _get_floris_cpp(solver=None, wake_model=None):
    """
    Import and return the floris_cpp extension.

    Tries ``floraf`` (the licensed distribution) first.  If floraf is installed
    and a valid license is present, ``get_licensed_floris_cpp`` returns an
    extension proxy that enforces feature and size limits.  Any
    ``LicenseError`` raised by floraf propagates directly to the caller — it
    is never silently swallowed.

    """
    try:
        from floraf import get_licensed_floris_cpp
        return get_licensed_floris_cpp(solver=solver, wake_model=wake_model)
    except ImportError as exc:
        raise ImportError(
            "The floris_cpp C++ extension is not available.  "
            "Install the floraf package:\n"
            "    pip install floraf\n"
            "or set backend: python in the solver block to use the pure-Python solver."
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_turbine_tables(floris_cpp, cpp_farm, py_farm, device="cpu"):
    """
    Populate ``cpp_farm.turbine_tables`` from the Python farm's
    ``turbine_power_thrust_tables`` dict.

    Phase 8 note: one table per *unique* turbine type is added.  For
    heterogeneous farms the C++ solver uses index 0 for all turbines; full
    per-turbine dispatch is deferred to Phase 11.
    """
    added_types: set[str] = set()
    for turb_type, tbl_dict in py_farm.turbine_power_thrust_tables.items():
        if turb_type in added_types:
            continue
        added_types.add(turb_type)
        tbl = floris_cpp.PowerThrustTable()
        tbl.wind_speed          = _to_f32(tbl_dict["wind_speed"], device)
        tbl.thrust_coefficient  = _to_f32(tbl_dict["thrust_coefficient"], device)
        tbl.power               = _to_f32(tbl_dict["power"], device)
        # ref_tilt: used for CosineLossTurbine CT correction CT *= cos(yaw)*cos(tilt)/cos(ref_tilt)
        tbl.ref_tilt = float(tbl_dict.get("ref_tilt", 5.0))
        floris_cpp.farm_add_turbine_table(cpp_farm, tbl)


def _build_model_config(floris_cpp, d: dict, cpp_solver_paradigm: str, device: str):
    """
    Construct a ``ModelConfig`` from the YAML dict ``d``.

    All wake-model hyperparameters are read from the nested ``wake.*``
    sub-dicts using the same key conventions as the Python ``WakeModelManager``.
    """
    cfg = floris_cpp.ModelConfig()

    # Solver identifiers
    cfg.cpp_solver_paradigm = cpp_solver_paradigm
    cfg.device = device

    # Grid
    solver_d = d.get("solver", {})
    cfg.turbine_grid_points = int(solver_d.get("turbine_grid_points", 3))

    # Wake model strings
    wake_d = d.get("wake", {})
    model_strings = wake_d.get("model_strings", {})
    cfg.velocity_model   = model_strings.get("velocity_model",   "gauss")
    cfg.deflection_model = model_strings.get("deflection_model", "gauss")
    cfg.turbulence_model = model_strings.get("turbulence_model", "crespo_hernandez")
    cfg.combination_model = model_strings.get("combination_model", "sosfs")

    # Optional physics flags
    cfg.enable_secondary_steering    = bool(wake_d.get("enable_secondary_steering", False))
    cfg.enable_yaw_added_recovery    = bool(wake_d.get("enable_yaw_added_recovery", False))

    # Gaussian velocity parameters
    vp = wake_d.get("wake_velocity_parameters", {}).get("gauss", {})
    cfg.alpha = float(vp.get("alpha", 0.58))
    cfg.beta  = float(vp.get("beta",  0.077))
    cfg.ka    = float(vp.get("ka",    0.38))
    cfg.kb    = float(vp.get("kb",    0.004))

    # Gaussian deflection parameters
    dp = wake_d.get("wake_deflection_parameters", {}).get("gauss", {})
    cfg.alpha = float(dp.get("alpha", cfg.alpha))   # deflection also carries α/β/ka/kb
    cfg.beta  = float(dp.get("beta",  cfg.beta))
    cfg.ka    = float(dp.get("ka",    cfg.ka))
    cfg.kb    = float(dp.get("kb",    cfg.kb))
    cfg.ad = float(dp.get("ad", 0.0))
    cfg.bd = float(dp.get("bd", 0.0))
    cfg.dm = float(dp.get("dm", 1.0))

    # Crespo-Hernandez turbulence parameters
    tp = wake_d.get("wake_turbulence_parameters", {}).get("crespo_hernandez", {})
    cfg.ch_initial    = float(tp.get("initial",    0.1))
    cfg.ch_constant   = float(tp.get("constant",   0.5))
    cfg.ch_ai_exp     = float(tp.get("ai",         0.8))
    cfg.ch_downstream = float(tp.get("downstream", -0.32))

    # sigmoid_k uses the compiled default (10.0 from model_config.hpp);
    # no standard YAML key — override here if needed in the future.

    # Jacobi parallel solver parameters
    cfg.jacobi_max_iters   = int(solver_d.get("jacobi_max_iters",   30))
    cfg.jacobi_chunk_size  = int(solver_d.get("jacobi_chunk_size",  8))
    cfg.jacobi_tol         = float(solver_d.get("jacobi_tol",       1e-6))
    cfg.jacobi_fixed_iters = bool(solver_d.get("jacobi_fixed_iters", False))

    return cfg


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CppCore:
    """
    Drop-in replacement for ``floris.core.core.Core`` backed by the C++
    libtorch solver.

    Construction is via the class methods ``from_dict`` and ``from_file``,
    matching the ``Core`` API that ``FlorisModel`` uses.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        device: str = "cpu",
        cpp_solver_paradigm: str = "wavefront"
    ) -> "CppCore":
        """Load from a YAML file path, identical signature to Core.from_file."""
        d = load_yaml(Path(path).resolve())
        return cls.from_dict(d, device=device, cpp_solver_paradigm=cpp_solver_paradigm)

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        device: str = "cpu",
        cpp_solver_paradigm: str = "wavefront"
    ) -> "CppCore":
        """
        Create a ``CppCore`` from the standard FLORIS YAML dict.

        Parameters
        ----------
        d:
            Parsed YAML dict (same schema as accepted by ``Core.from_dict``).
        device:
            Torch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
            All tensors passed to the C++ solver (farm layout, wind conditions,
            turbine tables, grid coordinates) are placed on this device before
            being forwarded to the C++ structs.
        cpp_solver_paradigm:
            Key into the C++ solver registry, e.g. ``"wavefront"``.
        """
        # Resolve the velocity_model for the license feature check.
        wake_model = (
            d.get("wake", {})
             .get("model_strings", {})
             .get("velocity_model", "gauss")
        )
        floris_cpp = _get_floris_cpp(solver=cpp_solver_paradigm, wake_model=wake_model)

        # Validate solver key early so errors surface at init time, not run().
        available = floris_cpp.available_solvers()
        if cpp_solver_paradigm not in available:
            raise ValueError(
                f"Unknown cpp_solver_paradigm '{cpp_solver_paradigm}'. "
                f"Available C++ solvers: {available}"
            )

        obj = cls.__new__(cls)
        obj._floris_cpp = floris_cpp
        obj._device = device
        obj._cpp_solver_paradigm = cpp_solver_paradigm

        # Python Core — owns all Python-specific state (power/thrust functions,
        # turbine type maps, AWC/setpoint arrays, as_dict() serialisation, etc.)
        obj._py_core = Core.from_dict(d)

        # C++ ModelConfig — wake hyperparameters + solver selection
        obj._model_config = _build_model_config(
            floris_cpp,
            d,
            cpp_solver_paradigm,
            device
        )

        # Build C++ structs from Python Core initial state
        obj._cpp_farm = cls._make_cpp_farm(floris_cpp, obj._py_core.farm, device)
        obj._cpp_flow_field = cls._make_cpp_flow_field(obj._py_core.flow_field, device)
        obj._cpp_grid = cls._make_cpp_grid(
            obj._py_core.farm, obj._py_core.flow_field, obj._py_core.solver, device
        )

        return obj

    # ------------------------------------------------------------------
    # Private: C++ struct factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cpp_farm(floris_cpp, py_farm, device="cpu") -> object:
        f = floris_cpp.Farm()
        f.layout_x        = _to_f32(py_farm.layout_x, device)
        f.layout_y        = _to_f32(py_farm.layout_y, device)
        f.hub_heights     = _to_f32(py_farm.hub_heights, device)
        f.rotor_diameters = _to_f32(py_farm.rotor_diameters, device)
        f.yaw_angles      = _to_f32(py_farm.yaw_angles, device)
        # tilt_angles: populated by set_tilt_to_ref_tilt() in the Python solver.
        # Fall back to ref_tilts (gives cos(tilt)/cos(ref_tilt) = 1) if unavailable.
        tilt = _get_tilt_angles(py_farm)
        if tilt is not None:
            f.tilt_angles = _to_f32(tilt, device)
        f.n_turbines      = py_farm.n_turbines
        _build_turbine_tables(floris_cpp, f, py_farm, device)
        return f

    @staticmethod
    def _make_cpp_flow_field(py_ff, device="cpu") -> object:
        floris_cpp = _get_floris_cpp()
        ff = floris_cpp.FlowField()
        ff.wind_speeds            = _to_f32(py_ff.wind_speeds, device)
        ff.wind_directions        = _to_f32(py_ff.wind_directions, device)
        ff.turbulence_intensities = _to_f32(py_ff.turbulence_intensities, device)
        ff.wind_shear             = float(py_ff.wind_shear)
        ff.wind_veer              = float(py_ff.wind_veer)
        ff.air_density            = float(py_ff.air_density)
        ff.reference_wind_height  = float(py_ff.reference_wind_height)
        ff.n_findex               = int(py_ff.n_findex)
        return ff

    @staticmethod
    def _make_cpp_grid(py_farm, py_ff, solver_dict: dict, device="cpu") -> object:
        floris_cpp = _get_floris_cpp()
        g = floris_cpp.TurbineGrid()
        g.turbine_coordinates = _to_f32(py_farm.coordinates, device)
        g.turbine_diameters   = _to_f32(py_farm.rotor_diameters, device)
        g.wind_directions     = _to_f32(py_ff.wind_directions, device)
        g.grid_resolution     = int(solver_dict.get("turbine_grid_points", 3))
        g.n_turbines          = py_farm.n_turbines
        g.n_findex            = int(py_ff.n_findex)
        return g

    # ------------------------------------------------------------------
    # Core interface — compute methods
    # ------------------------------------------------------------------

    def initialize_domain(self) -> None:
        """
        Sync mutable Python state to C++ structs, then call the C++
        initialization kernels (grid rotation/sorting + velocity-field setup).

        Autograd-aware: if the caller has already assigned a
        ``requires_grad=True`` tensor to ``_cpp_farm.layout_x``,
        ``_cpp_farm.layout_y``, or ``_cpp_farm.yaw_angles`` those tensors are
        preserved so that the grad graph remains connected through the C++
        solver.  The grid turbine coordinates are also rebuilt from the grad
        tensors in that case so that ``x_sorted`` / ``y_sorted`` carry
        ``grad_fn`` into the wake kernels.
        """
        py_farm = self._py_core.farm
        py_ff   = self._py_core.flow_field

        # Resolve reference_wind_height: the YAML sentinel -1 means "use hub height".
        # FlorisModel.__init__ calls _post_init_checks() → assign_hub_height_to_ref_height()
        # immediately for fresh constructions, so self._py_core.flow_field.reference_wind_height
        # should already be positive.  We add a guard here for the case where the C++ path
        # is constructed without going through FlorisModel (e.g. unit tests).
        ref_h = float(py_ff.reference_wind_height)
        if abs(ref_h + 1.0) < 1.0e-6:
            # Sentinel: use turbine hub height
            ref_h = float(py_farm.hub_heights[0])

        def _preserve_if_grad(attr: str, default_val: torch.Tensor) -> None:
            """Set *attr* on _cpp_farm unless it already holds a grad tensor."""
            existing = getattr(self._cpp_farm, attr, None)
            if (
                isinstance(existing, torch.Tensor)
                and existing.requires_grad
                and tuple(existing.shape) == tuple(default_val.shape)
            ):
                return  # User-injected differentiable tensor; preserve it.
            setattr(self._cpp_farm, attr, default_val)

        # ---- sync farm ----
        # layout_x / layout_y / yaw_angles are preserved if the caller has
        # already injected a requires_grad tensor (common in autograd tests).
        _preserve_if_grad('layout_x',  _to_f32(py_farm.layout_x, self._device))
        _preserve_if_grad('layout_y',  _to_f32(py_farm.layout_y, self._device))
        _preserve_if_grad('yaw_angles', _to_f32(py_farm.yaw_angles, self._device))
        self._cpp_farm.hub_heights     = _to_f32(py_farm.hub_heights, self._device)
        self._cpp_farm.rotor_diameters = _to_f32(py_farm.rotor_diameters, self._device)
        tilt = _get_tilt_angles(py_farm)
        if tilt is not None:
            self._cpp_farm.tilt_angles = _to_f32(tilt, self._device)
        self._cpp_farm.n_turbines      = py_farm.n_turbines

        # ---- sync flow field ----
        self._cpp_flow_field.wind_speeds            = _to_f32(py_ff.wind_speeds, self._device)
        self._cpp_flow_field.wind_directions        = _to_f32(py_ff.wind_directions, self._device)
        self._cpp_flow_field.turbulence_intensities = _to_f32(
            py_ff.turbulence_intensities, self._device
        )
        self._cpp_flow_field.wind_shear             = float(py_ff.wind_shear)
        self._cpp_flow_field.wind_veer              = float(py_ff.wind_veer)
        self._cpp_flow_field.air_density            = float(py_ff.air_density)
        self._cpp_flow_field.reference_wind_height  = ref_h  # resolved sentinel
        self._cpp_flow_field.n_findex               = int(py_ff.n_findex)

        # ---- sync grid (layout / wind dirs may have changed) ----
        # If layout tensors carry gradients, rebuild turbine_coordinates from
        # them so that set_grid() propagates grad_fn into x_sorted / y_sorted.
        lx = self._cpp_farm.layout_x
        ly = self._cpp_farm.layout_y
        has_layout_grad = (
            isinstance(lx, torch.Tensor) and lx.requires_grad
        ) or (
            isinstance(ly, torch.Tensor) and ly.requires_grad
        )
        if has_layout_grad:
            hub_h = _to_f32(py_farm.hub_heights, self._device)  # [T], no grad
            # torch.stack([T], [T], [T], dim=1) → [T, 3]
            self._cpp_grid.turbine_coordinates = torch.stack(
                [lx.float(), ly.float(), hub_h], dim=1
            )
        else:
            self._cpp_grid.turbine_coordinates = _to_f32(py_farm.coordinates, self._device)
        self._cpp_grid.turbine_diameters   = _to_f32(py_farm.rotor_diameters, self._device)
        self._cpp_grid.wind_directions     = _to_f32(py_ff.wind_directions, self._device)
        self._cpp_grid.n_turbines          = py_farm.n_turbines
        self._cpp_grid.n_findex            = int(py_ff.n_findex)

        # ---- License size check (no-op when floraf is not installed) ----
        try:
            from floraf.license import check_size, validate_license
            check_size(validate_license(), py_farm.n_turbines, int(py_ff.n_findex))
        except ImportError:
            pass

        # ---- C++ initialization ----
        # Step 1: compute the rotated grid + turbine sort order.
        self._cpp_grid.set_grid()

        # Step 2: sort yaw angles from most-upstream to most-downstream.
        self._cpp_farm.initialize(self._cpp_grid.sorted_coord_indices)

        # Step 3: build u_initial_sorted (wind-shear power-law profile).
        self._cpp_flow_field.initialize_velocity_field(self._cpp_grid)

        self._py_core.state = State.INITIALIZED

    def steady_state_atmospheric_condition(self) -> None:
        """
        Run the C++ wake solver and finalize.

        Matches ``Core.steady_state_atmospheric_condition()`` which also calls
        ``self.finalize()`` at the end—``FlorisModel.run()`` then calls
        ``self.core.finalize()`` again, which is idempotent.

        The solved velocity field is preserved as a tensor (with any grad_fn
        from differentiable inputs such as yaw_angles / layout_x / layout_y)
        in ``self._u_sorted_tensor`` before ``finalize()`` converts it to
        numpy.  Use ``get_farm_power_tensor()`` to obtain a differentiable
        farm-power output that supports ``backward()``.
        """
        solver = self._floris_cpp.get_solver(self._model_config.cpp_solver_paradigm)
        solver.solve(
            self._cpp_farm,
            self._cpp_flow_field,
            self._cpp_grid,
            self._model_config,
        )
        # Capture the solved velocity tensor BEFORE finalize() detaches and
        # converts it to numpy.  This is the hook for the differentiable
        # power output path (get_farm_power_tensor()).
        self._u_sorted_tensor = self._cpp_flow_field.u_sorted  # [F, T, nG, nG]
        self.finalize()

    def get_farm_power_tensor(self) -> torch.Tensor:
        """
        Return farm power as a differentiable ``torch.Tensor``.

        Uses the velocity field computed by the last ``run()`` call together
        with the first turbine type's power table (via the registered
        differentiable ``interp1d`` op) to produce a
        ``[n_findex, n_turbines]`` power tensor whose ``grad_fn`` is connected
        to any ``requires_grad=True`` inputs (e.g. ``yaw_angles``,
        ``layout_x``, ``layout_y``) that were set on ``_cpp_farm`` before
        calling ``run()``.

        Unlike ``FlorisModel.get_farm_power()`` — which goes through a numpy
        round-trip and returns a plain ndarray — this method stays entirely in
        PyTorch so that ``backward()`` can propagate gradients back to the
        input tensors.

        Raises ``RuntimeError`` if called before ``run()``.
        """
        if not hasattr(self, '_u_sorted_tensor'):
            raise RuntimeError(
                "get_farm_power_tensor() requires run() to be called first."
            )

        u = self._u_sorted_tensor  # [F, T_sorted, nG, nG], may carry grad_fn

        # Average velocity over rotor grid points → [F, T_sorted].
        # Arithmetic mean (rather than cubic-mean) preserves grad_fn correctly.
        u_avg = u.mean(dim=(2, 3))

        # Look up power from the first turbine type's power table using the
        # registered differentiable interp1d op (Phase 6).
        ptt = self._py_core.farm.turbine_power_thrust_tables
        first_type = next(iter(ptt))
        tbl = ptt[first_type]
        dtype = u_avg.dtype
        device = u_avg.device
        ws_t = torch.tensor(
            np.asarray(tbl['wind_speed'], dtype=np.float32),
            dtype=dtype, device=device,
        )
        pw_t = torch.tensor(
            np.asarray(tbl['power'], dtype=np.float32),
            dtype=dtype, device=device,
        )

        # interp1d accepts any-shape xnew and returns the same shape.
        # Power table values are in kW (per FLORIS convention); multiply by
        # 1000 to return Watts, consistent with get_farm_power().
        power = torch.ops.floris_cpp.interp1d(ws_t, pw_t, u_avg)  # [F, T], kW
        return power * 1000.0  # W

    def run_no_wake(self) -> None:
        """Initialize domain without applying the wake model (freestream only)."""
        self.initialize_domain()
        self.finalize()

    def finalize(self) -> None:
        """
        Finalize C++ state and sync results back to the Python Core.

        After this call, ``self._py_core.flow_field.u`` and
        ``self._py_core.flow_field.turbulence_intensity_field`` contain the
        post-solve velocity and turbulence fields that
        ``FlorisModel._get_turbine_powers()`` reads.
        """
        self._cpp_flow_field.finalize(self._cpp_grid.unsorted_indices)
        self._cpp_farm.finalize(self._cpp_grid.unsorted_indices)

        # unsorted_indices: [n_findex, n_turbines]  (long, may be on any device).
        # Used to map from sorted turbine order back to the original layout order.
        unsorted_np = self._cpp_grid.unsorted_indices.cpu().numpy()  # [fi, tu]

        # ── turbine_type_map ────────────────────────────────────────────────
        # _get_turbine_powers() dispatches power/thrust functions via
        # turbine_type_map[fi, ti], so it must be in original (unsorted) order.
        # We do NOT call _py_core.farm.finalize() because that also overwrites
        # rotor_diameters from its 1-D [n_turb] form to 2-D [n_fi, n_turb],
        # which breaks the next call to initialize_domain() → set_grid() because
        # the C++ TurbineGrid expects turbine_diameters[0] to be a scalar.
        self._py_core.farm.turbine_type_map = np.take_along_axis(
            self._py_core.farm.turbine_type_map_sorted,
            unsorted_np,
            axis=1,
        )

        # ── velocity field ──────────────────────────────────────────────────
        # The Phase-2 C++ FlowField::finalize() is a stub: u = u_sorted
        # (no spatial un-sorting is applied).  We apply the inverse permutation
        # here so that _get_turbine_powers() receives u in original turbine order.
        u_sorted_np = self._cpp_flow_field.u.detach().cpu().numpy()    # [fi, tu, ng, ng]
        unsorted_np_4d = unsorted_np[:, :, np.newaxis, np.newaxis]  # broadcast dims
        self._py_core.flow_field.u = np.take_along_axis(
            u_sorted_np, unsorted_np_4d, axis=1
        )

        # ── turbulence intensity field ───────────────────────────────────────
        # Average over rotor grid points → [n_fi, n_tu] in sorted turbine order;
        # then unsort back to the original layout order.
        ti_sorted_np = (
            self._cpp_flow_field.turbulence_intensity_field_sorted
            .mean(dim=(2, 3)).detach().cpu().numpy()
        )
        self._py_core.flow_field.turbulence_intensity_field = np.take_along_axis(
            ti_sorted_np, unsorted_np, axis=1
        )

        # ── yaw angles ───────────────────────────────────────────────────────
        # C++ Farm::finalize() already applied unsort via torch::gather so
        # cpp_farm.yaw_angles is in the original layout order.
        self._py_core.farm.yaw_angles = self._cpp_farm.yaw_angles.detach().cpu().numpy()

        self._py_core.state = State.USED

    # ------------------------------------------------------------------
    # Unsupported visualisation methods
    # ------------------------------------------------------------------

    def solve_for_viz(self, *args, **kwargs):
        raise NotImplementedError(
            "solve_for_viz is not supported by the C++ backend. "
            "Use FlorisModel(backend='python') for flow-field visualisations."
        )

    def solve_for_points(self, *args, **kwargs):
        raise NotImplementedError(
            "solve_for_points is not supported by the C++ backend. "
            "Use FlorisModel(backend='python') for point sampling."
        )

    def solve_for_velocity_deficit_profiles(self, *args, **kwargs):
        raise NotImplementedError(
            "solve_for_velocity_deficit_profiles is not supported by the "
            "C++ backend.  Use FlorisModel(backend='python')."
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """
        Reverse-serialise state to the YAML dict schema.

        Delegates to ``_py_core.as_dict()`` so that ``FlorisModel._reinitialize()``
        can call ``Core.from_dict(self.core.as_dict())`` without changes.
        """
        return self._py_core.as_dict()

    def __deepcopy__(self, memo):
        """
        Custom deep-copy that avoids copying the floris_cpp module object
        (which is not pickle-able).  Used by ``FlorisModel.calculate_horizontal_plane``
        and other viz helpers that do ``copy.deepcopy(fmodel)``.
        """
        new = CppCore.from_dict(
            self._py_core.as_dict(),
            device=self._device,
            cpp_solver_paradigm=self._cpp_solver_paradigm,
        )
        memo[id(self)] = new
        return new

    # ------------------------------------------------------------------
    # Attribute proxies — expose _py_core's farm / flow_field / grid / state
    # ------------------------------------------------------------------

    @property
    def farm(self):
        """Python Farm — owns power functions, type maps, AWC arrays, etc."""
        return self._py_core.farm

    @property
    def flow_field(self):
        """Python FlowField — post-finalize ``u`` and TI field live here."""
        return self._py_core.flow_field

    @property
    def grid(self):
        """Python Grid — used for visualisation helpers; C++ has its own grid."""
        return self._py_core.grid

    @property
    def state(self) -> State:
        return self._py_core.state

    @state.setter
    def state(self, value: State) -> None:
        self._py_core.state = value

    @property
    def solver(self) -> dict:
        """Solver settings dict (e.g. ``{"type": "turbine_grid", ...}``)."""
        return self._py_core.solver

    @property
    def wake(self):
        """``WakeModelManager`` instance from the Python Core."""
        return self._py_core.wake

    @property
    def logging(self) -> dict:
        return self._py_core.logging

    @property
    def name(self) -> str:
        return self._py_core.name

    @property
    def description(self) -> str:
        return self._py_core.description

    @property
    def floris_version(self) -> str:
        return self._py_core.floris_version

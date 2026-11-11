#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
General simulation code for
Distributed Dynamic Optimal Encirclement Control of Multi-Agent Systems
Using Classical and Quantum Optimization.

This script is intentionally built from the current manuscript formulation,
not from the earlier prototype simulation codes.

Supported modes
---------------
ARCHITECTURE = "centralized" or "distributed"
QP_SOLVER    = "quadprog" or "qqp"

Thus the same file runs the four paper cases:
    centralized + classical QP
    centralized + QQP/VQE
    distributed ADMM + classical local QPs
    distributed ADMM + QQP/VQE local QPs

Paper-faithful components
-------------------------
1. Differential-drive model and controlled front point.
2. Fixed-time distributed target-center estimator.
3. Cascaded fixed-time distributed target-group-radius estimator.
4. Rotating encirclement reference constructed from local center estimate.
5. One-step quadratic optimal encirclement cost.
6. Wheel speed, wheel acceleration, body-speed, and yaw-rate constraints.
7. Local robot-robot affine safety constraints.
8. Target-group affine safety constraint using the controlled point and only estimated center/radius.
   The controlled-point exclusion radius is enlarged by ell so that body-center
   target safety is guaranteed by the triangle inequality.
9. Centralized joint QP.
10. Distributed neighbor-copy consensus ADMM formulation.
11. Classical quadprog and quantum-compatible RAISELab QQP/DVQE backends.

Outputs
-------
The run directory contains raw CSV/JSON data, publication-quality figures,
estimator-convergence diagnostics, solver/ADMM diagnostics, and optional
2-D/3-D animations. The reports are designed to contain the quantities that
would normally be needed in an IEEE TAC numerical-experiments section.

Dependencies
------------
Required: numpy, pandas, matplotlib
Classical solver: quadprog
Quantum solver: raiselab (qqp with qubo_solver="dvqe"), as in the old code.
Optional animation: imageio-ffmpeg and pillow.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# USER SETTINGS
# ============================================================================

ARCHITECTURE = "distributed"   # "centralized" or "distributed"
QP_SOLVER = "qqp"         # "quadprog" or "qqp"
RUN_NAME = "encir"
BASE_OUTPUT_ROOT = Path.cwd() / "Encirclement_Results"

SAVE_PLOTS = True
SHOW_PLOTS = False
SAVE_VIDEO = True
SAVE_FRAMES = False
ANIMATION_STRIDE = None  # derived as cfg.animation_stride
VIDEO_FPS = 15
FRAME_DPI = 140
VIDEO_DPI = 110
VIEW_ELEVATION = 24
VIEW_AZIMUTH = -58
SHOW_ANIMATIONS_IN_NOTEBOOK = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Automatically scaled simulation configuration.

    The user-facing quantities are intentionally limited to the scenario and
    physical robot/target specifications. Quantities that are coupled to the
    number of agents, sampling period, simulation horizon, graph, or physical
    limits are derived automatically in ``__post_init__``.

    This keeps the optimization/control methodology unchanged while preventing
    accidental combinations of manually chosen hyperparameters from creating
    unreachable references, insufficient sensing look-ahead, poor estimator
    scaling, or badly scaled numerical subproblems.
    """

    # ------------------------------------------------------------------
    # PRIMARY USER SETTINGS
    # ------------------------------------------------------------------
    # These are the quantities normally changed between numerical experiments.
    n: int = 6
    T: float = 50.0
    dt: float = 0.05
    seed: int = 7

    # ------------------------------------------------------------------
    # PHYSICAL ROBOT SPECIFICATION
    # ------------------------------------------------------------------
    # These are physical properties/limits, not tuning knobs. They are NOT
    # silently increased when n or dt changes.
    wheel_radius: float = 0.05       # r [m]
    track_width: float = 0.30        # b [m]
    lookahead: float = 0.25          # ell [m]
    wheel_speed_max: float = 16.0    # bar omega [rad/s]
    wheel_accel_max: float = 360.0   # bar alpha [rad/s^2]
    speed_forward_max: float = 0.80  # bar v_f [m/s]
    speed_reverse_max: float = 0.30  # bar v_r [m/s]
    yaw_rate_max: float = 3.0        # bar nu [rad/s]

    # ------------------------------------------------------------------
    # PHYSICAL SAFETY SPECIFICATION
    # ------------------------------------------------------------------
    robot_robot_min_distance: float = 0.80  # d_rr [m]
    robot_target_min_distance: float = 0.80 # d_rt [m]

    # ------------------------------------------------------------------
    # TARGET MOTION SPECIFICATION
    # ------------------------------------------------------------------
    # The trajectory itself is a numerical-experiment choice. The velocity
    # bound used by the estimators is derived from these quantities.
    target_center_speed_x: float = 0.10
    target_center_y_amplitude: float = 0.35
    target_center_y_frequency: float = 0.08
    target_radius_base: float = 0.70
    target_radius_amplitude: float = 0.12
    target_radius_frequency: float = 0.10
    target_phase_rate: float = 0.015

    # ------------------------------------------------------------------
    # REFERENCE-POLICY PREFERENCES
    # ------------------------------------------------------------------
    # These are desired upper/preferred values. The actual values are reduced
    # or enlarged automatically when required by geometry or actuator limits.
    preferred_min_radius: float = 2.00
    preferred_radius_spacing: float = 0.2
    preferred_max_radial_span: float = 2.00
    preferred_omega_e: float = 0.14
    reference_speed_utilization: float = 0.72
    target_safety_margin: float = 0.2
    robot_robot_reference_margin: float = 0.1
    phi0: float = 0.0

    # ------------------------------------------------------------------
    # FIXED ALGORITHM EXPONENTS / NUMERICAL POLICY CONSTANTS
    # ------------------------------------------------------------------
    # mu and beta are part of the fixed-time estimator structure. The remaining
    # quantities below are policy factors, not case-by-case tuning parameters.
    mu: float = 0.50
    beta: float = 2.00
    center_k3_margin: float = 1.25
    radius_k3_margin: float = 1.25
    estimator_center_time_fraction: float = 0.03
    estimator_radius_time_fraction: float = 0.08
    base_center_gain: float = 25.0
    base_radius_gain: float = 50.0

    # ------------------------------------------------------------------
    # STAGE-COST BASELINE
    # ------------------------------------------------------------------
    # q_p and q_delta are automatically scaled relative to dt=0.05 s so that
    # changing the sample time does not drastically change the effective cost.
    nominal_dt: float = 0.05
    nominal_q_p: float = 10000.0
    nominal_q_u: float = 0.01
    nominal_q_delta: float = 0.10
    nominal_q_t: float = 0.01

    # ------------------------------------------------------------------
    # DERIVED VALUES -- DO NOT EDIT CASE BY CASE
    # ------------------------------------------------------------------
    adjacency: np.ndarray = field(init=False, repr=False)
    laplacian: np.ndarray = field(init=False, repr=False)
    lambda2: float = field(init=False)
    edge_count: int = field(init=False)
    communication_hops_each_side: int = field(init=False)

    target_velocity_bound: float = field(init=False)
    target_center_speed_bound: float = field(init=False)
    max_controlled_point_speed: float = field(init=False)

    desired_radius: float = field(init=False)
    desired_radius_spacing: float = field(init=False)
    omega_e: float = field(init=False)
    maximum_desired_radius: float = field(init=False)

    q_p: float = field(init=False)
    q_u: float = field(init=False)
    q_delta: float = field(init=False)
    q_t: float = field(init=False)

    random_x_min: float = field(init=False)
    random_x_max: float = field(init=False)
    random_y_min: float = field(init=False)
    random_y_max: float = field(init=False)
    initial_minimum_spacing: float = field(init=False)
    initial_heading_noise: float = field(init=False)
    initial_placement_attempts: int = field(init=False)
    safety_sensing_radius: float = field(init=False)

    center_k1: float = field(init=False)
    center_k2: float = field(init=False)
    center_k3: float = field(init=False)
    radius_k1: float = field(init=False)
    radius_k2: float = field(init=False)
    radius_k3: float = field(init=False)
    center_time_bound: float = field(init=False)
    radius_stage_time_bound: float = field(init=False)
    radius_time_bound: float = field(init=False)
    estimator_capture_tol: float = field(init=False)
    estimator_internal_dt_max: float = field(init=False)

    admm_rho: float = field(init=False)
    admm_max_iters: int = field(init=False)
    admm_abs_tol: float = field(init=False)
    admm_rel_tol: float = field(init=False)
    admm_warm_start: bool = field(init=False)
    admm_balance_mu: float = field(init=False)
    admm_rho_tau: float = field(init=False)
    admm_rho_min: float = field(init=False)
    admm_rho_max: float = field(init=False)
    admm_adapt_interval: int = field(init=False)

    qp_regularization: float = field(init=False)
    feasibility_tol: float = field(init=False)

    qqp_mu0: float = field(init=False)
    qqp_max_al_iters: int = field(init=False)
    qqp_max_region_iters: int = field(init=False)
    qqp_rho0: float = field(init=False)
    qqp_rho_decay: float = field(init=False)
    qqp_max_outer_iters: int = field(init=False)
    qqp_max_inner_iters: int = field(init=False)
    qqp_qubo_solver: str = field(init=False)

    figure_dpi: int = field(init=False)
    steady_state_fraction: float = field(init=False)
    animation_stride: int = field(init=False)

    def __post_init__(self):
        self._validate_primary_inputs()
        self._derive_graph()
        self._derive_target_bounds()
        self._derive_reference()
        self._derive_cost_weights()
        self._derive_safety_and_initialization()
        self._derive_estimators()
        self._derive_optimizer_settings()
        self._validate_derived_configuration()

    # ------------------------------------------------------------------
    # Geometry / physical capability helpers
    # ------------------------------------------------------------------
    def _max_controlled_point_speed_from_limits(self) -> float:
        """Exact maximum ||p_dot|| over the linear (v,nu) physical polytope."""
        r = self.wheel_radius
        b = self.track_width
        ell = self.lookahead
        vw = r * self.wheel_speed_max

        # A [v, nu]^T <= bvec.
        A = np.array([
            [ 1.0,  0.0],   # v <= v_f
            [-1.0,  0.0],   # -v <= v_r
            [ 0.0,  1.0],   # nu <= nu_max
            [ 0.0, -1.0],   # -nu <= nu_max
            [ 1.0,  b/2.0], # |v + b nu/2| <= r omega_max
            [-1.0, -b/2.0],
            [ 1.0, -b/2.0], # |v - b nu/2| <= r omega_max
            [-1.0,  b/2.0],
        ], dtype=float)
        bvec = np.array([
            self.speed_forward_max,
            self.speed_reverse_max,
            self.yaw_rate_max,
            self.yaw_rate_max,
            vw, vw, vw, vw,
        ], dtype=float)

        candidates = [np.zeros(2)]
        for i in range(len(bvec)):
            for j in range(i + 1, len(bvec)):
                M = np.vstack([A[i], A[j]])
                if abs(np.linalg.det(M)) < 1.0e-12:
                    continue
                x = np.linalg.solve(M, np.array([bvec[i], bvec[j]]))
                if np.all(A @ x <= bvec + 1.0e-10):
                    candidates.append(x)

        return max(
            math.hypot(float(x[0]), ell * float(x[1]))
            for x in candidates
        )

    def _center_bound_for_gains(self, k1: float, k2: float) -> float:
        return (
            2.0
            / (
                k1
                * (1.0 - self.mu)
                * (2.0 * self.lambda2) ** ((1.0 + self.mu) / 2.0)
            )
            + 2.0
            / (
                k2
                * (self.beta - 1.0)
                * (2.0 * self.edge_count) ** ((1.0 - self.beta) / 2.0)
                * (2.0 * self.lambda2) ** ((1.0 + self.beta) / 2.0)
            )
        )

    def _radius_stage_bound_for_gains(self, k1: float, k2: float) -> float:
        return (
            1.0 / (k1 * (1.0 - self.mu))
            + 1.0 / (k2 * (self.beta - 1.0))
        )

    # ------------------------------------------------------------------
    # Automatic derivation pipeline
    # ------------------------------------------------------------------
    def _derive_graph(self):
        # Keep communication local while increasing graph connectivity mildly
        # with n. n=6 and n=12 both use two neighbors per side; larger systems
        # gradually receive one additional hop when useful.
        suggested = max(1, int(math.ceil(math.log2(self.n) / 2.0)))
        self.communication_hops_each_side = min(
            suggested, max(1, (self.n - 1) // 2)
        )
        self.adjacency = make_ring_graph(
            self.n, self.communication_hops_each_side
        )
        self.laplacian = np.diag(self.adjacency.sum(axis=1)) - self.adjacency
        eig = np.linalg.eigvalsh(self.laplacian)
        self.lambda2 = float(eig[1])
        self.edge_count = int(np.sum(self.adjacency) // 2)

    def _derive_target_bounds(self):
        rmax = self.target_radius_base + abs(self.target_radius_amplitude)
        radial_speed = abs(
            self.target_radius_amplitude * self.target_radius_frequency
        )
        orbital_speed = rmax * abs(self.target_phase_rate)
        self.target_center_speed_bound = math.hypot(
            abs(self.target_center_speed_x),
            abs(self.target_center_y_amplitude * self.target_center_y_frequency),
        )
        # Conservative analytic bound for every individual target velocity,
        # with a small numerical margin. This replaces the manually entered
        # bar v_t and therefore cannot become inconsistent with the trajectory.
        self.target_velocity_bound = 1.05 * (
            self.target_center_speed_bound + radial_speed + orbital_speed
        )
        self.max_controlled_point_speed = (
            self._max_controlled_point_speed_from_limits()
        )

    def _derive_reference(self):
        target_rmax = self.target_radius_base + abs(self.target_radius_amplitude)

        # The controlled-point target-safety constraint requires this radius.
        target_safe_radius = (
            target_rmax
            + self.robot_target_min_distance
            + self.lookahead
            + self.target_safety_margin
        )

        # Conservative geometry for the steady encirclement references. If two
        # controlled points are separated by d_rr+2 ell+margin, then the robot
        # centers retain at least d_rr even under the worst orientation offset.
        required_cp_pair_distance = (
            self.robot_robot_min_distance
            + 2.0 * self.lookahead
            + self.robot_robot_reference_margin
        )
        sin_term = max(math.sin(math.pi / self.n), 1.0e-6)
        rr_safe_reference_radius = required_cp_pair_distance / (2.0 * sin_term)

        self.desired_radius = max(
            self.preferred_min_radius,
            target_safe_radius,
            rr_safe_reference_radius,
        )

        # Prevent the outer orbit from growing linearly without bound as n
        # increases. The total radial spread is capped while preserving the
        # preferred spacing for small systems.
        if self.n <= 1:
            self.desired_radius_spacing = 0.0
        else:
            self.desired_radius_spacing = min(
                self.preferred_radius_spacing,
                self.preferred_max_radial_span / (self.n - 1),
            )
        self.maximum_desired_radius = (
            self.desired_radius
            + (self.n - 1) * self.desired_radius_spacing
        )

        # The reference speed is center translation plus orbital tangential
        # speed. Use only a conservative fraction of the physically attainable
        # controlled-point speed, leaving authority for tracking and safety.
        usable_cp_speed = (
            self.reference_speed_utilization * self.max_controlled_point_speed
        )
        tangential_budget = usable_cp_speed - self.target_center_speed_bound
        if tangential_budget <= 1.0e-6:
            raise ValueError(
                "Physical limits leave no controlled-point speed margin after "
                "the target-center motion. Reduce target motion or increase the "
                "specified physical actuator limits."
            )
        omega_feasible = tangential_budget / self.maximum_desired_radius
        self.omega_e = min(self.preferred_omega_e, 0.98 * omega_feasible)

    def _derive_cost_weights(self):
        ratio = self.nominal_dt / self.dt
        self.q_p = self.nominal_q_p * ratio**2
        self.q_u = self.nominal_q_u
        self.q_delta = self.nominal_q_delta * ratio
        self.q_t = self.nominal_q_t

    def _derive_safety_and_initialization(self):
        # A safety neighbor is detected early enough for several samples of
        # maximum relative closure. This automatically grows with dt.
        closure_margin = max(
            0.20,
            4.0 * self.speed_forward_max * self.dt,
        )
        self.safety_sensing_radius = (
            self.robot_robot_min_distance + closure_margin
        )

        self.initial_minimum_spacing = (
            self.robot_robot_min_distance
            + max(0.15, 2.0 * self.speed_forward_max * self.dt)
        )
        self.initial_heading_noise = min(
            0.55, max(0.15, 3.0 * self.omega_e)
        )
        self.initial_placement_attempts = max(20000, 1000 * self.n)

        # Scale the remote cloud with both system size and horizon. The cloud
        # remains remote, but short simulations are not initialized needlessly
        # far away and larger systems receive enough area for separated starts.
        size_scale = math.sqrt(max(self.n, 2) / 6.0)
        travel_scale = max(2.0, self.speed_forward_max * self.T)
        near_gap = max(1.5, min(3.0, 0.05 * travel_scale))
        x_width = max(3.0 * size_scale, min(8.0 * size_scale, 0.16 * travel_scale))
        y_half = max(4.0 * size_scale, min(14.0 * size_scale, 0.24 * travel_scale))

        self.random_x_min = self.maximum_desired_radius + near_gap
        self.random_x_max = self.random_x_min + x_width
        self.random_y_min = -y_half
        self.random_y_max = y_half

    def _derive_estimators(self):
        self.center_k3 = (
            self.center_k3_margin
            * math.sqrt(self.n / self.lambda2)
            * self.target_velocity_bound
        )
        self.radius_k3 = (
            self.radius_k3_margin * 2.0 * self.target_velocity_bound
        )

        # Scale k1,k2 only upward from well-tested baseline gains so the
        # theoretical settling bounds remain a small fraction of T even as n
        # and graph connectivity change.
        center_target = max(5.0 * self.dt, self.estimator_center_time_fraction * self.T)
        base_center_bound = self._center_bound_for_gains(
            self.base_center_gain, self.base_center_gain
        )
        center_scale = max(1.0, base_center_bound / center_target)
        self.center_k1 = self.base_center_gain * center_scale
        self.center_k2 = self.base_center_gain * center_scale
        self.center_time_bound = self._center_bound_for_gains(
            self.center_k1, self.center_k2
        )

        radius_target = max(10.0 * self.dt, self.estimator_radius_time_fraction * self.T)
        base_stage = self._radius_stage_bound_for_gains(
            self.base_radius_gain, self.base_radius_gain
        )
        base_total = (self.n - 1) * base_stage
        radius_scale = max(1.0, base_total / radius_target)
        self.radius_k1 = self.base_radius_gain * radius_scale
        self.radius_k2 = self.base_radius_gain * radius_scale
        self.radius_stage_time_bound = self._radius_stage_bound_for_gains(
            self.radius_k1, self.radius_k2
        )
        self.radius_time_bound = (
            (self.n - 1) * self.radius_stage_time_bound
        )

        self.estimator_internal_dt_max = min(0.0025, self.dt / 20.0)
        self.estimator_capture_tol = max(2.0e-4, 0.04 * self.dt)

    def _derive_optimizer_settings(self):
        # Scale ADMM rho with the local quadratic curvature at the nominal
        # heading, so dt-dependent changes in q_p do not destroy the balance
        # between the local objective and consensus penalty.
        th = 0.0
        r, b, ell = self.wheel_radius, self.track_width, self.lookahead
        G0 = r * np.array([
            [0.5 * math.cos(th) - (ell / b) * math.sin(th),
             0.5 * math.cos(th) + (ell / b) * math.sin(th)],
            [0.5 * math.sin(th) + (ell / b) * math.cos(th),
             0.5 * math.sin(th) - (ell / b) * math.cos(th)],
        ])
        A0 = self.dt * G0
        d = np.array([1.0, -1.0])
        M0 = (
            self.q_p * (A0.T @ A0)
            + (self.q_u + self.q_delta) * np.eye(2)
            + self.q_t * np.outer(d, d)
        )
        H0 = 2.0 * M0
        local_curvature = float(np.max(np.linalg.eigvalsh(H0)))
        # Initial ADMM penalty. During distributed solves this value is adapted
        # by primal/dual residual balancing. The centralized QP is unaffected.
        self.admm_rho = max(0.5, 5.0 * local_curvature)

        # Tight consensus is important because finite consensus error can appear
        # directly as a violation of coupled robot-to-robot safety constraints.
        self.admm_max_iters = int(min(3000, max(800, 500 + 50 * self.n)))
        self.admm_abs_tol = max(1.0e-8, 2.0e-6 * self.dt)
        self.admm_rel_tol = 1.0e-6
        self.admm_warm_start = True

        # Residual-balancing policy for rho_A. These settings affect only the
        # numerical convergence of ADMM and do not alter the distributed QP.
        self.admm_balance_mu = 10.0
        self.admm_rho_tau = 2.0
        self.admm_rho_min = 1.0e-3
        self.admm_rho_max = 1.0e4
        self.admm_adapt_interval = 10

        self.qp_regularization = 1.0e-9
        self.feasibility_tol = max(5.0e-7, 4.0e-4 * self.dt)

        # QQP internally normalizes constraints/objective. Increase only the
        # iteration budgets mildly with problem size; do not change the QQP
        # formulation itself.
        complexity = int(math.ceil(math.log2(self.n + 1)))
        self.qqp_mu0 = 100.0
        self.qqp_max_al_iters = max(15, 10 + 2 * complexity)
        self.qqp_max_region_iters = max(10, 6 + complexity)
        self.qqp_rho0 = 0.20
        self.qqp_rho_decay = 0.40
        self.qqp_max_outer_iters = max(10, 7 + complexity)
        self.qqp_max_inner_iters = max(20, 14 + 2 * complexity)
        self.qqp_qubo_solver = "dvqe"

        self.figure_dpi = 600
        self.steady_state_fraction = min(0.35, max(0.20, 15.0 / self.T))
        nsteps = max(1, int(round(self.T / self.dt)))
        self.animation_stride = max(1, int(math.ceil(nsteps / 450.0)))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_primary_inputs(self):
        if self.n < 2:
            raise ValueError("n must be at least 2")
        if self.dt <= 0 or self.T <= 0:
            raise ValueError("dt and T must be positive")
        if self.T < 10.0 * self.dt:
            raise ValueError("T must contain at least 10 sampling intervals")
        if not (0.0 < self.mu < 1.0):
            raise ValueError("mu must satisfy 0 < mu < 1")
        if self.beta <= 1.0:
            raise ValueError("beta must exceed 1")
        if min(self.wheel_radius, self.track_width, self.lookahead) <= 0.0:
            raise ValueError("Robot geometry parameters must be positive")
        if min(self.wheel_speed_max, self.wheel_accel_max,
               self.speed_forward_max, self.speed_reverse_max,
               self.yaw_rate_max) <= 0.0:
            raise ValueError("Physical limits must be positive")
        if min(self.robot_robot_min_distance,
               self.robot_target_min_distance) <= 0.0:
            raise ValueError("Safety distances must be positive")
        if not (0.20 <= self.reference_speed_utilization < 1.0):
            raise ValueError("reference_speed_utilization must lie in [0.20,1)")

    def _validate_derived_configuration(self):
        if self.lambda2 <= 0.0:
            raise ValueError("The automatically generated communication graph is disconnected")
        if self.safety_sensing_radius <= self.robot_robot_min_distance:
            raise ValueError("Automatic R_s must exceed d_rr")

        target_rmax = self.target_radius_base + abs(self.target_radius_amplitude)
        required_target_radius = (
            target_rmax + self.robot_target_min_distance + self.lookahead
        )
        if self.desired_radius <= required_target_radius:
            raise ValueError(
                "Automatic minimum desired radius does not satisfy target safety"
            )

        # Reference dynamic-feasibility audit.
        required_ref_speed = (
            self.target_center_speed_bound
            + self.maximum_desired_radius * abs(self.omega_e)
        )
        allowed_ref_speed = (
            self.reference_speed_utilization * self.max_controlled_point_speed
        )
        if required_ref_speed > allowed_ref_speed + 1.0e-10:
            raise ValueError(
                "Automatic reference construction failed its speed-feasibility audit: "
                f"required={required_ref_speed:.6g}, allowed={allowed_ref_speed:.6g}."
            )

    def auto_configuration_report(self) -> str:
        return (
            f"graph hops/side={self.communication_hops_each_side}, "
            f"d_e=[{self.desired_radius:.3f}, {self.maximum_desired_radius:.3f}] m, "
            f"Delta d={self.desired_radius_spacing:.3f} m, "
            f"omega_e={self.omega_e:.4f} rad/s, "
            f"R_s={self.safety_sensing_radius:.3f} m, "
            f"v_p,max={self.max_controlled_point_speed:.3f} m/s, "
            f"v_t,bar={self.target_velocity_bound:.3f} m/s, "
            f"ADMM rho0={self.admm_rho:.3g}, "
            f"ADMM tol=({self.admm_abs_tol:.1e},{self.admm_rel_tol:.1e})"
        )


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def make_ring_graph(n: int, hops: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    hops = min(hops, max(1, (n - 1) // 2))
    for i in range(n):
        for h in range(1, hops + 1):
            j1 = (i + h) % n
            j2 = (i - h) % n
            if j1 != i:
                A[i, j1] = A[j1, i] = 1.0
            if j2 != i:
                A[i, j2] = A[j2, i] = 1.0
    # For n=2, explicitly connect the pair.
    if n == 2:
        A[0, 1] = A[1, 0] = 1.0
    return A


def neighbors(A: np.ndarray, i: int) -> np.ndarray:
    return np.flatnonzero(A[i] > 0.0)


def sig_power(x: np.ndarray, q: float) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** q


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def controlled_point(X: np.ndarray, cfg: Config) -> np.ndarray:
    out = np.empty((cfg.n, 2), dtype=float)
    out[:, 0] = X[:, 0] + cfg.lookahead * np.cos(X[:, 2])
    out[:, 1] = X[:, 1] + cfg.lookahead * np.sin(X[:, 2])
    return out


def G_matrix(theta: float, cfg: Config) -> np.ndarray:
    r, b, ell = cfg.wheel_radius, cfg.track_width, cfg.lookahead
    c, s = math.cos(theta), math.sin(theta)
    return r * np.array([
        [0.5 * c - (ell / b) * s, 0.5 * c + (ell / b) * s],
        [0.5 * s + (ell / b) * c, 0.5 * s - (ell / b) * c],
    ])


def center_displacement_matrix(theta: float, cfg: Config) -> np.ndarray:
    return (
        cfg.dt
        * cfg.wheel_radius
        / 2.0
        * np.array([[math.cos(theta), math.cos(theta)],
                    [math.sin(theta), math.sin(theta)]])
    )


def integrate_robots(X: np.ndarray, U: np.ndarray, cfg: Config) -> np.ndarray:
    Xn = X.copy()
    r, b, dt = cfg.wheel_radius, cfg.track_width, cfg.dt
    for i in range(cfg.n):
        wr, wl = U[i]
        v = 0.5 * r * (wr + wl)
        nu = (r / b) * (wr - wl)
        th = X[i, 2]
        Xn[i, 0] += dt * v * math.cos(th)
        Xn[i, 1] += dt * v * math.sin(th)
        Xn[i, 2] = float(wrap_angle(th + dt * nu))
    return Xn


# ============================================================================
# TARGET TRAJECTORY FOR NUMERICAL EXPERIMENTS
# ============================================================================

def target_state(t: float, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return target positions and velocities; m=n as assumed in the paper."""
    cx = cfg.target_center_speed_x * t
    cy = cfg.target_center_y_amplitude * math.sin(cfg.target_center_y_frequency * t)
    dcx = cfg.target_center_speed_x
    dcy = (
        cfg.target_center_y_amplitude
        * cfg.target_center_y_frequency
        * math.cos(cfg.target_center_y_frequency * t)
    )
    rad = cfg.target_radius_base + cfg.target_radius_amplitude * math.sin(
        cfg.target_radius_frequency * t
    )
    drad = cfg.target_radius_amplitude * cfg.target_radius_frequency * math.cos(
        cfg.target_radius_frequency * t
    )

    S = np.zeros((cfg.n, 2))
    V = np.zeros((cfg.n, 2))
    for i in range(cfg.n):
        phi = 2.0 * np.pi * i / cfg.n + cfg.target_phase_rate * t
        dphi = cfg.target_phase_rate
        cp, sp = math.cos(phi), math.sin(phi)
        S[i] = [cx + rad * cp, cy + rad * sp]
        V[i] = [
            dcx + drad * cp - rad * dphi * sp,
            dcy + drad * sp + rad * dphi * cp,
        ]
    vmax = float(np.max(np.linalg.norm(V, axis=1)))
    if vmax > cfg.target_velocity_bound + 1.0e-10:
        raise RuntimeError(
            f"Target trajectory violates configured velocity bound: {vmax:.6f} > "
            f"{cfg.target_velocity_bound:.6f}. Increase target_velocity_bound or "
            "reduce target-motion parameters."
        )
    return S, V


# ============================================================================
# PAPER FIXED-TIME ESTIMATORS
# ============================================================================

class FixedTimeCenterEstimator:
    """Eq. (center_estimate) and (fixed_time_center_estimator)."""

    def __init__(self, S0: np.ndarray, cfg: Config):
        self.cfg = cfg
        self.eta = np.zeros_like(S0)
        self.c_hat = self.eta + S0
        self.last_delta = np.zeros_like(S0)

    def advance(self, S0: np.ndarray, S1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate one control interval with small internal Filippov steps."""
        cfg = self.cfg
        old = self.c_hat.copy()
        nsub = max(1, int(math.ceil(cfg.dt / cfg.estimator_internal_dt_max)))
        h = cfg.dt / nsub
        for ss in range(nsub):
            a0 = ss / nsub
            a1 = (ss + 1) / nsub
            S = (1.0 - a0) * S0 + a0 * S1
            Snext = (1.0 - a1) * S0 + a1 * S1
            c_hat = self.eta + S
            eta_dot = np.zeros_like(self.eta)
            for i in range(cfg.n):
                for j in neighbors(cfg.adjacency, i):
                    d = c_hat[i] - c_hat[j]
                    eta_dot[i] -= cfg.center_k1 * sig_power(d, cfg.mu)
                    eta_dot[i] -= cfg.center_k2 * sig_power(d, cfg.beta)
                    eta_dot[i] -= cfg.center_k3 * np.sign(d)
            self.eta += h * eta_dot
            self.c_hat = self.eta + Snext

            max_edge = 0.0
            for i in range(cfg.n):
                for j in neighbors(cfg.adjacency, i):
                    max_edge = max(max_edge, float(np.linalg.norm(self.c_hat[i] - self.c_hat[j], ord=np.inf)))
            if max_edge <= cfg.estimator_capture_tol:
                avg = np.mean(self.c_hat, axis=0)
                self.c_hat[:] = avg
                self.eta = self.c_hat - Snext

        self.last_delta = self.c_hat - old
        return self.c_hat.copy(), self.last_delta.copy()


class CascadedFixedTimeRadiusEstimator:
    """Cascaded estimator in Eq. (fixed_time_radius_estimator)."""

    def __init__(self, S0: np.ndarray, c_hat0: np.ndarray, cfg: Config):
        self.cfg = cfg
        self.z = np.zeros((cfg.n, cfg.n), dtype=float)
        self.z[:, 0] = np.linalg.norm(S0 - c_hat0, axis=1)
        self.r_hat = self.z[:, 0].copy()
        self.last_delta = np.zeros(cfg.n)

    def _active_stage(self, t: float) -> int:
        cfg = self.cfg
        if t < cfg.center_time_bound:
            return 0
        elapsed = t - cfg.center_time_bound
        return min(cfg.n - 1, int(elapsed // cfg.radius_stage_time_bound) + 1)

    def advance(self, S0: np.ndarray, S1: np.ndarray, c0: np.ndarray, c1: np.ndarray, t0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        old = self.r_hat.copy()
        nsub = max(1, int(math.ceil(cfg.dt / cfg.estimator_internal_dt_max)))
        h = cfg.dt / nsub
        rho = self.z[:, 0].copy()
        for ss in range(nsub):
            a1 = (ss + 1) / nsub
            t = t0 + a1 * cfg.dt
            S = (1.0 - a1) * S0 + a1 * S1
            C = (1.0 - a1) * c0 + a1 * c1
            rho = np.linalg.norm(S - C, axis=1)
            self.z[:, 0] = rho
            active = self._active_stage(t)
            for sigma in range(1, active + 1):
                refs = np.zeros(cfg.n)
                for i in range(cfg.n):
                    inds = np.r_[i, neighbors(cfg.adjacency, i)]
                    refs[i] = float(np.max(self.z[inds, sigma - 1]))
                e0 = self.z[:, sigma] - refs
                zdot = (
                    -cfg.radius_k1 * sig_power(e0, cfg.mu)
                    -cfg.radius_k2 * sig_power(e0, cfg.beta)
                    -cfg.radius_k3 * np.sign(e0)
                )
                znew = self.z[:, sigma] + h * zdot
                e1 = znew - refs
                crossed = (e0 * e1 <= 0.0) | (np.abs(e1) <= cfg.estimator_capture_tol)
                znew[crossed] = refs[crossed]
                self.z[:, sigma] = znew
            self.r_hat = self.z[:, active].copy()

        self.r_hat = np.maximum(self.r_hat, 0.0)
        self.last_delta = self.r_hat - old
        return self.r_hat.copy(), self.last_delta.copy(), rho.copy()


# ============================================================================
# REFERENCE GENERATION
# ============================================================================

def desired_radii(cfg: Config) -> np.ndarray:
    return cfg.desired_radius + cfg.desired_radius_spacing * np.arange(cfg.n)


def desired_reference(t_next: float, c_hat: np.ndarray, delta_c: np.ndarray, cfg: Config) -> np.ndarray:
    """One-step reference using the paper phase law and local center estimate."""
    c_pred = c_hat + delta_c
    radii = desired_radii(cfg)
    pd = np.zeros((cfg.n, 2))
    for i in range(cfg.n):
        phi0_i = cfg.phi0 + 2.0 * np.pi * i / cfg.n
        phi = phi0_i + cfg.omega_e * t_next
        pd[i] = c_pred[i] + radii[i] * np.array([math.cos(phi), math.sin(phi)])
    return pd


# ============================================================================
# QP CONSTRUCTION: CENTRALIZED PROBLEM
# ============================================================================

def local_cost_terms(i: int, X: np.ndarray, P: np.ndarray, Pd_next: np.ndarray,
                     U_prev: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, float]:
    A = cfg.dt * G_matrix(X[i, 2], cfg)
    e0 = P[i] - Pd_next[i]
    d = np.array([1.0, -1.0])
    M = (
        cfg.q_p * (A.T @ A)
        + (cfg.q_u + cfg.q_delta) * np.eye(2)
        + cfg.q_t * np.outer(d, d)
    )
    H = 2.0 * M
    f = 2.0 * (cfg.q_p * A.T @ e0 - cfg.q_delta * U_prev[i])
    c = (
        cfg.q_p * float(e0 @ e0)
        + cfg.q_delta * float(U_prev[i] @ U_prev[i])
    )
    return H, f, c


def wheel_bounds(U_prev: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    du = cfg.dt * cfg.wheel_accel_max
    lb = np.maximum(-cfg.wheel_speed_max, U_prev.reshape(-1) - du)
    ub = np.minimum(cfg.wheel_speed_max, U_prev.reshape(-1) + du)
    return lb, ub


def safety_neighbors(X: np.ndarray, cfg: Config) -> List[List[int]]:
    out: List[List[int]] = [[] for _ in range(cfg.n)]
    for i in range(cfg.n):
        for a in range(cfg.n):
            if a == i:
                continue
            if np.linalg.norm(X[i, :2] - X[a, :2]) <= cfg.safety_sensing_radius:
                out[i].append(a)
    return out


def add_row(rows: List[np.ndarray], rhs: List[float], row: np.ndarray, b: float):
    rows.append(np.asarray(row, dtype=float))
    rhs.append(float(b))


def build_centralized_qp(X: np.ndarray, P: np.ndarray, Pd_next: np.ndarray,
                         U_prev: np.ndarray, c_hat: np.ndarray, delta_c: np.ndarray,
                         r_hat: np.ndarray, delta_r: np.ndarray,
                         cfg: Config):
    """Build the manuscript centralized QP and retain row metadata for diagnostics.

    The optimization problem is unchanged.  The parallel label/group/detail lists
    are used only after a failed feasibility check to identify the responsible
    hard constraints.
    """
    nv = 2 * cfg.n
    Q = np.zeros((nv, nv))
    q = np.zeros(nv)
    c0 = 0.0
    for i in range(cfg.n):
        Hi, fi, ci = local_cost_terms(i, X, P, Pd_next, U_prev, cfg)
        sl = slice(2 * i, 2 * i + 2)
        Q[sl, sl] += Hi
        q[sl] += fi
        c0 += ci

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    labels: List[str] = []
    groups: List[str] = []
    details: List[str] = []

    def add_qp_row(row, bb, label, group, detail=""):
        add_row(rows, rhs, row, bb)
        labels.append(str(label))
        groups.append(str(group))
        details.append(str(detail))

    # Body forward/reverse and yaw constraints.
    for i in range(cfg.n):
        idx = slice(2 * i, 2 * i + 2)
        row = np.zeros(nv)
        row[idx] = cfg.wheel_radius / 2.0 * np.array([1.0, 1.0])
        add_qp_row(row, cfg.speed_forward_max,
                   f"robot_{i+1}_forward_speed", "body_speed",
                   f"robot={i+1}; v <= {cfg.speed_forward_max:.6g}")
        add_qp_row(-row, cfg.speed_reverse_max,
                   f"robot_{i+1}_reverse_speed", "body_speed",
                   f"robot={i+1}; v >= {-cfg.speed_reverse_max:.6g}")

        row = np.zeros(nv)
        row[idx] = cfg.wheel_radius / cfg.track_width * np.array([1.0, -1.0])
        add_qp_row(row, cfg.yaw_rate_max,
                   f"robot_{i+1}_yaw_positive", "yaw_rate",
                   f"robot={i+1}; nu <= {cfg.yaw_rate_max:.6g}")
        add_qp_row(-row, cfg.yaw_rate_max,
                   f"robot_{i+1}_yaw_negative", "yaw_rate",
                   f"robot={i+1}; nu >= {-cfg.yaw_rate_max:.6g}")

    Ns = safety_neighbors(X, cfg)
    pair_list = []
    for i in range(cfg.n):
        Di = center_displacement_matrix(X[i, 2], cfg)
        for a in Ns[i]:
            if i >= a:
                continue
            Da = center_displacement_matrix(X[a, 2], cfg)
            d = X[i, :2] - X[a, :2]
            distance = float(np.linalg.norm(d))
            hrr = float(d @ d - cfg.robot_robot_min_distance ** 2)
            row = np.zeros(nv)
            row[2 * i:2 * i + 2] = -2.0 * (d @ Di)
            row[2 * a:2 * a + 2] = +2.0 * (d @ Da)
            add_qp_row(
                row, hrr,
                f"rr_safety_r{i+1}_r{a+1}", "robot_robot_safety",
                f"robots={i+1},{a+1}; distance={distance:.9g}; "
                f"d_rr={cfg.robot_robot_min_distance:.9g}; h_rr={hrr:.9g}"
            )
            pair_list.append((i, a))

    # Target-group safety on the controlled point p_i.
    #
    # We impose
    #   ||p_i-c_hat_i|| >= r_hat_i + d_rt + ell.
    # Because ||xi_i-c_hat_i|| >= ||p_i-c_hat_i|| - ell, this guarantees
    # the original body-center requirement ||xi_i-c_hat_i|| >= r_hat_i+d_rt.
    # Importantly, Delta p_i = dt G(theta_i) omega_i uses the full-rank G, so
    # both wheel inputs can act directly on the safety constraint in one sample.
    for i in range(cfg.n):
        Di = cfg.dt * G_matrix(X[i, 2], cfg)
        d = P[i] - c_hat[i]
        center_distance = float(np.linalg.norm(d))
        R = float(r_hat[i] + cfg.robot_target_min_distance + cfg.lookahead)
        htg = float(d @ d - R * R)
        row = np.zeros(nv)
        row[2 * i:2 * i + 2] = -2.0 * (d @ Di)
        bb = htg - 2.0 * float(d @ delta_c[i]) - 2.0 * R * float(delta_r[i])
        add_qp_row(
            row, bb,
            f"target_safety_r{i+1}", "target_safety",
            f"robot={i+1}; controlled_point_distance={center_distance:.9g}; "
            f"required_controlled_point_radius={R:.9g}; h_tg={htg:.9g}; "
            f"delta_r={float(delta_r[i]):.9g}"
        )

    A = np.vstack(rows) if rows else np.zeros((0, nv))
    b = np.asarray(rhs, dtype=float)
    lb, ub = wheel_bounds(U_prev, cfg)
    meta = {
        "safety_pairs": pair_list,
        "num_variables": nv,
        "num_constraints": len(rhs),
        "constraint_labels": labels,
        "constraint_groups": groups,
        "constraint_details": details,
    }
    return Q, q, c0, A, b, lb, ub, meta


# ============================================================================
# QP SOLVER BACKENDS (same backends as old code)
# ============================================================================

def objective_value(Q, q, c, z):
    z = np.asarray(z, dtype=float)
    return 0.5 * float(z @ Q @ z) + float(q @ z) + float(c)


def _linear_feasibility_check(A, b, lb, ub):
    """Check A x <= b and bounds independently of the QP objective.

    This is used only as a diagnostic when quadprog reports inconsistent
    constraints.  It does not replace the manuscript QP or alter its solution.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None, None
    n = len(lb)
    res = linprog(
        np.zeros(n),
        A_ub=A if A.size else None,
        b_ub=b if A.size else None,
        bounds=list(zip(lb, ub)),
        method="highs",
    )
    if res.success:
        return True, np.asarray(res.x, dtype=float)
    return False, str(res.message)


def solve_quadprog(Q, q, c, A, b, lb, ub):
    try:
        import quadprog
    except ImportError as exc:
        raise ImportError(
            "quadprog is required for QP_SOLVER='quadprog'. Install it with "
            "`pip install quadprog`."
        ) from exc

    n = len(q)

    # quadprog is an active-set solver and is sensitive to badly scaled rows.
    # Scale every affine inequality by a positive number. This leaves the
    # feasible set EXACTLY unchanged but greatly improves numerical robustness.
    if A.size:
        Awork = np.asarray(A, dtype=float).copy()
        bwork = np.asarray(b, dtype=float).copy()
        row_scale = np.maximum(np.linalg.norm(Awork, axis=1), np.abs(bwork))
        row_scale = np.maximum(row_scale, 1.0)
        Awork /= row_scale[:, None]
        bwork /= row_scale
    else:
        Awork = np.zeros((0, n), dtype=float)
        bwork = np.zeros(0, dtype=float)

    # Ensure a numerically positive-definite Hessian without changing the
    # practical optimizer. The manuscript cost is convex; this is only the
    # regularization required by Goldfarb-Idnani quadprog.
    Qs = 0.5 * (Q + Q.T)
    eigmin = float(np.min(np.linalg.eigvalsh(Qs)))
    reg = max(1.0e-9, -eigmin + 1.0e-9)
    Qs = Qs + reg * np.eye(n)

    # quadprog solves 1/2 x' G x - a' x subject to C' x >= d.
    C_blocks, d_blocks = [], []
    if Awork.size:
        C_blocks.append(-Awork.T)
        d_blocks.append(-bwork)

    # Bounds are also scaled variable-by-variable only through their own rows.
    # No variable transformation is performed, so the returned x is unchanged.
    C_blocks.append(np.eye(n))
    d_blocks.append(np.asarray(lb, dtype=float))
    C_blocks.append(-np.eye(n))
    d_blocks.append(-np.asarray(ub, dtype=float))
    C = np.concatenate(C_blocks, axis=1)
    d = np.concatenate(d_blocks)

    t0 = time.perf_counter()
    try:
        out = quadprog.solve_qp(Qs, -q, C, d, meq=0)
    except ValueError as exc:
        feasible, detail = _linear_feasibility_check(A, b, lb, ub)
        if feasible is True:
            # The affine constraints are genuinely feasible. Retry from an
            # equivalent uniformly rescaled constraint system. This addresses
            # occasional Goldfarb-Idnani false infeasibility reports.
            global_scale = max(
                1.0,
                float(np.max(np.abs(C))) if C.size else 1.0,
                float(np.max(np.abs(d))) if d.size else 1.0,
            )
            try:
                out = quadprog.solve_qp(Qs, -q, C / global_scale, d / global_scale, meq=0)
            except ValueError as exc2:
                raise RuntimeError(
                    "quadprog reported inconsistent constraints even though an "
                    "independent linear feasibility check found a feasible point. "
                    "This is a numerical active-set failure, not a manuscript-QP "
                    "infeasibility. Try a newer quadprog build or QP_SOLVER='qqp'."
                ) from exc2
        elif feasible is False:
            raise RuntimeError(
                "The manuscript QP is genuinely infeasible at this sample. "
                f"Linear feasibility diagnostic: {detail}"
            ) from exc
        else:
            raise

    runtime = time.perf_counter() - t0
    z = np.asarray(out[0], dtype=float)
    return z, objective_value(Q, q, c, z), runtime, 0, "quadprog solved"


def solve_qqp(Q, q, c, A, b, lb, ub, x0, cfg: Config):
    try:
        from raiselab import qqp
    except ImportError as exc:
        raise ImportError(
            "RAISELab QQP is required for QP_SOLVER='qqp'. Install/use the "
            "same open QQP package used in the previous code and manuscript."
        ) from exc

    n = len(q)
    G = np.zeros((0, n))
    r_eq = np.zeros(0)
    t0 = time.perf_counter()
    z, obj, al_it, info = qqp(
        A=Q, b=q, c=c,
        G=G, r=r_eq,
        H=A, h=b,
        x0=x0, lb=lb, ub=ub,
        normalize_constraints=True,
        scale_objective=True,
        regularize_weak_cost=True,
        eps_reg=1e-8,
        bound_active_tol=1e-7,
        lambda0=None,
        mu0=cfg.qqp_mu0,
        mu_min=1.0,
        mu_max=1e8,
        constraint_tol=1e-6,
        stationarity_tol=None,
        min_al_iters=2,
        max_al_iters=cfg.qqp_max_al_iters,
        phr_region_tol=1e-12,
        max_region_iters=cfg.qqp_max_region_iters,
        update_mu=True,
        mu_update_rule="primal_dual_balance",
        balance_factor=10.0,
        mu_increase=2.0,
        mu_decrease=1.5,
        allow_mu_decrease=True,
        primal_stall_ratio=0.90,
        rho0=cfg.qqp_rho0,
        rho_decay=cfg.qqp_rho_decay,
        rho_tol=1e-9,
        obj_tol=1e-10,
        grad_tol=None,
        max_outer_iters=cfg.qqp_max_outer_iters,
        max_consecutive_outer_rejections=5,
        alpha0=None,
        alpha_step0=None,
        alpha_step_fraction=1.0,
        alpha_step_decay=0.20,
        alpha_step_tol=1e-8,
        max_inner_iters=cfg.qqp_max_inner_iters,
        max_consecutive_inner_rejections=10,
        alpha_warm_start=True,
        alpha_warm_start_scale=0.5,
        qubo_solver=cfg.qqp_qubo_solver,
        classical_time_limit=None,
        classical_mip_gap=0.0,
        classical_threads=None,
        silent=True,
        verbose=False,
        store_full_history=False,
        store_z_history=False,
        return_info=True,
    )
    runtime = time.perf_counter() - t0
    msg = info.get("message", "qqp solved") if isinstance(info, dict) else "qqp solved"
    return np.asarray(z, float), float(obj), runtime, int(al_it), msg


def solve_selected_qp(Q, q, c, A, b, lb, ub, x0, cfg: Config, solver_name: str):
    if solver_name.lower() == "quadprog":
        return solve_quadprog(Q, q, c, A, b, lb, ub)
    if solver_name.lower() == "qqp":
        return solve_qqp(Q, q, c, A, b, lb, ub, x0, cfg)
    raise ValueError("QP solver must be 'quadprog' or 'qqp'")


def qp_residuals(z, A, b, lb, ub):
    ineq = float(np.max(np.maximum(A @ z - b, 0.0))) if A.size else 0.0
    bnd = float(max(np.max(np.maximum(lb - z, 0.0)), np.max(np.maximum(z - ub, 0.0))))
    return ineq, bnd



def _lp_feasible(A, b, bounds):
    """Return (feasible, solution/message) for a linear feasibility problem."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None, "scipy.optimize.linprog unavailable"
    n = len(bounds)
    res = linprog(
        np.zeros(n),
        A_ub=A if np.size(A) else None,
        b_ub=b if np.size(A) else None,
        bounds=bounds,
        method="highs",
    )
    if res.success:
        return True, np.asarray(res.x, dtype=float)
    return False, str(res.message)


def diagnose_hard_qp_infeasibility(A, b, lb, ub, meta, U_prev, cfg: Config):
    """Diagnose a hard-QP infeasibility without changing the controller.

    The routine performs auxiliary LPs only after a failed feasibility check.
    It tests constraint groups, wheel-acceleration bounds, and individual rows,
    then solves a minimum-normalized-slack LP to rank the rows participating in
    the conflict.  None of these auxiliary solutions is applied to the robots.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    m, n = A.shape

    labels = list(meta.get("constraint_labels", [f"row_{j}" for j in range(m)]))
    groups = list(meta.get("constraint_groups", ["unknown"] * m))
    details = list(meta.get("constraint_details", [""] * m))
    if len(labels) != m:
        labels = [f"row_{j}" for j in range(m)]
    if len(groups) != m:
        groups = ["unknown"] * m
    if len(details) != m:
        details = [""] * m

    standard_bounds = list(zip(lb, ub))
    full_feasible, full_detail = _lp_feasible(A, b, standard_bounds)

    # Group-removal tests: if dropping a group restores feasibility, that group
    # is directly implicated in the current infeasibility.
    unique_groups = list(dict.fromkeys(groups))
    group_tests = {}
    for g in unique_groups:
        keep = np.array([gg != g for gg in groups], dtype=bool)
        feas, _ = _lp_feasible(A[keep], b[keep], standard_bounds)
        group_tests[g] = feas

    # Separate the dynamic acceleration window from the absolute wheel-speed box.
    absolute_bounds = [(-cfg.wheel_speed_max, cfg.wheel_speed_max)] * n
    feas_without_accel, _ = _lp_feasible(A, b, absolute_bounds)
    unbounded = [(None, None)] * n
    feas_without_wheel_bounds, _ = _lp_feasible(A, b, unbounded)

    # Minimum normalized row slacks. Positive values rank constraints that need
    # relaxation under the actual wheel bounds. Row normalization prevents units
    # and coefficient magnitudes from dominating the ranking.
    row_scale = np.maximum(np.linalg.norm(A, axis=1), np.abs(b)) if m else np.zeros(0)
    if m:
        row_scale = np.maximum(row_scale, 1.0e-12)
        An = A / row_scale[:, None]
        bn = b / row_scale
    else:
        An, bn = A.copy(), b.copy()

    slack_values = np.full(m, np.nan)
    slack_solution = None
    try:
        from scipy.optimize import linprog
        if m:
            c = np.r_[np.zeros(n), np.ones(m)]
            A_aug = np.hstack([An, -np.eye(m)])
            bounds_aug = standard_bounds + [(0.0, None)] * m
            res = linprog(c, A_ub=A_aug, b_ub=bn, bounds=bounds_aug, method="highs")
            if res.success:
                slack_solution = np.asarray(res.x[:n], dtype=float)
                slack_values = np.asarray(res.x[n:], dtype=float)
    except ImportError:
        pass

    # Individual-row removal tests are performed only for rows with nontrivial
    # minimum slack (or all rows if the slack LP is unavailable). This keeps the
    # diagnostic inexpensive even for larger multi-agent cases.
    candidate_rows = np.flatnonzero(np.nan_to_num(slack_values, nan=1.0) > 1.0e-9)
    if candidate_rows.size == 0 and full_feasible is False:
        candidate_rows = np.arange(m)
    row_restore = np.zeros(m, dtype=bool)
    for j in candidate_rows:
        keep = np.ones(m, dtype=bool)
        keep[j] = False
        feas, _ = _lp_feasible(A[keep], b[keep], standard_bounds)
        row_restore[j] = (feas is True)

    implicated_groups = [g for g, feas in group_tests.items() if feas is True]
    if implicated_groups:
        primary = "+".join(implicated_groups)
    elif feas_without_accel is True:
        primary = "wheel_acceleration_bounds"
    elif feas_without_wheel_bounds is True:
        primary = "wheel_speed_or_acceleration_bounds"
    else:
        primary = "coupled_hard_constraints"

    rows = []
    order = np.argsort(-np.nan_to_num(slack_values, nan=-1.0)) if m else []
    for rank, j in enumerate(order, start=1):
        if np.isfinite(slack_values[j]) and slack_values[j] <= 1.0e-10 and not row_restore[j]:
            continue
        lhs = float(A[j] @ slack_solution) if slack_solution is not None else np.nan
        rows.append({
            "constraint_row": int(j),
            "rank": int(rank),
            "constraint_label": labels[j],
            "constraint_group": groups[j],
            "constraint_detail": details[j],
            "rhs": float(b[j]),
            "lhs_at_min_slack_solution": lhs,
            "normalized_slack_required": float(slack_values[j]) if np.isfinite(slack_values[j]) else np.nan,
            "remove_this_row_restores_feasibility": bool(row_restore[j]),
        })

    # Wheel-bound state for reporting. Acceleration bounds are embedded in lb/ub.
    du = cfg.dt * cfg.wheel_accel_max
    bound_info = {
        "du_max_per_sample": float(du),
        "min_lb": float(np.min(lb)),
        "max_ub": float(np.max(ub)),
        "num_variables_at_lower_absolute_speed_limit": int(np.sum(np.isclose(lb, -cfg.wheel_speed_max))),
        "num_variables_at_upper_absolute_speed_limit": int(np.sum(np.isclose(ub, cfg.wheel_speed_max))),
    }

    summary = {
        "full_feasible": full_feasible,
        "full_feasibility_detail": str(full_detail),
        "primary_diagnosis": primary,
        "implicated_groups": implicated_groups,
        "group_removal_tests": group_tests,
        "feasible_if_acceleration_bounds_removed": feas_without_accel,
        "feasible_if_all_wheel_bounds_removed": feas_without_wheel_bounds,
        "minimum_total_normalized_slack": float(np.nansum(slack_values)) if m else 0.0,
        "number_rows_requiring_slack": int(np.sum(np.nan_to_num(slack_values, nan=0.0) > 1.0e-9)),
        "bound_info": bound_info,
        "rows": rows,
    }
    return summary


def print_first_infeasibility_diagnostic(step, t, diag, max_rows=8):
    """Compact console report for the first genuine hard-QP infeasibility."""
    print("\n" + "=" * 78)
    print("FIRST HARD-QP INFEASIBILITY DIAGNOSTIC")
    print(f"step / time : {step} / {t:.6f} s")
    print(f"primary     : {diag.get('primary_diagnosis', 'unknown')}")
    print(f"groups      : {diag.get('implicated_groups', [])}")
    print(f"no accel bd : feasible={diag.get('feasible_if_acceleration_bounds_removed')}")
    print(f"no wheel bd : feasible={diag.get('feasible_if_all_wheel_bounds_removed')}")
    print(f"min slack   : {diag.get('minimum_total_normalized_slack', np.nan):.6g}")
    print("Top conflicting rows:")
    for row in diag.get("rows", [])[:max_rows]:
        print(
            f"  [{row['constraint_row']:>3}] {row['constraint_label']:<28} "
            f"group={row['constraint_group']:<20} "
            f"slack={row['normalized_slack_required']:.3e} "
            f"single-row-fix={row['remove_this_row_restores_feasibility']}"
        )
        if row.get("constraint_detail"):
            print(f"        {row['constraint_detail']}")
    print("=" * 78 + "\n")


# ============================================================================
# DISTRIBUTED ADMM EXACTLY IN THE PAPER'S NEIGHBOR-COPY FORM
# ============================================================================

def local_physical_constraints(i: int, U_prev: np.ndarray, cfg: Config):
    rows, rhs = [], []
    row = cfg.wheel_radius / 2.0 * np.array([1.0, 1.0])
    rows += [row.copy(), -row.copy()]
    rhs += [cfg.speed_forward_max, cfg.speed_reverse_max]
    row = cfg.wheel_radius / cfg.track_width * np.array([1.0, -1.0])
    rows += [row.copy(), -row.copy()]
    rhs += [cfg.yaw_rate_max, cfg.yaw_rate_max]
    du = cfg.dt * cfg.wheel_accel_max
    lb = np.maximum(-cfg.wheel_speed_max, U_prev[i] - du)
    ub = np.minimum(cfg.wheel_speed_max, U_prev[i] + du)
    return rows, rhs, lb, ub


def distributed_solve(X, P, Pd_next, U_prev, c_hat, delta_c, r_hat, delta_r,
                      cfg: Config, solver_name: str, admm_memory: dict):
    """Solve the paper's neighbor-copy distributed QP using consensus ADMM.

    The optimization model is unchanged from the manuscript. This implementation
    differs from the earlier numerical version only in the ADMM convergence layer:
    (i) robot-to-robot safety uses robot-center displacement for both robots,
    (ii) rho_A is adapted by residual balancing, and
    (iii) a consensus iterate is returned only after the prescribed primal and
    dual tolerances are actually satisfied.
    """
    Ns = safety_neighbors(X, cfg)
    owned = {i: sorted([a for a in Ns[i] if i < a]) for i in range(cfg.n)}

    # Warm start the consensus variable and, when available, the previous final
    # penalty. The multipliers below are unscaled Lagrange multipliers, so they
    # remain valid when rho_A changes and must not be rescaled.
    if cfg.admm_warm_start and admm_memory.get("z") is not None:
        zcons = admm_memory["z"].copy()
    else:
        zcons = U_prev.copy()

    rhoA = float(admm_memory.get("rho", cfg.admm_rho)) if cfg.admm_warm_start else float(cfg.admm_rho)
    rhoA = min(max(rhoA, cfg.admm_rho_min), cfg.admm_rho_max)

    # Duals: own lambda_i^i and copy lambda_a^i. These are unscaled multipliers.
    lam_own = np.zeros((cfg.n, 2))
    lam_copy: Dict[Tuple[int, int], np.ndarray] = {}
    if cfg.admm_warm_start:
        old_own = admm_memory.get("lam_own")
        if isinstance(old_own, np.ndarray) and old_own.shape == lam_own.shape:
            lam_own[:] = old_own
        for key, val in admm_memory.get("lam_copy", {}).items():
            if key[1] in owned.get(key[0], []):
                lam_copy[key] = np.asarray(val, dtype=float).copy()
    for i in range(cfg.n):
        for a in owned[i]:
            lam_copy.setdefault((i, a), np.zeros(2))

    x_own = U_prev.copy()
    x_copy: Dict[Tuple[int, int], np.ndarray] = {
        (i, a): U_prev[a].copy()
        for i in range(cfg.n)
        for a in owned[i]
    }

    history = []
    total_qp_runtime = 0.0
    total_qp_iterations = 0
    local_messages = []
    converged = False

    for s in range(cfg.admm_max_iters):
        # ----- local primal updates -----
        for i in range(cfg.n):
            copies = owned[i]
            m = 1 + len(copies)
            nv = 2 * m
            Q = np.zeros((nv, nv))
            q = np.zeros(nv)
            c0 = 0.0

            Hi, fi, ci = local_cost_terms(i, X, P, Pd_next, U_prev, cfg)
            Q[:2, :2] += Hi + rhoA * np.eye(2)
            q[:2] += fi + lam_own[i] - rhoA * zcons[i]
            c0 += (
                ci
                - float(lam_own[i] @ zcons[i])
                + 0.5 * rhoA * float(zcons[i] @ zcons[i])
            )

            for k, a in enumerate(copies, start=1):
                sl = slice(2 * k, 2 * k + 2)
                lam = lam_copy[(i, a)]
                Q[sl, sl] += rhoA * np.eye(2)
                q[sl] += lam - rhoA * zcons[a]
                c0 += (
                    -float(lam @ zcons[a])
                    + 0.5 * rhoA * float(zcons[a] @ zcons[a])
                )

            rows, rhs = [], []
            prows, prhs, lb0, ub0 = local_physical_constraints(i, U_prev, cfg)
            for rr, bb in zip(prows, prhs):
                row = np.zeros(nv)
                row[:2] = rr
                add_row(rows, rhs, row, bb)

            # --------------------------------------------------------------
            # Target-group safety: controlled-point displacement, exactly as
            # in the paper's target-group affine constraint.
            # --------------------------------------------------------------
            Dp_i = cfg.dt * G_matrix(X[i, 2], cfg)
            d_tg = P[i] - c_hat[i]
            R = r_hat[i] + cfg.robot_target_min_distance + cfg.lookahead
            htg = float(d_tg @ d_tg - R * R)

            row = np.zeros(nv)
            row[:2] = -2.0 * (d_tg @ Dp_i)
            bb = (
                htg
                - 2.0 * float(d_tg @ delta_c[i])
                - 2.0 * R * float(delta_r[i])
            )
            add_row(rows, rhs, row, bb)

            # --------------------------------------------------------------
            # Robot-to-robot safety: robot-center displacement for BOTH robots.
            # This is the same affine constraint used by the centralized QP and
            # by Eq. (rr_affine_safety) in the manuscript.
            # --------------------------------------------------------------
            Dxi_i = center_displacement_matrix(X[i, 2], cfg)
            for k, a in enumerate(copies, start=1):
                Dxi_a = center_displacement_matrix(X[a, 2], cfg)
                d_rr = X[i, :2] - X[a, :2]
                hrr = float(
                    d_rr @ d_rr
                    - cfg.robot_robot_min_distance ** 2
                )

                row = np.zeros(nv)
                row[:2] = -2.0 * (d_rr @ Dxi_i)
                row[2 * k:2 * k + 2] = +2.0 * (d_rr @ Dxi_a)
                add_row(rows, rhs, row, hrr)

            A = np.vstack(rows) if rows else np.zeros((0, nv))
            b = np.asarray(rhs)

            # Finite bounds are required by QQP. Copy bounds are redundant at
            # consensus and use the same physical/acceleration limits as the
            # corresponding actual robot.
            lbs, ubs = [lb0], [ub0]
            for a in copies:
                _, _, lba, uba = local_physical_constraints(a, U_prev, cfg)
                lbs.append(lba)
                ubs.append(uba)
            lb = np.concatenate(lbs)
            ub = np.concatenate(ubs)

            x0 = np.concatenate(
                [x_own[i]] + [x_copy[(i, a)] for a in copies]
            )
            sol, obj, rt, it, msg = solve_selected_qp(
                Q, q, c0, A, b, lb, ub, x0, cfg, solver_name
            )
            total_qp_runtime += rt
            total_qp_iterations += it
            local_messages.append(msg)

            x_own[i] = sol[:2]
            for k, a in enumerate(copies, start=1):
                x_copy[(i, a)] = sol[2 * k:2 * k + 2]

        # ----- consensus update z_a = average(x_a + lambda_a/rho_A) -----
        z_old = zcons.copy()
        for a in range(cfg.n):
            vals = [x_own[a] + lam_own[a] / rhoA]
            for i in range(a):
                if a in owned[i]:
                    vals.append(
                        x_copy[(i, a)] + lam_copy[(i, a)] / rhoA
                    )
            zcons[a] = np.mean(np.vstack(vals), axis=0)

        # ----- dual updates -----
        r_sq = 0.0
        for i in range(cfg.n):
            r_own = x_own[i] - zcons[i]
            lam_own[i] += rhoA * r_own
            r_sq += float(r_own @ r_own)

            for a in owned[i]:
                r_copy = x_copy[(i, a)] - zcons[a]
                lam_copy[(i, a)] += rhoA * r_copy
                r_sq += float(r_copy @ r_copy)

        primal = math.sqrt(r_sq)

        # For consensus ADMM, z_a can occur in several local copies. Weight the
        # dual residual by that multiplicity rather than counting each z_a once.
        dual_sq = 0.0
        for a in range(cfg.n):
            multiplicity = 1 + sum(
                1 for i in range(cfg.n) if a in owned[i]
            )
            dz = zcons[a] - z_old[a]
            dual_sq += multiplicity * float(dz @ dz)
        dual = rhoA * math.sqrt(dual_sq)

        # Dimension-aware primal/dual stopping thresholds.
        number_consensus_blocks = (
            cfg.n + sum(len(v) for v in owned.values())
        )
        consensus_dimension = 2 * number_consensus_blocks

        xnorm = math.sqrt(
            sum(float(x_own[i] @ x_own[i]) for i in range(cfg.n))
            + sum(float(v @ v) for v in x_copy.values())
        )
        znorm = math.sqrt(
            sum(float(zcons[i] @ zcons[i]) for i in range(cfg.n))
            + sum(
                float(zcons[a] @ zcons[a])
                for i in range(cfg.n)
                for a in owned[i]
            )
        )
        eps_pri = (
            math.sqrt(consensus_dimension) * cfg.admm_abs_tol
            + cfg.admm_rel_tol * max(xnorm, znorm)
        )

        dual_scale = math.sqrt(
            sum(float(lam_own[i] @ lam_own[i]) for i in range(cfg.n))
            + sum(float(v @ v) for v in lam_copy.values())
        )
        eps_dual = (
            math.sqrt(consensus_dimension) * cfg.admm_abs_tol
            + cfg.admm_rel_tol * dual_scale
        )

        history.append({
            "admm_iteration": s + 1,
            "primal_residual": primal,
            "dual_residual": dual,
            "primal_tolerance": eps_pri,
            "dual_tolerance": eps_dual,
            "rho": rhoA,
        })

        if primal <= eps_pri and dual <= eps_dual:
            converged = True
            break

        # ----- residual-balancing adaptation of rho_A -----
        if (s + 1) % cfg.admm_adapt_interval == 0:
            if primal > cfg.admm_balance_mu * max(dual, 1.0e-16):
                rhoA = min(
                    cfg.admm_rho_tau * rhoA,
                    cfg.admm_rho_max,
                )
            elif dual > cfg.admm_balance_mu * max(primal, 1.0e-16):
                rhoA = max(
                    rhoA / cfg.admm_rho_tau,
                    cfg.admm_rho_min,
                )
            # lam_own and lam_copy are unscaled multipliers; do NOT rescale them.

    if not converged:
        raise RuntimeError(
            "ADMM failed to satisfy the prescribed primal and dual tolerances "
            f"within {cfg.admm_max_iters} iterations. "
            f"Final primal residual={primal:.3e}, tolerance={eps_pri:.3e}; "
            f"dual residual={dual:.3e}, tolerance={eps_dual:.3e}."
        )

    admm_memory["z"] = zcons.copy()
    admm_memory["lam_own"] = lam_own.copy()
    admm_memory["lam_copy"] = {k: v.copy() for k, v in lam_copy.items()}
    admm_memory["rho"] = float(rhoA)

    return zcons.copy(), history, {
        "admm_iterations": len(history),
        "qp_runtime_sum": total_qp_runtime,
        "qp_inner_iterations_sum": total_qp_iterations,
        "messages": local_messages,
        "active_pairs": sum(len(v) for v in owned.values()),
        "converged": converged,
        "final_rho": float(rhoA),
    }

# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

def initial_robot_states(cfg: Config) -> np.ndarray:
    """Generate separated remote initial conditions without manual box retuning.

    The automatically derived rectangle is attempted first. If a dense large-n
    case cannot be packed, the rectangle is expanded gradually and regenerated.
    This changes only the numerical initial condition, not the controller.
    """
    rng = np.random.default_rng(cfg.seed)
    S0, _ = target_state(0.0, cfg)
    c0 = np.mean(S0, axis=0)

    base_xmin = cfg.random_x_min
    base_xmax = cfg.random_x_max
    base_ymin = cfg.random_y_min
    base_ymax = cfg.random_y_max

    X = None
    used_bounds = None
    for expansion in range(8):
        scale = 1.0 + 0.18 * expansion
        xmid = 0.5 * (base_xmin + base_xmax)
        xhalf = 0.5 * (base_xmax - base_xmin) * scale
        yhalf = 0.5 * (base_ymax - base_ymin) * scale
        xmin = max(cfg.maximum_desired_radius + 0.75, xmid - xhalf)
        xmax = xmid + xhalf
        ymin, ymax = -yhalf, yhalf

        candidate_X = np.zeros((cfg.n, 3), dtype=float)
        success = True
        for i in range(cfg.n):
            placed = False
            candidate = None
            for _ in range(cfg.initial_placement_attempts):
                candidate = c0 + np.array([
                    rng.uniform(xmin, xmax),
                    rng.uniform(ymin, ymax),
                ])
                if all(
                    np.linalg.norm(candidate - candidate_X[j, :2])
                    >= cfg.initial_minimum_spacing
                    for j in range(i)
                ):
                    placed = True
                    break
            if not placed:
                success = False
                break

            candidate_X[i, :2] = candidate
            heading_to_group = math.atan2(
                c0[1] - candidate[1], c0[0] - candidate[0]
            )
            candidate_X[i, 2] = wrap_angle(
                heading_to_group
                + rng.uniform(-cfg.initial_heading_noise,
                              cfg.initial_heading_noise)
            )

        if success:
            X = candidate_X
            used_bounds = (xmin, xmax, ymin, ymax)
            break

    if X is None:
        raise RuntimeError(
            "Automatic initial-condition generator could not place all robots "
            "after adaptive expansion. The requested physical d_rr is too large "
            "for this scenario or n is extreme."
        )

    rr = [np.linalg.norm(X[i, :2] - X[j, :2])
          for i in range(cfg.n) for j in range(i + 1, cfg.n)]
    rt = [np.linalg.norm(X[i, :2] - S0[j])
          for i in range(cfg.n) for j in range(cfg.n)]
    print("Initial condition type       : auto-scaled random remote cloud")
    print("Initial robot cloud center  : "
          f"[{np.mean(X[:,0]):.3f}, {np.mean(X[:,1]):.3f}] m")
    print(f"Initial minimum R-R distance : {min(rr):.3f} m")
    print(f"Initial minimum R-T distance : {min(rt):.3f} m")
    if used_bounds is not None:
        print("Auto initial offset box      : "
              f"x=[{used_bounds[0]:.2f},{used_bounds[1]:.2f}], "
              f"y=[{used_bounds[2]:.2f},{used_bounds[3]:.2f}] m")
    return X


# ============================================================================
# SIMULATION
# ============================================================================

def simulate(cfg: Config, architecture: str, solver_name: str):
    architecture = architecture.lower()
    solver_name = solver_name.lower()
    if architecture not in {"centralized", "distributed"}:
        raise ValueError("ARCHITECTURE must be 'centralized' or 'distributed'")
    if solver_name not in {"quadprog", "qqp"}:
        raise ValueError("QP_SOLVER must be 'quadprog' or 'qqp'")

    X = initial_robot_states(cfg)
    U_prev = np.zeros((cfg.n, 2))
    S0, _ = target_state(0.0, cfg)
    center_est = FixedTimeCenterEstimator(S0, cfg)
    radius_est = CascadedFixedTimeRadiusEstimator(S0, center_est.c_hat, cfg)
    admm_memory: dict = {}

    robot_rows, target_rows, estimator_rows, solver_rows, admm_rows = [], [], [], [], []
    failure_rows = []
    infeasibility_rows = []
    first_infeasibility_printed = False
    nsteps = int(round(cfg.T / cfg.dt))

    for k in range(nsteps + 1):
        t = k * cfg.dt
        S, Vt = target_state(t, cfg)
        c_true = np.mean(S, axis=0)
        rt_true = float(np.max(np.linalg.norm(S - c_true, axis=1)))

        c_hat = center_est.c_hat.copy()
        delta_c = center_est.last_delta.copy()
        r_hat = radius_est.r_hat.copy()
        delta_r = radius_est.last_delta.copy()
        rho = np.linalg.norm(S - c_hat, axis=1)

        P = controlled_point(X, cfg)
        Pd = desired_reference(t, c_hat, np.zeros_like(delta_c), cfg)
        Pd_next = desired_reference(t + cfg.dt, c_hat, delta_c, cfg)

        # Save target and estimator state at current sample before advancing robot.
        center_errors = np.linalg.norm(c_hat - c_true, axis=1)
        radius_errors = np.abs(r_hat - rt_true)
        for i in range(cfg.n):
            estimator_rows.append({
                "step": k, "time_s": t, "robot": i,
                "center_hat_x_m": c_hat[i, 0], "center_hat_y_m": c_hat[i, 1],
                "center_true_x_m": c_true[0], "center_true_y_m": c_true[1],
                "center_error_m": center_errors[i],
                "rho_i_m": rho[i], "radius_hat_m": r_hat[i],
                "radius_true_m": rt_true, "radius_error_m": radius_errors[i],
                "delta_center_x_m": delta_c[i, 0], "delta_center_y_m": delta_c[i, 1],
                "delta_radius_m": delta_r[i],
            })
            target_rows.append({
                "step": k, "time_s": t, "target": i,
                "target_x_m": S[i, 0], "target_y_m": S[i, 1],
                "target_vx_mps": Vt[i, 0], "target_vy_mps": Vt[i, 1],
                "true_center_x_m": c_true[0], "true_center_y_m": c_true[1],
                "true_radius_m": rt_true,
            })

        if k == nsteps:
            U = U_prev.copy()
            solve_info = {
                "objective": np.nan, "runtime": np.nan, "iterations": np.nan,
                "ineq_residual": np.nan, "bound_residual": np.nan,
                "num_variables": 2 * cfg.n, "num_constraints": 0,
                "active_pairs": 0, "message": "terminal sample",
                "solve_failed": False, "failure_reason": "",
            }
            admm_hist = []
        else:
            # Any solver/feasibility failure at this sample is handled by holding
            # the previously applied wheel command. The simulation then advances
            # normally and a fresh optimization is attempted at the next sample.
            # This does not modify the optimization problem or add slack/fallback
            # controllers; it is only a simulation-level hold-last-command policy.
            meta = {}
            try:
                if architecture == "centralized":
                    Q, q, c0, A, b, lb, ub, meta = build_centralized_qp(
                        X, P, Pd_next, U_prev, c_hat, delta_c, r_hat, delta_r, cfg
                    )
                    x0 = U_prev.reshape(-1)
                    feasible_before_solve, feasibility_detail = _linear_feasibility_check(A, b, lb, ub)
                    if feasible_before_solve is False:
                        raise RuntimeError(
                            f"The hard manuscript QP is infeasible before solver call at "
                            f"step={k}, t={t:.6f} s. Diagnostic={feasibility_detail}."
                        )

                    z, obj, runtime, iters, msg = solve_selected_qp(
                        Q, q, c0, A, b, lb, ub, x0, cfg, solver_name
                    )
                    ineq, bnd = qp_residuals(z, A, b, lb, ub)
                    if max(ineq, bnd) > cfg.feasibility_tol:
                        raise RuntimeError(
                            f"Centralized QP returned an infeasible solution: "
                            f"ineq={ineq:.3e}, bound={bnd:.3e}."
                        )

                    U = z.reshape(cfg.n, 2)
                    solve_info = {
                        "objective": obj, "runtime": runtime, "iterations": iters,
                        "ineq_residual": ineq, "bound_residual": bnd,
                        "num_variables": meta["num_variables"],
                        "num_constraints": meta["num_constraints"],
                        "active_pairs": len(meta["safety_pairs"]), "message": msg,
                        "solve_failed": False, "failure_reason": "",
                    }
                    admm_hist = []

                else:
                    U, admm_hist, dinfo = distributed_solve(
                        X, P, Pd_next, U_prev, c_hat, delta_c, r_hat, delta_r,
                        cfg, solver_name, admm_memory
                    )
                    # Evaluate the converged distributed solution against the centralized
                    # QP expression solely for diagnostics; it is not used to compute U.
                    Q, q, c0, A, b, lb, ub, meta = build_centralized_qp(
                        X, P, Pd_next, U_prev, c_hat, delta_c, r_hat, delta_r, cfg
                    )
                    z = U.reshape(-1)
                    ineq, bnd = qp_residuals(z, A, b, lb, ub)
                    obj = objective_value(Q, q, c0, z)
                    distributed_acceptance_tol = max(cfg.feasibility_tol, 1.0e-6)
                    if max(ineq, bnd) > distributed_acceptance_tol:
                        raise RuntimeError(
                            f"ADMM satisfied its consensus stopping test but the returned "
                            f"distributed wheel input failed the centralized-equivalent "
                            f"feasibility audit at t={t:.3f}: "
                            f"ineq={ineq:.3e}, bound={bnd:.3e}, "
                            f"tol={distributed_acceptance_tol:.3e}."
                        )
                    solve_info = {
                        "objective": obj,
                        "runtime": dinfo["qp_runtime_sum"],
                        "iterations": dinfo["qp_inner_iterations_sum"],
                        "ineq_residual": ineq, "bound_residual": bnd,
                        "num_variables": meta["num_variables"],
                        "num_constraints": meta["num_constraints"],
                        "active_pairs": dinfo["active_pairs"],
                        "message": f"ADMM converged in {dinfo['admm_iterations']} iterations",
                        "admm_iterations": dinfo["admm_iterations"],
                        "admm_converged": bool(dinfo.get("converged", False)),
                        "admm_final_rho": float(dinfo.get("final_rho", np.nan)),
                        "solve_failed": False, "failure_reason": "",
                    }
                    for row in admm_hist:
                        admm_rows.append({"step": k, "time_s": t, **row})

            except Exception as exc:
                # Hold the command that was successfully applied at the preceding
                # sample. On the next sample the same controller/solver is called
                # again using the newly evolved robot/target/estimator states.
                U = U_prev.copy()
                admm_hist = []
                reason = f"{type(exc).__name__}: {exc}"

                # Rebuild the centralized-equivalent hard constraints if needed and
                # diagnose genuine infeasibility. This is analysis only; no auxiliary
                # LP solution is ever applied as a control command.
                diag = None
                try:
                    if 'A' not in locals() or np.shape(A)[1] != 2 * cfg.n:
                        Qd, qd, c0d, Ad, bd, lbd, ubd, metad = build_centralized_qp(
                            X, P, Pd_next, U_prev, c_hat, delta_c, r_hat, delta_r, cfg
                        )
                    else:
                        Ad, bd, lbd, ubd, metad = A, b, lb, ub, meta
                    feas_diag, _ = _linear_feasibility_check(Ad, bd, lbd, ubd)
                    if feas_diag is False:
                        diag = diagnose_hard_qp_infeasibility(
                            Ad, bd, lbd, ubd, metad, U_prev, cfg
                        )
                        if not first_infeasibility_printed:
                            print_first_infeasibility_diagnostic(k, t, diag)
                            first_infeasibility_printed = True
                        for rr in diag.get("rows", []):
                            infeasibility_rows.append({
                                "step": k, "time_s": t,
                                "architecture": architecture, "solver": solver_name,
                                "primary_diagnosis": diag.get("primary_diagnosis", ""),
                                "implicated_groups": ";".join(diag.get("implicated_groups", [])),
                                "feasible_if_acceleration_bounds_removed": diag.get("feasible_if_acceleration_bounds_removed"),
                                "feasible_if_all_wheel_bounds_removed": diag.get("feasible_if_all_wheel_bounds_removed"),
                                "minimum_total_normalized_slack": diag.get("minimum_total_normalized_slack", np.nan),
                                **rr,
                            })
                except Exception as diag_exc:
                    diag = {"primary_diagnosis": "diagnostic_failed",
                            "diagnostic_error": f"{type(diag_exc).__name__}: {diag_exc}"}

                failure_rows.append({
                    "step": k,
                    "time_s": t,
                    "architecture": architecture,
                    "solver": solver_name,
                    "failure_reason": reason,
                    "primary_diagnosis": "" if diag is None else diag.get("primary_diagnosis", ""),
                    "implicated_groups": "" if diag is None else ";".join(diag.get("implicated_groups", [])),
                    "minimum_total_normalized_slack": np.nan if diag is None else diag.get("minimum_total_normalized_slack", np.nan),
                })
                warnings.warn(
                    f"Optimization failure at step={k}, t={t:.6f} s; "
                    f"holding previous wheel command and continuing. {reason}"
                    + ("" if diag is None else f" Diagnosis={diag.get('primary_diagnosis', '')}.")
                )

                # Preserve as much QP dimension/constraint information as was
                # available before the failure. NaNs identify quantities that were
                # not produced by a successful solve and are ignored by pandas means.
                meta_local = locals().get("meta", {})
                solve_info = {
                    "objective": np.nan,
                    "runtime": np.nan,
                    "iterations": np.nan,
                    "ineq_residual": np.nan,
                    "bound_residual": np.nan,
                    "num_variables": int(meta_local.get("num_variables", 2 * cfg.n)),
                    "num_constraints": int(meta_local.get("num_constraints", 0)),
                    "active_pairs": int(len(meta_local.get("safety_pairs", [])))
                        if "safety_pairs" in meta_local
                        else int(meta_local.get("active_pairs", 0)),
                    "message": "solve failed; previous control held",
                    "solve_failed": True,
                    "failure_reason": reason,
                    "primary_diagnosis": "" if diag is None else diag.get("primary_diagnosis", ""),
                }

        # Diagnostics based on current state and command.
        actual_center_dist = np.linalg.norm(P - c_true, axis=1)
        desired_rad = desired_radii(cfg)
        angles = np.arctan2(P[:, 1] - c_true[1], P[:, 0] - c_true[0])
        vbody = 0.5 * cfg.wheel_radius * np.sum(U, axis=1)
        yaw = cfg.wheel_radius / cfg.track_width * (U[:, 0] - U[:, 1])
        track_err = np.linalg.norm(P - Pd, axis=1)

        pair_min = np.inf
        for i in range(cfg.n):
            for j in range(i + 1, cfg.n):
                pair_min = min(pair_min, float(np.linalg.norm(X[i, :2] - X[j, :2])))
        target_min_global = np.inf
        for i in range(cfg.n):
            for j in range(cfg.n):
                target_min_global = min(target_min_global, float(np.linalg.norm(X[i, :2] - S[j])))

        for i in range(cfg.n):
            min_rr = min(
                [np.linalg.norm(X[i, :2] - X[j, :2]) for j in range(cfg.n) if j != i],
                default=np.nan,
            )
            min_rt = min(np.linalg.norm(X[i, :2] - S[j]) for j in range(cfg.n))
            robot_rows.append({
                "step": k, "time_s": t, "robot": i,
                "robot_x_m": X[i, 0], "robot_y_m": X[i, 1], "theta_rad": X[i, 2],
                "front_x_m": P[i, 0], "front_y_m": P[i, 1],
                "desired_front_x_m": Pd[i, 0], "desired_front_y_m": Pd[i, 1],
                "right_wheel_speed_radps": U[i, 0], "left_wheel_speed_radps": U[i, 1],
                "linear_velocity_mps": vbody[i], "yaw_rate_radps": yaw[i],
                "controlled_position_error_m": track_err[i],
                "encirclement_radius_m": actual_center_dist[i],
                "desired_encirclement_radius_m": desired_rad[i],
                "radial_error_m": actual_center_dist[i] - desired_rad[i],
                "encirclement_angle_rad": angles[i],
                "min_robot_robot_distance_m": min_rr,
                "min_robot_target_distance_m": min_rt,
                "global_min_robot_robot_distance_m": pair_min,
                "global_min_robot_target_distance_m": target_min_global,
                "center_hat_x_m": c_hat[i, 0], "center_hat_y_m": c_hat[i, 1],
                "radius_hat_m": r_hat[i],
            })

        solver_rows.append({
            "step": k, "time_s": t, "architecture": architecture,
            "solver": solver_name, **solve_info,
        })

        if k < nsteps:
            X = integrate_robots(X, U, cfg)
            U_prev = U.copy()
            S_next, _ = target_state(t + cfg.dt, cfg)
            c_old = center_est.c_hat.copy()
            c_new, _ = center_est.advance(S, S_next)
            radius_est.advance(S, S_next, c_old, c_new, t)

    return {
        "robots": pd.DataFrame(robot_rows),
        "targets": pd.DataFrame(target_rows),
        "estimators": pd.DataFrame(estimator_rows),
        "solver": pd.DataFrame(solver_rows),
        "admm": pd.DataFrame(admm_rows),
        "failures": pd.DataFrame(failure_rows),
        "infeasibility_diagnostics": pd.DataFrame(infeasibility_rows),
    }


# ============================================================================
# METRICS AND REPORTING
# ============================================================================

def empirical_settling_time(time_values, error_values, tol, hold_samples=10):
    time_values = np.asarray(time_values)
    error_values = np.asarray(error_values)
    ok = error_values <= tol
    if len(ok) < hold_samples:
        return np.nan
    for i in range(len(ok) - hold_samples + 1):
        if np.all(ok[i:i + hold_samples]) and np.all(ok[i:]):
            return float(time_values[i])
    return np.nan


def compute_summary(data, cfg: Config, architecture: str, solver_name: str):
    R = data["robots"]
    E = data["estimators"]
    S = data["solver"]
    steady_start = (1.0 - cfg.steady_state_fraction) * cfg.T
    Rs = R[R.time_s >= steady_start]

    center_by_t = E.groupby("time_s")["center_error_m"].max()
    radius_by_t = E.groupby("time_s")["radius_error_m"].max()
    center_emp = empirical_settling_time(center_by_t.index, center_by_t.values,
                                         5.0 * cfg.estimator_capture_tol)
    radius_emp = empirical_settling_time(radius_by_t.index, radius_by_t.values,
                                         5.0 * cfg.estimator_capture_tol)

    summary = {
        "architecture": architecture,
        "solver": solver_name,
        "n_robots": cfg.n,
        "simulation_time_s": cfg.T,
        "sampling_period_s": cfg.dt,
        "mean_steady_position_error_m": float(Rs.controlled_position_error_m.mean()),
        "max_steady_position_error_m": float(Rs.controlled_position_error_m.max()),
        "mean_steady_abs_radial_error_m": float(Rs.radial_error_m.abs().mean()),
        "max_steady_abs_radial_error_m": float(Rs.radial_error_m.abs().max()),
        "minimum_robot_robot_distance_m": float(R.min_robot_robot_distance_m.min()),
        "minimum_robot_target_distance_m": float(R.min_robot_target_distance_m.min()),
        "robot_robot_safety_margin_m": float(R.min_robot_robot_distance_m.min() - cfg.robot_robot_min_distance),
        "robot_target_safety_margin_m": float(R.min_robot_target_distance_m.min() - cfg.robot_target_min_distance),
        "max_center_estimation_error_m": float(E.center_error_m.max()),
        "max_radius_estimation_error_m": float(E.radius_error_m.max()),
        "final_max_center_estimation_error_m": float(E[E.time_s == E.time_s.max()].center_error_m.max()),
        "final_max_radius_estimation_error_m": float(E[E.time_s == E.time_s.max()].radius_error_m.max()),
        "center_theoretical_settling_bound_s": cfg.center_time_bound,
        "radius_theoretical_additional_bound_s": cfg.radius_time_bound,
        "radius_theoretical_total_bound_s": cfg.center_time_bound + cfg.radius_time_bound,
        "center_empirical_settling_time_s": center_emp,
        "radius_empirical_settling_time_s": radius_emp,
        "mean_qp_runtime_s": float(S.runtime.mean()),
        "max_qp_runtime_s": float(S.runtime.max()),
        "mean_qp_iterations": float(S.iterations.mean()),
        "max_inequality_residual": float(S.ineq_residual.max()),
        "max_bound_residual": float(S.bound_residual.max()),
        "mean_active_robot_pairs": float(S.active_pairs.mean()),
        "max_qp_variables": int(S.num_variables.max()),
        "max_qp_constraints": int(S.num_constraints.max()),
        "qp_failure_count": int(S.solve_failed.fillna(False).astype(bool).sum())
            if "solve_failed" in S.columns else 0,
        "qp_failure_fraction": float(S.solve_failed.fillna(False).astype(bool).sum() / max(len(S) - 1, 1))
            if "solve_failed" in S.columns else 0.0,
    }
    F = data.get("failures", pd.DataFrame())
    if not F.empty and "primary_diagnosis" in F.columns:
        counts = F.primary_diagnosis.fillna("").replace("", "unclassified").value_counts()
        summary["qp_failure_diagnosis_counts"] = {str(k): int(v) for k, v in counts.items()}
    else:
        summary["qp_failure_diagnosis_counts"] = {}
    if architecture == "distributed" and "admm_iterations" in S.columns:
        summary["mean_admm_iterations"] = float(S.admm_iterations.fillna(0).mean())
        summary["max_admm_iterations"] = int(S.admm_iterations.fillna(0).max())
    return summary


def per_robot_metrics(R: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    steady_start = (1.0 - cfg.steady_state_fraction) * cfg.T
    rows = []
    for i, d in R[R.time_s >= steady_start].groupby("robot"):
        rows.append({
            "robot": int(i),
            "mean_controlled_position_error_m": float(d.controlled_position_error_m.mean()),
            "max_controlled_position_error_m": float(d.controlled_position_error_m.max()),
            "mean_absolute_radial_error_m": float(d.radial_error_m.abs().mean()),
            "max_absolute_radial_error_m": float(d.radial_error_m.abs().max()),
            "minimum_robot_robot_distance_m": float(d.min_robot_robot_distance_m.min()),
            "minimum_robot_target_distance_m": float(d.min_robot_target_distance_m.min()),
        })
    return pd.DataFrame(rows)


# ============================================================================
# FIGURES AND ANIMATIONS
# ============================================================================

# Publication-oriented style.  The palette is colorblind-friendly and remains
# distinguishable when printed in grayscale because line styles/markers are also
# varied.  Figure sizes are selected to scale cleanly into IEEE one/two-column
# layouts; raster exports are saved at 600 dpi by default.
IEEE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
]
IEEE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
IEEE_LINESTYLES = ["-", "--", "-.", ":"]


def configure_ieee_matplotlib():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

        # Enlarged for half-column subfigures
        "font.size": 14.0,
        "axes.labelsize": 14.0,
        "axes.titlesize": 14.0,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,

        # Stronger axes/curves after reduction in LaTeX
        "axes.linewidth": 1.1,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.0,

        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.75",

        "grid.linewidth": 0.7,
        "grid.alpha": 0.28,

        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "mathtext.fontset": "dejavuserif",
    })


def style_axis(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True, length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def style_3d(ax, title=None):
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.set_zlabel("Time [s]")
    if title:
        ax.set_title(title)
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    ax.grid(True, alpha=0.25)


def savefig(fig, path: Path, cfg: Config):
    if SAVE_PLOTS:
        fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight", facecolor="white")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# Consistent physical-agent marker convention used in every figure/animation.
ROBOT_MARKER = "o"
ROBOT_MARKER_FACE = "black"
ROBOT_MARKER_EDGE = "black"
TARGET_MARKER = "o"
TARGET_MARKER_FACE = "red"
TARGET_MARKER_EDGE = "red"


def _robot_style(i):
    # Colors distinguish trajectories/signals only. Physical robot markers are
    # always black circles and physical target markers are always red circles.
    return IEEE_COLORS[i % len(IEEE_COLORS)], ROBOT_MARKER


def _add_robot_legend(ax, n, include_target=False, include_reference=False, outside=False):
    from matplotlib.lines import Line2D
    handles = []
    for i in range(n):
        color, _ = _robot_style(i)
        handles.append(Line2D([0], [0], color=color, linewidth=1.4,
                              label=f"R{i+1} traj."))
    handles.append(Line2D([0], [0], color="none", marker=ROBOT_MARKER,
                          markerfacecolor=ROBOT_MARKER_FACE,
                          markeredgecolor=ROBOT_MARKER_EDGE, markersize=5.0,
                          label="Robots"))
    if include_target:
        handles.append(Line2D([0], [0], color="red", linestyle=":",
                              marker=TARGET_MARKER,
                              markerfacecolor=TARGET_MARKER_FACE,
                              markeredgecolor=TARGET_MARKER_EDGE, markersize=5.0,
                              label="Targets"))
    if include_reference:
        handles.append(Line2D([0], [0], color="0.15", linestyle="--",
                              label="Desired/reference"))
    if outside:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.015, 0.5),
                  borderaxespad=0.0, ncol=1)
    else:
        ax.legend(handles=handles, loc="upper left", ncol=1)


def axis_limits(data):
    R, T = data["robots"], data["targets"]
    x = np.concatenate([R.robot_x_m.to_numpy(), R.front_x_m.to_numpy(),
                        R.desired_front_x_m.to_numpy(), T.target_x_m.to_numpy()])
    y = np.concatenate([R.robot_y_m.to_numpy(), R.front_y_m.to_numpy(),
                        R.desired_front_y_m.to_numpy(), T.target_y_m.to_numpy()])
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    pad = 0.06 * span + 0.35
    return (float(x.min()-pad), float(x.max()+pad)), (float(y.min()-pad), float(y.max()+pad))


def plot_all(data, cfg: Config, outdir: Path, architecture: str):
    configure_ieee_matplotlib()
    R, T, E, S, A = data["robots"], data["targets"], data["estimators"], data["solver"], data["admm"]
    pdir = outdir / "figures"; pdir.mkdir(exist_ok=True)
    robots = sorted(R.robot.unique())
    xlim, ylim = axis_limits(data)

    # 01: full planar motion with initial/final states and target center trajectory.
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step")
        c, _ = _robot_style(idx)
        ax.plot(d.robot_x_m, d.robot_y_m, color=c, linewidth=1.45,
                label=f"Robot {int(i)+1} trajectory")
        ax.plot(d.robot_x_m.iloc[-1], d.robot_y_m.iloc[-1], marker=ROBOT_MARKER,
                markerfacecolor=ROBOT_MARKER_FACE, markeredgecolor=ROBOT_MARKER_EDGE,
                markeredgewidth=0.45, markersize=6.0, linestyle="None")
    for idx, j in enumerate(sorted(T.target.unique())):
        d = T[T.target == j].sort_values("step")
        ax.plot(d.target_x_m, d.target_y_m, color="red", linestyle=":", linewidth=1.0,
                alpha=0.55)
        ax.plot(d.target_x_m.iloc[-1], d.target_y_m.iloc[-1], marker=TARGET_MARKER,
                markerfacecolor=TARGET_MARKER_FACE, markeredgecolor=TARGET_MARKER_EDGE,
                markersize=5.4, linestyle="None")
    tc = T[T.target == T.target.min()].sort_values("step")
    ax.plot(tc.true_center_x_m, tc.true_center_y_m, color="black", linestyle="--",
            linewidth=1.4, label="Target-group center")
    style_axis(ax, "x position [m]", "y position [m]", "Robot and target trajectories")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")
    _add_robot_legend(ax, len(robots), include_target=True, include_reference=False)
    savefig(fig, pdir / "01_robot_target_trajectories.png", cfg)

    # 02: controlled point vs desired reference.
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step")
        c, _ = _robot_style(idx)
        ax.plot(d.front_x_m, d.front_y_m, color=c, linewidth=1.45,
                label=f"Robot {int(i)+1}")
        ax.plot(d.desired_front_x_m, d.desired_front_y_m, color=c, linestyle="--",
                linewidth=1.05, alpha=0.85)
        ax.plot(d.front_x_m.iloc[0], d.front_y_m.iloc[0], marker=ROBOT_MARKER,
                markerfacecolor=ROBOT_MARKER_FACE, markeredgecolor=ROBOT_MARKER_EDGE,
                markersize=4.8, linestyle="None")
    style_axis(ax, "x position [m]", "y position [m]", "Controlled-point and desired trajectories")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")
    _add_robot_legend(ax, len(robots), include_reference=True)
    savefig(fig, pdir / "02_controlled_point_reference_trajectories.png", cfg)

    # 03 tracking error.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.controlled_position_error_m, color=c, label=f"Robot {int(i)+1}")
    style_axis(ax, "Time [s]", "Position error [m]", "Controlled-point tracking error")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "03_position_tracking_errors.png", cfg)

    # 04 radii.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.encirclement_radius_m, color=c, label=f"Robot {int(i)+1}")
        ax.plot(d.time_s, d.desired_encirclement_radius_m, color=c, linestyle="--", linewidth=0.95)
    style_axis(ax, "Time [s]", "Radius [m]", "Encirclement radii")
    _add_robot_legend(ax, len(robots), include_reference=True)
    savefig(fig, pdir / "04_encirclement_radii.png", cfg)

    # 05 radial error.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.radial_error_m, color=c, label=f"Robot {int(i)+1}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, label="Zero error")
    style_axis(ax, "Time [s]", "Radial error [m]", "Encirclement radial error")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "05_radial_errors.png", cfg)

    # 06 wheel speeds.
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.right_wheel_speed_radps, color=c, linestyle="-", label=f"R{int(i)+1} right")
        ax.plot(d.time_s, d.left_wheel_speed_radps, color=c, linestyle="--", label=f"R{int(i)+1} left")
    ax.axhline(cfg.wheel_speed_max, color="black", linestyle=":", linewidth=1.0, label="Wheel-speed limits")
    ax.axhline(-cfg.wheel_speed_max, color="black", linestyle=":", linewidth=1.0)
    style_axis(ax, "Time [s]", "Wheel speed [rad/s]", "Wheel angular velocities")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "06_wheel_speeds.png", cfg)

    # 07 wheel increments.
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.right_wheel_speed_radps.diff().fillna(0), color=c, label=f"R{int(i)+1} right")
        ax.plot(d.time_s, d.left_wheel_speed_radps.diff().fillna(0), color=c, linestyle="--", label=f"R{int(i)+1} left")
    lim = cfg.dt * cfg.wheel_accel_max
    ax.axhline(lim, color="black", linestyle=":", linewidth=1.0, label="Increment limits")
    ax.axhline(-lim, color="black", linestyle=":", linewidth=1.0)
    style_axis(ax, "Time [s]", "Wheel-speed increment [rad/s]", "Wheel-speed increments")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "07_wheel_speed_increments.png", cfg)

    # 08 linear velocity.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.linear_velocity_mps, color=c, label=f"Robot {int(i)+1}")
    ax.axhline(cfg.speed_forward_max, color="black", linestyle="--", linewidth=1.0, label="Forward/reverse limits")
    ax.axhline(-cfg.speed_reverse_max, color="black", linestyle="--", linewidth=1.0)
    style_axis(ax, "Time [s]", "Linear velocity [m/s]", "Robot linear velocities")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "08_linear_velocities.png", cfg)

    # 09 yaw rates.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.yaw_rate_radps, color=c, label=f"Robot {int(i)+1}")
    ax.axhline(cfg.yaw_rate_max, color="black", linestyle="--", linewidth=1.0, label="Yaw-rate limits")
    ax.axhline(-cfg.yaw_rate_max, color="black", linestyle="--", linewidth=1.0)
    style_axis(ax, "Time [s]", "Yaw rate [rad/s]", "Robot yaw rates")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "09_yaw_rates.png", cfg)

    # 10 center estimates, both components.
    for comp, colhat, coltrue in [("x", "center_hat_x_m", "center_true_x_m"), ("y", "center_hat_y_m", "center_true_y_m")]:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for idx, i in enumerate(robots):
            d = E[E.robot == i].sort_values("step"); c, _ = _robot_style(idx)
            ax.plot(d.time_s, d[colhat], color=c, label=f"Robot {int(i)+1}")
        truth = E[E.robot == min(robots)].sort_values("step")
        ax.plot(truth.time_s, truth[coltrue], color="black", linewidth=1.6, linestyle="--", label=f"True $c_{comp}$")
        ax.axvline(cfg.center_time_bound, color="0.35", linestyle=":", label="$T_c$ bound")
        style_axis(ax, "Time [s]", f"Center {comp} [m]", f"Target-center estimator: {comp} component")
        ax.legend(ncol=2, loc="best")
        savefig(fig, pdir / f"10{comp}_center_estimator_{comp}.png", cfg)

    # 11 center estimator convergence.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = E[E.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.semilogy(d.time_s, np.maximum(d.center_error_m, 1e-12), color=c, label=f"Robot {int(i)+1}")
    ax.axvline(cfg.center_time_bound, color="black", linestyle="--", linewidth=1.0, label="$T_c$ theoretical bound")
    style_axis(ax, "Time [s]", "Center estimation error [m]", "Fixed-time center-estimator convergence")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "11_center_estimator_convergence.png", cfg)

    # 12 radius estimates.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = E[E.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.time_s, d.radius_hat_m, color=c, label=f"Robot {int(i)+1}")
    truth = E[E.robot == min(robots)].sort_values("step")
    ax.plot(truth.time_s, truth.radius_true_m, color="black", linewidth=1.6, linestyle="--", label="True $r_t$")
    ax.axvline(cfg.center_time_bound + cfg.radius_time_bound, color="0.35", linestyle=":", label="$T_c+T_r$ bound")
    style_axis(ax, "Time [s]", "Target-group radius [m]", "Target-group radius estimator")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "12_radius_estimator.png", cfg)

    # 13 radius estimator convergence.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for idx, i in enumerate(robots):
        d = E[E.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.semilogy(d.time_s, np.maximum(d.radius_error_m, 1e-12), color=c, label=f"Robot {int(i)+1}")
    ax.axvline(cfg.center_time_bound + cfg.radius_time_bound, color="black", linestyle="--", linewidth=1.0,
               label="$T_c+T_r$ theoretical bound")
    style_axis(ax, "Time [s]", "Radius estimation error [m]", "Fixed-time radius-estimator convergence")
    ax.legend(ncol=2, loc="best")
    savefig(fig, pdir / "13_radius_estimator_convergence.png", cfg)

    # 14 safety distances.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    rr = R.groupby("time_s").global_min_robot_robot_distance_m.min()
    rt = R.groupby("time_s").global_min_robot_target_distance_m.min()
    ax.plot(rr.index, rr.values, color=IEEE_COLORS[0], label="Minimum robot-robot distance")
    ax.plot(rt.index, rt.values, color=IEEE_COLORS[1], label="Minimum robot-target distance")
    ax.axhline(cfg.robot_robot_min_distance, color=IEEE_COLORS[0], linestyle="--", linewidth=1.0, label="$d_{rr}$")
    ax.axhline(cfg.robot_target_min_distance, color=IEEE_COLORS[1], linestyle="--", linewidth=1.0, label="$d_{rt}$")
    style_axis(ax, "Time [s]", "Distance [m]", "Safety distances")
    ax.legend(loc="best", ncol=2)
    savefig(fig, pdir / "14_safety_distances.png", cfg)

    # 15 objective.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(S.time_s, S.objective, color=IEEE_COLORS[0], label="QP objective")
    style_axis(ax, "Time [s]", "Objective value", "Optimal encirclement objective")
    ax.legend(loc="best")
    savefig(fig, pdir / "15_qp_objective.png", cfg)

    # 16 runtime.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(S.time_s, S.runtime, color=IEEE_COLORS[0], label="Solver runtime")
    ax.axhline(S.runtime.mean(), color="black", linestyle="--", linewidth=1.0,
               label=f"Mean = {S.runtime.mean():.2e} s")
    style_axis(ax, "Time [s]", "Solver runtime [s]", "Optimization runtime")
    ax.legend(loc="best")
    savefig(fig, pdir / "16_solver_runtime.png", cfg)

    # 17 feasibility residuals.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.semilogy(S.time_s, np.maximum(S.ineq_residual, 1e-14), color=IEEE_COLORS[0], label="Inequality residual")
    ax.semilogy(S.time_s, np.maximum(S.bound_residual, 1e-14), color=IEEE_COLORS[1], label="Bound residual")
    ax.axhline(cfg.feasibility_tol, color="black", linestyle="--", linewidth=1.0, label="Feasibility tolerance")
    style_axis(ax, "Time [s]", "Residual", "QP feasibility residuals")
    ax.legend(loc="best")
    savefig(fig, pdir / "17_qp_feasibility_residuals.png", cfg)

    # 18 QP size.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(S.time_s, S.active_pairs, color=IEEE_COLORS[0], label="Active robot-pair constraints")
    ax.plot(S.time_s, S.num_constraints, color=IEEE_COLORS[1], label="Total QP constraints")
    ax.plot(S.time_s, S.num_variables, color=IEEE_COLORS[2], linestyle="--", label="QP variables")
    style_axis(ax, "Time [s]", "Count", "Optimization size and local safety coupling")
    ax.legend(loc="best")
    savefig(fig, pdir / "18_qp_size_active_pairs.png", cfg)

    # 19 ADMM convergence.
    if architecture == "distributed" and not A.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        sampled_steps = sorted(A.step.unique())
        if len(sampled_steps) > 12:
            sampled_steps = sampled_steps[::max(1, len(sampled_steps)//12)]
        for st in sampled_steps:
            d = A[A.step == st]
            ax.semilogy(d.admm_iteration, np.maximum(d.primal_residual, 1e-14), color="0.75", linewidth=0.7)
        last_step = int(A.step.max()); d = A[A.step == last_step]
        ax.semilogy(d.admm_iteration, np.maximum(d.primal_residual, 1e-14), color=IEEE_COLORS[0], linewidth=1.6, label="Primal residual")
        ax.semilogy(d.admm_iteration, np.maximum(d.dual_residual, 1e-14), color=IEEE_COLORS[1], linewidth=1.6, label="Dual residual")
        ax.semilogy(d.admm_iteration, np.maximum(d.primal_tolerance, 1e-14), color=IEEE_COLORS[0], linestyle="--", label="Primal tolerance")
        ax.semilogy(d.admm_iteration, np.maximum(d.dual_tolerance, 1e-14), color=IEEE_COLORS[1], linestyle="--", label="Dual tolerance")
        style_axis(ax, "ADMM iteration", "Residual", "Distributed optimization convergence")
        ax.legend(loc="best", ncol=2)
        savefig(fig, pdir / "19_admm_convergence.png", cfg)

    # 20 steady-state per-robot errors.
    M = per_robot_metrics(R, cfg)
    if not M.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = np.arange(len(M)); w = 0.34
        ax.bar(x-w/2, M.mean_controlled_position_error_m, w, color=IEEE_COLORS[0], label="Position error")
        ax.bar(x+w/2, M.mean_absolute_radial_error_m, w, color=IEEE_COLORS[1], label="Absolute radial error")
        ax.set_xticks(x); ax.set_xticklabels([f"R{int(r)+1}" for r in M.robot])
        style_axis(ax, "Robot", "Steady-state mean error [m]", "Per-robot steady-state errors")
        ax.legend(loc="best")
        savefig(fig, pdir / "20_per_robot_steady_state_errors.png", cfg)

    # 21 3D robot trajectories.
    fig = plt.figure(figsize=(7.5, 5.7)); ax = fig.add_subplot(111, projection="3d")
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.robot_x_m, d.robot_y_m, d.time_s, color=c, linewidth=1.3, label=f"Robot {int(i)+1}")
    style_3d(ax, "Robot trajectories in space-time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    savefig(fig, pdir / "21_3d_robot_time_trajectories.png", cfg)

    # 22 3D controlled-point/reference trajectories.
    fig = plt.figure(figsize=(7.5, 5.7)); ax = fig.add_subplot(111, projection="3d")
    for idx, i in enumerate(robots):
        d = R[R.robot == i].sort_values("step"); c, _ = _robot_style(idx)
        ax.plot(d.front_x_m, d.front_y_m, d.time_s, color=c, linewidth=1.3, label=f"Robot {int(i)+1}")
        ax.plot(d.desired_front_x_m, d.desired_front_y_m, d.time_s, color=c, linestyle="--", linewidth=0.9, alpha=0.8)
    style_3d(ax, "Controlled-point and reference space-time trajectories")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    savefig(fig, pdir / "22_3d_controlled_point_time_trajectories.png", cfg)


def save_animation(animation, mp4_path: Path, gif_path: Path):
    if not SAVE_VIDEO:
        return None
    from matplotlib.animation import FFMpegWriter, PillowWriter
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=VIDEO_FPS, codec="libx264",
                              extra_args=["-preset", "medium", "-crf", "23",
                                          "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                          "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        animation.save(str(mp4_path), writer=writer, dpi=VIDEO_DPI,
                       savefig_kwargs={"facecolor": "white"})
        return mp4_path
    except Exception as exc:
        warnings.warn(f"MP4 creation failed ({exc}); saving GIF instead")
        writer = PillowWriter(fps=min(VIDEO_FPS, 12))
        animation.save(str(gif_path), writer=writer, dpi=min(VIDEO_DPI, 90),
                       savefig_kwargs={"facecolor": "white"})
        return gif_path


def save_animation_2d(data, cfg: Config, outdir: Path):
    if not SAVE_VIDEO and not SAVE_FRAMES and not SHOW_ANIMATIONS_IN_NOTEBOOK:
        return None
    configure_ieee_matplotlib()
    from matplotlib.animation import FuncAnimation
    R, T = data["robots"], data["targets"]
    robots = sorted(R.robot.unique())
    all_steps = sorted(R.step.unique()); steps = all_steps[::cfg.animation_stride]
    if steps[-1] != all_steps[-1]: steps.append(all_steps[-1])
    xlim, ylim = axis_limits(data)

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    fig.subplots_adjust(left=0.08, right=0.74, bottom=0.10, top=0.90)
    style_axis(ax, "x position [m]", "y position [m]")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")

    robot_lines=[]; target_lines=[]; reference_lines=[]; robot_markers=[]; target_markers=[]; front_markers=[]; heading_lines=[]
    for idx, i in enumerate(robots):
        c, _ = _robot_style(idx)
        robot_lines.append(ax.plot([], [], color=c, linewidth=1.55, label=f"Robot {int(i)+1}")[0])
        target_lines.append(ax.plot([], [], color="red", linestyle=":", linewidth=1.0, alpha=0.55,
                                    label=f"Target {int(i)+1}")[0])
        reference_lines.append(ax.plot([], [], color=c, linestyle="--", linewidth=0.95, alpha=0.65)[0])
        robot_markers.append(ax.plot([], [], marker=ROBOT_MARKER, markersize=7.0,
                                     markerfacecolor=ROBOT_MARKER_FACE,
                                     markeredgecolor=ROBOT_MARKER_EDGE,
                                     markeredgewidth=0.55, linestyle="None")[0])
        target_markers.append(ax.plot([], [], marker=TARGET_MARKER, markersize=5.5,
                                      markerfacecolor=TARGET_MARKER_FACE,
                                      markeredgecolor=TARGET_MARKER_EDGE,
                                      markeredgewidth=0.4, linestyle="None")[0])
        front_markers.append(ax.plot([], [], marker="o", markersize=3.5, markerfacecolor="white",
                                     markeredgecolor=c, markeredgewidth=0.9, linestyle="None")[0])
        heading_lines.append(ax.plot([], [], color="black", linewidth=1.0)[0])

    # Trajectory colors identify robots; physical-agent markers have one common convention.
    from matplotlib.lines import Line2D
    handles=[]
    for idx,i in enumerate(robots):
        c,_=_robot_style(idx)
        handles.append(Line2D([0],[0], color=c, linewidth=1.5, label=f"Robot {int(i)+1} trajectory"))
    handles += [
        Line2D([0],[0], color="none", marker=ROBOT_MARKER,
               markerfacecolor=ROBOT_MARKER_FACE, markeredgecolor=ROBOT_MARKER_EDGE,
               markersize=5.5, label="Robots"),
        Line2D([0],[0], color="red", linestyle=":", marker=TARGET_MARKER,
               markerfacecolor=TARGET_MARKER_FACE, markeredgecolor=TARGET_MARKER_EDGE,
               markersize=5.0, label="Targets"),
        Line2D([0],[0], color="0.25", linestyle="--", label="Desired controlled-point paths")
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02,0.5), borderaxespad=0.0,
              title="Agents and signals")

    frames_dir = outdir / "frames_2d"
    if SAVE_FRAMES: frames_dir.mkdir(exist_ok=True)

    def update(frame_idx):
        step=steps[frame_idx]
        cur_time=float(R[R.step==step].time_s.iloc[0])
        ax.set_title(f"Dynamic optimal rotating encirclement, t = {cur_time:.2f} s")
        for idx,i in enumerate(robots):
            d=R[(R.robot==i)&(R.step<=step)].sort_values("step")
            tg=T[(T.target==i)&(T.step<=step)].sort_values("step")
            cur=d.iloc[-1]
            robot_lines[idx].set_data(d.robot_x_m, d.robot_y_m)
            target_lines[idx].set_data(tg.target_x_m, tg.target_y_m)
            reference_lines[idx].set_data(d.desired_front_x_m, d.desired_front_y_m)
            robot_markers[idx].set_data([cur.robot_x_m],[cur.robot_y_m])
            target_markers[idx].set_data([tg.target_x_m.iloc[-1]],[tg.target_y_m.iloc[-1]])
            front_markers[idx].set_data([cur.front_x_m],[cur.front_y_m])
            Lh=0.38
            hx=cur.robot_x_m+Lh*np.cos(cur.theta_rad); hy=cur.robot_y_m+Lh*np.sin(cur.theta_rad)
            heading_lines[idx].set_data([cur.robot_x_m,hx],[cur.robot_y_m,hy])
        if SAVE_FRAMES:
            fig.savefig(frames_dir/f"frame_{frame_idx:05d}.png", dpi=FRAME_DPI, facecolor="white", bbox_inches="tight")
        return robot_lines+target_lines+reference_lines+robot_markers+target_markers+front_markers+heading_lines

    anim=FuncAnimation(fig, update, frames=len(steps), interval=1000/VIDEO_FPS,
                       blit=False, repeat=False, cache_frame_data=False)
    vdir=outdir/"animations"; vdir.mkdir(exist_ok=True)
    saved=save_animation(anim, vdir/"encirclement_2d.mp4", vdir/"encirclement_2d.gif") if SAVE_VIDEO else None
    if SHOW_ANIMATIONS_IN_NOTEBOOK: plt.show()
    else: plt.close(fig)
    return saved


def save_animation_3d(data, cfg: Config, outdir: Path):
    if not SAVE_VIDEO and not SHOW_ANIMATIONS_IN_NOTEBOOK:
        return None
    configure_ieee_matplotlib()
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D
    R, T = data["robots"], data["targets"]
    robots = sorted(R.robot.unique())
    all_steps = sorted(R.step.unique())
    steps = all_steps[::cfg.animation_stride]
    if steps[-1] != all_steps[-1]:
        steps.append(all_steps[-1])
    xlim, ylim = axis_limits(data)
    fig = plt.figure(figsize=(11.0, 6.8))
    fig.subplots_adjust(left=0.03, right=0.75, bottom=0.06, top=0.91)
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(0, cfg.T)

    robot_lines=[]; reference_lines=[]; target_lines=[]; robot_markers=[]; target_markers=[]
    for idx, i in enumerate(robots):
        c, _ = _robot_style(idx)
        robot_lines.append(ax.plot([], [], [], color=c, linewidth=1.4,
                                   label=f"Robot {int(i)+1} controlled point")[0])
        reference_lines.append(ax.plot([], [], [], color=c, linestyle="--",
                                       linewidth=0.9, alpha=0.7)[0])
        target_lines.append(ax.plot([], [], [], color="red", linestyle=":",
                                    linewidth=0.85, alpha=0.50)[0])
        robot_markers.append(ax.plot([], [], [], marker=ROBOT_MARKER, markersize=5.8,
                                     markerfacecolor=ROBOT_MARKER_FACE,
                                     markeredgecolor=ROBOT_MARKER_EDGE,
                                     linestyle="None")[0])
        target_markers.append(ax.plot([], [], [], marker=TARGET_MARKER, markersize=5.0,
                                      markerfacecolor=TARGET_MARKER_FACE,
                                      markeredgecolor=TARGET_MARKER_EDGE,
                                      linestyle="None")[0])

    handles=[]
    for idx, i in enumerate(robots):
        c, _ = _robot_style(idx)
        handles.append(Line2D([0],[0], color=c, linewidth=1.4,
                              label=f"Robot {int(i)+1} controlled-point path"))
    handles += [
        Line2D([0],[0], color="none", marker=ROBOT_MARKER,
               markerfacecolor=ROBOT_MARKER_FACE, markeredgecolor=ROBOT_MARKER_EDGE,
               markersize=5.5, label="Robots"),
        Line2D([0],[0], color="red", linestyle=":", marker=TARGET_MARKER,
               markerfacecolor=TARGET_MARKER_FACE, markeredgecolor=TARGET_MARKER_EDGE,
               markersize=5.0, label="Targets"),
        Line2D([0],[0], color="0.25", linestyle="--", label="Desired controlled-point paths")
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02,0.5),
              borderaxespad=0.0, title="Agents and signals")

    def update(frame_idx):
        step = steps[frame_idx]
        cur_time = float(R[R.step == step].time_s.iloc[0])
        ax.set_title(f"Controlled-point space-time trajectories, t = {cur_time:.2f} s")
        for idx, i in enumerate(robots):
            d = R[(R.robot == i) & (R.step <= step)].sort_values("step")
            tg = T[(T.target == i) & (T.step <= step)].sort_values("step")
            robot_lines[idx].set_data(d.front_x_m, d.front_y_m)
            robot_lines[idx].set_3d_properties(d.time_s)
            reference_lines[idx].set_data(d.desired_front_x_m, d.desired_front_y_m)
            reference_lines[idx].set_3d_properties(d.time_s)
            target_lines[idx].set_data(tg.target_x_m, tg.target_y_m)
            target_lines[idx].set_3d_properties(tg.time_s)
            cur = d.iloc[-1]
            tcur = tg.iloc[-1]
            robot_markers[idx].set_data([cur.front_x_m], [cur.front_y_m])
            robot_markers[idx].set_3d_properties([cur.time_s])
            target_markers[idx].set_data([tcur.target_x_m], [tcur.target_y_m])
            target_markers[idx].set_3d_properties([tcur.time_s])
        return robot_lines + reference_lines + target_lines + robot_markers + target_markers

    anim = FuncAnimation(fig, update, frames=len(steps), interval=1000/VIDEO_FPS,
                         blit=False, repeat=False, cache_frame_data=False)
    vdir = outdir / "animations"; vdir.mkdir(exist_ok=True)
    saved = save_animation(anim, vdir/"encirclement_3d_time.mp4",
                           vdir/"encirclement_3d_time.gif")
    if SHOW_ANIMATIONS_IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    return saved


# ============================================================================
# EXPORT
# ============================================================================

def prepare_output_dir(architecture: str, solver_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = BASE_OUTPUT_ROOT / f"{RUN_NAME}_{architecture}_{solver_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def export_all(data, cfg: Config, outdir: Path, architecture: str, solver_name: str):
    ddir = outdir / "data"; ddir.mkdir(exist_ok=True)
    data["robots"].to_csv(ddir / "robot_timeseries.csv", index=False)
    data["targets"].to_csv(ddir / "target_timeseries.csv", index=False)
    data["estimators"].to_csv(ddir / "estimator_timeseries.csv", index=False)
    data["solver"].to_csv(ddir / "solver_timeseries.csv", index=False)
    data["admm"].to_csv(ddir / "admm_convergence_history.csv", index=False)
    data["failures"].to_csv(ddir / "solver_failures.csv", index=False)
    data["infeasibility_diagnostics"].to_csv(
        ddir / "infeasibility_diagnostics.csv", index=False
    )

    prm = per_robot_metrics(data["robots"], cfg)
    prm.to_csv(ddir / "per_robot_steady_state_metrics.csv", index=False)

    summary = compute_summary(data, cfg, architecture, solver_name)
    pd.DataFrame([summary]).to_csv(ddir / "summary_metrics.csv", index=False)
    with open(ddir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    config_dict = asdict(cfg)
    # ndarray fields are not JSON serializable and are stored separately.
    config_dict.pop("adjacency", None); config_dict.pop("laplacian", None)
    with open(ddir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"architecture": architecture, "solver": solver_name, **config_dict}, f, indent=2, default=float)
    np.savetxt(ddir / "communication_adjacency.csv", cfg.adjacency, delimiter=",")
    np.savetxt(ddir / "communication_laplacian.csv", cfg.laplacian, delimiter=",")

    # A compact TAC-facing table with the principal quantities typically cited
    # in the numerical-experiments text.
    tac_rows = [
        ("Steady-state mean position error [m]", summary["mean_steady_position_error_m"]),
        ("Steady-state max position error [m]", summary["max_steady_position_error_m"]),
        ("Steady-state mean |radial error| [m]", summary["mean_steady_abs_radial_error_m"]),
        ("Minimum robot-robot distance [m]", summary["minimum_robot_robot_distance_m"]),
        ("Minimum robot-target distance [m]", summary["minimum_robot_target_distance_m"]),
        ("Center estimator theoretical bound [s]", summary["center_theoretical_settling_bound_s"]),
        ("Radius estimator total theoretical bound [s]", summary["radius_theoretical_total_bound_s"]),
        ("Center estimator empirical settling [s]", summary["center_empirical_settling_time_s"]),
        ("Radius estimator empirical settling [s]", summary["radius_empirical_settling_time_s"]),
        ("Mean solver runtime [s/successful sample]", summary["mean_qp_runtime_s"]),
        ("QP/ADMM failure count", summary["qp_failure_count"]),
        ("QP/ADMM failure fraction", summary["qp_failure_fraction"]),
        ("Most common QP failure diagnosis",
         max(summary["qp_failure_diagnosis_counts"], key=summary["qp_failure_diagnosis_counts"].get)
         if summary["qp_failure_diagnosis_counts"] else "none"),
        ("Max inequality residual", summary["max_inequality_residual"]),
    ]
    pd.DataFrame(tac_rows, columns=["metric", "value"]).to_csv(ddir / "tac_report_metrics.csv", index=False)

    plot_all(data, cfg, outdir, architecture)
    save_animation_2d(data, cfg, outdir)
    save_animation_3d(data, cfg, outdir)
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    architecture = ARCHITECTURE.lower()
    solver_name = QP_SOLVER.lower()

    print("=" * 78)
    print("TAC Dynamic Optimal Encirclement Simulation")
    print(f"Architecture : {architecture}")
    print(f"QP solver    : {solver_name}")
    print(f"Robots       : {cfg.n}")
    print(f"dt / T       : {cfg.dt:.4f} s / {cfg.T:.2f} s")
    print(f"lambda_2(L)  : {cfg.lambda2:.6g}")
    print(f"center k3    : {cfg.center_k3:.6g}")
    print(f"radius k3    : {cfg.radius_k3:.6g}")
    print(f"T_c bound    : {cfg.center_time_bound:.4f} s")
    print(f"T_r bound    : {cfg.radius_time_bound:.4f} s")
    print("Target safety: controlled point with radius r_hat + d_rt + ell")
    print(f"Auto config   : {cfg.auto_configuration_report()}")
    print("=" * 78)

    outdir = prepare_output_dir(architecture, solver_name)
    t0 = time.perf_counter()
    data = simulate(cfg, architecture, solver_name)
    total_runtime = time.perf_counter() - t0
    summary = export_all(data, cfg, outdir, architecture, solver_name)
    summary["total_wall_runtime_s"] = total_runtime
    with open(outdir / "data" / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nSimulation completed.")
    print(f"Output directory: {outdir}")
    print(f"Total wall runtime: {total_runtime:.3f} s")
    print(f"Mean steady position error: {summary['mean_steady_position_error_m']:.6g} m")
    print(f"Minimum robot-robot distance: {summary['minimum_robot_robot_distance_m']:.6g} m")
    print(f"Minimum robot-target distance: {summary['minimum_robot_target_distance_m']:.6g} m")
    print(f"Mean QP runtime: {summary['mean_qp_runtime_s']:.6g} s/successful sample")
    print(f"QP/ADMM failures: {summary['qp_failure_count']} "
          f"({100.0 * summary['qp_failure_fraction']:.3f}% of solve attempts)")
    if summary["qp_failure_count"] > 0:
        F = data["failures"]
        times = ", ".join(f"{x:.3f}" for x in F.time_s.to_numpy())
        print(f"Failure times [s]: {times}")
        print(f"Failure details saved to: {outdir / 'data' / 'solver_failures.csv'}")
        print(f"Failure diagnosis counts: {summary.get('qp_failure_diagnosis_counts', {})}")
        print(f"Constraint diagnostics saved to: {outdir / 'data' / 'infeasibility_diagnostics.csv'}")


if __name__ == "__main__":
    main()


# In[ ]:





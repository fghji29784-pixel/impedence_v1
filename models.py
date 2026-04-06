"""
models.py — Battery equivalent circuit models and parameter fitting.

Equivalent circuit: Rs + R1||C1 + R2||C2 (Extended Randles)

Changes from DCIM_claude:
  - FitResult에 converged: bool 필드 추가 (curve_fit 실패 시 크래시 방지)
  - _fit_2rc, _fit_1rc: curve_fit 예외 처리 추가 (RuntimeError 발생 시 앱 전체 크래시 방지)
  - voltage_response_2rc, voltage_response_1rc: 사용되지 않는 Rs 파라미터 제거
  - compute_nyquist: f_range를 고정값 대신 fitted τ1, τ2에서 자동 계산
    (대형 셀에서 저주파 아크가 주파수 범위 밖으로 잘리는 문제 수정)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────
# Data transfer object
# ──────────────────────────────────────────────

@dataclass
class FitResult:
    Rs: float
    R1: float
    C1: float
    R2: float
    C2: float
    sigma_R1: float
    sigma_C1: float
    sigma_R2: float
    sigma_C2: float
    r2: float
    rmse_mv: float          # RMSE in millivolts
    converged: bool = True  # False if curve_fit failed to converge
    sigma_W: float = 0.0    # Warburg coefficient [Ω·s^(-1/2)]; 0 for non-Warburg models
    tau1: float = field(init=False)
    tau2: float = field(init=False)
    f1: float = field(init=False)
    f2: float = field(init=False)

    def __post_init__(self):
        self.tau1 = self.R1 * self.C1
        self.tau2 = self.R2 * self.C2
        self.f1 = 1.0 / (2.0 * math.pi * self.tau1) if self.tau1 > 0 else float("nan")
        self.f2 = 1.0 / (2.0 * math.pi * self.tau2) if self.tau2 > 0 else float("nan")


# ──────────────────────────────────────────────
# Cell type presets
# ──────────────────────────────────────────────
# p0 = [R1, C1, R2, C2] initial guess values
# lb / ub = lower / upper bounds
# fit_window_s = recommended fitting window (seconds from p2)
#   - Larger cells (4680/4695) need longer windows to capture slow τ₂

CELL_PRESETS: dict[str, dict] = {
    "18650_3Ah": {
        "label": "18650  (3 Ah)",
        "nominal_capacity_ah": 3.0,
        "p0":    [0.025, 0.8,  0.020, 80.0],
        "lb":    [1e-6,  0.01, 1e-6,  1.0],
        "ub":    [0.5,   100.0, 0.5,  1000.0],
        "p0_1rc": [0.025, 0.8],
        "lb_1rc": [1e-6,  0.01],
        "ub_1rc": [0.5,   100.0],
        "fit_window_s": 5.0,
    },
    "21700_5Ah": {
        "label": "21700  (5 Ah)",
        "nominal_capacity_ah": 5.0,
        "p0":    [0.005, 2.0,  0.010, 150.0],
        "lb":    [1e-6,  1e-3, 1e-6,  1e-3],
        "ub":    [1.0,   500.0, 1.0,  2000.0],
        "p0_1rc": [0.005, 2.0],
        "lb_1rc": [1e-6,  1e-3],
        "ub_1rc": [1.0,   500.0],
        "fit_window_s": 5.0,
    },
    "4680_27Ah": {
        "label": "4680  (27 Ah)",
        "nominal_capacity_ah": 27.0,
        "p0":    [0.003, 20.0, 0.003, 800.0],
        "lb":    [1e-6,  0.5,  1e-6,  50.0],
        "ub":    [0.1,   500.0, 0.1,  5000.0],
        "p0_1rc": [0.003, 20.0],
        "lb_1rc": [1e-6,  0.5],
        "ub_1rc": [0.1,   500.0],
        "fit_window_s": 15.0,
    },
    "4695_32Ah": {
        "label": "4695  (32 Ah)",
        "nominal_capacity_ah": 32.0,
        "p0":    [0.002, 25.0, 0.002, 1000.0],
        "lb":    [1e-6,  0.5,  1e-6,  50.0],
        "ub":    [0.1,   500.0, 0.1,  8000.0],
        "p0_1rc": [0.002, 25.0],
        "lb_1rc": [1e-6,  0.5],
        "ub_1rc": [0.1,   500.0],
        "fit_window_s": 20.0,
    },
    "custom": {
        "label": "Custom",
        "nominal_capacity_ah": None,
        "p0":    [0.005, 2.0,  0.010, 150.0],
        "lb":    [1e-6,  1e-3, 1e-6,  1e-3],
        "ub":    [1.0,   500.0, 1.0,  2000.0],
        "p0_1rc": [0.005, 2.0],
        "lb_1rc": [1e-6,  1e-3],
        "ub_1rc": [1.0,   500.0],
        "fit_window_s": 5.0,
    },
}

# ──────────────────────────────────────────────
# Expected parameter ranges per cell type
# ──────────────────────────────────────────────
# Reference ranges for FORMATION STATE cells (post 1st–2nd cycle).
# ⚠️ These are guideline values — calibrate with your actual measurement data.
# Units: mΩ for resistances.

CELL_EXPECTED_RANGES: dict[str, dict] = {
    "18650_3Ah": {
        "Rs_mohm": (20.0, 120.0),
        "R1_mohm": (8.0,  80.0),
        "R2_mohm": (5.0,  60.0),
    },
    "21700_5Ah": {
        "Rs_mohm": (8.0, 60.0),
        "R1_mohm": (2.0, 30.0),
        "R2_mohm": (1.5, 20.0),
    },
    "4680_27Ah": {
        "Rs_mohm": (0.8, 12.0),
        "R1_mohm": (0.3,  6.0),
        "R2_mohm": (0.3,  6.0),
    },
    "4695_32Ah": {
        "Rs_mohm": (0.6, 10.0),
        "R1_mohm": (0.2,  5.0),
        "R2_mohm": (0.2,  5.0),
    },
    "custom": {
        "Rs_mohm": (0.0, 9999.0),
        "R1_mohm": (0.0, 9999.0),
        "R2_mohm": (0.0, 9999.0),
    },
}


# ──────────────────────────────────────────────
# Time-domain voltage response models
# ──────────────────────────────────────────────

def voltage_response_2rc(
    t: np.ndarray,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
    Vp2: float,
    I: float,
) -> np.ndarray:
    """Extended Randles time-domain response.

    V(t) = Vp2 + R1*I*(1 - exp(-t/τ1)) + R2*I*(1 - exp(-t/τ2))

    Vp2 is the measured voltage at p2, which already includes the ohmic drop
    (Rs*I), so Rs does NOT appear in this equation.
    Vp2, I are fixed constants (pre-calculated before fitting).
    R1, C1, R2, C2 are free parameters fitted by curve_fit.
    """
    tau1 = R1 * C1
    tau2 = R2 * C2
    return (
        Vp2
        + R1 * I * (1.0 - np.exp(-t / tau1))
        + R2 * I * (1.0 - np.exp(-t / tau2))
    )


def voltage_response_1rc(
    t: np.ndarray,
    R1: float,
    C1: float,
    Vp2: float,
    I: float,
) -> np.ndarray:
    """Simple Randles time-domain response (Rs + R1||C1 only)."""
    tau1 = R1 * C1
    return Vp2 + R1 * I * (1.0 - np.exp(-t / tau1))


def voltage_response_2rc_warburg(
    t: np.ndarray,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
    sigma_W: float,
    Vp2: float,
    I: float,
) -> np.ndarray:
    """Extended Randles + Warburg diffusion time-domain response.

    V(t) = Vp2 + R1*I*(1-exp(-t/τ1)) + R2*I*(1-exp(-t/τ2)) + I*sigma_W*sqrt(t)

    The Warburg √t term captures slow solid-state diffusion that appears as a
    monotonically rising voltage beyond the RC exponential transients.  Without
    this term, the 2-RC model absorbs the diffusion voltage into a spuriously
    large R2, causing ~200 % overestimation vs EIS Rct values.

    sigma_W : Warburg coefficient [Ω·s^(-1/2)], ≥ 0
    """
    tau1 = R1 * C1
    tau2 = R2 * C2
    rc_part = R1 * I * (1.0 - np.exp(-t / tau1)) + R2 * I * (1.0 - np.exp(-t / tau2))
    warburg_part = I * sigma_W * np.sqrt(np.maximum(t, 0.0))
    return Vp2 + rc_part + warburg_part


# ──────────────────────────────────────────────
# Frequency-domain impedance model
# ──────────────────────────────────────────────

def impedance_2rc_warburg(
    f: np.ndarray,
    Rs: float,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
    sigma_W: float,
) -> np.ndarray:
    """Complex impedance Z(f) for Extended Randles + Warburg circuit.

    Z(f) = Rs + R1/(1+jωτ1) + R2/(1+jωτ2) + sigma_W*(1-j)/sqrt(2ω)

    Derived from V(t) = I*sigma_W*sqrt(t) in time domain →
    Z_W(ω) = sigma_W * sqrt(π)/2 / sqrt(j*ω)
            = sigma_W * sqrt(π)/2 * (1-j) / sqrt(2*ω)
    """
    _SQRT_PI_OVER_2 = math.sqrt(math.pi) / 2.0   # ≈ 0.8862
    omega = 2.0 * math.pi * np.asarray(f, dtype=float)
    Z1 = R1 / (1.0 + 1j * omega * R1 * C1)
    Z2 = R2 / (1.0 + 1j * omega * R2 * C2)
    Z_W = _SQRT_PI_OVER_2 * sigma_W * (1.0 - 1j) / np.sqrt(2.0 * omega)
    return Rs + Z1 + Z2 + Z_W


def impedance_2rc(
    f: np.ndarray,
    Rs: float,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
) -> np.ndarray:
    """Complex impedance Z(f) for Extended Randles circuit.

    Z(f) = Rs + R1/(1 + j·ω·R1·C1) + R2/(1 + j·ω·R2·C2)
    """
    omega = 2.0 * math.pi * np.asarray(f, dtype=float)
    Z1 = R1 / (1.0 + 1j * omega * R1 * C1)
    Z2 = R2 / (1.0 + 1j * omega * R2 * C2)
    return Rs + Z1 + Z2


# ──────────────────────────────────────────────
# Parameter fitting
# ──────────────────────────────────────────────

def fit_parameters(
    t_fit: np.ndarray,
    V_fit: np.ndarray,
    Rs: float,
    I: float,
    Vp2: float,
    model: str = "extended",
    use_lmfit: bool = False,
    cell_preset: dict | None = None,
) -> FitResult:
    """Fit equivalent circuit parameters to measured voltage transient.

    Parameters
    ----------
    t_fit       : time array offset so that t[0] == 0 (at p2)
    V_fit       : measured voltage array (same length as t_fit)
    Rs          : pre-calculated ohmic resistance (fixed, not fitted)
    I           : applied current in Amperes (positive)
    Vp2         : voltage at p2 (initial condition, fixed)
    model       : 'extended' (Rs+R1C1+R2C2) or 'simple' (Rs+R1C1)
    use_lmfit   : use lmfit instead of scipy (richer uncertainty output)
    cell_preset : dict from CELL_PRESETS — controls initial guesses & bounds

    Returns
    -------
    FitResult with fitted params, 1-sigma errors, R², RMSE(mV).
    If curve_fit fails to converge, result.converged == False and
    parameters are set to initial guess values.
    """
    if cell_preset is None:
        cell_preset = CELL_PRESETS["21700_5Ah"]

    if model == "simple":
        return _fit_1rc(t_fit, V_fit, Rs, I, Vp2, cell_preset)

    if model == "warburg":
        return _fit_2rc_warburg(t_fit, V_fit, Rs, I, Vp2, use_lmfit, cell_preset)

    return _fit_2rc(t_fit, V_fit, Rs, I, Vp2, use_lmfit, cell_preset)


def _fit_2rc(
    t_fit: np.ndarray,
    V_fit: np.ndarray,
    Rs: float,
    I: float,
    Vp2: float,
    use_lmfit: bool,
    cell_preset: dict,
) -> FitResult:
    p0 = cell_preset.get("p0", [0.005, 2.0, 0.010, 150.0])
    lb = cell_preset.get("lb", [1e-6, 1e-3, 1e-6, 1e-3])
    ub = cell_preset.get("ub", [1.0, 500.0, 1.0, 2000.0])

    def model_fixed(t, R1, C1, R2, C2):
        return voltage_response_2rc(t, R1, C1, R2, C2, Vp2, I)

    if use_lmfit:
        try:
            import lmfit
        except ImportError:
            use_lmfit = False

    if use_lmfit:
        import lmfit
        params = lmfit.Parameters()
        for name, p0v, lbv, ubv in zip(["R1", "C1", "R2", "C2"], p0, lb, ub):
            params.add(name, value=p0v, min=lbv, max=ubv)

        def residual(p):
            return model_fixed(t_fit, p["R1"], p["C1"], p["R2"], p["C2"]) - V_fit

        lm_result = lmfit.minimize(residual, params, method="least_squares")
        p = lm_result.params
        R1, C1 = p["R1"].value, p["C1"].value
        R2, C2 = p["R2"].value, p["C2"].value

        def _s(par):
            return par.stderr if par.stderr is not None else float("nan")

        sigma = [_s(p["R1"]), _s(p["C1"]), _s(p["R2"]), _s(p["C2"])]
        V_pred = model_fixed(t_fit, R1, C1, R2, C2)
        converged = lm_result.success
    else:
        converged = True
        R1, C1, R2, C2 = p0
        sigma = [float("nan")] * 4
        try:
            popt, pcov = curve_fit(
                model_fixed,
                t_fit,
                V_fit,
                p0=p0,
                bounds=(lb, ub),
                method="trf",
                maxfev=10000,
            )
            R1, C1, R2, C2 = popt
            sigma = np.sqrt(np.diag(pcov)).tolist()
        except (RuntimeError, ValueError):
            converged = False

    V_pred = model_fixed(t_fit, R1, C1, R2, C2)

    return FitResult(
        Rs=Rs, R1=R1, C1=C1, R2=R2, C2=C2,
        sigma_R1=sigma[0], sigma_C1=sigma[1],
        sigma_R2=sigma[2], sigma_C2=sigma[3],
        r2=_r2(V_fit, V_pred),
        rmse_mv=_rmse_mv(V_fit, V_pred),
        converged=converged,
    )


def _fit_2rc_warburg(
    t_fit: np.ndarray,
    V_fit: np.ndarray,
    Rs: float,
    I: float,
    Vp2: float,
    use_lmfit: bool,
    cell_preset: dict,
) -> FitResult:
    """Fit 2-RC + Warburg model.

    Adds sigma_W (Warburg coefficient) as a 5th free parameter.  The √t term
    absorbs slow diffusion voltage, preventing R2 from inflating to absorb it.
    Expected result: R1+R2 ≈ EIS Rct, sigma_W captures the Warburg portion.
    """
    p0_rc = cell_preset.get("p0", [0.005, 2.0, 0.010, 150.0])
    lb_rc = cell_preset.get("lb", [1e-6, 1e-3, 1e-6, 1e-3])
    ub_rc = cell_preset.get("ub", [1.0, 500.0, 1.0, 2000.0])

    # Initial Warburg guess: small positive value; upper bound capped at 1 Ω·s^-1/2
    p0 = p0_rc + [1e-4]
    lb = lb_rc + [0.0]
    ub = ub_rc + [1.0]

    def model_fixed(t, R1, C1, R2, C2, sigma_W):
        return voltage_response_2rc_warburg(t, R1, C1, R2, C2, sigma_W, Vp2, I)

    if use_lmfit:
        try:
            import lmfit
        except ImportError:
            use_lmfit = False

    if use_lmfit:
        import lmfit
        names = ["R1", "C1", "R2", "C2", "sigma_W"]
        params = lmfit.Parameters()
        for name, p0v, lbv, ubv in zip(names, p0, lb, ub):
            params.add(name, value=p0v, min=lbv, max=ubv)

        def residual(p):
            return model_fixed(t_fit, p["R1"], p["C1"], p["R2"], p["C2"], p["sigma_W"]) - V_fit

        lm_result = lmfit.minimize(residual, params, method="least_squares")
        p = lm_result.params
        R1, C1, R2, C2, sigma_W = (p["R1"].value, p["C1"].value,
                                    p["R2"].value, p["C2"].value, p["sigma_W"].value)
        def _s(par):
            return par.stderr if par.stderr is not None else float("nan")
        sigma = [_s(p["R1"]), _s(p["C1"]), _s(p["R2"]), _s(p["C2"])]
        converged = lm_result.success
    else:
        converged = True
        R1, C1, R2, C2, sigma_W = p0
        sigma = [float("nan")] * 4
        try:
            popt, pcov = curve_fit(
                model_fixed,
                t_fit,
                V_fit,
                p0=p0,
                bounds=(lb, ub),
                method="trf",
                maxfev=20000,
            )
            R1, C1, R2, C2, sigma_W = popt
            sigma = np.sqrt(np.diag(pcov))[:4].tolist()
        except (RuntimeError, ValueError):
            converged = False

    V_pred = model_fixed(t_fit, R1, C1, R2, C2, sigma_W)

    result = FitResult(
        Rs=Rs, R1=R1, C1=C1, R2=R2, C2=C2,
        sigma_R1=sigma[0], sigma_C1=sigma[1],
        sigma_R2=sigma[2], sigma_C2=sigma[3],
        r2=_r2(V_fit, V_pred),
        rmse_mv=_rmse_mv(V_fit, V_pred),
        converged=converged,
    )
    result.sigma_W = float(sigma_W)
    return result


def _fit_1rc(
    t_fit: np.ndarray,
    V_fit: np.ndarray,
    Rs: float,
    I: float,
    Vp2: float,
    cell_preset: dict,
) -> FitResult:
    p0 = cell_preset.get("p0_1rc", [0.005, 2.0])
    lb = cell_preset.get("lb_1rc", [1e-6, 1e-3])
    ub = cell_preset.get("ub_1rc", [1.0, 500.0])

    def model_fixed(t, R1, C1):
        return voltage_response_1rc(t, R1, C1, Vp2, I)

    converged = True
    R1, C1 = p0
    sigma = np.array([float("nan"), float("nan")])
    try:
        popt, pcov = curve_fit(
            model_fixed,
            t_fit,
            V_fit,
            p0=p0,
            bounds=(lb, ub),
            method="trf",
            maxfev=10000,
        )
        R1, C1 = popt
        sigma = np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        converged = False

    V_pred = model_fixed(t_fit, R1, C1)

    return FitResult(
        Rs=Rs, R1=R1, C1=C1, R2=0.0, C2=0.0,
        sigma_R1=sigma[0], sigma_C1=sigma[1],
        sigma_R2=0.0, sigma_C2=0.0,
        r2=_r2(V_fit, V_pred),
        rmse_mv=_rmse_mv(V_fit, V_pred),
        converged=converged,
    )


# ──────────────────────────────────────────────
# Nyquist curve generation
# ──────────────────────────────────────────────

def compute_nyquist(
    Rs: float,
    R1: float,
    C1: float,
    R2: float,
    C2: float,
    f_range: tuple[float, float] | None = None,
    n_points: int = 500,
    sigma_W: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nyquist plot arrays from fitted parameters.

    f_range is auto-derived from the fitted time constants τ1, τ2 when not
    specified. This ensures the full semicircle is always visible, even for
    large cells with very slow τ2 (e.g., τ2 = 30 s → f2 ≈ 0.005 Hz, which
    was outside the old hardcoded lower bound of 0.01 Hz).

    Returns
    -------
    re_z     : Re(Z) array [Ohm]
    neg_im_z : -Im(Z) array [Ohm]
    """
    if f_range is None:
        tau1 = R1 * C1
        tau2 = R2 * C2
        tau_max = max(tau1, tau2, 1e-6)
        tau_min = min(tau1, tau2, 1e-6)
        f_lo = 0.05 / (2.0 * math.pi * tau_max)
        f_hi = 20.0 / (2.0 * math.pi * tau_min)
        f_range = (max(f_lo, 1e-4), min(f_hi, 1e6))

    f_array = np.logspace(
        math.log10(f_range[0]), math.log10(f_range[1]), n_points
    )
    if sigma_W > 0.0:
        Z = impedance_2rc_warburg(f_array, Rs, R1, C1, R2, C2, sigma_W)
    else:
        Z = impedance_2rc(f_array, Rs, R1, C1, R2, C2)
    return np.real(Z), -np.imag(Z)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _r2(y_meas: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_meas - y_pred) ** 2)
    ss_tot = np.sum((y_meas - np.mean(y_meas)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _rmse_mv(y_meas: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_meas - y_pred) ** 2)) * 1000.0)

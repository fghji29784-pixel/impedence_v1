"""
eis_fitter.py — EIS Equivalent Circuit Fitting (CNLS)

Models
------
  2RC        : Rs + R1||C1 + R2||C2                    (5 params)
  2RC_CPE    : Rs + R1||CPE1 + R2||CPE2                (7 params)
  3RC        : Rs + R1||C1 + R2||C2 + R3||C3           (7 params)
  Randles_W  : Rs + (Rct + Zw) || Cdl                  (4 params)

Fitting Method
--------------
  Complex Nonlinear Least Squares (CNLS)
  scipy.optimize.least_squares with residuals:
    res = [Re(Z_model - Z_meas), Im(Z_model - Z_meas)] for each frequency

Model Selection
---------------
  AIC = N_obs * ln(SSR / N_obs) + 2k
  BIC = N_obs * ln(SSR / N_obs) + k * ln(N_obs)
  where N_obs = 2 * len(freq)  (Re + Im per point),  k = number of params

Changes from DCIM_claude:
  - p0 클리핑 방식 변경: lo*1.01, hi*0.99 → lo + 1e-12, hi - 1e-12
    (CPE alpha 파라미터처럼 상한이 1.0인 경우 hi*0.99 = 0.99가 상한 근처 탐색을 막는 문제 수정)

References
----------
  Boukamp 1995 — Solid State Ionics 18-19, CNLS method
  Orazem & Tribollet 2008 — Electrochemical Impedance Spectroscopy, Wiley
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


# ──────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class EISFitResult:
    model_name: str
    model_label: str
    param_names: list[str]
    param_values: np.ndarray      # fitted parameter values (SI units: Ω, F, etc.)
    param_errors: np.ndarray      # ±1σ from Jacobian covariance
    Z_fit: np.ndarray             # complex Z predicted at measured frequencies
    freq: np.ndarray
    re_z_meas: np.ndarray         # measured Re(Z) in Ω
    neg_im_z_meas: np.ndarray     # measured -Im(Z) in Ω
    aic: float
    bic: float
    r2_re: float
    r2_im: float
    r2_total: float               # combined Re+Im R²
    rmse_mohm: float              # RMSE of |Z_model − Z_meas| in mΩ
    converged: bool = True
    message: str = ""

    @property
    def params_dict(self) -> dict[str, float]:
        return dict(zip(self.param_names, self.param_values))

    @property
    def errors_dict(self) -> dict[str, float]:
        return dict(zip(self.param_names, self.param_errors))

    @property
    def Rs(self) -> float:
        return float(self.param_values[0])

    @property
    def R_total(self) -> float:
        """Sum of all resistances (Rs + R1 + R2 [+ R3])."""
        d = self.params_dict
        return sum(v for k, v in d.items() if k.startswith("R"))


# ──────────────────────────────────────────────────────────────
# Impedance functions
# ──────────────────────────────────────────────────────────────

def _z_rc(R: float, C: float, omega: np.ndarray) -> np.ndarray:
    """Ideal RC parallel: Z = R / (1 + jωRC)"""
    return R / (1.0 + 1j * omega * R * C)


def _z_cpe_parallel(R: float, Q: float, alpha: float, omega: np.ndarray) -> np.ndarray:
    """CPE in parallel with R: Z = R / (1 + R·Q·(jω)^α)
    Q [S·s^α], α ∈ (0.5, 1.0]  → CPE admittance Y = Q·(jω)^α
    """
    return R / (1.0 + R * Q * (1j * omega) ** alpha)


def _z_warburg_series(sigma: float, omega: np.ndarray) -> np.ndarray:
    """Semi-infinite Warburg (series): Zw = σ(1−j)/√ω
    σ [Ω·s^(-1/2)] = Warburg coefficient
    """
    return sigma * (1.0 - 1j) / np.sqrt(omega)


def z_model_2rc(params: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, R1, C1, R2, C2 = params
    return Rs + _z_rc(R1, C1, omega) + _z_rc(R2, C2, omega)


def z_model_2rc_cpe(params: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, R1, Q1, a1, R2, Q2, a2 = params
    return Rs + _z_cpe_parallel(R1, Q1, a1, omega) + _z_cpe_parallel(R2, Q2, a2, omega)


def z_model_3rc(params: np.ndarray, omega: np.ndarray) -> np.ndarray:
    Rs, R1, C1, R2, C2, R3, C3 = params
    return Rs + _z_rc(R1, C1, omega) + _z_rc(R2, C2, omega) + _z_rc(R3, C3, omega)


def z_model_randles_w(params: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Classic Randles: Rs + (Rct + Zw) || Cdl
    Z = Rs + (Rct + Zw) / (1 + jω·Cdl·(Rct + Zw))
    """
    Rs, Rct, Cdl, sigma = params
    Zw = _z_warburg_series(sigma, omega)
    Zfaradic = Rct + Zw
    return Rs + Zfaradic / (1.0 + 1j * omega * Cdl * Zfaradic)


# ──────────────────────────────────────────────────────────────
# Geometry-based initial parameter estimation
# ──────────────────────────────────────────────────────────────

def _estimate_p0_from_geometry(
    freq: np.ndarray,
    re_z: np.ndarray,
    neg_im_z: np.ndarray,
) -> dict[str, float]:
    """Rough initial guess from Nyquist geometry."""
    Rs_est = float(re_z.min())
    R_total_est = max(float(re_z.max()) - Rs_est, 1e-4)

    # Split high/low frequency halves
    n = len(freq)
    half = max(n // 2, 1)

    # High-freq arc (first half, higher freq)
    hi_half = neg_im_z[:half]
    lo_half = neg_im_z[half:]
    idx_peak_hi = int(np.argmax(hi_half))
    idx_peak_lo = int(np.argmax(lo_half)) + half

    # Peak frequency → time constant estimate
    f_peak_hi = float(freq[idx_peak_hi]) if idx_peak_hi < n else 1.0
    f_peak_lo = float(freq[idx_peak_lo]) if idx_peak_lo < n else 0.01

    tau1_est = 1.0 / (2 * np.pi * max(f_peak_hi, 1e-3))
    tau2_est = 1.0 / (2 * np.pi * max(f_peak_lo, 1e-5))

    R1_est = max(R_total_est * 0.35, 1e-5)
    R2_est = max(R_total_est * 0.65, 1e-5)
    C1_est = tau1_est / max(R1_est, 1e-6)
    C2_est = tau2_est / max(R2_est, 1e-6)

    # Clamp capacitance to physically reasonable range
    C1_est = float(np.clip(C1_est, 1e-4, 1e5))
    C2_est = float(np.clip(C2_est, 1e-2, 1e6))

    return {
        "Rs": max(Rs_est, 1e-6),
        "R1": R1_est,
        "C1": C1_est,
        "R2": R2_est,
        "C2": C2_est,
        "tau1": tau1_est,
        "tau2": tau2_est,
    }


# ──────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────

def _p0_2rc(geo):
    return [geo["Rs"], geo["R1"], geo["C1"], geo["R2"], geo["C2"]]

def _p0_2rc_cpe(geo):
    return [geo["Rs"], geo["R1"], 1.0 / max(geo["C1"] * geo["R1"], 1e-9), 0.85,
            geo["R2"], 1.0 / max(geo["C2"] * geo["R2"], 1e-9), 0.80]

def _p0_3rc(geo):
    return [geo["Rs"],
            geo["R1"] * 0.5, geo["C1"] * 0.5,
            geo["R1"] * 0.5, geo["C1"] * 5.0,
            geo["R2"],       geo["C2"]]

def _p0_randles_w(geo):
    Rct = geo["R1"] + geo["R2"]
    Cdl = geo["C1"]
    sigma = geo["R2"] * np.sqrt(2 * np.pi * max(1.0 / geo["tau2"], 1e-4))
    sigma = float(np.clip(sigma, 1e-5, 10.0))
    return [geo["Rs"], Rct, Cdl, sigma]


MODELS: dict[str, dict] = {
    "2RC": {
        "label": "2-RC  (Rs + R₁||C₁ + R₂||C₂)",
        "description": "Extended Randles 2-RC 회로. DCIM 기본 모델과 동일.",
        "n_params": 5,
        "param_names": ["Rs", "R1", "C1", "R2", "C2"],
        "z_func": z_model_2rc,
        "make_p0": _p0_2rc,
        "bounds_lo": [1e-7,  1e-7, 1e-5,  1e-7, 1e-3],
        "bounds_hi": [5.0,   5.0,  1e6,   5.0,  1e7 ],
    },
    "2RC_CPE": {
        "label": "2-RC+CPE  (Rs + R₁||CPE₁ + R₂||CPE₂)",
        "description": "CPE(α)로 비이상적 전극 표면 분산 효과 반영. 눌린 반원 재현.",
        "n_params": 7,
        "param_names": ["Rs", "R1", "Q1", "α1", "R2", "Q2", "α2"],
        "z_func": z_model_2rc_cpe,
        "make_p0": _p0_2rc_cpe,
        "bounds_lo": [1e-7, 1e-7, 1e-8, 0.50, 1e-7, 1e-8, 0.50],
        "bounds_hi": [5.0,  5.0,  1e6,  1.00, 5.0,  1e6,  1.00],
    },
    "3RC": {
        "label": "3-RC  (Rs + R₁||C₁ + R₂||C₂ + R₃||C₃)",
        "description": "3개 시정수 분리. 4680/4695 대형 셀 또는 뚜렷한 3-반원 스펙트럼에 유효.",
        "n_params": 7,
        "param_names": ["Rs", "R1", "C1", "R2", "C2", "R3", "C3"],
        "z_func": z_model_3rc,
        "make_p0": _p0_3rc,
        "bounds_lo": [1e-7, 1e-7, 1e-5, 1e-7, 1e-3, 1e-7, 1e-1],
        "bounds_hi": [5.0,  5.0,  1e4,  5.0,  1e5,  5.0,  1e7 ],
    },
    "Randles_W": {
        "label": "Randles+W  (Rs + (Rct+Zw)||Cdl)",
        "description": "고전적 Randles 회로 + Warburg 확산. 저주파 45° 직선 특성 재현.",
        "n_params": 4,
        "param_names": ["Rs", "Rct", "Cdl", "σ"],
        "z_func": z_model_randles_w,
        "make_p0": _p0_randles_w,
        "bounds_lo": [1e-7, 1e-7, 1e-5, 1e-6],
        "bounds_hi": [5.0,  5.0,  1e5,  100.0],
    },
}


# ──────────────────────────────────────────────────────────────
# Core fitting function
# ──────────────────────────────────────────────────────────────

def _compute_metrics(
    re_z: np.ndarray,
    neg_im_z: np.ndarray,
    Z_fit: np.ndarray,
    n_params: int,
) -> dict:
    re_fit = np.real(Z_fit)
    im_fit = -np.imag(Z_fit)   # convert to -Im(Z) convention

    # R² for Re part
    ss_res_re = np.sum((re_z - re_fit) ** 2)
    ss_tot_re = np.sum((re_z - re_z.mean()) ** 2)
    r2_re = 1 - ss_res_re / (ss_tot_re + 1e-30)

    # R² for -Im part
    ss_res_im = np.sum((neg_im_z - im_fit) ** 2)
    ss_tot_im = np.sum((neg_im_z - neg_im_z.mean()) ** 2)
    r2_im = 1 - ss_res_im / (ss_tot_im + 1e-30)

    # Combined (complex) SSR and R²
    ssr = ss_res_re + ss_res_im
    ss_tot = ss_tot_re + ss_tot_im
    r2_total = 1 - ssr / (ss_tot + 1e-30)

    # RMSE of |Z_model - Z_meas| in mΩ
    Z_meas = re_z - 1j * neg_im_z
    rmse_mohm = float(np.sqrt(np.mean(np.abs(Z_fit - Z_meas) ** 2))) * 1000

    # AIC / BIC  (N_obs = 2*n for complex data)
    n = len(re_z)
    N_obs = 2 * n
    k = n_params
    log_term = np.log(max(ssr / N_obs, 1e-30))
    aic = N_obs * log_term + 2 * k
    bic = N_obs * log_term + k * np.log(N_obs)

    return dict(r2_re=r2_re, r2_im=r2_im, r2_total=r2_total,
                rmse_mohm=rmse_mohm, aic=aic, bic=bic)


def fit_eis_model(
    freq: np.ndarray,
    re_z: np.ndarray,
    neg_im_z: np.ndarray,
    model_name: str,
    p0_override: Optional[list] = None,
) -> EISFitResult:
    """
    Fit a single equivalent circuit model to EIS data using CNLS.

    Parameters
    ----------
    freq       : frequency array [Hz]
    re_z       : Re(Z) array [Ω]
    neg_im_z   : -Im(Z) array [Ω]  (positive for capacitive arc)
    model_name : one of MODELS keys
    p0_override: optional manual initial guess

    Returns
    -------
    EISFitResult
    """
    m = MODELS[model_name]
    omega = 2.0 * np.pi * freq

    geo = _estimate_p0_from_geometry(freq, re_z, neg_im_z)
    p0 = p0_override if p0_override is not None else m["make_p0"](geo)
    lo = m["bounds_lo"]
    hi = m["bounds_hi"]

    # Clamp p0 inside bounds with additive epsilon.
    # Additive (not multiplicative) ensures CPE alpha near 1.0 is reachable.
    p0 = [float(np.clip(v, lo[i] + 1e-12, hi[i] - 1e-12)) for i, v in enumerate(p0)]

    # Reference scale for residual normalisation (improves convergence)
    z_scale = float(np.median(np.abs(re_z))) + 1e-9

    def residuals(params):
        try:
            Z_m = m["z_func"](params, omega)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return np.full(2 * len(freq), 1e6)
        res_re = (np.real(Z_m) - re_z) / z_scale
        res_im = (-np.imag(Z_m) - neg_im_z) / z_scale
        return np.concatenate([res_re, res_im])

    converged = True
    message = "OK"

    try:
        sol = least_squares(
            residuals,
            p0,
            bounds=(lo, hi),
            method="trf",
            max_nfev=20000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        popt = sol.x

        # Parameter uncertainty from Jacobian covariance
        try:
            J = sol.jac
            cov = np.linalg.pinv(J.T @ J)
            # MSE = cost / (2*N - k)
            N = 2 * len(freq)
            k = len(popt)
            mse = 2 * sol.cost / max(N - k, 1)
            perr = np.sqrt(np.maximum(np.diag(cov) * mse, 0))
        except np.linalg.LinAlgError:
            perr = np.full_like(popt, np.nan)

        if not sol.success and sol.cost > 1.0:
            converged = False
            message = sol.message

    except Exception as exc:
        popt = np.array(p0, dtype=float)
        perr = np.full_like(popt, np.nan)
        converged = False
        message = str(exc)

    Z_fit = m["z_func"](popt, omega)
    metrics = _compute_metrics(re_z, neg_im_z, Z_fit, m["n_params"])

    return EISFitResult(
        model_name=model_name,
        model_label=m["label"],
        param_names=m["param_names"],
        param_values=popt,
        param_errors=perr,
        Z_fit=Z_fit,
        freq=freq,
        re_z_meas=re_z,
        neg_im_z_meas=neg_im_z,
        aic=metrics["aic"],
        bic=metrics["bic"],
        r2_re=metrics["r2_re"],
        r2_im=metrics["r2_im"],
        r2_total=metrics["r2_total"],
        rmse_mohm=metrics["rmse_mohm"],
        converged=converged,
        message=message,
    )


# ──────────────────────────────────────────────────────────────
# Fit all models and rank by AIC
# ──────────────────────────────────────────────────────────────

def fit_eis_all_models(
    freq: np.ndarray,
    re_z: np.ndarray,
    neg_im_z: np.ndarray,
) -> list[EISFitResult]:
    """
    Fit all 4 equivalent circuit models to EIS data.

    Returns
    -------
    List of EISFitResult sorted by AIC (best = lowest AIC first).
    """
    # Sort freq descending (high → low, standard EIS convention)
    idx_sort = np.argsort(freq)[::-1]
    freq = freq[idx_sort]
    re_z = re_z[idx_sort]
    neg_im_z = neg_im_z[idx_sort]

    # ── Filter out inductive region (neg_im_z < 0) ─────────────────────
    # At high frequencies, cable/contact inductance causes Im(Z) > 0
    # (i.e. neg_im_z < 0). The RC-based equivalent circuit models do not
    # include inductance, so fitting against inductive data causes wildly
    # wrong parameter estimates (R² << 0).  Keep only the capacitive arc
    # region for fitting, but return the full data in EISFitResult for
    # display purposes.
    cap_mask = neg_im_z >= 0
    if cap_mask.sum() >= 5:
        freq_fit  = freq[cap_mask]
        re_z_fit  = re_z[cap_mask]
        nim_fit   = neg_im_z[cap_mask]
    else:
        # Fallback: use all data if almost no capacitive points
        freq_fit, re_z_fit, nim_fit = freq, re_z, neg_im_z

    results = []
    for name in MODELS:
        try:
            r = fit_eis_model(freq_fit, re_z_fit, nim_fit, name)
            # Store the FULL (unfiltered) measured data for display
            r.freq          = freq
            r.re_z_meas     = re_z
            r.neg_im_z_meas = neg_im_z
            results.append(r)
        except Exception as exc:
            # Build a dummy failed result
            m = MODELS[name]
            dummy_params = np.zeros(m["n_params"])
            dummy_errs   = np.full(m["n_params"], np.nan)
            dummy_Z      = np.zeros(len(freq), dtype=complex)
            results.append(EISFitResult(
                model_name=name,
                model_label=m["label"],
                param_names=m["param_names"],
                param_values=dummy_params,
                param_errors=dummy_errs,
                Z_fit=dummy_Z,
                freq=freq,
                re_z_meas=re_z,
                neg_im_z_meas=neg_im_z,
                aic=np.inf,
                bic=np.inf,
                r2_re=-np.inf,
                r2_im=-np.inf,
                r2_total=-np.inf,
                rmse_mohm=np.inf,
                converged=False,
                message=str(exc),
            ))

    # Rank by AIC (lower is better)
    results.sort(key=lambda r: r.aic)
    return results


# ──────────────────────────────────────────────────────────────
# DCIM vs EIS parameter comparison helper
# ──────────────────────────────────────────────────────────────

def compare_dcim_eis(dcim_result, eis_result: EISFitResult) -> list[dict]:
    """
    Compare DCIM fit result with EIS fit result for 2-RC model.
    Returns list of comparison rows for display.

    dcim_result : models.FitResult (has .Rs, .R1, .C1, .R2, .C2, .r2)
    eis_result  : EISFitResult (model_name == '2RC')
    """
    if eis_result.model_name != "2RC":
        return []

    pd_eis = eis_result.params_dict
    rows = []
    for name, dcim_val, eis_key in [
        ("Rs",  dcim_result.Rs * 1000,  "Rs"),
        ("R₁",  dcim_result.R1 * 1000,  "R1"),
        ("C₁",  dcim_result.C1,         "C1"),
        ("R₂",  dcim_result.R2 * 1000,  "R2"),
        ("C₂",  dcim_result.C2,         "C2"),
    ]:
        eis_val_raw = pd_eis.get(eis_key, np.nan)
        # Convert R to mΩ for display
        eis_val = eis_val_raw * 1000 if eis_key.startswith("R") else eis_val_raw
        unit = "mΩ" if eis_key.startswith("R") else "F"

        if not np.isnan(eis_val) and abs(eis_val) > 1e-10:
            diff = dcim_val - eis_val
            pct  = abs(diff) / abs(eis_val) * 100
        else:
            diff = np.nan
            pct  = np.nan

        rows.append({
            "파라미터": name,
            "단위": unit,
            "DCIM": dcim_val,
            "EIS 피팅": eis_val,
            "차이": diff,
            "오차율": pct,
        })
    return rows

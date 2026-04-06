"""
exporter.py — Export DCIM analysis results to Excel and plain text.

Changes from DCIM_claude:
  - openpyxl 누락 시 명확한 ImportError 메시지 추가
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

try:
    import openpyxl  # noqa: F401
except ImportError as _openpyxl_err:
    raise ImportError(
        "Excel 내보내기에 openpyxl이 필요합니다. "
        "설치 명령: pip install openpyxl"
    ) from _openpyxl_err

from models import FitResult


def export_results_excel(
    result: FitResult,
    t_fit: np.ndarray,
    V_meas: np.ndarray,
    V_pred: np.ndarray,
    nyquist_data: tuple[np.ndarray, np.ndarray],
) -> bytes:
    """Create a multi-sheet Excel workbook with analysis results.

    Sheets
    ------
    Parameters : fitted circuit parameters + quality metrics
    Fit Data   : time-domain measurement vs model
    Nyquist    : frequency-domain Nyquist curve data

    Returns
    -------
    bytes of the .xlsx file
    """
    buf = BytesIO()
    re_z, neg_im_z = nyquist_data

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # ── Sheet 1: Parameters ──────────────────────────────────────────
        param_rows = [
            ("Rs [Ω]",  result.Rs,       "—"),
            ("Rs [mΩ]", result.Rs * 1000, "—"),
            ("R1 [Ω]",  result.R1,       result.sigma_R1),
            ("C1 [F]",  result.C1,       result.sigma_C1),
            ("R2 [Ω]",  result.R2,       result.sigma_R2),
            ("C2 [F]",  result.C2,       result.sigma_C2),
            ("τ1 [s]",  result.tau1,     "—"),
            ("τ2 [s]",  result.tau2,     "—"),
            ("f1 [Hz]", result.f1,       "—"),
            ("f2 [Hz]", result.f2,       "—"),
            ("R²",      result.r2,       "—"),
            ("RMSE [mV]", result.rmse_mv, "—"),
            ("수렴 여부", "예" if result.converged else "아니오 (초기값 사용)", "—"),
        ]
        df_params = pd.DataFrame(param_rows, columns=["Parameter", "Value", "1σ Error"])
        df_params.to_excel(writer, sheet_name="Parameters", index=False)

        # ── Sheet 2: Fit Data ────────────────────────────────────────────
        df_fit = pd.DataFrame({
            "time_s": t_fit,
            "time_ms": t_fit * 1000.0,
            "V_measured_V": V_meas,
            "V_fitted_V": V_pred,
            "residual_mV": (V_meas - V_pred) * 1000.0,
        })
        df_fit.to_excel(writer, sheet_name="Fit Data", index=False)

        # ── Sheet 3: Nyquist ─────────────────────────────────────────────
        df_nyq = pd.DataFrame({
            "Re_Z_Ohm": re_z,
            "neg_Im_Z_Ohm": neg_im_z,
        })
        df_nyq.to_excel(writer, sheet_name="Nyquist", index=False)

    return buf.getvalue()


def export_report_text(result: FitResult) -> str:
    """Generate a human-readable plain-text analysis report.

    Returns
    -------
    str containing the formatted report
    """
    converged_str = "예" if result.converged else "아니오 (초기값 — 결과 신뢰 불가)"
    lines = [
        "=" * 50,
        "  DCIM Battery Analysis Report",
        "=" * 50,
        "",
        f"  피팅 수렴: {converged_str}",
        "",
        "── Ohmic Resistance ──",
        f"  Rs  = {result.Rs * 1000:.4f} mΩ",
        "",
        "── High-Frequency RC (SEI / interface) ──",
        f"  R1  = {result.R1 * 1000:.4f} mΩ  ±  {result.sigma_R1 * 1000:.4f} mΩ",
        f"  C1  = {result.C1:.4f} F      ±  {result.sigma_C1:.4f} F",
        f"  τ1  = {result.tau1 * 1000:.3f} ms  →  f1 = {result.f1:.2f} Hz",
        "",
        "── Low-Frequency RC (charge transfer / double layer) ──",
        f"  R2  = {result.R2 * 1000:.4f} mΩ  ±  {result.sigma_R2 * 1000:.4f} mΩ",
        f"  C2  = {result.C2:.4f} F      ±  {result.sigma_C2:.4f} F",
        f"  τ2  = {result.tau2:.4f} s    →  f2 = {result.f2:.4f} Hz",
        "",
        "── Fit Quality ──",
        f"  R²   = {result.r2:.6f}",
        f"  RMSE = {result.rmse_mv:.4f} mV",
        "",
        "── Impedance at DC (f→0) ──",
        f"  Rs + R1 + R2 = {(result.Rs + result.R1 + result.R2) * 1000:.4f} mΩ",
        "=" * 50,
    ]
    return "\n".join(lines)

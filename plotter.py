"""
plotter.py — Matplotlib figure generators for DCIM analysis.

All functions return matplotlib.figure.Figure objects for use with
st.pyplot() in the Streamlit app.

Changes from DCIM_claude:
  - voltage_response_2rc / voltage_response_1rc 호출에서
    사용되지 않는 Rs 파라미터 제거 (models.py 변경 반영)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from models import FitResult, voltage_response_2rc, voltage_response_2rc_warburg, voltage_response_1rc

matplotlib.use("Agg")  # non-interactive backend for Streamlit


# ──────────────────────────────────────────────
# Graph 1 — Raw data with p0/p1/p2 markers
# ──────────────────────────────────────────────

def plot_raw_data(
    df: pd.DataFrame,
    idx_p0: int,
    idx_p1: int,
    idx_p2: int,
) -> Figure:
    """Two-panel figure: voltage (top) and current (bottom) vs time.

    Vertical markers at p0 (gray dashed), p1 (orange dotted), p2 (red solid).
    """
    t = df["time_s"].values
    V = df["voltage_V"].values
    I = df["current_A"].values * 1000.0  # display in mA

    t_p0 = df.loc[idx_p0, "time_s"]
    t_p1 = df.loc[idx_p1, "time_s"]
    t_p2 = df.loc[idx_p2, "time_s"]

    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # Voltage panel
    ax_v.plot(t, V, color="#2196F3", linewidth=0.8, label="Voltage")
    _add_pmarkers(ax_v, t_p0, t_p1, t_p2)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.legend(loc="upper left", fontsize=8)
    ax_v.grid(True, alpha=0.3)

    # Current panel
    ax_i.plot(t, I, color="#FF5722", linewidth=0.8, label="Current")
    _add_pmarkers(ax_i, t_p0, t_p1, t_p2)
    ax_i.set_ylabel("Current (mA)")
    ax_i.set_xlabel("Time (s)")
    ax_i.legend(loc="upper left", fontsize=8)
    ax_i.grid(True, alpha=0.3)

    # Shared legend for markers (add to bottom panel)
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], color="gray",   linestyle="--", label=f"p0 (rest)      t={t_p0:.4f} s"),
        Line2D([0], [0], color="orange", linestyle=":",  label=f"p1 (50% I_set) t={t_p1:.4f} s"),
        Line2D([0], [0], color="red",    linestyle="-",  label=f"p2 (settled)   t={t_p2:.4f} s"),
    ]
    ax_i.legend(handles=marker_handles, loc="lower right", fontsize=7)

    fig.suptitle("Raw Charge Data — Pulse Detection", fontsize=11, fontweight="bold")
    return fig


def _add_pmarkers(ax, t_p0: float, t_p1: float, t_p2: float):
    ax.axvline(t_p0, color="gray",   linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(t_p1, color="orange", linestyle=":",  linewidth=1.2, alpha=0.9)
    ax.axvline(t_p2, color="red",    linestyle="-",  linewidth=1.2, alpha=0.9)


# ──────────────────────────────────────────────
# Graph 2 — Fitting result with residuals
# ──────────────────────────────────────────────

def plot_fit_result(
    t_fit: np.ndarray,
    V_meas: np.ndarray,
    V_pred: np.ndarray,
    result: FitResult,
    Vp2: float | None = None,
    I: float | None = None,
    model: str = "extended",
) -> Figure:
    """Two-panel figure: measured vs fitted voltage (top), residuals in mV (bottom).

    A parameter table is annotated as text on the right side of the top panel.

    Parameters
    ----------
    Vp2, I, model : optional — if provided, the fitted curve is recomputed on a
                    dense 1000-point grid so it always appears smooth, even when
                    the measured data has very sparse or non-uniform sampling.
    """
    t_ms = t_fit * 1000.0       # convert to milliseconds for display
    residuals_mv = (V_meas - V_pred) * 1000.0

    # ── Smooth fitted curve on dense grid ────────────────────────────────
    if Vp2 is not None and I is not None and len(t_fit) > 1:
        t_dense = np.linspace(0.0, float(t_fit[-1]), 1000)
        if model == "simple":
            V_dense = voltage_response_1rc(t_dense, result.R1, result.C1, Vp2, I)
        elif model == "warburg":
            V_dense = voltage_response_2rc_warburg(
                t_dense, result.R1, result.C1, result.R2, result.C2,
                result.sigma_W, Vp2, I,
            )
        else:
            V_dense = voltage_response_2rc(
                t_dense, result.R1, result.C1, result.R2, result.C2, Vp2, I
            )
        t_dense_ms = t_dense * 1000.0
        V_dense_mv = V_dense * 1000.0
    else:
        t_dense_ms = t_ms
        V_dense_mv = V_pred * 1000.0

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[3, 1],
                           height_ratios=[3, 1],
                           hspace=0.08, wspace=0.05)

    ax_fit = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_fit)
    ax_tbl = fig.add_subplot(gs[:, 1])

    # ── Fit overlay ──────────────────────────────────────────────────────
    n_pts = len(t_ms)
    marker_size = max(2, min(8, 300 // max(n_pts, 1)))
    ax_fit.scatter(t_ms, V_meas * 1000, color="#2196F3", s=marker_size,
                   linewidths=0, alpha=0.7, label="Measured", zorder=3)
    if n_pts <= 500:
        ax_fit.plot(t_ms, V_meas * 1000, color="#2196F3", linewidth=0.6,
                    alpha=0.4, zorder=2)
    ax_fit.plot(t_dense_ms, V_dense_mv, color="#F44336", linewidth=1.8,
                linestyle="--", label="Fitted", zorder=4)

    converge_note = "" if result.converged else "  ⚠️ 수렴 실패 — 초기값"
    ax_fit.set_ylabel("Voltage (mV)")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.tick_params(labelbottom=False)
    ax_fit.set_title(
        f"Voltage Transient Fit  (R² = {result.r2:.5f},  RMSE = {result.rmse_mv:.3f} mV){converge_note}",
        fontsize=10,
    )

    # ── Residuals ─────────────────────────────────────────────────────────
    ax_res.scatter(t_ms, residuals_mv, color="#9C27B0", s=marker_size,
                   linewidths=0, alpha=0.7, zorder=3)
    if n_pts <= 500:
        ax_res.plot(t_ms, residuals_mv, color="#9C27B0", linewidth=0.6, alpha=0.4, zorder=2)
    ax_res.axhline(0, color="black", linewidth=0.6)
    ax_res.set_ylabel("Residual\n(mV)", fontsize=8)
    ax_res.set_xlabel("Time from p2 (ms)")
    ax_res.grid(True, alpha=0.3)

    # ── Parameter table ───────────────────────────────────────────────────
    ax_tbl.axis("off")
    table_data = [
        ["Parameter", "Value", "±1σ"],
        ["Rs",        f"{result.Rs * 1000:.3f} mΩ",  "—"],
        ["R1",        f"{result.R1 * 1000:.3f} mΩ",  f"{result.sigma_R1 * 1000:.3f} mΩ"],
        ["C1",        f"{result.C1:.3f} F",           f"{result.sigma_C1:.3f} F"],
        ["R2",        f"{result.R2 * 1000:.3f} mΩ",  f"{result.sigma_R2 * 1000:.3f} mΩ"],
        ["C2",        f"{result.C2:.2f} F",           f"{result.sigma_C2:.2f} F"],
        ["τ1",        f"{result.tau1 * 1000:.2f} ms", "—"],
        ["τ2",        f"{result.tau2:.3f} s",         "—"],
        ["f1",        f"{result.f1:.2f} Hz",          "—"],
        ["f2",        f"{result.f2:.4f} Hz",          "—"],
        ["R²",        f"{result.r2:.5f}",             "—"],
        ["RMSE",      f"{result.rmse_mv:.3f} mV",     "—"],
    ]
    tbl = ax_tbl.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    # Style header row
    for j in range(3):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("DCIM Parameter Fitting Result", fontsize=11, fontweight="bold")
    return fig


# ──────────────────────────────────────────────
# Graph 3 — Nyquist plot
# ──────────────────────────────────────────────

def plot_nyquist(
    re_z: np.ndarray,
    neg_im_z: np.ndarray,
    eis_df: pd.DataFrame | None = None,
    result: FitResult | None = None,
) -> Figure:
    """Single-panel Nyquist plot.

    Parameters
    ----------
    re_z, neg_im_z : DCIM model curve arrays [Ohm]
    eis_df         : optional DataFrame with 're_z' and 'neg_im_z' columns
                     (measured EIS overlay)
    result         : optional FitResult for annotating Rs, peak frequencies
    """
    # Compute data range using only the capacitive portion (neg_im_z ≥ 0).
    all_re = list(re_z * 1000)
    all_im = list(neg_im_z * 1000)
    if eis_df is not None and len(eis_df) > 0:
        eis_cap_df = eis_df[eis_df["neg_im_z"] >= 0]
        all_re += list(eis_cap_df["re_z"] * 1000)
        all_im += list(eis_cap_df["neg_im_z"] * 1000)
    data_span = max(
        max(all_re) - min(all_re),
        max(all_im) - min(all_im),
        1e-9,
    )
    fig_size = min(max(6.0, data_span * 0.6), 10.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # DCIM model curve
    ax.plot(re_z * 1000, neg_im_z * 1000,
            color="#2196F3", linewidth=1.8, label="DCIM model", zorder=3)

    # EIS measured overlay (optional)
    if eis_df is not None and len(eis_df) > 0:
        eis_cap = eis_df[eis_df["neg_im_z"] >= 0]
        eis_ind = eis_df[eis_df["neg_im_z"] < 0]
        if len(eis_cap) > 0:
            ax.scatter(
                eis_cap["re_z"] * 1000,
                eis_cap["neg_im_z"] * 1000,
                color="#F44336", s=18, zorder=4,
                label="EIS measured", marker="o",
            )
        if len(eis_ind) > 0:
            ax.scatter(
                eis_ind["re_z"] * 1000,
                eis_ind["neg_im_z"] * 1000,
                color="#F44336", s=10, zorder=4,
                label=f"EIS inductive ({len(eis_ind)} pts, excluded from fit)",
                marker="x", alpha=0.4,
            )

    # Annotate Rs (DCIM) on x-axis
    if result is not None:
        rs_mohm = result.Rs * 1000
        ax.axvline(rs_mohm, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"Rs(DCIM)={rs_mohm:.2f} mΩ",
            xy=(rs_mohm, 0),
            xytext=(rs_mohm + data_span * 0.03,
                    data_span * 0.05),
            fontsize=7,
            color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
        )

    # Annotate Rs (EIS) — high-frequency real-axis intercept
    if eis_df is not None and len(eis_df) > 0:
        rs_eis_mohm = float(eis_df["re_z"].min()) * 1000
        ax.axvline(rs_eis_mohm, color="#F44336", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"Rs(EIS)={rs_eis_mohm:.2f} mΩ",
            xy=(rs_eis_mohm, 0),
            xytext=(rs_eis_mohm + data_span * 0.03,
                    data_span * 0.12),
            fontsize=7,
            color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336", lw=0.6),
        )

    ax.set_xlabel("Re(Z) (mΩ)", fontsize=10)
    ax.set_ylabel("-Im(Z) (mΩ)", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Nyquist Plot — DCIM Reconstruction", fontsize=11, fontweight="bold")

    ymin, ymax = ax.get_ylim()
    if ymin > 0:
        ax.set_ylim(bottom=-0.02 * ymax)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Graph 4 — EIS Fitting Nyquist
# ──────────────────────────────────────────────

def plot_eis_fit(eis_result) -> Figure:
    """Nyquist plot comparing measured EIS data vs fitted model curve.

    Parameters
    ----------
    eis_result : EISFitResult from eis_fitter.py
    """
    re_meas  = eis_result.re_z_meas * 1000
    im_meas  = eis_result.neg_im_z_meas * 1000
    re_fit   = np.real(eis_result.Z_fit) * 1000
    im_fit   = -np.imag(eis_result.Z_fit) * 1000

    # Compute data span for square aspect
    all_re = np.concatenate([re_meas, re_fit])
    all_im = np.concatenate([im_meas, im_fit])
    data_span = max(all_re.max() - all_re.min(), all_im.max() - all_im.min(), 1e-9)
    fig_size = float(np.clip(data_span * 0.6, 5.0, 10.0))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Measured data
    ax.scatter(re_meas, im_meas,
               color="#F44336", s=22, zorder=4, label="EIS 실측", alpha=0.85)

    # Fitted curve: recompute on dense log-spaced grid to avoid index-length
    # mismatch (Z_fit was computed on filtered freq_fit, not full eis_result.freq)
    from eis_fitter import MODELS as _EIS_MODELS
    import math as _math
    try:
        _f_lo = max(float(eis_result.freq.min()), 1e-4)
        _f_hi = float(eis_result.freq.max())
        _f_dense = np.logspace(_math.log10(_f_lo), _math.log10(_f_hi), 400)
        _omega_dense = 2.0 * np.pi * _f_dense
        _Z_dense = _EIS_MODELS[eis_result.model_name]["z_func"](
            eis_result.param_values, _omega_dense
        )
        re_fit_line = np.real(_Z_dense) * 1000
        im_fit_line = -np.imag(_Z_dense) * 1000
    except Exception:
        # fallback: use Z_fit directly (may be shorter than freq)
        re_fit_line = re_fit
        im_fit_line = im_fit
    ax.plot(re_fit_line, im_fit_line,
            color="#2196F3", linewidth=2.0, zorder=3, label=f"피팅: {eis_result.model_name}")

    # Rs annotation
    rs_val = eis_result.params_dict.get("Rs", eis_result.re_z_meas.min())
    ax.axvline(rs_val * 1000, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.annotate(
        f"Rs = {rs_val * 1000:.2f} mΩ",
        xy=(rs_val * 1000, 0),
        xytext=(rs_val * 1000 + data_span * 0.04, data_span * 0.05),
        fontsize=7, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
    )

    # Metrics annotation box
    txt = (f"AIC = {eis_result.aic:.2f}\n"
           f"R²  = {eis_result.r2_total:.5f}\n"
           f"RMSE= {eis_result.rmse_mohm:.3f} mΩ")
    ax.text(0.03, 0.97, txt,
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.85))

    ax.set_xlabel("Re(Z) (mΩ)", fontsize=10)
    ax.set_ylabel("-Im(Z) (mΩ)", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(
        f"EIS Nyquist — {eis_result.model_label}",
        fontsize=10, fontweight="bold",
    )

    ymin, ymax = ax.get_ylim()
    if ymin > 0:
        ax.set_ylim(bottom=-0.02 * ymax)

    fig.tight_layout()
    return fig

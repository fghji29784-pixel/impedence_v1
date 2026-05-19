"""
preprocessor.py — Signal pre-processing for DCIM analysis.

Key fixes over prior known-buggy implementations:
  Bug 2: t is offset to p2=0 (not stored as absolute time)
  Bug 3: 5-second window derived from auto-detected dt (not 1000 fixed samples)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Pulse detection
# ──────────────────────────────────────────────

def detect_pulse(df: pd.DataFrame, threshold_fraction: float = 0.05) -> int:
    """Find the index where the current pulse begins.

    Looks for the first sample where |current_A| exceeds
    ``threshold_fraction`` × max(|current_A|).

    Parameters
    ----------
    df                 : DataFrame with 'current_A' column
    threshold_fraction : fraction of max current to use as onset threshold

    Returns
    -------
    int index of first above-threshold sample
    """
    i_abs = df["current_A"].abs()
    threshold = threshold_fraction * i_abs.max()
    mask = i_abs > threshold
    if not mask.any():
        raise ValueError(
            "No current pulse detected. Check that the file contains charge data "
            "and that the correct current unit (A/mA) is selected."
        )
    return int(mask.idxmax())


# ──────────────────────────────────────────────
# Critical point detection: p0, p1, p2
# ──────────────────────────────────────────────

def find_p0_p1_p2(
    df: pd.DataFrame,
    I_set: float,
) -> tuple[int, int, int]:
    """Locate the three critical points in the charge pulse.

    Definitions
    -----------
    p0 : last sample **before** the current pulse (rest, I ≈ 0)
    p1 : first sample (after pulse onset) where |I_set - I| / I_set < 50%
    p2 : first sample where |I_set - I| / I_set < 1%
         (relaxed to 5% if strict threshold is never met)

    Parameters
    ----------
    df    : DataFrame with 'current_A' column (standardised by loader.py)
    I_set : target / set-point current in Amperes (positive value)

    Returns
    -------
    (idx_p0, idx_p1, idx_p2) — integer row-label indices into df
    """
    pulse_start = detect_pulse(df)

    # p0: one sample before the pulse starts (rest state)
    idx_p0 = max(0, pulse_start - 1)

    # Search only from pulse_start onward to avoid pre-pulse false-positives
    search = df.iloc[pulse_start:].copy()
    rel_err = (I_set - search["current_A"]).abs() / abs(I_set)

    # p1: first where within 50% of I_set
    mask_p1 = rel_err < 0.50
    if not mask_p1.any():
        raise ValueError(
            "Cannot find p1: current never reaches 50% of I_set. "
            "Check I_set or data integrity."
        )
    idx_p1 = int(mask_p1.idxmax())

    # p2: first where within 1% (strict), fall back to 5%
    mask_p2_strict = rel_err < 0.01
    if mask_p2_strict.any():
        idx_p2 = int(mask_p2_strict.idxmax())
    else:
        mask_p2_loose = rel_err < 0.05
        if mask_p2_loose.any():
            idx_p2 = int(mask_p2_loose.idxmax())
        else:
            raise ValueError(
                "Cannot find p2: current never settles within 5% of I_set. "
                "Check I_set value or try setting it manually."
            )

    return idx_p0, idx_p1, idx_p2


# ──────────────────────────────────────────────
# Rs calculation
# ──────────────────────────────────────────────

def calculate_Rs(df: pd.DataFrame, idx_p0: int, idx_p1: int) -> float:
    """Compute ohmic resistance from instantaneous voltage/current jump.

    Rs = ΔV / ΔI  measured between p0 (rest) and p1 (first 50% point)

    Parameters
    ----------
    df              : DataFrame with 'voltage_V' and 'current_A'
    idx_p0, idx_p1  : integer indices (row labels) returned by find_p0_p1_p2

    Returns
    -------
    Rs in Ohms (positive for a normal ohmic drop)
    """
    dV = float(df.loc[idx_p1, "voltage_V"] - df.loc[idx_p0, "voltage_V"])
    dI = float(df.loc[idx_p1, "current_A"] - df.loc[idx_p0, "current_A"])
    if abs(dI) < 1e-9:
        raise ValueError(
            "ΔI ≈ 0 between p0 and p1; cannot compute Rs. "
            "Check that current unit conversion is correct."
        )
    return dV / dI


# ──────────────────────────────────────────────
# Fit data preparation
# ──────────────────────────────────────────────

def prepare_fit_data(
    df: pd.DataFrame,
    idx_p2: int,
    dt: float | None = None,
    window_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract and prepare the voltage transient window for curve fitting.

    Window selection uses the actual elapsed time after p2. BioLogic files can
    mix a fast DCIM burst with slower CC samples, so sample-count windows can
    accidentally capture tens of seconds when the user asked for only a few.

    Parameters
    ----------
    df       : DataFrame with 'time_s' and 'voltage_V'
    idx_p2   : row-label index of p2 point
    dt       : sampling interval in seconds; auto-detected if None
    window_s : length of fitting window in seconds (default 5.0 s)

    Returns
    -------
    t_fit   : time array [s], starts at 0 (p2 = t=0)
    V_fit   : voltage array [V], same length as t_fit
    Vp2     : voltage at p2 [V]
    dt      : detected (or provided) sampling interval [s]
    """
    time_all = df["time_s"].values
    volt_all = df["voltage_V"].values

    # ── Auto-detect dt (global median) ───────────────────────────────────
    # Global median is correct when the DCIM burst dominates the recording.
    # Using local dt around p2 caused too-small n_samples when p2 sits at
    # the transition between fast burst and slow CC data.
    if dt is None:
        diffs = np.diff(time_all)
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) > 0:
            dt = float(np.median(positive_diffs))
        else:
            dt = 1.0
        if dt <= 0:
            raise ValueError(
                f"Auto-detected dt = {dt:.6g} s is non-positive. "
                "Check that the time column is in seconds and monotonically increasing."
            )

    pos_p2 = df.index.get_loc(idx_p2)
    pos_end = _window_end_by_elapsed_time(time_all, pos_p2, window_s)

    if pos_end <= pos_p2:
        raise ValueError(
            "p2 is at or beyond the end of the data. "
            "Cannot extract a fitting window."
        )

    t_window = time_all[pos_p2:pos_end]
    V_window = volt_all[pos_p2:pos_end]

    Vp2 = float(V_window[0])

    # ── t offset: 0-based from p2 ─────────────────────────────────────────
    t_fit = t_window - float(t_window[0])   # t_fit[0] == 0.0

    # ── Mixed-rate resampling ─────────────────────────────────────────────
    # BioLogic 파일은 흔히 DCIM 버스트(≈192 µs 간격)와 일반 CC 데이터(≈0.1–1 s 간격)가
    # 섞여 있습니다. 리샘플링 없이 사용하면 curve_fit이 밀집 구간에 편중되어
    # 느린 τ₂를 제대로 못 잡음. max/min 간격 비율이 10 초과 시 로그 간격으로 리샘플링.
    diffs = np.diff(t_fit)
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) >= 2:
        dt_min_w = float(pos_diffs.min())
        dt_max_w = float(pos_diffs.max())
        if dt_max_w / dt_min_w > 10.0 and float(t_fit[-1]) > 0.0:
            n_pts = max(500, min(2000, len(t_fit)))
            t_new = np.concatenate([
                [0.0],
                np.geomspace(dt_min_w, float(t_fit[-1]), n_pts - 1),
            ])
            V_window = np.interp(t_new, t_fit, V_window)
            t_fit = t_new

    return t_fit, V_window.copy(), Vp2, dt


def _window_end_by_elapsed_time(time_all: np.ndarray, pos_start: int, window_s: float) -> int:
    """Return an exclusive end position using elapsed time from pos_start.

    If the time column resets at a BioLogic sequence boundary, stop just before
    that reset so interpolation and fitting see a monotonic time vector.
    """
    if pos_start >= len(time_all):
        return len(time_all)

    rel = np.asarray(time_all[pos_start:], dtype=float) - float(time_all[pos_start])
    resets = np.flatnonzero(np.diff(rel) < 0)
    local_limit = int(resets[0] + 1) if len(resets) else len(rel)
    rel_mono = rel[:local_limit]

    pos_local_end = int(np.searchsorted(rel_mono, window_s, side="right"))
    pos_local_end = max(2, pos_local_end)
    return min(pos_start + pos_local_end, len(time_all))


def prepare_joint_fit_data(
    df: pd.DataFrame,
    idx_p0: int,
    idx_p2: int,
    window_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract ramp(p0→p2) and CC(p2→window) arrays for joint Warburg fitting."""
    time_all = df["time_s"].values
    volt_all = df["voltage_V"].values
    curr_all = df["current_A"].values
    pos_p0 = df.index.get_loc(idx_p0)
    pos_p2 = df.index.get_loc(idx_p2)
    if pos_p2 <= pos_p0:
        raise ValueError("idx_p2 must be after idx_p0 for joint fitting.")

    pos_cc_end = _window_end_by_elapsed_time(time_all, pos_p2, window_s)
    t0 = float(time_all[pos_p2])
    t_ramp = time_all[pos_p0:pos_p2 + 1] - t0
    V_ramp = volt_all[pos_p0:pos_p2 + 1]
    I_ramp = curr_all[pos_p0:pos_p2 + 1]
    t_cc = time_all[pos_p2:pos_cc_end] - t0
    V_cc = volt_all[pos_p2:pos_cc_end]
    Vp2 = float(V_cc[0])
    return t_ramp, V_ramp, I_ramp, t_cc, V_cc, Vp2


def find_relaxation_start(
    df: pd.DataFrame,
    I_set: float,
    search_after_idx: int,
) -> int | None:
    """Find first current-interruption point after p2 (|I| < 5% I_set)."""
    pos_start = df.index.get_loc(search_after_idx)
    search = df.iloc[pos_start + 1:]
    if search.empty:
        return None
    mask = search["current_A"].abs() < 0.05 * abs(I_set)
    if not mask.any():
        return None
    return int(mask.idxmax())


def prepare_relaxation_data(
    df: pd.DataFrame,
    idx_relax_start: int,
    window_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract voltage relaxation after current is interrupted."""
    time_all = df["time_s"].values
    volt_all = df["voltage_V"].values
    pos_start = df.index.get_loc(idx_relax_start)
    pos_end = _window_end_by_elapsed_time(time_all, pos_start, window_s)
    t_relax = time_all[pos_start:pos_end] - float(time_all[pos_start])
    V_relax = volt_all[pos_start:pos_end]
    if len(t_relax) < 3:
        raise ValueError("Not enough relaxation data after current interruption.")
    return t_relax, V_relax.copy(), float(V_relax[0])


# ──────────────────────────────────────────────
# Convenience: auto-detect I_set from data
# ──────────────────────────────────────────────

def detect_I_set(df: pd.DataFrame) -> float:
    """Estimate the target current as the 95th percentile of |current_A|.

    Using the 95th percentile (rather than the maximum) guards against
    transient spikes inflating the estimate.
    """
    return float(np.percentile(df["current_A"].abs(), 95))

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

    Window selection uses sample count (pos_p2 + n_samples), NOT searchsorted.
    searchsorted requires monotonically increasing time, which fails when
    BioLogic files reset time at the start of each sequence/technique.
    dt is the global median sampling interval of the whole file.

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

    # ── Auto-detect dt (global median) — kept for return value only ──────
    if dt is None:
        diffs = np.diff(time_all)
        positive_diffs = diffs[diffs > 0]
        dt = float(np.median(positive_diffs)) if len(positive_diffs) > 0 else 1.0
        if dt <= 0:
            dt = 1.0

    # ── Row-position of p2 ────────────────────────────────────────────────
    pos_p2 = df.index.get_loc(idx_p2)

    # ── Window size: time-based (handles mixed-rate & avoids over-capture) ─
    # Sample-count approach (n = window_s / dt) fails when the global median
    # dt comes from a fast DCIM burst: a 5 s window at dt=1 ms → 5000 samples
    # that may actually span 40+ seconds, capturing the relaxation phase and
    # causing the RC model to fail (voltage goes up then down → R1,R2 → 0).
    #
    # Instead: walk forward in real time and stop at window_s seconds.
    # BioLogic files can reset time at sequence boundaries; we stop at the
    # first backward jump so we never cross a reset.
    remaining_t = time_all[pos_p2:]
    t_p2_val = float(remaining_t[0])

    # Find first time-reset (backwards jump) after p2
    diffs_remaining = np.diff(remaining_t)
    resets = np.where(diffs_remaining < 0)[0]
    monotonic_len = int(resets[0]) + 1 if len(resets) > 0 else len(remaining_t)

    # searchsorted is safe on the guaranteed-monotonic slice
    t_mono = remaining_t[:monotonic_len] - t_p2_val
    n_in_window = int(np.searchsorted(t_mono, window_s, side="right"))
    n_samples = max(2, n_in_window)
    pos_end = min(pos_p2 + n_samples, len(df))

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


# ──────────────────────────────────────────────
# Joint fit data preparation
# ──────────────────────────────────────────────

def prepare_joint_fit_data(
    df: pd.DataFrame,
    idx_p0: int,
    idx_p2: int,
    window_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Prepare ramp + CC transient data for joint parameter fitting.

    Joint fitting treats Rs as a free parameter instead of computing it from a
    single ΔV/ΔI ratio.  The ramp phase (p0→p2) provides Rs constraints, and
    the CC transient (p2→p2+window) constrains R1, C1, R2, C2, σ_W.

    Returns
    -------
    t_ramp  : time array [s] from p0 (t=0) to p2, all ramp samples (3–5 pts)
    I_ramp  : current [A] at each ramp sample
    V_ramp  : voltage [V] at each ramp sample
    t_cc    : CC transient time [s], 0-based from p2 (resampled)
    V_cc    : voltage [V] during CC transient (same length as t_cc)
    V0      : voltage at p0 [V]
    dt      : median sampling interval [s]
    """
    time_all = df["time_s"].values
    volt_all = df["voltage_V"].values
    curr_all = df["current_A"].values

    # ── dt (global median) ──────────────────────────────────────────────
    diffs = np.diff(time_all)
    pos_diffs = diffs[diffs > 0]
    dt = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0

    # ── Ramp phase: p0 inclusive → p2 inclusive ──────────────────────────
    pos_p0 = df.index.get_loc(idx_p0)
    pos_p2 = df.index.get_loc(idx_p2)
    # Include p0..p2 (small slice, typically 3 rows)
    t_ramp_abs = time_all[pos_p0 : pos_p2 + 1]
    I_ramp     = curr_all[pos_p0 : pos_p2 + 1]
    V_ramp     = volt_all[pos_p0 : pos_p2 + 1]
    # Offset time so p0 = t=0
    t_ramp = t_ramp_abs - t_ramp_abs[0]
    V0 = float(V_ramp[0])

    # ── CC transient: p2 → p2 + window_s (time-based) ───────────────────
    remaining_t_cc = time_all[pos_p2:]
    t_p2_val_cc = float(remaining_t_cc[0])
    diffs_cc_r = np.diff(remaining_t_cc)
    resets_cc = np.where(diffs_cc_r < 0)[0]
    mono_len_cc = int(resets_cc[0]) + 1 if len(resets_cc) > 0 else len(remaining_t_cc)
    t_mono_cc = remaining_t_cc[:mono_len_cc] - t_p2_val_cc
    n_cc = max(2, int(np.searchsorted(t_mono_cc, window_s, side="right")))
    pos_cc_end = min(pos_p2 + n_cc, len(df))
    t_cc_abs = time_all[pos_p2 : pos_cc_end]
    V_cc_raw = volt_all[pos_p2 : pos_cc_end]
    t_cc = t_cc_abs - t_cc_abs[0]   # 0-based from p2

    # Resample CC if mixed rates
    diffs_cc = np.diff(t_cc)
    pos_diffs_cc = diffs_cc[diffs_cc > 0]
    if len(pos_diffs_cc) >= 2:
        dt_min_w = float(pos_diffs_cc.min())
        dt_max_w = float(pos_diffs_cc.max())
        if dt_max_w / dt_min_w > 10.0 and float(t_cc[-1]) > 0.0:
            n_pts = max(500, min(2000, len(t_cc)))
            t_cc_new = np.concatenate([
                [0.0],
                np.geomspace(dt_min_w, float(t_cc[-1]), n_pts - 1),
            ])
            V_cc_raw = np.interp(t_cc_new, t_cc, V_cc_raw)
            t_cc = t_cc_new

    return (
        t_ramp,
        I_ramp.astype(float),
        V_ramp.astype(float),
        t_cc,
        V_cc_raw.copy(),
        V0,
        dt,
    )


# ──────────────────────────────────────────────
# Convenience: auto-detect I_set from data
# ──────────────────────────────────────────────

def detect_I_set(df: pd.DataFrame) -> float:
    """Estimate the target current as the 95th percentile of |current_A|.

    Using the 95th percentile (rather than the maximum) guards against
    transient spikes inflating the estimate.
    """
    return float(np.percentile(df["current_A"].abs(), 95))


# ──────────────────────────────────────────────
# Relaxation phase extraction
# ──────────────────────────────────────────────

def find_relaxation_start(
    df: pd.DataFrame,
    I_set: float,
    search_after_idx: int | None = None,
    threshold_fraction: float = 0.05,
) -> int | None:
    """Find the first index where current drops back to near-zero after the pulse.

    This marks the start of the relaxation (rest) phase that follows a
    charge pulse.  The relaxation voltage recovery contains the RC discharge
    transient, which provides a second independent constraint for R1, C1,
    R2, C2 and allows sequential peeling to work on a cleaner signal.

    Parameters
    ----------
    df                  : DataFrame with 'current_A'
    I_set               : settled charge current [A]
    search_after_idx    : row-label index; only search after this point.
                          Typically idx_p2 or the end of the CC window.
    threshold_fraction  : current < threshold_fraction * I_set → relaxation

    Returns
    -------
    Row-label index of relaxation start, or None if not found.
    """
    threshold = threshold_fraction * abs(I_set)
    if search_after_idx is not None:
        pos_start = df.index.get_loc(search_after_idx) + 1
        search = df.iloc[pos_start:]
    else:
        search = df

    mask = search["current_A"].abs() < threshold
    if not mask.any():
        return None
    return int(mask.idxmax())


def prepare_relaxation_data(
    df: pd.DataFrame,
    idx_relax_start: int,
    window_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract voltage recovery data from the relaxation phase.

    The relaxation phase starts when the charge current is removed.
    The voltage decays back toward OCV following:
      V(t) = V_relax0 - R1*I_pre*(1-exp(-t/τ1)) - R2*I_pre*(1-exp(-t/τ2))
    where I_pre is the pre-relaxation current (= I_set, positive).

    A longer window captures the slow RC (τ2 ~ seconds) and makes R2, C2
    estimation far more reliable than using the charge transient alone
    (Hust et al. 2021; HPPC methodology).

    Parameters
    ----------
    df              : DataFrame with 'time_s', 'voltage_V'
    idx_relax_start : row-label index of first relaxation sample
    window_s        : relaxation window length [s] (default 30 s)

    Returns
    -------
    t_relax  : time [s], 0-based from relaxation start
    V_relax  : voltage [V]
    V_relax0 : voltage at start of relaxation [V]
    """
    time_all = df["time_s"].values
    volt_all = df["voltage_V"].values

    pos_r0 = df.index.get_loc(idx_relax_start)

    # Time-based window (same approach as prepare_fit_data)
    remaining_t_r = time_all[pos_r0:]
    t_r0_val = float(remaining_t_r[0])
    diffs_r_r = np.diff(remaining_t_r)
    resets_r = np.where(diffs_r_r < 0)[0]
    mono_len_r = int(resets_r[0]) + 1 if len(resets_r) > 0 else len(remaining_t_r)
    t_mono_r = remaining_t_r[:mono_len_r] - t_r0_val
    n_relax = max(2, int(np.searchsorted(t_mono_r, window_s, side="right")))
    pos_end = min(pos_r0 + n_relax, len(df))

    # dt still needed for return (use local estimate near relaxation start)
    diffs = np.diff(time_all)
    pos_diffs = diffs[diffs > 0]
    dt = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0

    t_abs   = time_all[pos_r0 : pos_end]
    V_raw   = volt_all[pos_r0 : pos_end]
    t_relax = t_abs - t_abs[0]
    V_relax0 = float(V_raw[0])

    # Mixed-rate resample (same logic as prepare_fit_data)
    diffs_r = np.diff(t_relax)
    pos_dr  = diffs_r[diffs_r > 0]
    if len(pos_dr) >= 2:
        dt_min_r = float(pos_dr.min())
        dt_max_r = float(pos_dr.max())
        if dt_max_r / dt_min_r > 10.0 and float(t_relax[-1]) > 0.0:
            n_pts = max(500, min(2000, len(t_relax)))
            t_new = np.concatenate([
                [0.0],
                np.geomspace(dt_min_r, float(t_relax[-1]), n_pts - 1),
            ])
            V_raw = np.interp(t_new, t_relax, V_raw)
            t_relax = t_new

    return t_relax, V_raw.copy(), V_relax0

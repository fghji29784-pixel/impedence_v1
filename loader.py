"""
loader.py — Load BioLogic EC-Lab charge data and EIS data files.

Handles:
- Multi-line BioLogic header ("Nb header lines : N")
- Tab or comma delimiters
- UTF-8 / CP949 / latin-1 encoding fallback
- mA → A current conversion (Bug 1 fix)
- EIS column auto-detection with sign-flip correction
"""

from __future__ import annotations

from io import StringIO

import pandas as pd


# ──────────────────────────────────────────────
# Charge data loader
# ──────────────────────────────────────────────

def load_charge_data(file, current_unit: str = "mA") -> pd.DataFrame:
    """Load a BioLogic GCPL charge data file.

    Parameters
    ----------
    file         : file-like object (from st.file_uploader or open())
    current_unit : 'mA' or 'A' — unit used in the file for current column.
                   If 'mA', values are divided by 1000 to convert to Amperes.

    Returns
    -------
    DataFrame with columns: time_s, voltage_V, current_A
    """
    # ── 1. Read raw bytes and decode ──────────────────────────────────────
    if hasattr(file, "read"):
        raw_bytes = file.read()
    else:
        with open(file, "rb") as fh:
            raw_bytes = fh.read()

    text = None
    for encoding in ("utf-8", "cp949", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        raise ValueError("Cannot decode file with UTF-8, CP949, or latin-1 encodings.")

    lines = text.splitlines()

    # ── 2. BioLogic header line count ────────────────────────────────────
    nb_header = _parse_nb_header(lines)

    # ── 3. Delimiter auto-detection ──────────────────────────────────────
    # Sample the first data line (right after header)
    data_start = nb_header
    sample_line = lines[data_start] if data_start < len(lines) else ""
    delimiter = "\t" if "\t" in sample_line else ","

    # ── 4. Parse CSV ──────────────────────────────────────────────────────
    data_text = "\n".join(lines[data_start:])
    df = _try_read_csv(data_text, delimiter)

    # ── 5. Standardise column names ───────────────────────────────────────
    col_time, col_voltage, col_current = _detect_columns(df.columns)
    if col_time is None or col_voltage is None or col_current is None:
        raise ValueError(
            f"Cannot identify time/voltage/current columns.\n"
            f"Detected columns: {list(df.columns)}"
        )

    df = df[[col_time, col_voltage, col_current]].copy()
    df.columns = ["time_s", "voltage_V", "current_raw"]

    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 6. Current unit conversion (Bug 1 fix) ────────────────────────────
    if current_unit == "mA":
        df["current_A"] = df["current_raw"] / 1000.0
    else:
        df["current_A"] = df["current_raw"]

    df.drop(columns=["current_raw"], inplace=True)
    return df[["time_s", "voltage_V", "current_A"]]


# ──────────────────────────────────────────────
# EIS data loader
# ──────────────────────────────────────────────

def load_eis_data(file) -> pd.DataFrame:
    """Load a BioLogic GEIS EIS data file.

    Returns
    -------
    DataFrame with columns: freq, re_z, neg_im_z
    """
    if hasattr(file, "read"):
        raw_bytes = file.read()
    else:
        with open(file, "rb") as fh:
            raw_bytes = fh.read()

    text = None
    for encoding in ("utf-8", "cp949", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        raise ValueError("Cannot decode EIS file.")

    lines = text.splitlines()
    nb_header = _parse_nb_header(lines)
    data_start = nb_header

    sample_line = lines[data_start] if data_start < len(lines) else ""
    delimiter = "\t" if "\t" in sample_line else ","

    data_text = "\n".join(lines[data_start:])
    df = _try_read_csv(data_text, delimiter)

    col_freq, col_re, col_im = _detect_eis_columns(df.columns)
    if col_freq is None or col_re is None or col_im is None:
        raise ValueError(
            f"Cannot identify freq/Re(Z)/-Im(Z) columns.\n"
            f"Detected columns: {list(df.columns)}"
        )

    df = df[[col_freq, col_re, col_im]].copy()
    df.columns = ["freq", "re_z", "neg_im_z"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Sign-flip: Nyquist convention requires -Im(Z) > 0 for capacitive arc
    if df["neg_im_z"].median() < 0:
        df["neg_im_z"] = -df["neg_im_z"]

    return df[["freq", "re_z", "neg_im_z"]]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _parse_nb_header(lines: list[str]) -> int:
    """Parse BioLogic 'Nb header lines : N' to get number of header rows."""
    for line in lines:
        if "nb header lines" in line.lower():
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return 0


def _try_read_csv(data_text: str, delimiter: str) -> pd.DataFrame:
    """Try reading CSV; retry with decimal='.' if columns are non-numeric."""
    # First attempt: BioLogic sometimes uses comma as decimal separator
    try:
        df = pd.read_csv(
            StringIO(data_text),
            sep=delimiter,
            decimal=",",
            engine="python",
            on_bad_lines="skip",
        )
        # Check if numeric columns are actually numeric
        numeric_cols = sum(
            1 for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        )
        if numeric_cols >= 2:
            return df
    except Exception:
        pass

    # Fallback: standard decimal point
    df = pd.read_csv(
        StringIO(data_text),
        sep=delimiter,
        decimal=".",
        engine="python",
        on_bad_lines="skip",
    )
    return df


def _detect_columns(columns) -> tuple[str | None, str | None, str | None]:
    """Auto-detect time, voltage, and current column names."""
    col_time = col_voltage = col_current = None

    for col in columns:
        cl = col.lower().strip()
        # Time
        if col_time is None and (
            cl in ("time/s", "t/s", "time_s", "time (s)", "time")
            or cl.startswith("time")
        ):
            col_time = col
        # Voltage — BioLogic uses 'Ewe/V'
        elif col_voltage is None and (
            cl in ("ewe/v", "voltage/v", "voltage_v", "v", "u/v", "<ewe>/v")
            or "ewe" in cl
            or cl.startswith("voltage")
        ):
            col_voltage = col
        # Current — BioLogic uses 'I/mA' or '<I>/mA'
        elif col_current is None and (
            cl in ("i/ma", "<i>/ma", "i/a", "<i>/a", "current/a", "current/ma",
                   "current_a", "current_ma", "i (a)", "i (ma)")
            or (cl.startswith("i/") or cl.startswith("<i>"))
            or cl.startswith("current")
        ):
            col_current = col

    return col_time, col_voltage, col_current


def _detect_eis_columns(columns) -> tuple[str | None, str | None, str | None]:
    """Auto-detect freq, Re(Z), and -Im(Z) column names in EIS file."""
    col_freq = col_re = col_im = None

    freq_patterns = ("freq/hz", "frequency", "f/hz", "freq")
    re_patterns = ("re(z)/ohm", "re(z)", "zre", "z_re", "real", "z'/ohm", "z'")
    neg_im_patterns = (
        "-im(z)/ohm", "-im(z)", "zim", "z_im", "-imag",
        "-imaginary", "z\"/ohm", "z\"", "-z\"/ohm",
    )
    pos_im_patterns = ("im(z)/ohm", "im(z)", "imaginary")

    for col in columns:
        cl = col.lower().strip()
        if col_freq is None and any(p in cl for p in freq_patterns):
            col_freq = col
        elif col_re is None and any(p in cl for p in re_patterns):
            col_re = col
        elif col_im is None and any(p in cl for p in neg_im_patterns):
            col_im = col

    # If -Im(Z) not found, look for Im(Z) and negate later (sign handled in caller)
    if col_im is None:
        for col in columns:
            cl = col.lower().strip()
            if any(p in cl for p in pos_im_patterns):
                col_im = col
                break

    return col_freq, col_re, col_im

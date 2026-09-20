# Version 7: isolated spectrum configurations and an integrated Pickett workspace.
# The v6 numerical/assignment implementation is preserved below and extended.
if __name__ == "__main__":
    import argparse
    from spectrum_workspace_v7 import launch
    parser = argparse.ArgumentParser(description="Hyperfine Interactive Spectrum Assigner v7")
    parser.add_argument("--port", type=int, default=8053)
    launch(__file__, port=parser.parse_args().port)
    raise SystemExit

import os
import json
import time
import datetime
import re
from functools import lru_cache
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import dash
from dash import dcc, html, dash_table, Output, Input, State, ALL, ctx, callback_context
import dash.exceptions
import plotly.graph_objs as go
from dash_extensions import Keyboard

import tempfile



DEBUG_UNC = False  # set to True if you want detailed per-line debug prints

INACTIVE_COLOR = "#BBBBBB"
INACTIVE_OPACITY = 0.35  # dim them so they sit visually “behind”

# =========================
# Init & Config
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
# Configuration and cache paths are explicit; never chdir the application.
from pathlib import Path
from spectrum_workspace_v7 import identity, discover, remap_assignments
WORKSPACE_CONFIG = globals().get("WORKSPACE_CONFIG", os.path.join(script_dir, "config.json"))
WORKSPACE_PREFIX = globals().get("WORKSPACE_PREFIX", "/")
WORKSPACE_OPTIONS = globals().get("WORKSPACE_OPTIONS", discover(script_dir))
WORKSPACE_STATE_DIR = os.path.join(script_dir, "autosave", "v7", identity(WORKSPACE_CONFIG))
os.makedirs(WORKSPACE_STATE_DIR, exist_ok=True)

with open(WORKSPACE_CONFIG, "r", encoding="utf-8") as f:
    config = json.load(f)

# === Uncertainty config (ΔF from FID, base sigma) ===
_unc = config.get("uncertainty", {}) or {}
try:
    FID_TIME_US = float(_unc.get("fid_time_us", 10.0))
    if FID_TIME_US <= 0:
        FID_TIME_US = 10.0
except Exception:
    FID_TIME_US = 10.0

# ΔF in MHz from FID time (µs): ΔF ≈ 1 / T_FID(µs)
DELTA_F_DEFAULT = 1.0 / FID_TIME_US

try:
    BASE_SIGMA_INSTR = float(_unc.get("base_sigma_instr_mhz", 0.01))
    if BASE_SIGMA_INSTR <= 0:
        BASE_SIGMA_INSTR = 0.01
except Exception:
    BASE_SIGMA_INSTR = 0.01


qn_label_map = config.get("qn_labels", {})
# Accept either one or many cat files (backwards-compatible)
cat_files = config.get("cat_files")
if cat_files is None:
    cat_files = [config["cat_file"]]
def _resolve_input(path):
    value = Path(path).expanduser()
    return str(value.resolve() if value.is_absolute() else (Path(WORKSPACE_CONFIG).parent / value).resolve())
cat_files = [_resolve_input(path) for path in cat_files]
csv_file_path = _resolve_input(config["csv_file"])

# =========================
# Parse Simulated Spectrum (.cat)
# =========================
def parse_cat_file(filepath):
    sim_data = []
    max_qns = 0
    bad_lines = []
    with open(filepath) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                freq = float(line[0:13].strip())
                logI = float(line[21:30].strip())
                E_low = float(line[32:41].strip())
                countQN = int(line[54:55].strip())

                entry = {"Freq": freq, "Intensity": 10 ** logI, "Eu": E_low}
                max_qns = max(max_qns, countQN)

                label_map = {0: "J", 1: "Ka", 2: "Kc"}
                upper_start, lower_start = 55, 67

                for i in range(countQN):
                    qn_name = label_map.get(i, f"Q{i - 2}") if i > 2 else label_map[i]
                    uq_label = f"Upper{qn_name}"
                    lq_label = f"Lower{qn_name}"

                    if upper_start + 2 <= len(line):
                        entry[uq_label] = int(line[upper_start:upper_start + 2].strip())
                    if lower_start + 2 <= len(line):
                        entry[lq_label] = int(line[lower_start:lower_start + 2].strip())
                    upper_start += 2
                    lower_start += 2

                sim_data.append(entry)
            except Exception:
                bad_lines.append(line_number)

    if bad_lines:
        raise ValueError(f"{filepath}: {len(bad_lines)} unparseable CAT records; first lines: {bad_lines[:8]}")
    if not sim_data:
        raise ValueError(f"{filepath}: empty catalog")
    df = pd.DataFrame(sim_data)

    # Build QN column order
    qn_order = []
    for i in range(max_qns):
        name = {0: "J", 1: "Ka", 2: "Kc"}.get(i, f"Q{i-2}" if i > 2 else f"Q{i}")
        qn_order.append(f"Upper{name}")
    for i in range(max_qns):
        name = {0: "J", 1: "Ka", 2: "Kc"}.get(i, f"Q{i-2}" if i > 2 else f"Q{i}")
        qn_order.append(f"Lower{name}")

    return df, qn_order


# Build hover text for simulated sticks (includes QN info)
def generate_hover(row):
    parts = [
        f"<b>Freq:</b> {row['Freq']:.4f} MHz",
        f"<b>Intensity:</b> {row['Intensity']:.2e}",
        f"<b>Eu:</b> {row['Eu']:.2f} cm⁻¹"
    ]
    upper_qns, lower_qns = [], []
    for col in row.index:
        if col.startswith("Upper"):
            upper_qns.append(f"{qn_label_map.get(col, col)}={row[col]}")
        elif col.startswith("Lower"):
            lower_qns.append(f"{qn_label_map.get(col, col)}={row[col]}")
    if upper_qns:
        parts.append("<b>Upper:</b> " + " ".join(upper_qns))
    if lower_qns:
        parts.append("<b>Lower:</b> " + " ".join(lower_qns))
    return "<br>".join(parts)

# =========================
# Build per-catalog structures
# =========================
from spectrum_intensity_v7 import normalize_catalog, scale_figure, normalized_relayout

# Read the measured band before loading catalogs, including on first startup.
_csv_separator = config.get('csv_separator', ';')
meas_df = pd.read_csv(csv_file_path, sep=None, engine='python') if _csv_separator == 'auto' else pd.read_csv(csv_file_path, sep=_csv_separator)
meas_freqs = meas_df.iloc[:, 0].to_numpy(dtype=float)
meas_intensities_raw = meas_df.iloc[:, 1].to_numpy(dtype=float).copy()
MEAS_XMIN = float(np.nanmin(meas_freqs))
MEAS_XMAX = float(np.nanmax(meas_freqs))
_finite_intensities = meas_intensities_raw[np.isfinite(meas_intensities_raw)]
MEAS_INTENSITY_REFERENCE = float(_finite_intensities.max()) if _finite_intensities.size else 1.0
if MEAS_INTENSITY_REFERENCE <= 0:
    MEAS_INTENSITY_REFERENCE = float(np.max(np.abs(_finite_intensities))) if _finite_intensities.size else 1.0
MEAS_INTENSITY_REFERENCE = MEAS_INTENSITY_REFERENCE or 1.0
meas_intensities = meas_intensities_raw / MEAS_INTENSITY_REFERENCE

def load_catalog(path):
    cdf, qn_order = parse_cat_file(path)
    if not cdf.empty:
        normalization_reference = normalize_catalog(cdf, MEAS_XMIN, MEAS_XMAX)
        cdf["RoundedFreq"] = cdf["Freq"].round(4)
        cdf["StickX"] = cdf["Freq"].apply(lambda f: [f, f, None])  # repeated x values to draw a stick
        cdf["Hover"] = cdf.apply(generate_hover, axis=1)


        # --- NEW: stable row identifier for matching assigned sticks exactly ---
        cdf.reset_index(drop=True, inplace=True)
        cdf["SimUID"] = cdf.index.astype(int)

    return {
        "path": path,
        "df": cdf,
        "qn_order": qn_order,
        "name": os.path.basename(path),
        "normalization_reference": normalization_reference,
        "normalization_band": [MEAS_XMIN, MEAS_XMAX],
    }





catalogs = [load_catalog(path) for path in cat_files]

# --- Debug prints to confirm data loaded (appears in your terminal) ---
try:
    print(f"[INFO] Measured CSV: {csv_file_path}")
    #print(f"[INFO] Measured points: {len(meas_freqs)}")
except NameError:
    pass

print("[INFO] Catalogs loaded:")
for i, c in enumerate(catalogs):
    n = 0 if (c.get("df") is None) else len(c["df"])
    print(f"  [{i}] {c['name']}  rows={n}")

# =========================
# Measured spectrum
# =========================
# Measured arrays initialized above, before catalog normalization.


def _clamp_x_range(x0, x1):
    try:
        x0 = float(x0)
        x1 = float(x1)
    except Exception:
        return [MEAS_XMIN, MEAS_XMAX]

    if x1 < x0:
        x0, x1 = x1, x0

    x0 = max(x0, MEAS_XMIN)
    x1 = min(x1, MEAS_XMAX)

    if x1 <= x0:
        return [MEAS_XMIN, MEAS_XMAX]
    return [x0, x1]


# =========================
# Embedded Loomis-Wood helpers
# =========================
LW_MAX_ROWS = 80
LW_SERIES = {
    "aR(0,1)": (1, 0, 1),
    "bR(1,1)": (1, 1, 1),
    "bR(-1,1)": (1, -1, 1),
    "bR(1,-1)": (1, 1, -1),
    "bQ(1,-1)": (0, 1, -1),
    "cR(1,0)": (1, 1, 0),
    "cQ(1,0)": (0, 1, 0),
    "aQ(0,1)": (0, 0, 1),
}


def _lw_qn_names(cat):
    """Return catalog QN names (J, Ka, Kc, ...) in file order."""
    names = []
    for col in cat.get("qn_order", []):
        if col.startswith("Upper"):
            names.append(col[5:])
    return names


def _lw_parse_filter(text):
    """Parse the database-app Loomis-Wood filter mini-language."""
    clauses = []
    for token in [t.strip() for t in re.split(r"[,;]+", str(text or "")) if t.strip()]:
        component = re.fullmatch(
            r"Kc\s*(')?\s*=\s*J\s*(')?\s*-\s*Ka\s*(')?\s*(\+\s*1)?",
            token,
            flags=re.IGNORECASE,
        )
        if component:
            primes = [component.group(i) == "'" for i in (1, 2, 3)]
            if len(set(primes)) != 1:
                raise ValueError(f"mixed upper/lower primes in '{token}'")
            clauses.append({"kind": "component", "lower": primes[0],
                            "offset": 1 if component.group(4) else 0})
            continue

        match = re.fullmatch(
            r"([dΔ])?\s*([A-Za-z][A-Za-z0-9]*)\s*(')?\s*"
            r"(<=|>=|!=|=|<|>)\s*"
            r"(-?\d+(?:\.\d+)?(?:\|-?\d+(?:\.\d+)?)*)",
            token,
        )
        if not match:
            raise ValueError(f"cannot parse '{token}'")
        clauses.append({
            "kind": "value",
            "delta": bool(match.group(1)),
            "name": match.group(2),
            "lower": match.group(3) == "'",
            "op": match.group(4),
            "values": [float(v) for v in match.group(5).split("|")],
        })
    return clauses


def _lw_find_qn_column(df, name, lower=False):
    wanted = ("Lower" if lower else "Upper") + str(name)
    for col in df.columns:
        if str(col).lower() == wanted.lower():
            return col
    return None


def _lw_clause_mask(df, clause):
    if clause["kind"] == "component":
        side = "Lower" if clause["lower"] else "Upper"
        required = [side + "J", side + "Ka", side + "Kc"]
        if any(col not in df.columns for col in required):
            return pd.Series(False, index=df.index)
        return df[side + "Kc"] == (df[side + "J"] - df[side + "Ka"] + clause["offset"])

    name = clause["name"]
    if name.lower() == "logi":
        values = np.log10(np.maximum(df["Intensity"].astype(float), np.finfo(float).tiny))
    elif name.lower() == "freq":
        values = df["Freq"].astype(float)
    elif clause["delta"]:
        upper = _lw_find_qn_column(df, name, lower=False)
        lower = _lw_find_qn_column(df, name, lower=True)
        if upper is None or lower is None:
            return pd.Series(False, index=df.index)
        values = df[upper].astype(float) - df[lower].astype(float)
    else:
        col = _lw_find_qn_column(df, name, lower=clause["lower"])
        if col is None:
            return pd.Series(False, index=df.index)
        values = df[col].astype(float)

    targets = clause["values"]
    op = clause["op"]
    if op == "=":
        return values.isin(targets)
    if op == "!=":
        return ~values.isin(targets)
    target = targets[0]
    return {
        "<": values < target,
        ">": values > target,
        "<=": values <= target,
        ">=": values >= target,
    }[op]


def _lw_filter_catalog(cat, filter_text, sort_key, ignore_hf):
    df = cat.get("df")
    if df is None or df.empty:
        return df, 0, 0
    rows = df.copy()
    for clause in _lw_parse_filter(filter_text):
        rows = rows.loc[_lw_clause_mask(rows, clause)]

    # An L-W strip is useful only where the loaded measured spectrum exists.
    rows = rows.loc[(rows["Freq"] >= MEAS_XMIN) & (rows["Freq"] <= MEAS_XMAX)]

    before_hf = len(rows)
    if ignore_hf:
        rotational = [
            col for col in ("UpperJ", "UpperKa", "UpperKc", "LowerJ", "LowerKa", "LowerKc")
            if col in rows.columns
        ]
        if len(rotational) == 6 and len(rows):
            rows = (rows.sort_values("Intensity", ascending=False)
                        .drop_duplicates(rotational, keep="first"))

    if sort_key not in rows.columns:
        sort_key = "Freq"
    rows = rows.sort_values([sort_key, "Freq"], kind="stable")
    total = len(rows)
    return rows.head(LW_MAX_ROWS).copy(), total, before_hf


def _lw_transition_label(row, qn_order):
    upper = [str(int(row[c])) for c in qn_order if c.startswith("Upper") and c in row and pd.notna(row[c])]
    lower = [str(int(row[c])) for c in qn_order if c.startswith("Lower") and c in row and pd.notna(row[c])]
    return f"{' '.join(upper)} ← {' '.join(lower)}  |  {float(row['Freq']):.4f} MHz"


def _lw_empty_figure(message="Choose a filter and click Plot Loomis-Wood"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color="#f2f4f5"))
    fig.update_layout(template="simple_white", height=330,
                      plot_bgcolor="#202428", paper_bgcolor="#202428",
                      font_color="#ffffff", margin=dict(l=50, r=20, t=25, b=45),
                      xaxis_title="Offset from predicted frequency (MHz)")
    return fig

# =========================
# Helpers
# =========================
def compute_uncertainties(assignments, selection_range, delta_F=DELTA_F_DEFAULT):
    """
    Compute uncertainty components for each observed frequency.

    Terms:
      - sigma_merge  : spread of simulated lines mapped to same observed frequency
      - sigma_interf : crowding from nearby observed frequencies
      - sigma_instr  : SNR-based term (uses Rayleigh-estimated noise)

    Rayleigh noise:
      - FitCtx path: estimate noise from sideband residuals after subtracting fitted model .
      - No-FitCtx path: estimate noise from local sidebands using iterative Rayleigh clipping to remove peaks.

    Returns:
      {obs: {
           sigma_total,
           sigma_merge,
           sigma_interf,
           sigma_instr,
           SNR,
           baseline_rms,
           peak_height,
           error_method  # NEW: how this line's σ_instr was computed
      }}
    """
    if not assignments:
        return {}

    # ---- group simulated lines by observed frequency ----
    sim_by_obs = defaultdict(list)
    fitctx_by_obs = {}  # one FitCtx per obs if present
    for r in assignments:
        try:
            o = float(r["obs"])
            sim_by_obs[o].append(float(r["sim"]))
            fc = r.get("FitCtx")
            if fc and o not in fitctx_by_obs:
                fitctx_by_obs[o] = fc
        except Exception:
            continue

    obs = np.array(sorted(sim_by_obs.keys()), dtype=float)
    n = obs.size

    # ---- (1) merge term: spread of sims mapped to same obs ----
    sigma_merge = np.zeros(n, dtype=float)
    for i, oi in enumerate(obs):
        sims = np.asarray(sim_by_obs[oi], dtype=float)
        if sims.size > 1:
            smax = float(np.max(sims))
            smin = float(np.min(sims))
            sigma_merge[i] = 0.5 * (smax - smin)

    # ---- (2) interference term: crowding from nearby obs ----
    sigma_interf = _sigma_interf_local(obs, delta_F, cutoff_mult=6.0)


    # ---- (3) instrumental/SNR term ----
    try:
        x0, x1 = selection_range if selection_range else (0.0, 0.0)
        sel_w = float(x1 - x0)
    except Exception:
        sel_w = 0.0

    bw = (sel_w / 6.0) if sel_w > 0 else 0.30
    bw = max(0.05, min(bw, 0.80))  # 0.05–0.80 MHz

    SNR_FLOOR = 4.0  # keeps σ_instr from blowing up when noise is very small / fit is too "perfect"

    def nearest_idx(x):
        j = np.searchsorted(meas_freqs, x)
        if j <= 0:
            return 0
        if j >= len(meas_freqs):
            return len(meas_freqs) - 1
        return j if (abs(meas_freqs[j] - x) < abs(meas_freqs[j - 1] - x)) else (j - 1)

    # ---------- model + Rayleigh helpers ----------
    def model_sum(x, baseline, peaks):
        x = np.asarray(x, dtype=float)
        y = np.full_like(x, float(baseline), dtype=float)
        for p in (peaks or []):
            A = float(p["amp"]); M = float(p["mu"]); S = float(p["sigma"])
            y += A * np.exp(-0.5 * ((x - M) / S) ** 2)
        return y

    def rayleigh_sigma_from_median(r):
        """
        Underlying complex Gaussian sigma from Rayleigh magnitudes.
        median(R) = sigma * sqrt(2 ln 2)
        """
        r = np.asarray(r, dtype=float)
        r = r[np.isfinite(r) & (r >= 0)]
        if r.size < 5:
            return np.nan
        return float(np.median(r)) / np.sqrt(2.0 * np.log(2.0))

    def rayleigh_std_from_sigma(sigma):
        """std of Rayleigh magnitudes"""
        return float(sigma) * np.sqrt((4.0 - np.pi) / 2.0)

    def rayleigh_noise_std_from_residuals(r):
        """
        Residuals are treated as magnitude samples (>=0).
        Returns std(R) of Rayleigh magnitude noise (not the mean).
        """
        sigma = rayleigh_sigma_from_median(r)
        if not np.isfinite(sigma) or sigma <= 0:
            return np.nan
        return rayleigh_std_from_sigma(sigma)

    def rayleigh_std_clipped(y, clip_sigma=3.0, max_iter=8, min_pts=20):
        """
        Estimate Rayleigh std from y by iteratively removing likely peaks.
        Threshold uses Rayleigh mean + clip_sigma * Rayleigh std.
        """
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y) & (y >= 0)]
        if y.size < min_pts:
            return np.nan

        keep = y
        for _ in range(max_iter):
            sigma = rayleigh_sigma_from_median(keep)
            if not np.isfinite(sigma) or sigma <= 0:
                return np.nan

            mean_R = sigma * np.sqrt(np.pi / 2.0)
            std_R = rayleigh_std_from_sigma(sigma)
            thr = mean_R + clip_sigma * std_R

            new_keep = keep[keep <= thr]

            if new_keep.size == keep.size or new_keep.size < min_pts:
                keep = new_keep
                break
            keep = new_keep

        sigma = rayleigh_sigma_from_median(keep)
        if not np.isfinite(sigma) or sigma <= 0:
            return np.nan
        return rayleigh_std_from_sigma(sigma)

    # ---------- Rice de-bias helpers ----------
    def sigma_complex_from_rayleigh_std(std_R):
        """
        Convert Rayleigh magnitude std (std of |N|) to complex Gaussian sigma (per quadrature).
        For N = sqrt(X^2+Y^2), with X,Y~N(0,sigma_c^2):
            std(|N|) = sigma_c * sqrt((4-pi)/2)
        """
        std_R = float(std_R)
        if not np.isfinite(std_R) or std_R <= 0:
            return np.nan
        return std_R / np.sqrt((4.0 - np.pi) / 2.0)

    def rice_debias_nu(R, sigma_c):
        """
        Moment-based Rice de-bias:
            E[R^2] = nu^2 + 2*sigma_c^2  ->  nu_hat = sqrt(max(R^2 - 2*sigma_c^2, 0))
        Works well from moderate SNR upward; at low SNR it clamps to 0.
        """
        R = float(R)
        sigma_c = float(sigma_c)
        if (not np.isfinite(R)) or (not np.isfinite(sigma_c)) or sigma_c <= 0:
            return np.nan
        return float(np.sqrt(max(R*R - 2.0*sigma_c*sigma_c, 0.0)))


    # ---------- outputs ----------
    sigma_instr = np.zeros(n, dtype=float)
    snr_vals = np.zeros(n, dtype=float)
    base_rms_vals = np.zeros(n, dtype=float)
    peak_heights = np.zeros(n, dtype=float)
    err_method = {}  # obs -> string describing how σ_instr was derived

    for i, oi in enumerate(obs):
        fitctx = fitctx_by_obs.get(oi)

        # defaults
        baseline_rms = np.nan
        peak_height = np.nan
        method_label = "unknown"

        if fitctx and isinstance(fitctx, dict):
            baseline = float(fitctx.get("baseline", 0.0) or 0.0)
            peaks = fitctx.get("peaks", []) or []

            # --- Rayleigh noise from sidebands *without* subtracting baseline or model ---
            br = fitctx.get("baseline_range", None)
            y_side = np.array([], dtype=float)

            if isinstance(br, (list, tuple)) and len(br) == 2:
                x0b, x1b = float(br[0]), float(br[1])

                # baseline_range = [x0-margin, x1+margin]
                # middle third ~ fit window; outer thirds ~ sidebands
                window_width = max(1e-9, (x1b - x0b) / 3.0)
                x0 = x0b + window_width
                x1 = x1b - window_width

                side_left  = (meas_freqs >= x0b) & (meas_freqs <  x0)
                side_right = (meas_freqs >  x1)  & (meas_freqs <= x1b)

                x_side = np.concatenate([meas_freqs[side_left], meas_freqs[side_right]])
                y_side = np.concatenate([meas_intensities[side_left], meas_intensities[side_right]])

                if x_side.size >= 8:
                    # Treat sideband amplitudes directly as Rayleigh magnitudes
                    mag_vals = np.clip(y_side, 0.0, None)
                    baseline_rms = rayleigh_std_clipped(mag_vals, clip_sigma=3.0)
                    method_label = "FitCtx-Rayleigh"
                else:
                    baseline_rms = np.nan
                    method_label = "FitCtx-InsufficientSideband"

            # fallback if Rayleigh fails in FitCtx path
            if not np.isfinite(baseline_rms) or baseline_rms <= 0:
                try:
                    print(
                        f"[WARN] Rayleigh noise failed in FitCtx at obs={oi:.4f} MHz "
                        f"(baseline_range={br}); using default baseline_rms=0.05"
                    )
                except Exception:
                    print("[WARN] Rayleigh noise failed in FitCtx; using default baseline_rms=0.05")

                baseline_rms = 0.05
                method_label = "FitCtx-DefaultFallback"

            # observed peak magnitude at oi from fitted model (baseline + gauss sum)
            sig = 0.0
            for p in peaks:
                A = float(p["amp"]); M = float(p["mu"]); S = float(p["sigma"])
                sig += A * np.exp(-0.5 * ((oi - M) / S) ** 2)

            R_abs = float(baseline + sig)     # absolute fitted magnitude (display/debug)
            R_sig = float(max(sig, 0.0))      # peak above baseline (use for Rice/SNR)

            peak_height = R_abs



        else:
            # --- No FitCtx fallback: local sidebands + Rayleigh clipping ---
            j = nearest_idx(oi)
            peak_height = float(meas_intensities[j])

            left_mask  = (meas_freqs >= oi - 2*bw) & (meas_freqs <  oi - bw)
            right_mask = (meas_freqs >  oi + bw)  & (meas_freqs <= oi + 2*bw)
            baseline_vals = np.concatenate([meas_intensities[left_mask], meas_intensities[right_mask]])

            baseline_level = float(np.median(baseline_vals)) if baseline_vals.size else 0.0
            R_abs = float(meas_intensities[j])
            R_sig = float(max(R_abs - baseline_level, 0.0))  # keep for debug/display if you want


            peak_height = R_abs


            baseline_rms = rayleigh_std_clipped(baseline_vals, clip_sigma=3.0)
            if np.isfinite(baseline_rms) and baseline_rms > 0:
                method_label = "NoFit-Rayleigh"
            else:
                print(
                    f"[WARN] Rayleigh noise failed (no FitCtx) at obs={oi:.4f} MHz; "
                    f"using default baseline_rms=0.05"
                )
                baseline_rms = 0.05
                method_label = "NoFit-DefaultFallback"

        # --- ALWAYS clamp and compute SNR + sigma_instr ---
        baseline_rms = max(float(baseline_rms), 1e-6)

        # Convert Rayleigh magnitude std -> underlying complex sigma (per quadrature)
        sigma_c = sigma_complex_from_rayleigh_std(baseline_rms)

        if fitctx and isinstance(fitctx, dict):
            # fitted peak amplitude is treated as already noise-free
            nu_hat = float(max(R_sig, 0.0))
            method_label = method_label + "-NoRice"
        else:
            # No FitCtx: use Rice de-bias on the observed magnitude at the peak
            nu_hat = rice_debias_nu(R_abs, sigma_c)

        # SNR: compare signal estimate to Rayleigh magnitude-noise std
        SNR = nu_hat / sigma_c        # sigma_c = complex sigma


        if SNR_FLOOR is not None:
            SNR = max(SNR, SNR_FLOOR)

        sigma_instr[i] = np.sqrt((0.0575 / SNR) ** 2 + (BASE_SIGMA_INSTR) ** 2)


        snr_vals[i] = float(SNR if np.isfinite(SNR) else 1e6)
        base_rms_vals[i] = float(baseline_rms)
        peak_heights[i] = float(peak_height)
        err_method[float(oi)] = method_label


    sigma_total = np.sqrt(sigma_merge**2 + sigma_interf**2 + sigma_instr**2)

    out = {}
    for o, st, smg, sif, sin, snr, brms, ph in zip(
        obs, sigma_total, sigma_merge, sigma_interf, sigma_instr, snr_vals, base_rms_vals, peak_heights
    ):
        method = err_method.get(float(o), "unknown")
        out[float(o)] = {
            "sigma_total":  round(float(st), 4),
            "sigma_merge":  round(float(smg), 4),
            "sigma_interf": round(float(sif), 4),
            "sigma_instr":  round(float(sin), 4),
            "SNR":          round(float(snr), 2),
            "baseline_rms": round(float(brms), 6),
            "peak_height":  round(float(ph), 6),
            "error_method": method,   # <--- NEW field
        }

        if DEBUG_UNC:
            print(
                f"[UNC] obs={float(o):10.4f} MHz | "
                f"method={method:20s} | "
                f"merge={out[float(o)]['sigma_merge']:.4f} MHz, "
                f"interf={out[float(o)]['sigma_interf']:.4f} MHz, "
                f"instr={out[float(o)]['sigma_instr']:.4f} MHz (SNR={out[float(o)]['SNR']:.2f}) "
                f"-> total={out[float(o)]['sigma_total']:.4f} MHz | "
                f"baseline_rms={out[float(o)]['baseline_rms']:.3g}, "
                f"peak={out[float(o)]['peak_height']:.3g}"
            )

    return out

def _sigma_interf_local(obs_sorted, delta_F, cutoff_mult=6.0):
    obs_sorted = np.asarray(obs_sorted, dtype=float)
    n = obs_sorted.size
    if n <= 1:
        return np.zeros(n, dtype=float)

    R = cutoff_mult * float(delta_F)
    out = np.zeros(n, dtype=float)

    for i, oi in enumerate(obs_sorted):
        lo = np.searchsorted(obs_sorted, oi - R, side="left")
        hi = np.searchsorted(obs_sorted, oi + R, side="right")

        neigh = obs_sorted[lo:hi]
        D = np.abs(neigh - oi)
        D = D[D > 0.0]  # remove self

        if D.size == 0:
            out[i] = 0.0
            continue

        C = 0.5 * D * (1.0 - np.tanh(D / float(delta_F)))
        out[i] = np.sqrt(np.sum(C * C))

    return out


def _sanitize_for_table(rows):
    if not isinstance(rows, list):
        return rows
    clean = []
    for r in rows:
        rc = dict(r)
        rc.pop("FitCtx", None)

        # ensure flags exist
        rc["Include_SNR"]    = bool(rc.get("Include_SNR", True))
        rc["Include_Interf"] = bool(rc.get("Include_Interf", True))
        rc["Include_Merge"]  = bool(rc.get("Include_Merge", True))

        # ensure numeric mirrors exist
        rc["Include_SNR_num"]    = 1 if rc["Include_SNR"] else 0
        rc["Include_Interf_num"] = 1 if rc["Include_Interf"] else 0
        rc["Include_Merge_num"]  = 1 if rc["Include_Merge"] else 0
        try:
            rc["WeightedSim"] = None if rc.get("WeightedSim") is None else round(float(rc["WeightedSim"]), 4)
        except Exception:
            rc["WeightedSim"] = None
        try:
            rc["AbsDelta"] = None if rc.get("Delta") is None else round(abs(float(rc["Delta"])), 4)
        except Exception:
            rc["AbsDelta"] = None

        clean.append(rc)
    return clean



def _recalc_total_from_flags(row):
    """Return new total (MHz) using only selected components."""
    c = []
    if row.get("Include_SNR", True):
        c.append(float(row.get("Unc_SNR", 0.0) or 0.0))
    if row.get("Include_Interf", True):
        c.append(float(row.get("Unc_Interf", 0.0) or 0.0))
    if row.get("Include_Merge", True):
        c.append(float(row.get("Unc_Merge", 0.0) or 0.0))
    total = float(np.sqrt(np.sum(np.square(c)))) if c else 0.0
    return round(total, 4)


def decimate_xy(x, y, max_pts=35000):
    n = x.size
    if n <= max_pts:
        return x, y
    step = max(1, n // max_pts)
    return x[::step], y[::step]

def decimate_xy_preserve_extrema(x, y, max_pts=35000):
    """
    Downsample by keeping local min & max in coarse bins so narrow peaks survive.
    Emits ~max_pts points total (min+max pairs).
    """
    n = x.size
    if n <= max_pts:
        return x, y
    # number of bins is half of max_pts because each bin yields 2 points
    n_bins = max(1, max_pts // 2)
    # integer bin edges over indices (uniform in index → fast)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi - lo <= 1:
            continue
        xi = x[lo:hi]
        yi = y[lo:hi]
        jmin = int(np.argmin(yi))
        jmax = int(np.argmax(yi))
        pair = sorted([(xi[jmin], yi[jmin]), (xi[jmax], yi[jmax])], key=lambda t: t[0])
        for px, py in pair:
            xs.append(px); ys.append(py)
    return np.asarray(xs), np.asarray(ys)


@lru_cache(maxsize=64)
def _fit_sum_cached(amps, mus, sigmas, baseline, x_min, x_max, npts):
    x = np.linspace(x_min, x_max, npts)
    y = np.full_like(x, baseline, dtype=float)
    for A, M, S in zip(amps, mus, sigmas):
        y += A * np.exp(-0.5 * ((x - M) / S) ** 2)
    return x, y

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def multi_gauss_with_offset(x, *params):
    offset = params[-1]
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        y += gaussian(x, params[i], params[i + 1], params[i + 2])
    return y + offset

def _apply_adaptive_xticks(fig, span):
    # Let Plotly choose by default
    if span is None:
        fig.update_xaxes(
            tickmode="auto",
            tickformatstops=[
                dict(dtickrange=[None,   1],  value=".4f"),  # <1 MHz → 4 decimals
                dict(dtickrange=[1,     10],  value=".2f"),  # 1–10 MHz → 2 decimals
                dict(dtickrange=[10,   None], value=".0f"),  # ≥10 MHz → integers
            ],
            showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.10)",
        )
        return

    # Wider windows → coarser ticks
    if span > 800:
        fig.update_xaxes(
            tickmode="linear", dtick=250, tickformat=".0f",
            showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.10)",
        )
    elif span > 200:
        fig.update_xaxes(tickmode="linear", dtick=50,  tickformat=".0f")
    elif span > 50:
        fig.update_xaxes(tickmode="linear", dtick=10,  tickformat=".1f")
    elif span > 5:
        fig.update_xaxes(tickmode="linear", dtick=1,   tickformat=".2f")
    elif span > 1:
        fig.update_xaxes(tickmode="linear", dtick=0.2, tickformat=".3f")  # 200 kHz
    else:
        fig.update_xaxes(tickmode="linear", dtick=0.05, tickformat=".4f") # 50 kHz

    # Helpful readout
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)



def build_assignment_columns(qn_field_order):
    static_columns = [
        {"name": "obs", "id": "obs", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "sim", "id": "sim", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "sim_w", "id": "WeightedSim", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "Delta (obs-sim_w)", "id": "Delta", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "Eu", "id": "Eu", "type": "numeric", "format": {"specifier": ".2f"}, "editable": False},
        {"name": "logI", "id": "logI", "type": "numeric", "format": {"specifier": ".2f"}, "editable": False},
    ]

    qn_columns = [
        {"name": qn_label_map.get(col, col), "id": col, "editable": False}
        for col in qn_field_order
    ]

    toggle_display_cols = [
        {"name": "SNR (MHz)",    "id": "Disp_SNR",    "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "Interf (MHz)", "id": "Disp_Interf", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "Merge (MHz)",  "id": "Disp_Merge",  "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
    ]

    extras = [
        {"name": "Uncertainty (MHz)", "id": "Uncertainty", "type": "numeric",
         "format": {"specifier": ".4f"}, "editable": True},
        {"name": "Weight", "id": "Weight", "type": "numeric",
         "format": {"specifier": ".2f"}, "editable": False},
    ]

    hidden_flags = [
        {"name": "AbsDelta",           "id": "AbsDelta",           "type": "numeric", "editable": False},
        {"name": "Include_SNR",        "id": "Include_SNR",        "type": "any",     "editable": False},
        {"name": "Include_Interf",     "id": "Include_Interf",     "type": "any",     "editable": False},
        {"name": "Include_Merge",      "id": "Include_Merge",      "type": "any",     "editable": False},
        {"name": "Include_SNR_num",    "id": "Include_SNR_num",    "type": "numeric", "editable": False},
        {"name": "Include_Interf_num", "id": "Include_Interf_num", "type": "numeric", "editable": False},
        {"name": "Include_Merge_num",  "id": "Include_Merge_num",  "type": "numeric", "editable": False},
    ]

    return static_columns + [{"name": "CAT status", "id": "CatalogStatus", "editable": False}] + qn_columns + toggle_display_cols + extras + hidden_flags


def recompute_peak_weights(assignments, recompute_weights=True):
    if not assignments:
        return assignments
    by_obs = defaultdict(list)
    for i, r in enumerate(assignments):
        by_obs[r["obs"]].append(i)
    for obs, idxs in by_obs.items():
        if recompute_weights:
            scores = []
            for i in idxs:
                r = assignments[i]
                try:
                    strength = 10.0 ** float(r["logI"])
                except (ValueError, TypeError):
                    strength = 0.0
                if not np.isfinite(strength) or strength < 0:
                    strength = 0.0
                scores.append(strength)
            ssum = sum(scores)
            if ssum <= 0:
                weights = [1.0 / len(idxs)] * len(idxs)
            else:
                weights = [s / ssum for s in scores]
        else:
            weights = []
            for i in idxs:
                try:
                    w = float(assignments[i].get("Weight", 0.0) or 0.0)
                except Exception:
                    w = 0.0
                if not np.isfinite(w) or w < 0:
                    w = 0.0
                weights.append(w)
            wsum = sum(weights)
            if wsum <= 0:
                weights = [1.0 / len(idxs)] * len(idxs)
            else:
                weights = [w / wsum for w in weights]
        weighted_sim = 0.0
        for i, w in zip(idxs, weights):
            try:
                sim_val = float(assignments[i].get("sim", 0.0) or 0.0)
            except Exception:
                sim_val = 0.0
            weighted_sim += w * sim_val

        unresolved = any(assignments[i].get("CatalogStatus") in ("missing", "ambiguous") for i in idxs)
        weighted_sim = None if unresolved else round(float(weighted_sim), 4)
        for i, w in zip(idxs, weights):
            assignments[i]["Weight"] = round(w, 4)
            assignments[i]["WeightedSim"] = weighted_sim
            try:
                assignments[i]["Delta"] = round(float(assignments[i]["obs"]) - weighted_sim, 4)
            except Exception:
                assignments[i]["Delta"] = None
    return assignments


def _decorate_display_flags(rows):
    if not isinstance(rows, list):
        return rows
    for r in rows:
        r["Include_SNR"]    = bool(r.get("Include_SNR", True))
        r["Include_Interf"] = bool(r.get("Include_Interf", True))
        r["Include_Merge"]  = bool(r.get("Include_Merge", True))

        r["Disp_SNR"]    = round(float(r.get("Unc_SNR",    0.0) or 0.0), 4)
        r["Disp_Interf"] = round(float(r.get("Unc_Interf", 0.0) or 0.0), 4)
        r["Disp_Merge"]  = round(float(r.get("Unc_Merge",  0.0) or 0.0), 4)
        try:
            r["WeightedSim"] = None if r.get("WeightedSim") is None else round(float(r["WeightedSim"]), 4)
        except Exception:
            r["WeightedSim"] = None
        try:
            r["AbsDelta"] = None if r.get("Delta") is None else round(abs(float(r["Delta"])), 4)
        except Exception:
            r["AbsDelta"] = None

        # NEW: numeric mirrors used by style filters
        r["Include_SNR_num"]    = 1 if r["Include_SNR"]    else 0
        r["Include_Interf_num"] = 1 if r["Include_Interf"] else 0
        r["Include_Merge_num"]  = 1 if r["Include_Merge"]  else 0
    return rows



# --- per-catalog sim-scale persistence ---
def _get_scale_cache_path():
    return os.path.join(WORKSPACE_STATE_DIR, "scales.json")

def _load_scale_cache():
    p = _get_scale_cache_path()
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # keys are catalog indexes as strings -> float scales
                    return {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass
    return {}

def _save_scale_cache(scales_dict):
    try:
        with open(_get_scale_cache_path(), "w", encoding="utf-8") as f:
            json.dump(scales_dict, f, indent=2)
    except Exception:
        pass


# --- assignment autosave persistence ---
def _get_assignment_autosave_path():
    autosave_dir = WORKSPACE_STATE_DIR
    return os.path.join(autosave_dir, "plotcomparison_assignments_autosave.json")


def _autosave_context_signature():
    try:
        csv_path = os.path.abspath(csv_file_path)
        csv_mtime = os.path.getmtime(csv_path)
    except Exception:
        csv_path = os.path.abspath(str(csv_file_path))
        csv_mtime = None

    catalogs_sig = []
    for cat in catalogs:
        path = os.path.abspath(cat.get("path", ""))
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = None
        catalogs_sig.append({
            "path": path,
            "name": cat.get("name"),
            "mtime": mtime,
        })
    return {
        "csv_file": {
            "path": csv_path,
            "mtime": csv_mtime,
        },
        "catalogs": catalogs_sig,
    }


def _autosave_context_matches(payload):
    if not isinstance(payload, dict):
        return False

    saved_context = payload.get("context")
    if saved_context is None:
        # Compatibility with the first autosave format from this script.
        saved_context = {
            "csv_file": None,
            "catalogs": payload.get("catalogs"),
        }

    current = _autosave_context_signature()

    saved_csv = saved_context.get("csv_file") if isinstance(saved_context, dict) else None
    if isinstance(saved_csv, dict):
        if saved_csv.get("path") != current["csv_file"]["path"]:
            return False
    else:
        # Missing CSV context is too ambiguous when switching molecules.
        return False

    saved_catalogs = saved_context.get("catalogs") if isinstance(saved_context, dict) else None
    if not isinstance(saved_catalogs, list):
        return False
    return all(isinstance(saved, dict) and saved.get('path') for saved in saved_catalogs)


def _load_assignment_autosave():
    path = _get_assignment_autosave_path()
    if not os.path.isfile(path):
        # Read-only migration: only the matching v6 spectrum context is accepted below.
        path = os.path.join(script_dir, 'autosave', 'plotcomparison_assignments_autosave.json')
        if not os.path.isfile(path):
            return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[AUTOSAVE] Could not read assignment autosave: {e}")
        return {}

    if not _autosave_context_matches(payload):
        print(f"[AUTOSAVE] Ignored autosave because cat/csv context does not match: {path}")
        return {}

    data = payload.get("percat_assignments", payload) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return {}

    cleaned = {}
    saved_catalogs = payload['context']['catalogs']
    current_paths = {os.path.normcase(os.path.abspath(cat['path'])): str(i) for i, cat in enumerate(catalogs)}
    for key, rows in data.items():
        try:
            key = current_paths.get(os.path.normcase(os.path.abspath(saved_catalogs[int(key)]['path'])))
        except (ValueError, IndexError, KeyError, TypeError):
            continue
        if key is None or not isinstance(rows, list):
            continue
        cleaned[key], _ = remap_assignments([dict(r) for r in rows if isinstance(r, dict)], catalogs[int(key)])
        cleaned[key] = recompute_peak_weights(cleaned[key], recompute_weights=False)

    total_rows = sum(len(v) for v in cleaned.values())
    if total_rows:
        print(f"[AUTOSAVE] Restored {total_rows} assignment row(s) from {path}")
    return cleaned


def _save_assignment_autosave(percat):
    if not isinstance(percat, dict):
        return
    path = _get_assignment_autosave_path()
    payload = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "context": _autosave_context_signature(),
        "percat_assignments": percat,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[AUTOSAVE] Could not save assignments: {e}")


def _get_cache_path():
    # a small temp JSON cache
    return os.path.join(WORKSPACE_STATE_DIR, "fitcache.json")

def _load_fitcache():
    p = _get_cache_path()
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_fitcache(cache):
    p = _get_cache_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

# --- Debug prints: where are the caches? ---
try:
    print(f"[CACHE] Scale cache path: {_get_scale_cache_path()}")
    print(f"[CACHE] Fit cache path:   {_get_cache_path()}")
except Exception as e:
    print(f"[CACHE] Could not determine cache paths: {e}")


def _build_fit_context(selection, fit_params, delta_F=DELTA_F_DEFAULT):
    # what we want to remember for repeatable recomputes
    sel_range = None
    if selection and isinstance(selection, dict) and "range" in selection and "x" in selection["range"]:
        sel_range = [float(selection["range"]["x"][0]), float(selection["range"]["x"][1])]
    elif fit_params and isinstance(fit_params, dict) and "baseline_range" in fit_params:
        br = fit_params.get("baseline_range")
        if isinstance(br, (list, tuple)) and len(br) == 2:
            sel_range = [float(br[0]), float(br[1])]
    # fallback: cause compute_uncertainties to use its default width
    if not sel_range:
        sel_range = [0.0, 0.0]

    peaks = []
    if fit_params and isinstance(fit_params, dict):
        for p in fit_params.get("multi", []) or []:
            try:
                peaks.append({"amp": float(p["amp"]), "mu": float(p["mu"]), "sigma": float(p["sigma"])})
            except Exception:
                pass

    ctx = {
        "selection_range": sel_range,                          # <- this is the critical piece
        "baseline": float(fit_params.get("baseline", 0.0)) if fit_params else None,
        "baseline_std": float(fit_params.get("baseline_std", 0.0)) if fit_params else None,
        "baseline_range": fit_params.get("baseline_range") if fit_params else None,
        "n_gauss": len(peaks) if peaks else None,
        "peaks": peaks,
        "delta_F": float(delta_F),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return ctx

def _cache_key(active_idx, obs, sim_uid):
    # stable identifier for this row in cache
    cat = catalogs[int(active_idx)]
    matches = cat['df'].loc[cat['df']['SimUID'] == sim_uid]
    transition = ",".join(str(int(matches.iloc[0][key])) for key in cat['qn_order']) if len(matches) == 1 else "unmatched"
    return f"{identity(cat['path'])}|obs{float(obs):.4f}|qn{transition}"

def _persist_existing_fitctx(rows, active_idx):
    """
    Persist any existing FitCtx per row into the JSON cache.
    Unlike _attach_and_persist_fitctx, this does NOT overwrite FitCtx
    on the rows; it only mirrors what's already there into the cache.
    """
    if not rows:
        return
    cache = _load_fitcache()
    for r in rows:
        fc = r.get("FitCtx")
        if not fc:
            continue
        k = _cache_key(active_idx, r.get("obs"), r.get("SimUID"))
        cache[k] = fc
    _save_fitcache(cache)


def _attach_and_persist_fitctx(rows, active_idx, fit_ctx):
    """
    Attach the same fit context to all given rows and persist it.
    If fit_ctx is None, do nothing (avoid wiping existing FitCtx).
    """
    if fit_ctx is None:
        return
    cache = _load_fitcache()
    for r in rows:
        r["FitCtx"] = fit_ctx
        k = _cache_key(active_idx, r.get("obs"), r.get("SimUID"))
        cache[k] = fit_ctx
    _save_fitcache(cache)

def _restore_fitctx_if_missing(rows, active_idx):
    """
    If a row is missing FitCtx (e.g., loaded from .lin or legacy items), try to restore it from cache.
    """
    cache = _load_fitcache()
    changed = False
    restored_count = 0
    for r in rows:
        if "FitCtx" not in r or not r["FitCtx"]:
            k = _cache_key(active_idx, r.get("obs"), r.get("SimUID"))
            if k in cache:
                r["FitCtx"] = cache[k]
                changed = True
                restored_count += 1
    if changed:
        print(f"[FITCTX] Restored {restored_count} fit context(s) for catalog {active_idx}")
    return rows, changed


def _fit_sigma_for_mu(mu, fit_params):
    if mu is None or not fit_params or "multi" not in fit_params:
        return None
    fits = fit_params.get("multi") or []
    if not fits:
        return None
    try:
        mu = float(mu)
    except Exception:
        return None

    best = None
    best_dist = np.inf
    for p in fits:
        try:
            pmu = float(p["mu"])
            psig = float(p["sigma"])
        except Exception:
            continue
        d = abs(pmu - mu)
        if d < best_dist:
            best_dist = d
            best = psig

    if best is None or not np.isfinite(best) or best <= 0:
        return None
    return float(best)


def _window_from_mu_sigma(mu, sigma, mult=10.0):
    try:
        mu = float(mu)
        sigma = float(sigma)
    except Exception:
        return None
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return None
    radius = mult * sigma
    return [mu - radius, mu + radius]


def _obs_in_window(obs_value, local_window):
    if not local_window or len(local_window) != 2:
        return True
    try:
        ov = float(obs_value)
        x0, x1 = map(float, local_window)
    except Exception:
        return False
    if x1 < x0:
        x0, x1 = x1, x0
    return x0 <= ov <= x1


def _recalc_uncertainties(rows, selection_range=None, delta_F=None, full=True, local_window=None):
    """
    Recompute uncertainty terms. Interference is computed once across ALL observed
    lines (so nearby obs can contribute), while SNR & Merge are still computed
    per-obs using that obs' saved selection range if available.
    """
    if not rows:
        return rows

    # --- compute interference once across obs (fast local-neighbor method) ---
    # pick a global delta_F: first valid from any FitCtx, else provided, else default
    df_candidates = []
    for r in rows:
        fc = r.get("FitCtx") or {}
        try:
            cand = float(fc.get("delta_F", np.nan))
            if np.isfinite(cand) and cand > 0:
                df_candidates.append(cand)
        except Exception:
            pass
    df_global = (df_candidates[0] if df_candidates
                else (float(delta_F) if (delta_F is not None) else DELTA_F_DEFAULT))

    # Build sorted unique obs list
    obs_all = np.array(sorted({float(r.get("obs")) for r in rows if r.get("obs") is not None}), dtype=float)

    # Compute sigma_interf for all obs once (already cheap now)
    sigma_interf_all = _sigma_interf_local(obs_all, df_global, cutoff_mult=6.0)
    obs_to_interf = {float(o): float(s) for o, s in zip(obs_all, sigma_interf_all)}



    # --- Now compute SNR & Merge per-obs as before ---
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[r.get("obs")].append(i)

    for obs_val, idxs in grouped.items():
        if (not full) and local_window is not None and (not _obs_in_window(obs_val, local_window)):
            continue

        # choose selection_range (prefer per-row FitCtx on this obs)
        sel_range = None
        for i in idxs:
            fitctx = rows[i].get("FitCtx")
            if fitctx and "selection_range" in fitctx:
                sel_range = fitctx["selection_range"]
                break
        if sel_range is None:
            sel_range = selection_range if selection_range else [0.0, 0.0]

        # choose delta_F for SNR/Merge calc (doesn't affect sigma_interf here)
        df_local = None
        for i in idxs:
            fitctx = rows[i].get("FitCtx")
            if fitctx and ("delta_F" in fitctx):
                try:
                    cand = float(fitctx["delta_F"])
                    df_local = cand if cand > 0 else None
                except Exception:
                    pass
                if df_local is not None:
                    break

        if df_local is None:
            df_local = float(delta_F) if (delta_F is not None) else DELTA_F_DEFAULT

        subset = [rows[i] for i in idxs]
        unc_map = compute_uncertainties(subset, sel_range, delta_F=df_local)

        for i in idxs:
            r = rows[i]
            u = unc_map.get(float(r.get("obs"))) if r.get("obs") is not None else None
            if u:
                # keep SNR & Merge from the per-obs calc
                r["Unc_SNR"]   = u["sigma_instr"]
                r["Unc_Merge"] = u["sigma_merge"]

            # overwrite Interf with the GLOBAL value so neighbors contribute
            try:
                ov = float(r.get("obs"))
                if ov in obs_to_interf:
                    r["Unc_Interf"] = obs_to_interf[ov]
            except Exception:
                pass

            # Ensure flags exist (preserve existing)
            if "Include_SNR"    not in r: r["Include_SNR"]    = True
            if "Include_Interf" not in r: r["Include_Interf"] = True
            if "Include_Merge"  not in r: r["Include_Merge"]  = True

            # total from flags
            r["Uncertainty"] = _recalc_total_from_flags(r)

    return rows


def parse_lin_line_flexible(line):
    if line.endswith("\n"):
        line = line[:-1]
    if not line.strip():
        raise ValueError("blank line")

    WT_W, UNC_W, FREQ_W = 6, 10, 12
    wt_str = line[-WT_W:]
    unc_str = line[-(WT_W + UNC_W):-WT_W]
    freq_str = line[-(WT_W + UNC_W + FREQ_W):-(WT_W + UNC_W)]
    qns_raw = line[:-(WT_W + UNC_W + FREQ_W)].rstrip()

    freq = float(freq_str.strip())
    unc = float(unc_str.strip())
    wt = float(wt_str.strip())

    qns = []
    for i in range(0, len(qns_raw), 3):
        chunk = qns_raw[i:i + 3]
        if chunk.strip():
            qns.append(int(chunk))
    return qns, freq, unc, wt

# =========================
# Dash App
# =========================
app = dash.Dash(__name__, assets_folder=os.path.join(script_dir, "assets"), url_base_pathname=WORKSPACE_PREFIX)
server = app.server

DEFAULT_XMIN = 5000.0
DEFAULT_XMAX = 18500.0
DEFAULT_MERGE_GUARD_MHZ = 0.05

MERGE_WARNING_HIDDEN_STYLE = {
    "display": "none",
}
MERGE_WARNING_VISIBLE_STYLE = {
    "position": "fixed",
    "inset": "0",
    "zIndex": 10000,
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "backgroundColor": "rgba(0, 0, 0, 0.72)",
    "padding": "24px",
}

_initial_scales = _load_scale_cache()
_initial_assignments = _load_assignment_autosave()
_initial_active_rows = _decorate_display_flags(list(_initial_assignments.get("0", [])))
_initial_assignment_columns = build_assignment_columns(catalogs[0]["qn_order"]) if catalogs else []
_lw_sort_columns = []
for _cat in catalogs:
    for _col in _cat.get("qn_order", []):
        if _col not in _lw_sort_columns:
            _lw_sort_columns.append(_col)
_lw_sort_options = [
    {
        "label": qn_label_map.get(
            col,
            col.replace("Upper", "").replace("Lower", "") + ("'" if col.startswith("Lower") else ""),
        ),
        "value": col,
    }
    for col in _lw_sort_columns
] + [{"label": "Frequency", "value": "Freq"}]
_lw_builder_qns = _lw_qn_names(catalogs[0]) if catalogs else ["J", "Ka", "Kc"]
app.layout = html.Div([
    html.H2("Interactive Spectrum Assigner"),

    # Active catalog label
    html.Div([
        html.Span("Active catalog: "),
        html.Strong(id="active-cat-label")
    ], style={"marginBottom": "8px"}),

    # Intensity scaling & default X range controls
    html.Div([
        html.Label("Simulated Spectrum Intensity Scale:"),
        dcc.Input(id="sim-scale", type="number", min=0, max=1, value=1.0, step=0.001, style={"width": "120px", "marginRight": "10px"}),
        html.Button("Apply Scale", id="apply-scale", n_clicks=0),
        dcc.Checklist(id="original-intensity", options=[{"label": " Use original experimental intensity", "value": "raw"}],
                      value=[], inline=True, style={"display": "inline-block", "marginLeft": "18px"}),
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Label("Default X-axis Range (MHz):"),
        dcc.Input(id="default-xmin", type="number", value=DEFAULT_XMIN, step=100, style={"width": "120px"}),
        dcc.Input(id="default-xmax", type="number", value=DEFAULT_XMAX, step=100, style={"width": "120px", "marginLeft": "8px", "marginRight": "10px"}),
        html.Button("Apply X-range", id="apply-xrange", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    # Keyboard listener (added 's')
    Keyboard(id="keyboard", captureKeys=["q", "w", "e", "r", "a", "d", "s", "p", "Enter"] + [str(i) for i in range(0, 11)]),

    # Fitting controls
    html.Div([
        html.Label("Number of Gaussians to Fit:"),
        dcc.RadioItems(
            id='num-gaussians',
            options=[{'label': f'{n}', 'value': n} for n in range(1, 11)],
            value=1,
            inline=True
        )
    ], style={'marginBottom': '10px'}),

    html.Div([
        dcc.Checklist(
            id="flip-sim-checkbox",
            options=[{"label": "Flip Simulated Spectrum", "value": "flip"}],
            value=[],
            style={"color": "white"}
        )
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Label("Choose Fitted Peak to Assign:"),
        html.Div(id="fit-mu-button-container",
                 style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginTop": "5px"})
    ], style={"marginBottom": "10px"}),

    html.Div("Tip: Press 's' to switch the active catalog.", style={"marginBottom": "6px", "fontStyle": "italic"}),

    html.Div([
        html.Button("Assign Region", id="assign-button", n_clicks=0),
        html.Label("Maximum merge (MHz):", htmlFor="merge-guard-threshold",
                   style={"marginLeft": "14px", "marginRight": "6px"}),
        dcc.Input(
            id="merge-guard-threshold",
            type="number",
            value=DEFAULT_MERGE_GUARD_MHZ,
            min=0,
            step=0.005,
            debounce=True,
            style={"width": "90px"},
        ),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

    # Custom confirmation modal: Enter always rejects; only the explicit
    # proceed button can approve an over-threshold assignment.
    html.Div([
        html.Div([
            html.H3("Merge threshold exceeded", style={"marginTop": 0, "color": "#ffcc66"}),
            html.Div(id="merge-warning-message", style={"whiteSpace": "pre-line", "lineHeight": "1.55"}),
            html.Div([
                html.Button(
                    "Reject (Enter)", id="merge-warning-reject", n_clicks=0,
                    style={"fontWeight": "700", "marginRight": "12px"},
                ),
                html.Button(
                    "Still proceed", id="merge-warning-proceed", n_clicks=0,
                    style={"backgroundColor": "#8b2f2f", "color": "white", "fontWeight": "700"},
                ),
            ], style={"marginTop": "20px", "textAlign": "right"}),
        ], style={
            "width": "min(620px, 92vw)",
            "backgroundColor": "#24272b",
            "color": "white",
            "border": "1px solid #676d75",
            "borderRadius": "10px",
            "boxShadow": "0 18px 55px rgba(0, 0, 0, 0.65)",
            "padding": "22px 24px",
        }),
    ], id="merge-warning-modal", style=MERGE_WARNING_HIDDEN_STYLE),

    dcc.RadioItems(
        id='mode-selector',
        options=[
            {'label': 'Zoom', 'value': 'zoom'},
            {'label': 'Select Region to Fit', 'value': 'select'},
            {'label': 'Assign by Region', 'value': 'assign_all'}
        ],
        value='zoom',
        inline=True,
        style={"marginBottom": "10px"}
    ),

    # html.Div(
    #     id="cursor-readout",
    #     style={
    #         "marginBottom": "6px",
    #         "fontFamily": "monospace",
    #         "fontSize": "14px",
    #         "color": "white",
    #         "background": "rgba(0,0,0,0.25)",
    #         "padding": "4px 8px",
    #         "display": "inline-block",
    #         "borderRadius": "6px",
    #     },
    #     children="Freq: — MHz | Intensity: —",
    # ),



    dcc.Graph(id='spectrum-plot', config={"modeBarButtonsToAdd": ["select2d", "zoom2d"]}),
    html.Div(id="fit-output", style={"marginBottom": 10}),

    # Zoom controls (X and Y)
    html.Div([
        html.Button("Undo Zoom", id="undo-zoom-button", n_clicks=0, style={"marginRight": "10px"}),

        html.Button("X+ (Zoom In)", id="x-zoom-in", n_clicks=0, style={"marginRight": "5px"}),
        html.Button("X– (Zoom Out)", id="x-zoom-out", n_clicks=0, style={"marginRight": "20px"}),

        html.Button("Y+ (Zoom In)", id="y-zoom-in", n_clicks=0, style={"marginRight": "5px"}),
        html.Button("Y– (Zoom Out)", id="y-zoom-out", n_clicks=0),
    ], style={"marginBottom": "15px"}),

    html.Details([
        html.Summary("Loomis-Wood plot", style={
            "fontSize": "20px", "fontWeight": "600", "cursor": "pointer", "padding": "10px 0"
        }),
        html.Div([
            html.Div([
                html.Label("Filter clauses"),
                dcc.Input(id="lw-filter", type="text", value="dJ=1",
                          placeholder="e.g. dJ=1, dKa=0, Ka=1|2, logI>=-7",
                          debounce=True, style={"width": "100%"}),
                html.Div(
                    "Syntax: upper Ka=1; lower Ka'=1; changes dJ=1; OR Ka=1|2; "
                    "components Kc=J-Ka or Kc=J-Ka+1.",
                    style={"fontSize": "12px", "opacity": 0.8, "marginTop": "4px"}),
            ], className="lw-filter-field"),
            html.Div([html.Label("Sort by"),
                      dcc.Dropdown(id="lw-sort", options=_lw_sort_options,
                                   value=(_lw_sort_columns[0] if _lw_sort_columns else "Freq"),
                                   clearable=False, className="lw-dropdown")], className="lw-field"),
            html.Div([html.Label("Span (MHz)"),
                      dcc.Input(id="lw-span", type="number", value=20.0, min=0.001, step=0.001,
                                )], className="lw-field"),
            html.Div([html.Label("Y zoom"),
                      dcc.Input(id="lw-y-zoom", type="number", value=1.0, min=0.05, step=0.05,
                                )], className="lw-field"),
            html.Div([html.Label("Strip height"),
                      dcc.Input(id="lw-strip-height", type="number", value=72, min=36, max=180, step=4,
                                )], className="lw-field"),
        ], className="lw-toolbar-grid"),

        html.Div([
            html.Div([
                html.Span("Selection rule: "),
                dcc.Dropdown(id="lw-rule-qn",
                             options=[{"label": f"Δ{q}", "value": q} for q in _lw_builder_qns],
                             value=(_lw_builder_qns[0] if _lw_builder_qns else "J"), clearable=False,
                             className="lw-dropdown lw-qn-dropdown"),
                dcc.Dropdown(id="lw-rule-op", options=[{"label": x, "value": x} for x in ("=", "!=", "<", ">", "<=", ">=")],
                             value="=", clearable=False, className="lw-dropdown lw-op-dropdown"),
                dcc.Input(id="lw-rule-value", type="number", value=1, step=1),
                html.Button("Add", id="lw-add-rule", n_clicks=0),
            ], className="lw-builder-controls"),
            html.Div([
                html.Span("Range/value: "),
                dcc.Dropdown(id="lw-range-qn",
                             options=([{"label": q, "value": q} for q in _lw_builder_qns] +
                                      [{"label": f"{q}'", "value": f"{q}'"} for q in _lw_builder_qns] +
                                      [{"label": "logI", "value": "logI"}, {"label": "freq", "value": "freq"}]),
                             value=(_lw_builder_qns[0] if _lw_builder_qns else "J"), clearable=False,
                             className="lw-dropdown lw-qn-dropdown"),
                dcc.Dropdown(id="lw-range-op", options=[{"label": x, "value": x} for x in ("=", "!=", "<", ">", "<=", ">=")],
                             value="=", clearable=False, className="lw-dropdown lw-op-dropdown"),
                dcc.Input(id="lw-range-value", type="number", value=0, step="any"),
                html.Button("Add", id="lw-add-range", n_clicks=0),
            ], className="lw-builder-controls"),
            html.Div([
                html.Span("Series preset: "),
                dcc.Dropdown(id="lw-series", options=[{"label": k, "value": k} for k in LW_SERIES],
                             value="aR(0,1)", clearable=False, className="lw-dropdown lw-series-dropdown"),
                dcc.Input(id="lw-series-ka", type="number", placeholder="upper Ka", step=1),
                dcc.Dropdown(id="lw-series-component", options=[
                    {"label": "Both components", "value": "both"},
                    {"label": "Kc=J-Ka", "value": "lo"},
                    {"label": "Kc=J-Ka+1", "value": "hi"},
                ], value="both", clearable=False, className="lw-dropdown lw-component-dropdown"),
                html.Button("Add", id="lw-add-series", n_clicks=0),
            ], className="lw-builder-controls"),
        ], className="lw-builder-grid"),

        html.Div([
            dcc.Checklist(id="lw-options", options=[
                {"label": " Ignore hyperfine (keep strongest J,Ka,Kc component)", "value": "ignore_hf"},
                {"label": " Show transition labels", "value": "labels"},
            ], value=["ignore_hf", "labels"], inline=True,
               style={"display": "inline-block", "marginRight": "16px"}),
            html.Button("Plot Loomis-Wood", id="lw-plot-button", n_clicks=0,
                        style={"marginRight": "8px", "fontWeight": "600"}),
            html.Button("Clear filter", id="lw-clear-filter", n_clicks=0),
            html.Span(id="lw-status", style={"marginLeft": "14px", "fontFamily": "monospace"}),
        ], className="lw-actions"),

        html.Div([dcc.Loading(dcc.Graph(
            id="lw-graph", figure=_lw_empty_figure(),
            config={"displaylogo": False, "modeBarButtonsToAdd": ["toImage"],
                    "toImageButtonOptions": {"format": "png", "filename": "loomis_wood", "scale": 2}},
        ))], className="lw-graph-frame"),
        html.Div("Click any L-W strip to jump the main spectrum to that transition window.",
                 style={"fontSize": "12px", "opacity": 0.8, "marginTop": "5px"}),
    ], open=False, className="lw-panel"),

    html.Div([
        html.Label("Simulated Intensity Threshold (0–1):"),
        dcc.Input(id='intensity-threshold', type='number', min=0, max=1, step=0.0001, value=0.001, debounce=True)
    ], style={'marginBottom': '15px'}),

    html.Div([
        html.Label("Load .lin file from path:"),
        dcc.Input(id="int-file-path", type="text", placeholder="Enter path to .lin file",
                  style={"width": "70%"}),
        html.Button("Load .lin File", id="load-int-button", n_clicks=0, style={"marginLeft": "8px"}),
        html.Button("Auto-load current .lin", id="auto-load-lin-button", n_clicks=0, style={"marginLeft": "8px"})
    ], style={"marginBottom": "15px"}),
    dcc.ConfirmDialog(
        id="auto-load-lin-confirm",
        message=(
            "Auto-load current .lin will replace the current assignment table with the active .lin file. "
            "Unsaved new fitting/assignment edits in the table may be lost. Continue?"
        ),
    ),

    html.H4("Assignments (Click row to delete; double-click Uncertainty to edit)"),
    html.Button("Save .lin file", id="save-lin-button", n_clicks=0, style={"marginTop": "8px"}),
    html.Button("Overwrite active .lin file", id="overwrite-lin-button", n_clicks=0, style={"marginTop": "8px", "marginLeft": "8px"}),
    dcc.ConfirmDialog(
        id="overwrite-lin-confirm",
        message=(
            "Overwrite active .lin file will write the current assignment table to disk and move the existing "
            ".lin to a timestamped .assigner-history backup. Continue?"
        ),
    ),
    html.Div(id="save-lin-confirmation", style={"marginTop": "8px", "color": "green"}),
    html.Button("Sort by obs", id="sort-obs-button", n_clicks=0, style={"marginTop": "8px", "marginLeft": "8px"}),
    html.Button("Recalculate Weights", id="recalc-weights-button", n_clicks=0, style={"marginTop": "8px"}),
    html.Div([
        html.Label("Delete rows with Weight <= "),
        dcc.Input(id="weight-delete-limit", type="number", value=0.01, step=0.0001,
                  style={"width": "100px", "marginLeft": "6px", "marginRight": "8px"}),
        html.Button("Delete Weak Rows", id="delete-weak-rows-button", n_clicks=0),
    ], style={"marginTop": "8px", "marginBottom": "8px"}),

    dash_table.DataTable(
        id='assignment-table',
        columns=_initial_assignment_columns,
        data=_sanitize_for_table(_initial_active_rows),
        hidden_columns=[
            "AbsDelta",
            "Include_SNR", "Include_Interf", "Include_Merge",
            "Include_SNR_num", "Include_Interf_num", "Include_Merge_num",
        ],
        row_selectable="single",
        selected_rows=[],
        style_table={'width': '95%'},
        style_cell={
            'textAlign': 'center',
            'padding': '4px 6px',
            'fontSize': 12,
            'backgroundColor': '#1c252f',
            'color': 'white',
            'border': '1px solid #444'
        },
        style_header={
            'fontSize': 12,
            'fontWeight': 'bold',
            'backgroundColor': '#3a3a3a',
            'color': 'white',
            'border': '1px solid #444'
        },
        virtualization=True,
        fixed_rows={'headers': True},
        editable=True,
        style_data_conditional=[
            # Delta exceeds allowed uncertainty
            {'if': {'filter_query': '{AbsDelta} > {Uncertainty}', 'column_id': 'Delta'}, 'color': '#ffbbb0', 'backgroundColor': '#482e35', 'fontWeight': '700'},
            {'if': {'filter_query': '{AbsDelta} > {Uncertainty}', 'column_id': 'Uncertainty'}, 'color': '#ffbbb0', 'backgroundColor': '#482e35', 'fontWeight': '700'},

            # SNR
            {'if': {'filter_query': '{Include_SNR_num} = 1', 'column_id': 'Disp_SNR'},    'color': 'green', 'fontWeight': '600'},
            {'if': {'filter_query': '{Include_SNR_num} = 0', 'column_id': 'Disp_SNR'},    'color': 'red',   'fontWeight': '600'},

            # Interf
            {'if': {'filter_query': '{Include_Interf_num} = 1', 'column_id': 'Disp_Interf'}, 'color': 'green', 'fontWeight': '600'},
            {'if': {'filter_query': '{Include_Interf_num} = 0', 'column_id': 'Disp_Interf'}, 'color': 'red',   'fontWeight': '600'},

            # Merge
            {'if': {'filter_query': '{Include_Merge_num} = 1', 'column_id': 'Disp_Merge'},  'color': 'green', 'fontWeight': '600'},
            {'if': {'filter_query': '{Include_Merge_num} = 0', 'column_id': 'Disp_Merge'},  'color': 'red',   'fontWeight': '600'},
        ],
    ),


    # Stores (needed by callbacks)
    dcc.Store(id="active-cat-idx", data=0),
    dcc.Store(id="percat-assignments", data=_initial_assignments),           # {str(idx): [rows]}
    dcc.Store(id="selected-fit-mu"),
    dcc.Store(id="stored-fit-params"),
    dcc.Store(id="stored-region-selection"),
    dcc.Store(id="stored-zoom", data={"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None}),
    dcc.Store(id="zoom-history", data=[]),
    dcc.Store(id="sim-scale-store", data=1.0),
    dcc.Store(id="last-y-range", data=None),
    dcc.Store(id="percat-scales", data=_initial_scales),
    dcc.Store(id="measured-trace-index", data=None),
    dcc.Store(id="assign-request"),
    dcc.Store(id="approved-assign-request"),
    dcc.Store(id="merge-warning-pending"),
    dcc.Store(id="deassign-request"),
    dcc.Store(id="lw-scroll-signal"),
])  # close app.layout



# =========================
# Publish the clicked L-W frequency and bring the main fitting plot into view.
app.clientside_callback(
    """
    function(clickData, spanValue) {
        if (!clickData || !clickData.points || !clickData.points.length) {
            return window.dash_clientside.no_update;
        }
        const point = clickData.points[0];
        const custom = point.customdata;
        let frequency = Number(Array.isArray(custom) ? custom[0] : custom);
        const span = Math.max(0.001, Number(spanValue) || 20.0);

        // Plotly versions differ in whether line-click events include
        // customdata.  Fall back to metadata on the clicked trace.
        if (!Number.isFinite(frequency)) {
            const wrapper = document.getElementById('lw-graph');
            const graph = wrapper ? wrapper.querySelector('.js-plotly-plot') : null;
            const curve = Number(point.curveNumber);
            const trace = graph && graph.data && Number.isInteger(curve)
                ? graph.data[curve]
                : null;
            frequency = Number(trace && trace.meta
                ? trace.meta.predicted_frequency_mhz
                : NaN);
        }
        if (!Number.isFinite(frequency)) {
            return window.dash_clientside.no_update;
        }

        window.setTimeout(function() {
            const plot = document.getElementById('spectrum-plot');
            if (plot) {
                plot.scrollIntoView({behavior: 'smooth', block: 'center'});
            }
        }, 250);
        return {frequency_mhz: frequency, span_mhz: span, timestamp: Date.now()};
    }
    """,
    Output("lw-scroll-signal", "data"),
    Input("lw-graph", "clickData"),
    State("lw-span", "value"),
    State("original-intensity", "value"),
    prevent_initial_call=True,
)


# Callbacks: embedded Loomis-Wood plot
# =========================
@app.callback(
    Output("lw-filter", "value"),
    Input("lw-add-rule", "n_clicks"), Input("lw-add-range", "n_clicks"),
    Input("lw-add-series", "n_clicks"), Input("lw-clear-filter", "n_clicks"),
    State("lw-filter", "value"), State("lw-rule-qn", "value"),
    State("lw-rule-op", "value"), State("lw-rule-value", "value"),
    State("lw-range-qn", "value"), State("lw-range-op", "value"),
    State("lw-range-value", "value"), State("lw-series", "value"),
    State("lw-series-ka", "value"), State("lw-series-component", "value"),
    prevent_initial_call=True,
)
def lw_build_filter(_rule_clicks, _range_clicks, _series_clicks, _clear_clicks,
                    current, rule_qn, rule_op, rule_value, range_qn, range_op,
                    range_value, series_name, series_ka, series_component):
    trigger = ctx.triggered_id
    if trigger == "lw-clear-filter":
        return ""
    additions = []
    if trigger == "lw-add-rule" and rule_qn is not None and rule_value is not None:
        additions = [f"d{rule_qn}{rule_op}{float(rule_value):g}"]
    elif trigger == "lw-add-range" and range_qn is not None and range_value is not None:
        additions = [f"{range_qn}{range_op}{float(range_value):g}"]
    elif trigger == "lw-add-series" and series_name in LW_SERIES:
        d_j, d_ka, d_kc = LW_SERIES[series_name]
        additions = [f"dJ={d_j:g}", f"dKa={d_ka:g}", f"dKc={d_kc:g}"]
        if series_ka is not None:
            additions.append(f"Ka={float(series_ka):g}")
        if series_component == "lo":
            additions.append("Kc=J-Ka")
        elif series_component == "hi":
            additions.append("Kc=J-Ka+1")
    else:
        raise dash.exceptions.PreventUpdate
    clauses = [part.strip() for part in re.split(r"[,;]+", str(current or "")) if part.strip()]
    for clause in additions:
        if clause not in clauses:
            clauses.append(clause)
    return ", ".join(clauses)


@app.callback(
    Output("lw-graph", "figure"), Output("lw-status", "children"),
    Input("lw-plot-button", "n_clicks"), Input("lw-y-zoom", "value"),
    Input("lw-strip-height", "value"), Input("lw-options", "value"),
    Input("active-cat-idx", "data"), Input("ws-revision", "data"),
    State("lw-filter", "value"), State("lw-sort", "value"),
    State("lw-span", "value"), State("percat-assignments", "data"),
    prevent_initial_call=True,
)
def draw_loomis_wood(_plot_clicks, y_zoom, strip_height, options, active_idx, catalog_revision,
                     filter_text, sort_key, span, percat_assignments):
    try:
        active_idx = int(active_idx or 0)
        if not (0 <= active_idx < len(catalogs)):
            raise ValueError("active catalog is unavailable")
        cat = catalogs[active_idx]
        span = max(0.001, float(span or 20.0))
        y_zoom = max(0.05, float(y_zoom or 1.0))
        strip_height = min(180, max(36, int(strip_height or 72)))
        options = options or []
        rows, total, before_hf = _lw_filter_catalog(cat, filter_text, sort_key, "ignore_hf" in options)
    except Exception as exc:
        return _lw_empty_figure(f"L-W filter error: {exc}"), f"Error: {exc}"
    if rows is None or rows.empty:
        return _lw_empty_figure("No transitions match the L-W filter"), "0 transitions match."

    assigned = percat_assignments or {}
    assigned_uids = {int(row["SimUID"]) for row in (assigned.get(str(active_idx), []) or [])
                     if row.get("SimUID") is not None}
    show_labels = "labels" in options
    qn_order = cat.get("qn_order", [])
    n_rows = len(rows)
    fig = go.Figure()
    tickvals, ticktext = [], []
    for row_number, (_, row) in enumerate(rows.iterrows()):
        base = float(n_rows - row_number - 1)
        predicted = float(row["Freq"])
        left = int(np.searchsorted(meas_freqs, predicted - span / 2.0, side="left"))
        right = int(np.searchsorted(meas_freqs, predicted + span / 2.0, side="right"))
        x = np.asarray(meas_freqs[left:right], dtype=float) - predicted
        y_raw = np.asarray(meas_intensities[left:right], dtype=float)
        if x.size > 900:
            x, y_raw = decimate_xy_preserve_extrema(x, y_raw, max_pts=900)
        finite = np.isfinite(y_raw)
        ymax = float(np.nanmax(y_raw[finite])) if finite.any() else 0.0
        normalized = (y_raw / ymax) if ymax > 0 else np.zeros_like(y_raw)
        y = base + np.clip(normalized * y_zoom, 0.0, 1.35) * 0.82
        sim_uid = int(row.get("SimUID", -1))
        label = _lw_transition_label(row, qn_order)
        custom = np.full((len(x), 2), [predicted, row_number], dtype=float) if len(x) else None
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="lines",
            line=dict(color="#ffffff",
                      width=(1.9 if sim_uid in assigned_uids else 1.05)),
            name=label, showlegend=False, customdata=custom,
            meta={"predicted_frequency_mhz": predicted, "row_number": row_number},
            hovertemplate=(f"{label}<br>offset: %{{x:.4f}} MHz"
                           "<br>normalized measured intensity: %{text}<extra></extra>"),
            text=[f"{v:.4f}" for v in normalized],
        ))
        tickvals.append(base + 0.32)
        ticktext.append(label if show_labels else str(row_number + 1))
    fig.add_vline(x=0.0, line_color="#ffbd73", line_dash="dash", line_width=1.5,
                  annotation_text="predicted", annotation_position="top right")
    fig.update_xaxes(range=[-span / 2.0, span / 2.0], zeroline=False,
                     title="Offset from predicted frequency (MHz)",
                     tickfont=dict(size=13, color="#ffffff"),
                     gridcolor="#3d454d", linecolor="#aeb8c2")
    fig.update_yaxes(range=[-0.15, n_rows - 0.02 + 0.9], tickmode="array",
                     tickvals=tickvals, ticktext=ticktext, fixedrange=True,
                     tickfont=dict(size=14, family="Consolas, monospace", color="#ffffff"),
                     gridcolor="#343b42", linecolor="#aeb8c2")
    fig.update_layout(
        template="simple_white", height=max(330, n_rows * strip_height),
        title=f"Loomis-Wood — {cat['name']} — {n_rows} strip(s)",
        plot_bgcolor="#202428", paper_bgcolor="#202428", font_color="#ffffff",
        hovermode="closest", hoverdistance=35,
        clickmode="event+select", uirevision="lw-view",
        margin=dict(l=(420 if show_labels else 70), r=25, t=50, b=55),
    )
    notes = [f"{total} match(es)", f"showing {n_rows}", f"span {span:g} MHz"]
    if before_hf > total:
        notes.append(f"HF collapsed {before_hf}→{total}")
    if total > LW_MAX_ROWS:
        notes.append(f"first {LW_MAX_ROWS} only")
    return fig, " · ".join(notes)


# =========================
# Callbacks: Fit peaks
# =========================
@app.callback(
    Output("fit-output", "children"),
    Output("stored-fit-params", "data"),
    Input("spectrum-plot", "selectedData"),
    State("num-gaussians", "value"),
    State("mode-selector", "value"),
    prevent_initial_call=True
)
def fit_peak(selection, num_gauss, mode):
    if mode != "select":
        raise dash.exceptions.PreventUpdate
    if not selection or "range" not in selection:
        return dash.no_update, dash.no_update

    x0, x1 = selection["range"]["x"]
    mask = (meas_freqs >= x0) & (meas_freqs <= x1)
    if np.sum(mask) < 5:
        return "❌ Too few points to fit.", dash.no_update

    x, y = meas_freqs[mask], meas_intensities[mask]
    x, y = decimate_xy_preserve_extrema(x, y, max_pts=1200)


    window_width = x1 - x0
    margin = window_width
    side_left = (meas_freqs >= (x0 - margin)) & (meas_freqs < x0)
    side_right = (meas_freqs > x1) & (meas_freqs <= (x1 + margin))
    side_y = np.concatenate([meas_intensities[side_left], meas_intensities[side_right]])

    if len(side_y) < 5:
        y_base = 0.0
        y_base_std = 0.05
    else:
        y_base = float(np.mean(side_y))
        y_base_std = float(np.std(side_y))

    mu_guesses = np.linspace(x0 + 0.1 * window_width, x1 - 0.1 * window_width, num_gauss)
    y_max = max(y.max(), 0.01)

    initial_p0, bounds_lower, bounds_upper = [], [], []
    for mu in mu_guesses:
        amp_guess = y_max - y_base
        sigma_guess = max(window_width / (3 * num_gauss), 0.01)
        initial_p0 += [amp_guess, mu, sigma_guess]
        bounds_lower += [0, x0, 0.01]
        bounds_upper += [1.5 * amp_guess, x1, window_width]

    initial_p0 += [y_base]
    bounds_lower += [y_base - y_base_std]
    bounds_upper += [y_base + y_base_std]

    try:
        popt, _ = curve_fit(
            multi_gauss_with_offset,
            x, y,
            p0=initial_p0,
            bounds=(bounds_lower, bounds_upper)
        )
        fits = [{"amp": popt[i], "mu": popt[i + 1], "sigma": popt[i + 2]}
                for i in range(0, len(popt) - 1, 3)]
        baseline = popt[-1]
        msg = "✅ Fitted peaks at: " + ", ".join(f"{p['mu']:.2f} MHz" for p in fits)
        msg += f"<br>Estimated baseline offset: {baseline:.4f} ± {y_base_std:.4f}"
        return msg, {
            "multi": fits,
            "baseline": baseline,
            "baseline_std": y_base_std,              # <-- NEW
            "baseline_range": [x0 - margin, x1 + margin]
        }
    except Exception as e:
        return f"❌ Fit failed: {str(e)}", dash.no_update

# =========================
# Callbacks: Zoom & controls (with Y-zoom)
# =========================
@app.callback(
    Output("stored-zoom", "data"),
    Output("zoom-history", "data"),
    Input("spectrum-plot", "relayoutData"),
    Input("lw-scroll-signal", "data"),
    Input("undo-zoom-button", "n_clicks"),
    Input("x-zoom-in", "n_clicks"),
    Input("x-zoom-out", "n_clicks"),
    Input("y-zoom-in", "n_clicks"),
    Input("y-zoom-out", "n_clicks"),
    Input("apply-xrange", "n_clicks"),
    State("default-xmin", "value"),
    State("default-xmax", "value"),
    State("stored-zoom", "data"),
    State("zoom-history", "data"),
    State("last-y-range", "data"),
    State("lw-span", "value"),
    prevent_initial_call=True
)
def handle_all_zoom_events(relayout, lw_jump, undo_clicks, zoom_in_clicks, zoom_out_clicks, y_in, y_out,
                           apply_xrange_clicks, xmin_val, xmax_val,
                           current_zoom, history, last_y, lw_span, original_intensity=None):
    if 'raw' in (original_intensity or []):
        relayout = normalized_relayout(relayout, MEAS_INTENSITY_REFERENCE)
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_prop = ctx.triggered[0]["prop_id"]
    trigger_id = trigger_prop.split(".")[0]
    history = history or []

    def default_y_range():
        if last_y and isinstance(last_y, list) and len(last_y) == 2:
            return last_y
        return [-0.1, 1.2]

    # Clicking a Loomis-Wood strip centers the main spectrum on its prediction.
    if trigger_id == "lw-scroll-signal" and lw_jump:
        try:
            predicted = float(lw_jump["frequency_mhz"])
            span = max(0.001, float(lw_jump.get("span_mhz", lw_span or 20.0)))
            x0, x1 = _clamp_x_range(predicted - span / 2.0, predicted + span / 2.0)
        except (KeyError, TypeError, ValueError):
            raise dash.exceptions.PreventUpdate
        if isinstance(current_zoom, dict):
            history.append(current_zoom)
        return {"x": [x0, x1], "y": None}, history

    # Apply-xrange button
    if trigger_id == "apply-xrange":
        try:
            xmin = float(xmin_val); xmax = float(xmax_val)
            if xmax <= xmin:
                raise ValueError
        except Exception:
            xmin, xmax = MEAS_XMIN, MEAS_XMAX
        xmin, xmax = _clamp_x_range(xmin, xmax)
        if isinstance(current_zoom, dict):
            history.append(current_zoom)
        return {"x": [xmin, xmax], "y": None}, history

    # Plotly relayouts (zoom, pan, double-click home, toolbar autorange)
    if trigger_id == "spectrum-plot" and relayout:
        # Any autorange flag → treat as reset unless explicit x range is provided
        if relayout.get("autosize") or relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
            if isinstance(current_zoom, dict):
                history.append(current_zoom)
            xmin, xmax = _clamp_x_range(DEFAULT_XMIN, DEFAULT_XMAX)
            return {"x": [xmin, xmax], "y": None}, history

        # Accept either indexed or list-style ranges
        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")
        if (x0 is None or x1 is None) and "xaxis.range" in relayout:
            xr = relayout.get("xaxis.range")
            if isinstance(xr, (list, tuple)) and len(xr) == 2:
                x0, x1 = xr[0], xr[1]

        y0 = relayout.get("yaxis.range[0]")
        y1 = relayout.get("yaxis.range[1]")
        if (y0 is None or y1 is None) and "yaxis.range" in relayout:
            yr = relayout.get("yaxis.range")
            if isinstance(yr, (list, tuple)) and len(yr) == 2:
                y0, y1 = yr[0], yr[1]

        if (x0 is None or x1 is None) and y0 is not None and y1 is not None:
            x0, x1 = (current_zoom or {}).get('x') or [MEAS_XMIN, MEAS_XMAX]
        if x0 is not None and x1 is not None:
            x0, x1 = _clamp_x_range(x0, x1)
            new_zoom = {"x": [float(x0), float(x1)],
                        "y": [float(y0), float(y1)] if (y0 is not None and y1 is not None) else None}
            if isinstance(current_zoom, dict):
                history.append(current_zoom)
            return new_zoom, history

        # Nothing actionable in this relayout payload
        raise dash.exceptions.PreventUpdate

    # Undo
    if trigger_id == "undo-zoom-button":
        if not history:
            xmin, xmax = _clamp_x_range(DEFAULT_XMIN, DEFAULT_XMAX)
            return {"x": [xmin, xmax], "y": None}, []
        last_zoom = history[-1]
        return last_zoom, history[:-1]

    # X-zooms
    if trigger_id in ("x-zoom-in", "x-zoom-out"):
        if not current_zoom or "x" not in current_zoom:
            raise dash.exceptions.PreventUpdate
        x0, x1 = current_zoom["x"]
        x_center = (x0 + x1) / 2
        x_width = (x1 - x0)
        zoom_factor = 0.3 if trigger_id == "x-zoom-in" else 2.5
        new_width = x_width * zoom_factor
        new_x0 = x_center - new_width / 2
        new_x1 = x_center + new_width / 2
        new_x0, new_x1 = _clamp_x_range(new_x0, new_x1)
        new_zoom = {"x": [new_x0, new_x1], "y": current_zoom.get("y")}
        history.append(current_zoom)
        return new_zoom, history

    # Y-zooms
    if trigger_id in ("y-zoom-in", "y-zoom-out"):
        yr = (current_zoom or {}).get("y") or default_y_range()
        y0, y1 = float(yr[0]), float(yr[1])
        y_center = (y0 + y1) / 2.0
        y_height = (y1 - y0)
        zoom_factor = 0.3 if trigger_id == "y-zoom-in" else 2.5
        new_height = y_height * zoom_factor
        new_y0 = y_center - new_height / 2.0
        new_y1 = y_center + new_height / 2.0
        new_zoom = {"x": (current_zoom or {}).get("x", [DEFAULT_XMIN, DEFAULT_XMAX]), "y": [new_y0, new_y1]}
        history.append(current_zoom or {"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None})
        return new_zoom, history

    raise dash.exceptions.PreventUpdate

# Apply intensity scale from input (per catalog + persist to disk)
@app.callback(
    Output("percat-scales", "data"),
    Output("sim-scale-store", "data"),  # kept for backward compatibility (not used by update_plot)
    Input("apply-scale", "n_clicks"),
    State("sim-scale", "value"),
    State("percat-scales", "data"),
    State("active-cat-idx", "data"),
    prevent_initial_call=True
)
def update_scale(n, scale_val, scales, active_idx):
    scales = dict(scales or {})
    try:
        s = float(scale_val)
        if s <= 0:
            s = 1.0
    except Exception:
        s = 1.0
    key = str(int(active_idx or 0))
    scales[key] = s
    _save_scale_cache(scales)
    return scales, s


# =========================
# Store selection (region for assign)
# =========================
@app.callback(
    Output("stored-region-selection", "data"),
    Input("spectrum-plot", "selectedData"),
    prevent_initial_call=True
)
def store_selection(selection):
    if selection and "range" in selection and "x" in selection["range"]:
        return selection
    return dash.no_update


@app.callback(
    Output("auto-load-lin-confirm", "displayed"),
    Input("auto-load-lin-button", "n_clicks"),
    prevent_initial_call=True
)
def confirm_auto_load_lin(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return True


@app.callback(
    Output("overwrite-lin-confirm", "displayed"),
    Input("overwrite-lin-button", "n_clicks"),
    prevent_initial_call=True
)
def confirm_overwrite_lin(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return True


def _prospective_merge_for_request(assign_req, percat):
    """Estimate the exact merge term before an assignment mutates the store."""
    if not assign_req:
        return None

    try:
        req_idx = int(assign_req.get("active_idx", 0))
        sel_mu = round(float(assign_req.get("mu")), 4)
        sel_range = assign_req.get("sel_range")
        thr = float(assign_req.get("thr", 0.01))
    except (TypeError, ValueError):
        return None

    if not (0 <= req_idx < len(catalogs)) or not sel_range or len(sel_range) != 2:
        return None

    x0, x1 = map(float, sel_range)
    if x1 < x0:
        x0, x1 = x1, x0

    current = list((percat or {}).get(str(req_idx), []))
    existing_group = []
    existing_pairs = set()
    for row in current:
        try:
            obs = round(float(row.get("obs")), 4)
            uid = row.get("SimUID")
            if obs == sel_mu:
                existing_group.append(float(row.get("sim")))
                if uid is not None:
                    existing_pairs.add((obs, int(uid)))
        except (TypeError, ValueError):
            continue

    sim_df = catalogs[req_idx]["df"]
    mask = (
        (sim_df["Norm_Intensity"] >= thr)
        & (sim_df["Freq"] >= x0)
        & (sim_df["Freq"] <= x1)
    )
    candidate_rows = sim_df.loc[mask, ["Freq", "SimUID"]]

    new_sims = []
    for row in candidate_rows.itertuples(index=False):
        pair = (sel_mu, int(row.SimUID))
        if pair not in existing_pairs:
            new_sims.append(float(row.Freq))

    combined = existing_group + new_sims
    if combined:
        sim_min = float(np.min(combined))
        sim_max = float(np.max(combined))
        merge = 0.5 * (sim_max - sim_min) if len(combined) > 1 else 0.0
    else:
        sim_min = sim_max = None
        merge = 0.0

    return {
        "merge_mhz": float(merge),
        "new_count": len(new_sims),
        "total_count": len(combined),
        "sim_min": sim_min,
        "sim_max": sim_max,
        "observed_mhz": sel_mu,
        "active_idx": req_idx,
    }


@app.callback(
    Output("merge-warning-reject", "n_clicks"),
    Input("keyboard", "n_keydowns"),
    State("keyboard", "keydown"),
    State("merge-warning-pending", "data"),
    State("merge-warning-reject", "n_clicks"),
    prevent_initial_call=True,
)
def reject_merge_warning_on_enter(n_keydowns, key_event, pending, reject_clicks):
    """Translate Enter into an explicit reject click without racing assign-request."""
    key = (key_event or {}).get("key", "")
    if not pending or str(key).lower() != "enter":
        raise dash.exceptions.PreventUpdate
    return int(reject_clicks or 0) + 1


@app.callback(
    Output("approved-assign-request", "data"),
    Output("merge-warning-pending", "data"),
    Output("merge-warning-modal", "style"),
    Output("merge-warning-message", "children"),
    Input("assign-request", "data"),
    Input("merge-warning-proceed", "n_clicks"),
    Input("merge-warning-reject", "n_clicks"),
    State("merge-guard-threshold", "value"),
    State("percat-assignments", "data"),
    State("merge-warning-pending", "data"),
    prevent_initial_call=True,
)
def guard_assignment_merge(assign_req, proceed_clicks, reject_clicks,
                           threshold, percat, pending):
    trigger = ctx.triggered_id

    if trigger == "merge-warning-reject":
        if not pending:
            raise dash.exceptions.PreventUpdate
        return dash.no_update, None, MERGE_WARNING_HIDDEN_STYLE, ""

    if trigger == "merge-warning-proceed":
        if not pending or not pending.get("request"):
            raise dash.exceptions.PreventUpdate
        approved = dict(pending["request"])
        approved["approval_ts"] = time.time_ns()
        approved["merge_guard_overridden"] = True
        return approved, None, MERGE_WARNING_HIDDEN_STYLE, ""

    if trigger != "assign-request" or not assign_req:
        raise dash.exceptions.PreventUpdate

    try:
        limit = max(0.0, float(threshold))
    except (TypeError, ValueError):
        limit = DEFAULT_MERGE_GUARD_MHZ

    estimate = _prospective_merge_for_request(assign_req, percat)
    if estimate is None:
        # Preserve the existing hard guardrails in mutate_assignments.
        approved = dict(assign_req)
        approved["approval_ts"] = time.time_ns()
        return approved, None, MERGE_WARNING_HIDDEN_STYLE, ""

    if estimate["merge_mhz"] <= limit:
        approved = dict(assign_req)
        approved["approval_ts"] = time.time_ns()
        approved["prospective_merge_mhz"] = estimate["merge_mhz"]
        return approved, None, MERGE_WARNING_HIDDEN_STYLE, ""

    pending_data = {"request": assign_req, "estimate": estimate, "limit_mhz": limit}
    if estimate["sim_min"] is None:
        span_text = "No simulated-frequency span was available."
    else:
        span_text = (
            f"Combined simulated-frequency range: {estimate['sim_min']:.6f}–"
            f"{estimate['sim_max']:.6f} MHz ({estimate['total_count']} line(s))."
        )
    message = (
        f"Prospective merge = {estimate['merge_mhz']:.6f} MHz, exceeding the "
        f"configured limit of {limit:.6f} MHz.\n"
        f"Observed fitted center: {estimate['observed_mhz']:.4f} MHz; "
        f"new candidate lines: {estimate['new_count']}.\n"
        f"{span_text}\n\n"
        "This assignment has not been written. Press Enter or click Reject to close. "
        "Only clicking Still proceed will include it in the fit."
    )
    return dash.no_update, pending_data, MERGE_WARNING_VISIBLE_STYLE, message

# =========================
# Mutate assignments (assign, delete, recalc, load, edits)
# =========================
@app.callback(
    Output("percat-assignments", "data"),
    Output("assignment-table", "selected_rows"),
    Input("approved-assign-request", "data"),
    Input("deassign-request", "data"),
    Input("assignment-table", "selected_rows"),
    Input("recalc-weights-button", "n_clicks"),
    Input("sort-obs-button", "n_clicks"),
    Input("delete-weak-rows-button", "n_clicks"),
    Input("assignment-table", "data_timestamp"),
    Input("load-int-button", "n_clicks"),
    Input("auto-load-lin-confirm", "submit_n_clicks"),
    State("percat-assignments", "data"),
    State("mode-selector", "value"),
    State("stored-region-selection", "data"),
    State("intensity-threshold", "value"),
    State("active-cat-idx", "data"),
    State("assignment-table", "data"),
    State("int-file-path", "value"),
    State("stored-fit-params", "data"),
    State("weight-delete-limit", "value"),
    prevent_initial_call=True
)
def mutate_assignments(assign_req, deassign_req, selected_rows, recalc_clicks, sort_obs_clicks, delete_weak_clicks, data_ts, load_clicks,
                       auto_load_submit_clicks,
                       percat, mode, selection, intensity_threshold, active_idx,
                       table_data, int_path, fit_params_state, weight_delete_limit):


    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    percat = percat or {}
    key = str(int(active_idx or 0))
    current = list(percat.get(key, []))

    # A) Atomic assign: use the SNAPSHOT embedded in assign_req
    if trig.startswith("approved-assign-request") and assign_req:
        req_idx = int(assign_req.get("active_idx", active_idx or 0))
        sel_range = assign_req.get("sel_range")
        thr = float(assign_req.get("thr", 0.01))
        sel_mu = assign_req.get("mu", None)

        # hard guardrails
        if sel_mu is None or not sel_range or len(sel_range) != 2:
            raise dash.exceptions.PreventUpdate

        x0, x1 = map(float, sel_range)
        if x1 < x0:
            x0, x1 = x1, x0
        key = str(req_idx)
        current = list((percat or {}).get(key, []))

        active = catalogs[req_idx]
        sim_df = active["df"]; qn_field_order = active["qn_order"]

        mask = (sim_df["Norm_Intensity"] >= thr) & (sim_df["Freq"] >= x0) & (sim_df["Freq"] <= x1)
        in_range = sim_df.loc[mask]

        fit_ctx = _build_fit_context({"range": {"x": sel_range}}, fit_params_state, delta_F=DELTA_F_DEFAULT)
        sel_mu = float(sel_mu)
        local_sigma = _fit_sigma_for_mu(sel_mu, fit_params_state)
        local_window = _window_from_mu_sigma(sel_mu, local_sigma, mult=10.0)

        for _, row in in_range.iterrows():
            freq_full = float(row["Freq"])
            new_entry = {
                "obs": round(sel_mu, 4),
                "sim": freq_full,
                "WeightedSim": round(freq_full, 4),
                "Delta": round(sel_mu - freq_full, 4),
                "Eu": round(float(row["Eu"]), 4),
                "logI": round(np.log10(float(row["Intensity"])), 4),
                "SimUID": int(row["SimUID"]),
                "FitCtx": fit_ctx,
            }
            # copy QNs in the same order as the active catalog
            for k in qn_field_order:
                if k in row:
                    new_entry[k] = int(row[k])

            # avoid duplicates: (same observed + same simulated line)
            if not any((r.get("obs") == new_entry["obs"] and r.get("SimUID") == new_entry["SimUID"]) for r in current):
                current.append(new_entry)

        current = recompute_peak_weights(current)

        current, _ = _restore_fitctx_if_missing(current, req_idx)
        current = _recalc_uncertainties(
            current,
            selection_range=[0.0, 0.0],
            delta_F=DELTA_F_DEFAULT,
            full=False,
            local_window=local_window,
        )
        percat[key] = current
        _save_assignment_autosave(percat)
        return percat, []

    # A2) Atomic deassign: remove everything at this observed mu
    if trig.startswith("deassign-request") and deassign_req:
        req_idx = int(deassign_req.get("active_idx", active_idx or 0))
        mu = deassign_req.get("mu", None)
        sel_range = deassign_req.get("sel_range", None)

        if mu is None:
            raise dash.exceptions.PreventUpdate

        mu = round(float(mu), 4)
        key = str(req_idx)
        current = list((percat or {}).get(key, []))

        # Remove all assignments with obs == mu
        #new_rows = [r for r in current if round(float(r.get("obs", -1e99)), 4) != mu]

        # If we have a selection range, deassign sims inside that x-range (REGARDLESS of obs).
        if sel_range and isinstance(sel_range, (list, tuple)) and len(sel_range) == 2:
            x0, x1 = map(float, sel_range)
            if x1 < x0:
                x0, x1 = x1, x0

            def _keep(r):
                try:
                    uid = r.get("SimUID")
                    if uid is not None:
                        # if SimUID exists, prefer exact identity match by freq range is still OK,
                        # but UID is safer when selection aligns to sticks
                        sim = float(r.get("sim"))
                        return not (x0 <= sim <= x1)
                    sim = float(r.get("sim"))
                    return not (x0 <= sim <= x1)
                except Exception:
                    return True


            new_rows = [r for r in current if _keep(r)]

        else:
            # No selection -> fallback: remove everything at this mu (old behavior)
            new_rows = [r for r in current if round(float(r.get("obs", -1e99)), 4) != mu]


        # Recompute
        new_rows = recompute_peak_weights(new_rows)
        new_rows, _ = _restore_fitctx_if_missing(new_rows, req_idx)
        new_rows = _recalc_uncertainties(new_rows, selection_range=None, delta_F=None, full=False)
        new_rows = _decorate_display_flags(new_rows)

        percat[key] = new_rows
        _save_assignment_autosave(percat)
        return percat, []


    # B) Delete selected rows
    if trig.startswith("assignment-table.selected_rows"):
        if selected_rows:
            drop = set(selected_rows)
            current = [row for i, row in enumerate(current) if i not in drop]
            current = recompute_peak_weights(current)


            # Restore any missing FitCtx, then recompute using per-row saved settings
            current, _ = _restore_fitctx_if_missing(current, active_idx)
            current = _recalc_uncertainties(current, selection_range=None, delta_F=None, full=False)
            current = _decorate_display_flags(current)

            percat[key] = current
            _save_assignment_autosave(percat)
            return percat, []

        raise dash.exceptions.PreventUpdate

    # C) Recompute weights (button)
    if trig.startswith("recalc-weights-button"):
        current = recompute_peak_weights(current)
        current, _ = _restore_fitctx_if_missing(current, active_idx)
        current = _recalc_uncertainties(current, selection_range=None, delta_F=None, full=True)
        current = _decorate_display_flags(current)

        percat[key] = current
        _save_assignment_autosave(percat)
        return percat, []

    # C2) Sort rows by observed frequency (ascending)
    if trig.startswith("sort-obs-button"):
        def _obs_sort_key(row):
            try:
                return (0, float(row.get("obs")))
            except Exception:
                return (1, float("inf"))

        current = sorted(current, key=_obs_sort_key)
        percat[key] = current
        _save_assignment_autosave(percat)
        return percat, []

    # C3) Delete rows with weak weight
    if trig.startswith("delete-weak-rows-button"):
        try:
            limit = float(weight_delete_limit if weight_delete_limit is not None else 0.01)
        except Exception:
            limit = 0.01

        filtered = []
        for r in current:
            try:
                w = float(r.get("Weight", 0.0) or 0.0)
            except Exception:
                w = 0.0
            if w > limit:
                filtered.append(r)

        if len(filtered) == len(current):
            raise dash.exceptions.PreventUpdate

        filtered = recompute_peak_weights(filtered)
        filtered, _ = _restore_fitctx_if_missing(filtered, active_idx)
        filtered = _recalc_uncertainties(filtered, selection_range=None, delta_F=None, full=False)
        filtered = _decorate_display_flags(filtered)

        percat[key] = filtered
        _save_assignment_autosave(percat)
        return percat, []




    # D) Inline edits via data_timestamp
    if trig.startswith("assignment-table.data_timestamp"):
        if not isinstance(table_data, list):
            raise dash.exceptions.PreventUpdate
        current_in_store = _sanitize_for_table(percat.get(key, []))
        if table_data == current_in_store:
            raise dash.exceptions.PreventUpdate

        table_data = recompute_peak_weights(table_data, recompute_weights=False)
        table_data, _ = _restore_fitctx_if_missing(table_data, active_idx)
        table_data = _recalc_uncertainties(table_data, selection_range=None, delta_F=None, full=False)
        table_data = _decorate_display_flags(table_data)

        percat[key] = table_data
        _save_assignment_autosave(percat)
        return percat, dash.no_update


    # E) Load .lin file
    if trig.startswith("load-int-button") or trig.startswith("auto-load-lin-confirm"):
        if trig.startswith("auto-load-lin-confirm"):
            cat_path = catalogs[int(active_idx or 0)]["path"]
            lin_path = os.path.splitext(cat_path)[0] + ".lin"
        else:
            lin_path = int_path

        if not lin_path or not os.path.isfile(lin_path) or not str(lin_path).lower().endswith(".lin"):
            raise dash.exceptions.PreventUpdate

        sim_df = catalogs[int(active_idx or 0)]["df"]
        qn_field_order = catalogs[int(active_idx or 0)]["qn_order"]
        loaded = []
        try:
            with open(lin_path, "r") as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    try:
                        qns, freq, unc, wt = parse_lin_line_flexible(raw)
                        qn_fields = qn_field_order[:len(qns)]
                        qn_values = qns[:len(qn_fields)]
                        match_df = sim_df.copy()
                        for field, value in zip(qn_fields, qn_values):
                            if field in match_df.columns:
                                match_df = match_df[match_df[field] == value]
                            else:
                                match_df = match_df.iloc[0:0]
                                break
                        if match_df.empty:
                            continue
                        sim_row = match_df.iloc[0]
                        assignment = {
                            "obs": round(float(freq), 4),
                            "sim": float(sim_row["Freq"]),
                            "WeightedSim": round(float(sim_row["Freq"]), 4),
                            "Delta": round(float(freq) - float(sim_row["Freq"]), 4),
                            "Eu": round(float(sim_row["Eu"]), 4),
                            "logI": round(np.log10(float(sim_row["Intensity"])), 4),
                            "Uncertainty": round(float(unc), 4),
                            "Weight": round(float(wt), 4),
                            # --- NEW: stable ID carried through loads as well ---
                            "SimUID": int(sim_row["SimUID"]),
                        }

                        for field, value in zip(qn_fields, qn_values):
                            assignment[field] = value
                        loaded.append(assignment)
                    except Exception:
                        continue
        except Exception:
            raise dash.exceptions.PreventUpdate

        loaded = recompute_peak_weights(loaded, recompute_weights=False)

        # after loading, compute components + set flags + compute total from flags
        unc = compute_uncertainties(loaded, [0.0, 0.0], delta_F=DELTA_F_DEFAULT)
        for r in loaded:
            u = unc.get(float(r["obs"]))
            if u:
                r["Unc_SNR"]     = u["sigma_instr"]
                r["Unc_Interf"]  = u["sigma_interf"]
                r["Unc_Merge"]   = u["sigma_merge"]
            if "Include_SNR"    not in r: r["Include_SNR"]    = True
            if "Include_Interf" not in r: r["Include_Interf"] = True
            if "Include_Merge"  not in r: r["Include_Merge"]  = True
            r["Uncertainty"] = _recalc_total_from_flags(r)

        # After you've built 'loaded' list, do only one recompute path:
        loaded, _ = _restore_fitctx_if_missing(loaded, active_idx)
        loaded = _recalc_uncertainties(loaded, selection_range=[0.0, 0.0], delta_F=DELTA_F_DEFAULT)
        loaded = _decorate_display_flags(loaded)
        percat[key] = loaded
        _save_assignment_autosave(percat)
        return percat, []


    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("assignment-table", "data"),
    Output("percat-assignments", "data", allow_duplicate=True),
    Output("assignment-table", "active_cell"),
    Input("assignment-table", "active_cell"),
    State("assignment-table", "data"),
    State("active-cat-idx", "data"),
    State("percat-assignments", "data"),
    prevent_initial_call=True,
)
def toggle_flags_on_click(active_cell, ui_rows, active_idx, percat):
    if not active_cell:
        raise dash.exceptions.PreventUpdate

    r_idx = active_cell.get("row")
    c_id  = active_cell.get("column_id")
    if c_id not in ("Disp_SNR", "Disp_Interf", "Disp_Merge"):
        raise dash.exceptions.PreventUpdate

    # 1) Work on the authoritative store (with FitCtx), not the UI copy
    active_idx = int(active_idx or 0)
    key = str(active_idx)
    store_rows = list((percat or {}).get(key, []))


    if not isinstance(ui_rows, list) or not ui_rows:
        raise dash.exceptions.PreventUpdate


    # Identify the row using stable fields present in the UI rows
    if r_idx is None or r_idx < 0 or r_idx >= len(ui_rows):
        raise dash.exceptions.PreventUpdate

    ui_row = ui_rows[r_idx]
    target_obs = ui_row.get("obs")
    target_uid = ui_row.get("SimUID")


    # Find the corresponding row in the store
    target_i = None
    for i, sr in enumerate(store_rows):
        if sr.get("obs") == target_obs and sr.get("SimUID") == target_uid:
            target_i = i
            break
    if target_i is None:
        raise dash.exceptions.PreventUpdate  # can't map back safely

    # 2) Toggle flag on the store row
    field = {"Disp_SNR": "Include_SNR",
             "Disp_Interf": "Include_Interf",
             "Disp_Merge": "Include_Merge"}[c_id]
    store_rows[target_i][field] = not bool(store_rows[target_i].get(field, True))

    # 3) Recompute ONLY the displayed total using existing components
    store_rows[target_i]["Uncertainty"] = _recalc_total_from_flags(store_rows[target_i])

    # 4) Refresh numeric mirrors (for conditional cell styling) but DO NOT touch Unc_* components
    _decorate_display_flags(store_rows)  # in-place is fine

    # 5) Write back the authoritative store and emit sanitized UI rows
    percat[key] = store_rows
    _save_assignment_autosave(percat)
    ui_out = _sanitize_for_table(store_rows)
    return ui_out, percat, None




# =========================
# Main plot
# =========================
@app.callback(
    Output("spectrum-plot", "figure"),
    Output("last-y-range", "data"),
    Output("measured-trace-index", "data"),
    Input("stored-fit-params", "data"),
    Input("percat-assignments", "data"),
    Input("stored-region-selection", "data"),
    Input("stored-zoom", "data"),
    Input("mode-selector", "value"),
    Input("intensity-threshold", "value"),
    Input("selected-fit-mu", "data"),
    Input("flip-sim-checkbox", "value"),
    Input("percat-scales", "data"),
    Input("active-cat-idx", "data"),
    Input("lw-scroll-signal", "data"),
    Input("ws-revision", "data"),
    Input("original-intensity", "value"),
)
def update_plot(fit_params, percat, selection, zoom, mode, intensity_threshold, selected_mu,
                flip_checkbox, percat_scales, active_idx, lw_jump, catalog_revision=0, original_intensity=None):
    raw_display = 'raw' in (original_intensity or [])
    display_factor = MEAS_INTENSITY_REFERENCE if raw_display else 1.0
    unit = config.get('intensity_unit')
    intensity_label = (f'Intensity ({unit})' if unit else 'Intensity (original units)') if raw_display else 'Intensity'

    # --- y range helper (axis coords, never paper) ---
    def _shape_y_range(zoom, y_meas, flip, sim_scale):
        if zoom and zoom.get("y") is not None:
            y0, y1 = zoom["y"]
            return float(y0), float(y1)
        if isinstance(y_meas, np.ndarray) and y_meas.size:
            ymin = float(np.nanmin(y_meas))
            ymax = float(np.nanmax(y_meas))
        else:
            ymin, ymax = 0.0, 1.0
        s = float(sim_scale or 1.0)
        if ("flip" in (flip_checkbox or [])):
            ymin = min(ymin, -s); ymax = max(ymax, 0.0)
        else:
            ymin = min(ymin, 0.0);  ymax = max(ymax, s)
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = 0.0, 1.0
        pad = 0.02 * (ymax - ymin)
        return ymin - pad, ymax + pad

    try:
        # ---- inputs ----
        flip_vals = flip_checkbox or []
        thr = float(intensity_threshold if (intensity_threshold is not None) else 0.01)
        active_idx = int(active_idx or 0)
        percat_scales = percat_scales or {}
        sim_scale = float(percat_scales.get(str(active_idx), 1.0))
        flip = ("flip" in flip_vals)
        percat = percat or {}

        # On an L-W click, use the click payload itself as the authoritative
        # window for this render.  Changing uirevision below forces Plotly to
        # accept it instead of preserving the previous browser-side zoom.
        jump_revision = "base"
        triggered_ids = {
            item.get("prop_id", "").split(".")[0]
            for item in (callback_context.triggered or [])
        }
        if "lw-scroll-signal" in triggered_ids and isinstance(lw_jump, dict):
            try:
                jump_frequency = float(lw_jump["frequency_mhz"])
                jump_span = max(0.001, float(lw_jump.get("span_mhz", 20.0)))
                jump_x0, jump_x1 = _clamp_x_range(
                    jump_frequency - jump_span / 2.0,
                    jump_frequency + jump_span / 2.0,
                )
                zoom = {"x": [jump_x0, jump_x1], "y": None}
                jump_revision = str(lw_jump.get("timestamp", time.time_ns()))
            except (KeyError, TypeError, ValueError):
                pass

        if not isinstance(meas_freqs, np.ndarray) or not isinstance(meas_intensities, np.ndarray):
            raise ValueError("Measured arrays not initialized.")
        if meas_freqs.size == 0 or meas_intensities.size == 0:
            raise ValueError("Measured arrays are empty.")

        # measured subset by zoom + decimate
        if zoom and "x" in zoom and zoom["x"] is not None:
            x0z, x1z = _clamp_x_range(zoom["x"][0], zoom["x"][1])
            mask_meas = (meas_freqs >= x0z) & (meas_freqs <= x1z)
            x_meas = meas_freqs[mask_meas]
            y_meas = meas_intensities[mask_meas]
        else:
            x_meas = meas_freqs
            y_meas = meas_intensities
        x_meas, y_meas = decimate_xy_preserve_extrema(x_meas, y_meas, max_pts=20000)

        # how wide is the current x window? (None if not set)
        span = None
        if zoom and "x" in zoom and zoom["x"] is not None:
            span = float(zoom["x"][1] - zoom["x"][0])
        show_stick_hover = (span is not None and span < 200.0)  # show hover only when zoomed in

        # helper: stick trace
        def get_stick_trace(df, color, name, dash=None, opacity=1.0, scale=1.0):
            if df is None or df.empty:
                return go.Scattergl(x=[], y=[], name=name, opacity=opacity)

            freq = df["Freq"].to_numpy(dtype=float)
            norm = df["Norm_Intensity"].to_numpy(dtype=float)
            sign = -1.0 if flip else 1.0
            amp  = norm * float(scale if scale is not None else 1.0)

            # Build [f, f, nan] pattern without Python loops
            n = freq.size
            x = np.empty(n * 3, dtype=float)
            y = np.empty(n * 3, dtype=float)
            x[0::3] = freq
            x[1::3] = freq
            x[2::3] = np.nan
            y[0::3] = 0.0
            y[1::3] = sign * amp
            y[2::3] = np.nan

            line_kwargs = dict(color=color)
            if dash:
                line_kwargs["dash"] = dash

            # Only send hover strings when zoomed in
            if show_stick_hover:
                hv = df["Hover"].to_numpy(object)
                text = np.empty(n * 3, dtype=object)
                text[0::3] = hv
                text[1::3] = hv
                text[2::3] = None
                hover_kwargs = dict(hoverinfo="text", text=text, hovertemplate="%{text}<extra></extra>")
            else:
                hover_kwargs = dict(hoverinfo="skip")

            return go.Scattergl(
                x=x, y=y, mode="lines",
                line=line_kwargs, name=name,
                opacity=opacity,
                **hover_kwargs
            )





        # ---- build traces: inactive first (dim), then active on top ----
        inactive_traces, active_traces = [], []

        for idx, cat in enumerate(catalogs):
            sim_df = cat.get("df")
            if sim_df is None or sim_df.empty or "Norm_Intensity" not in sim_df.columns:
                continue

            cat_scale = float((percat_scales or {}).get(str(idx), 1.0))

            # x-window
            if zoom and "x" in zoom and zoom["x"] is not None:
                x0z, x1z = _clamp_x_range(zoom["x"][0], zoom["x"][1])
            else:
                x0z, x1z = MEAS_XMIN, MEAS_XMAX

            # unassigned: respect threshold
            mask_unassigned = (sim_df["Norm_Intensity"] >= thr) & (sim_df["Freq"] >= x0z) & (sim_df["Freq"] <= x1z)
            unassigned_lines = sim_df.loc[mask_unassigned]

            # assigned: IGNORE threshold so they are always visible
            assigned_rows = percat.get(str(idx), []) or []
            assigned_uids = {int(r["SimUID"]) for r in assigned_rows if "SimUID" in r}

            mask_x = (sim_df["Freq"] >= x0z) & (sim_df["Freq"] <= x1z)
            cands = sim_df.loc[mask_x]

            if assigned_uids and not cands.empty:
                assigned_lines = cands[cands["SimUID"].isin(assigned_uids)]
            else:
                assigned_lines = cands.iloc[0:0]

            # remove assigned from unassigned to avoid double-drawing
            if assigned_uids and not unassigned_lines.empty:
                unassigned_lines = unassigned_lines[~unassigned_lines["SimUID"].isin(assigned_uids)]


            # limit sticks for speed on very wide windows
            if span is None or span > 500:
                if not unassigned_lines.empty:
                    unassigned_lines = unassigned_lines.nlargest(min(2500, len(unassigned_lines)), "Norm_Intensity")
                #if not assigned_lines.empty:
                #    assigned_lines = assigned_lines.nlargest(min(2500, len(assigned_lines)), "Norm_Intensity")


            if idx == active_idx:
                active_traces.append(get_stick_trace(unassigned_lines, "#ef8b77",  f"{cat['name']} (unassigned)", scale=cat_scale))
                active_traces.append(get_stick_trace(assigned_lines,   "#73b4ff", f"{cat['name']} (assigned)", dash="dash", scale=cat_scale))
            else:
                inactive_traces.append(get_stick_trace(unassigned_lines, INACTIVE_COLOR,
                                                    f"{cat['name']} (inactive unassigned)",
                                                    dash=None, opacity=INACTIVE_OPACITY, scale=cat_scale))
                inactive_traces.append(get_stick_trace(assigned_lines, INACTIVE_COLOR,
                                                    f"{cat['name']} (inactive assigned)",
                                                    dash="dash", opacity=INACTIVE_OPACITY, scale=cat_scale))


        traces = inactive_traces + active_traces

        measured_idx = len(traces)

        # measured on top (use the downsampled arrays!)
        traces.append(go.Scattergl(
            x=x_meas, y=y_meas, mode="lines",
            name="Measured",
            hovertemplate="Freq: %{x:.4f} MHz<br>Intensity: %{y:.4f}<extra></extra>",
            line=dict(color="#d7e2ec", width=1.6)
        ))



        fig = go.Figure(traces)

        # axes / zoom
        if zoom and "x" in zoom and zoom["x"] is not None:
            fig.update_xaxes(range=_clamp_x_range(zoom["x"][0], zoom["x"][1]))
            if zoom.get("y") is not None:
                fig.update_yaxes(range=zoom["y"])
        else:
            fig.update_xaxes(range=_clamp_x_range(DEFAULT_XMIN, DEFAULT_XMAX))



        _apply_adaptive_xticks(fig, span)


        fig.update_layout(
            dragmode="select" if mode in ["select", "assign_all"] else "zoom",
            template="simple_white",
            height=600,
            xaxis_title="Frequency (MHz)",
            yaxis_title=intensity_label + (" (Sim Flipped)" if flip else ""),
            uirevision=f"zoom-lock-{jump_revision}-raw-{raw_display}",
            plot_bgcolor="#17212b",
            paper_bgcolor="#1c232c",
            font_color='white',
            margin=dict(t=20, b=40, l=60, r=20)
        )
        # Put legend inside the spectrum, translucent background
        fig.update_layout(
            legend=dict(
                x=0.01, y=0.99, xanchor="left", yanchor="top",
                bgcolor="rgba(0,0,0,0.35)",    # semi-transparent background
                bordercolor="rgba(255,255,255,0.25)",
                borderwidth=1,
                font=dict(size=11),
                itemclick="toggleothers",
                itemdoubleclick="toggle"
            )
        )

        # Make sure we have the current x-range handy (used below)
        xr = fig.layout.xaxis.range if fig.layout.xaxis.range else [DEFAULT_XMIN, DEFAULT_XMAX]

        # current selection guides
        if selection and "range" in selection and "x" in selection["range"]:
            try:
                sx0, sx1 = map(float, selection["range"]["x"])
                if sx1 < sx0:
                    sx0, sx1 = sx1, sx0
                y0_shape, y1_shape = _shape_y_range(zoom, y_meas, flip, sim_scale)
                for sx, label in ((sx0, "Selection start"), (sx1, "Selection end")):
                    fig.add_trace(go.Scatter(
                        x=[sx, sx, None],
                        y=[y0_shape, y1_shape, None],
                        mode="lines",
                        name=label,
                        line=dict(color="#D3D3D3", dash="dash", width=2),
                        opacity=0.9,
                        hovertemplate=f"{label}: {sx:.4f} MHz<extra></extra>",
                        showlegend=False
                    ))
            except Exception:
                pass



        # fitted peaks (individual + sum + baseline)
        if fit_params and "multi" in fit_params and isinstance(fit_params["multi"], list):
            baseline = float(fit_params.get("baseline", 0.0))
            baseline_std = float(fit_params.get("baseline_std", 0.0))
            brange = fit_params.get("baseline_range", None)
            fit_mus = []

            # individual Gaussians (offset by baseline)
            for p in fit_params["multi"]:
                mu = float(p["mu"]); sig = float(p["sigma"]); amp = float(p["amp"])
                fit_mus.append(round(mu, 4))
                x_fit = np.linspace(mu - 4 * sig, mu + 4 * sig, 120)
                y_fit = gaussian(x_fit, amp, mu, sig) + baseline
                fig.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode="lines",
                    name=f"μ={mu:.2f}",
                    line=dict(color="green", dash="dot"),
                    hoverinfo="skip"
                ))

            # envelope for the sum curve
            x_env_min = min(float(p["mu"]) - 4.0 * float(p["sigma"]) for p in fit_params["multi"])
            x_env_max = max(float(p["mu"]) + 4.0 * float(p["sigma"]) for p in fit_params["multi"])
            if zoom and "x" in zoom and zoom["x"] is not None:
                x_vis_min, x_vis_max = zoom["x"]
                x_min = max(x_env_min, x_vis_min)
                x_max = min(x_env_max, x_vis_max)
            else:
                x_min, x_max = x_env_min, x_env_max

            if x_max > x_min:
                amps = tuple(float(p["amp"]) for p in fit_params["multi"])
                mus = tuple(float(p["mu"]) for p in fit_params["multi"])
                sigmas = tuple(float(p["sigma"]) for p in fit_params["multi"])
                baseline_val = baseline
                x_grid, y_sum = _fit_sum_cached(amps, mus, sigmas, baseline_val, float(x_min), float(x_max), 800)
                fig.add_trace(go.Scatter(
                    x=x_grid, y=y_sum, mode="lines",
                    name="Fit sum",
                    line=dict(color="yellow", width=1, dash="dot"),
                    opacity=0.5, hoverinfo="skip"
                ))

            # --- NEW: baseline line across the fitted/sideband region
            if brange and len(brange) == 2:
                x0b, x1b = float(brange[0]), float(brange[1])
                fig.add_trace(go.Scatter(
                    x=[x0b, x1b], y=[baseline, baseline],
                    mode="lines", name="Baseline",
                    line=dict(color="gray", dash="dash"),
                    hoverinfo="skip"
                ))
                # optional ±σ band (visible if baseline_std > 0)
                if baseline_std > 0:
                    fig.add_hrect(
                        y0=baseline - baseline_std, y1=baseline + baseline_std,
                        x0=x0b, x1=x1b,
                        line_width=0, fillcolor="gray", opacity=0.15, layer="below"
                    )

            # selected fitted peak guide
            highlight_mu = None
            try:
                mu_candidates = sorted(fit_mus)
                if mu_candidates:
                    if selected_mu is not None:
                        sel = round(float(selected_mu), 4)
                        if sel in mu_candidates:
                            highlight_mu = sel
                    if highlight_mu is None:
                        highlight_mu = mu_candidates[0]
            except Exception:
                highlight_mu = None

            if highlight_mu is not None:
                y0_shape, y1_shape = _shape_y_range(zoom, y_meas, flip, sim_scale)
                fig.add_trace(go.Scatter(
                    x=[highlight_mu, highlight_mu, None],
                    y=[y0_shape, y1_shape, None],
                    mode="lines",
                    name="Selected fit",
                    line=dict(color="#32CD32", dash="dash", width=3),
                    opacity=0.95,
                    hovertemplate=f"Selected fit: {highlight_mu:.4f} MHz<extra></extra>",
                    showlegend=False
                ))


        # observed (active catalog) vertical markers as one scatter
        active_rows = percat.get(str(active_idx), []) or []
        if active_rows:
            if not zoom or (zoom["x"][1] - zoom["x"][0] < 300):
                obs_vals = sorted({float(r["obs"]) for r in active_rows if "obs" in r})
                if zoom and "x" in zoom and zoom["x"] is not None:
                    xv0, xv1 = zoom["x"]
                    obs_vals = [v for v in obs_vals if xv0 <= v <= xv1]
                if len(obs_vals) > 200:
                    cx = sum(fig.layout.xaxis.range)/2.0 if fig.layout.xaxis.range else 0.0
                    obs_vals.sort(key=lambda v: abs(v - cx))
                    obs_vals = obs_vals[:200]
                x_obs, y_obs = [], []
                y0_shape, y1_shape = _shape_y_range(zoom, y_meas, flip, sim_scale)
                for v in obs_vals:
                    x_obs += [v, v, None]
                    y_obs += [y0_shape, y1_shape*1.02, None]
                fig.add_trace(go.Scatter(
                    x=x_obs, y=y_obs, mode="lines",
                    line=dict(color="#73b4ff", dash="dot", width=3.5),
                    opacity=0.7, name="Assigned Obs",
                    hoverinfo="skip", showlegend=False
                ))

        # uncertainty bands (active catalog)
        if active_rows:
            obs_to_unc = {}
            for r in active_rows:
                ov = r.get("obs"); uv = r.get("Uncertainty")
                if ov is None or uv is None:
                    continue
                try:
                    ov = float(ov); uv = float(uv)
                except (TypeError, ValueError):
                    continue
                if ov not in obs_to_unc:
                    obs_to_unc[ov] = uv

            MAX_VRECTS = 40
            MAX_SPAN_FOR_VRECTS = 100.0


            draw_now = True
            x0z, x1z = (xr[0], xr[1])
            if zoom and "x" in zoom and zoom["x"] is not None:
                x0z, x1z = zoom["x"]
                draw_now = (x1z - x0z) < MAX_SPAN_FOR_VRECTS

            if draw_now and obs_to_unc:
                visible_obs = [o for o in obs_to_unc if x0z <= o <= x1z]
                if visible_obs:
                    x_center = (x0z + x1z) / 2.0
                    visible_obs.sort(key=lambda v: abs(v - x_center))
                    visible_obs = visible_obs[:MAX_VRECTS]
                    y0_shape, y1_shape = _shape_y_range(zoom, y_meas, flip, sim_scale)
                    for ov in visible_obs:
                        uv = float(obs_to_unc[ov])
                        if not np.isfinite(uv) or uv <= 0:
                            continue
                        fig.add_vrect(
                            x0=ov - uv, x1=ov + uv,
                            y0=0, y1=1,
                            xref="x", yref="y",
                            fillcolor="lightblue", opacity=0.15,
                            layer="below", line_width=0,
                        )

        fig.update_xaxes(gridcolor='#30404f', linecolor='#526477', zerolinecolor='#526477')
        fig.update_yaxes(gridcolor='#30404f', linecolor='#526477', zerolinecolor='#526477')
        fig.update_layout(font=dict(family='Segoe UI, Arial, sans-serif', color='#d7e2ec'))
        yr = fig.layout.yaxis.range
        last_y = list(yr) if yr else None
        scale_figure(fig, display_factor, intensity_label + (" (Sim Flipped)" if flip else ""))
        return fig, last_y, measured_idx

    except Exception as e:
        print(f"[ERROR] update_plot crashed: {e}")
        try:
            fig = go.Figure([
                go.Scatter(x=meas_freqs, y=meas_intensities, mode="lines", name="Measured", line=dict(color="#d7e2ec"))
            ])
            fig.update_xaxes(range=[DEFAULT_XMIN, DEFAULT_XMAX])
            fig.update_layout(height=600, template="plotly_dark", plot_bgcolor="#17212b",
                              paper_bgcolor="#1c232c", font_color="#d7e2ec")
            scale_figure(fig, display_factor, intensity_label)
            return fig, None, None
        except Exception as ee:
            print(f"[FATAL] Could not even draw measured trace: {ee}")
            return go.Figure(), None, None

# =========================
# Save .lin (active catalog)
# =========================
def generate_lin_file(assignments, qn_field_order):
    if any(row.get('CatalogStatus') in ('missing', 'ambiguous') for row in assignments):
        raise ValueError('Resolve unmatched or ambiguous CAT assignments before exporting LIN.')
    lines = []
    for row in assignments:
        freq = float(row["obs"])
        unc = float(row.get("Uncertainty", 0.0100))
        wt = float(row.get("Weight", 1.00))
        qn_values = [int(row[k]) for k in qn_field_order if k in row]
        qn_str = "".join(f"{q:3d}" for q in qn_values)
        line = (
            f"{qn_str:<24s}"
            f"{'':25s}"
            f"{freq:12.4f}{unc:10.4f}{wt:6.2f}"
        )
        lines.append(line)
    return "\n".join(lines)


def _active_lin_filepath(active_idx):
    cat_path = catalogs[int(active_idx or 0)]["path"]
    base, _ = os.path.splitext(cat_path)
    return base + ".lin"

@app.callback(
    Output("save-lin-confirmation", "children"),
    Input("save-lin-button", "n_clicks"),
    Input("overwrite-lin-confirm", "submit_n_clicks"),
    State("percat-assignments", "data"),
    State("active-cat-idx", "data"),
    prevent_initial_call=True
)
def save_lin_file(save_n_clicks, overwrite_submit_clicks, percat, active_idx):
    active_idx = int(active_idx or 0)
    key = str(active_idx)
    percat = percat or {}
    assignments = percat.get(key, [])
    if not assignments:
        return "❌ No assignments to save for this catalog."

    # NEW: persist any existing FitCtx for these rows into the cache
    _persist_existing_fitctx(assignments, active_idx)

    qn_field_order = catalogs[active_idx]["qn_order"]
    try:
        content = generate_lin_file(assignments, qn_field_order)
    except ValueError as exc:
        return str(exc)

    if ctx.triggered_id == "overwrite-lin-confirm":
        filepath = _active_lin_filepath(active_idx)
        from spectrum_workspace_v7 import directory_lock, save_document, file_snapshot
        guard = directory_lock(Path(filepath).parent)
        if not guard.acquire(blocking=False):
            return 'Working directory is busy. Wait for SPFIT/SPCAT to finish.'
        try:
            save_document(filepath, content + '\n', file_snapshot(filepath))
            return f'Overwrote: {filepath} | Timestamped backup in .assigner-history'
        except Exception as exc:
            return f'Could not save LIN: {exc}'
        finally:
            guard.release()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    assignments_dir = os.path.join(script_dir, "assignments", identity(WORKSPACE_CONFIG))
    os.makedirs(assignments_dir, exist_ok=True)

    cat_name = os.path.splitext(os.path.basename(catalogs[active_idx]["path"]))[0]
    filename = f"{cat_name}_assignments_{timestamp}.lin"
    filepath = os.path.join(assignments_dir, filename)

    with open(filepath, "w") as f:
        f.write(content + "\n")

    print(f"[LIN] Saved {filepath}")  # optional but handy
    return f"✅ Saved: {filename}"


# =========================
# Active catalog label / table columns updater
# =========================
@app.callback(
    Output("active-cat-label", "children"),
    Output("assignment-table", "columns"),
    Output("assignment-table", "data", allow_duplicate=True),
    Input("active-cat-idx", "data"),
    Input("percat-assignments", "data"),
    prevent_initial_call=True,
)
def update_active_catalog_view(active_idx, percat):
    active_idx = int(active_idx or 0)
    if not catalogs:
        return "—", [], []

    qn_fields = catalogs[active_idx]["qn_order"]
    cols = build_assignment_columns(qn_fields)


    key = str(active_idx)
    data = (percat or {}).get(key, [])
    # If any row lacks Weight, recompute per observed frequency cluster
    if any(("Weight" not in r or r["Weight"] is None) for r in data):
        data = recompute_peak_weights(data, recompute_weights=True)
    elif any(("WeightedSim" not in r or r["WeightedSim"] is None or "Delta" not in r) for r in data):
        data = recompute_peak_weights(data, recompute_weights=False)
    data = _decorate_display_flags(data)
    return catalogs[active_idx]["name"], cols, _sanitize_for_table(data)



# Keep the sim-scale input showing the active catalog's saved scale
@app.callback(
    Output("sim-scale", "value"),
    Input("active-cat-idx", "data"),
    State("percat-scales", "data")
)
def sync_scale_input(active_idx, scales):
    scales = scales or {}
    key = str(int(active_idx or 0))
    return float(scales.get(key, 1.0))



# --- helper to build an atomic assign snapshot (selection + threshold + active cat) ---
def _build_assign_request(mu_to_use, selection_state, thr_state, active_idx):
    sel_range = None
    if selection_state and "range" in selection_state and "x" in selection_state["range"]:
        x0, x1 = selection_state["range"]["x"]
        sel_range = [float(x0), float(x1)]
    thr = float(thr_state) if thr_state is not None else 0.01
    return {
        "ts": time.time(),
        "mu": mu_to_use,
        "active_idx": int(active_idx or 0),
        "sel_range": sel_range,
        "thr": thr,
    }
def _build_deassign_request(mu_to_use, selection_state, active_idx):
    sel_range = None
    if selection_state and "range" in selection_state and "x" in selection_state["range"]:
        x0, x1 = selection_state["range"]["x"]
        sel_range = [float(x0), float(x1)]
    return {
        "ts": time.time(),
        "mu": mu_to_use,
        "active_idx": int(active_idx or 0),
        "sel_range": sel_range,
    }



# =========================
# μ buttons + keyboard (includes 's' to switch catalog)
# =========================
@app.callback(
    Output("fit-mu-button-container", "children"),
    Output("selected-fit-mu", "data"),
    Output("mode-selector", "value"),
    Output("undo-zoom-button", "n_clicks"),
    Output("num-gaussians", "value"),
    Output("active-cat-idx", "data"),
    Output("assign-request", "data"),
    Output("deassign-request", "data"),
    Input("stored-fit-params", "data"),
    Input({"type": "fit-mu-button", "index": ALL}, "n_clicks"),
    Input("assign-button", "n_clicks"),
    Input("keyboard", "n_keydowns"),
    Input("keyboard", "keydown"),
    State({"type": "fit-mu-button", "index": ALL}, "id"),
    State("selected-fit-mu", "data"),
    State("stored-fit-params", "data"),
    State("active-cat-idx", "data"),
    # NEW: snapshot sources
    State("stored-region-selection", "data"),
    State("intensity-threshold", "value"),
    State("ws-page", "data"),
    prevent_initial_call=True
)
def handle_fit_mu_and_keyboard(fit_params, n_clicks_list, assign_n_clicks, n_keydowns, key_event, ids,
                               selected_mu, fit_params_state, active_idx,
                               selection_state, thr_state, workspace_page="assigner"):
    trigger = ctx.triggered_id
    if trigger == "keyboard" and workspace_page != "assigner":
        raise dash.exceptions.PreventUpdate
    if trigger == 'keyboard' and globals().get('workspace_runner') and workspace_runner.running:
        raise dash.exceptions.PreventUpdate
    mode_value = dash.no_update
    undo_clicks = dash.no_update
    num_gauss = dash.no_update
    next_mu = selected_mu
    next_active_idx = dash.no_update

    assign_request = dash.no_update
    deassign_request = dash.no_update


    if isinstance(trigger, dict) and trigger.get("type") == "fit-mu-button":
        next_mu = trigger["index"]

    elif trigger == "stored-fit-params" and fit_params and "multi" in fit_params:
        fits = sorted(fit_params["multi"], key=lambda p: p["mu"])
        next_mu = round(fits[0]["mu"], 4)

    elif trigger == "keyboard" and key_event:
        key = key_event.get("key", "").lower()
        if key == "q":
            mode_value = "zoom"
        elif key == "w":
            mode_value = "select"
        elif key == "e":
            mode_value = "assign_all"
        elif key == "r":
            undo_clicks = int(time.time())
        elif key == "a":
            # Ensure we have a μ; if not, pick the first from current fit params.
            mu_to_use = next_mu
            if mu_to_use is None and fit_params_state and "multi" in fit_params_state and fit_params_state["multi"]:
                mu_to_use = round(sorted(fit_params_state["multi"], key=lambda p: p["mu"])[0]["mu"], 4)

            # Emit a single, self-contained request. Also force mode to assign_all
            mode_value = "assign_all"
            assign_request = _build_assign_request(mu_to_use, selection_state, thr_state, active_idx)
        elif key == "p":
            # choose mu
            mu_to_use = next_mu
            if mu_to_use is None and fit_params_state and "multi" in fit_params_state and fit_params_state["multi"]:
                mu_to_use = round(sorted(fit_params_state["multi"], key=lambda p: p["mu"])[0]["mu"], 4)

            # Only act if we actually have a mu
            if mu_to_use is not None:
                deassign_request = _build_deassign_request(mu_to_use, selection_state, active_idx)



        elif key in [str(n) for n in range(10)]:
            num_gauss = 10 if key == "0" else int(key)
        elif key == "d" and fit_params_state and "multi" in fit_params_state:
            fits_sorted = sorted(fit_params_state["multi"], key=lambda p: p["mu"])
            mu_list = [round(p["mu"], 4) for p in fits_sorted]
            if mu_list:
                if next_mu not in mu_list:
                    next_mu = mu_list[0]
                else:
                    current_idx = mu_list.index(next_mu)
                    next_mu = mu_list[(current_idx + 1) % len(mu_list)]
        elif key == "s":
            ncat = max(len(catalogs), 1)
            cur = int(active_idx or 0)
            next_active_idx = (cur + 1) % ncat

    elif trigger == "assign-button":             # <-- now reachable
        mu_to_use = next_mu
        if mu_to_use is None and fit_params_state and "multi" in fit_params_state and fit_params_state["multi"]:
            mu_to_use = round(sorted(fit_params_state["multi"], key=lambda p: p["mu"])[0]["mu"], 4)
        mode_value = "assign_all"
        assign_request = _build_assign_request(mu_to_use, selection_state, thr_state, active_idx)


    buttons = []
    if fit_params and "multi" in fit_params:
        fits = sorted(fit_params["multi"], key=lambda p: p["mu"])
        for p in fits:
            mu = round(p["mu"], 4)
            label = f"{mu:.2f} MHz"
            is_selected = (mu == next_mu)
            style = {
                "padding": "6px 10px",
                "border": "2px solid",
                "borderRadius": "6px",
                "cursor": "pointer",
                "fontWeight": "bold" if is_selected else "normal",
                "backgroundColor": "#285b72" if is_selected else "#273340",
                "color": "#f2faff" if is_selected else "#c4d2df",
            }
            buttons.append(html.Button(label, id={"type": "fit-mu-button", "index": mu}, n_clicks=0, style=style))

    return buttons, next_mu, mode_value, undo_clicks, num_gauss, next_active_idx, assign_request, deassign_request




# @app.callback(
#     Output("cursor-readout", "children"),
#     Input("spectrum-plot", "hoverData"),
#     State("measured-trace-index", "data")
# )
# def update_cursor_readout(hoverData, measured_idx):
#     default_msg = "Freq: — MHz | Height: —"
#     try:
#         if not hoverData or "points" not in hoverData or not hoverData["points"]:
#             return default_msg

#         pts = hoverData["points"]
#         target = None

#         # Prefer the measured trace if we know its curve index
#         if isinstance(measured_idx, int):
#             for p in pts:
#                 if p.get("curveNumber") == measured_idx:
#                     target = p
#                     break

#         # Fallback: just take the first under-cursor point
#         if target is None:
#             target = pts[0]

#         x = target.get("x", None)
#         y = target.get("y", None)
#         if x is None or y is None:
#             return default_msg

#         return f"Freq: {float(x):.4f} MHz | Intensity: {float(y):.4f}"
#     except Exception:
#         return default_msg


if __name__ == "__main__":
    app.run(debug=False, port=8053)


from spectrum_workspace_v7 import install
install(globals())

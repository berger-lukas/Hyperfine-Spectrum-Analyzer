import os
import json
import time
import datetime
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


INACTIVE_COLOR = "#BBBBBB"
INACTIVE_OPACITY = 0.35  # dim them so they sit visually “behind”

# =========================
# Init & Config
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

with open("config.json", "r", encoding="utf-8") as f:
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
csv_file_path = config["csv_file"]

# =========================
# Parse Simulated Spectrum (.cat)
# =========================
def parse_cat_file(filepath):
    sim_data = []
    max_qns = 0
    with open(filepath) as f:
        for line in f:
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
                continue

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
catalogs = []
for path in cat_files:
    cdf, qn_order = parse_cat_file(path)
    if not cdf.empty:
        cdf["Norm_Intensity"] = cdf["Intensity"] / cdf["Intensity"].max()
        cdf["RoundedFreq"] = cdf["Freq"].round(4)
        cdf["StickX"] = cdf["Freq"].apply(lambda f: [f, f, None])  # repeated x values to draw a stick
        cdf["Hover"] = cdf.apply(generate_hover, axis=1)


        # --- NEW: stable row identifier for matching assigned sticks exactly ---
        cdf.reset_index(drop=True, inplace=True)
        cdf["SimUID"] = cdf.index.astype(int)

    catalogs.append({
        "path": path,
        "df": cdf,
        "qn_order": qn_order,
        "name": os.path.basename(path)
    })




# --- Debug prints to confirm data loaded (appears in your terminal) ---
try:
    print(f"[INFO] Measured CSV: {csv_file_path}")
    print(f"[INFO] Measured points: {len(meas_freqs)}")
except NameError:
    pass

print("[INFO] Catalogs loaded:")
for i, c in enumerate(catalogs):
    n = 0 if (c.get("df") is None) else len(c["df"])
    print(f"  [{i}] {c['name']}  rows={n}")

# =========================
# Measured spectrum
# =========================
meas_df = pd.read_csv(csv_file_path, sep=";")
meas_freqs = meas_df.iloc[:, 0].values
meas_intensities = meas_df.iloc[:, 1].values
meas_intensities = meas_intensities / np.max(meas_intensities)

# =========================
# Helpers
# =========================
def compute_uncertainties(assignments, selection_range, delta_F=DELTA_F_DEFAULT):
    """
    Compute uncertainty components for each observed frequency.

    Now prefers the fitted context (FitCtx) if available:
      • baseline_rms := FitCtx['baseline_std']
      • signal height at obs := sum of fitted Gaussians at obs (above baseline)
      • SNR = signal_height / baseline_rms
    Falls back to a local sideband estimator if no FitCtx is present.

    Returns: {obs: {sigma_total, sigma_merge, sigma_interf, sigma_instr, SNR, baseline_rms, peak_height}}
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
            # snag a FitCtx for this obs if available
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
            smax = float(np.max(sims)); smin = float(np.min(sims))
            sigma_merge[i] = 0.5 * (smax - smin)

    # ---- (2) interference term: crowding from nearby obs ----
    if n > 1:
        D = np.abs(obs[:, None] - obs[None, :])
        np.fill_diagonal(D, 0.0)
        C = 0.5 * D * (1.0 - np.tanh(D / float(delta_F)))
        sigma_interf = np.sqrt(np.sum(C * C, axis=1))
    else:
        sigma_interf = np.zeros(n, dtype=float)

    # ---- (3) instrumental/SNR term ----
    # Params for fallbacks (when no FitCtx):
    # derive a local half-width (MHz) from the selection, but clamp tightly
    try:
        x0, x1 = selection_range if selection_range else (0.0, 0.0)
        sel_w = float(x1 - x0)
    except Exception:
        sel_w = 0.0
    bw = (sel_w / 6.0) if sel_w > 0 else 0.30
    bw = max(0.05, min(bw, 0.80))  # 0.05–0.80 MHz

    SNR_FLOOR = 4.0  # keep σ_instr near your base error when data is decent

    def nearest_idx(x):
        j = np.searchsorted(meas_freqs, x)
        if j <= 0: return 0
        if j >= len(meas_freqs): return len(meas_freqs) - 1
        return j if (abs(meas_freqs[j] - x) < abs(meas_freqs[j-1] - x)) else (j-1)

    def robust_rms(vals):
        if vals.size >= 12:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            return max(1.4826 * mad, 0.005)
        return max(float(np.std(vals)) if vals.size > 0 else 0.01, 0.005)

    sigma_instr = np.zeros(n, dtype=float)
    snr_vals = np.zeros(n, dtype=float)
    base_rms_vals = np.zeros(n, dtype=float)
    peak_heights = np.zeros(n, dtype=float)

    for i, oi in enumerate(obs):
        fitctx = fitctx_by_obs.get(oi)

        if fitctx and isinstance(fitctx, dict):
            # --- Preferred path: use fitted context ---
            baseline_std = fitctx.get("baseline_std", None)
            peaks = fitctx.get("peaks", None)

            # baseline RMS from fit (already estimated from sidebands during fit)
            if isinstance(baseline_std, (int, float)) and np.isfinite(baseline_std) and baseline_std > 0:
                baseline_rms = float(baseline_std)
            else:
                # fallback local sidebands
                left_mask  = (meas_freqs >= oi - 2*bw) & (meas_freqs <  oi - bw)
                right_mask = (meas_freqs >  oi + bw)  & (meas_freqs <= oi + 2*bw)
                baseline_vals = np.concatenate([meas_intensities[left_mask], meas_intensities[right_mask]])
                baseline_rms = robust_rms(baseline_vals)

            # signal height above baseline at the observed frequency
            if peaks and isinstance(peaks, list):
                try:
                    # sum of all fitted Gaussians evaluated at oi (height above baseline)
                    sig = 0.0
                    for p in peaks:
                        A = float(p["amp"]); M = float(p["mu"]); S = float(p["sigma"])
                        sig += A * np.exp(-0.5 * ((oi - M) / S) ** 2)
                    peak_height = float(sig)
                except Exception:
                    # very safe fallback if peaks malformed
                    j = nearest_idx(oi)
                    peak_height = float(meas_intensities[j])
            else:
                # no peaks in context → fallback to measured
                j = nearest_idx(oi)
                peak_height = float(meas_intensities[j])

        else:
            # --- Fallback path: no FitCtx for this obs ---
            j = nearest_idx(oi)
            peak_height = float(meas_intensities[j])
            left_mask  = (meas_freqs >= oi - 2*bw) & (meas_freqs <  oi - bw)
            right_mask = (meas_freqs >  oi + bw)  & (meas_freqs <= oi + 2*bw)
            baseline_vals = np.concatenate([meas_intensities[left_mask], meas_intensities[right_mask]])
            baseline_rms = robust_rms(baseline_vals)

        # SNR and σ_instr
        SNR = (peak_height / baseline_rms) if baseline_rms > 0 else float("inf")
        if SNR_FLOOR is not None:
            SNR = max(SNR, SNR_FLOOR)

        #sigma_instr[i] = np.sqrt((0.0575 / SNR) ** 2 + (0.01) ** 2)
        sigma_instr[i] = np.sqrt((0.0575 / SNR) ** 2 + (BASE_SIGMA_INSTR) ** 2)

        snr_vals[i] = float(SNR if np.isfinite(SNR) else 1e6)
        base_rms_vals[i] = float(baseline_rms)
        peak_heights[i] = float(peak_height)

    sigma_total = np.sqrt(sigma_merge**2 + sigma_interf**2 + sigma_instr**2)

    out = {}
    for o, st, smg, sif, sin, snr, brms, ph in zip(
        obs, sigma_total, sigma_merge, sigma_interf, sigma_instr, snr_vals, base_rms_vals, peak_heights
    ):
        out[float(o)] = {
            "sigma_total": round(float(st), 4),
            "sigma_merge": round(float(smg), 4),
            "sigma_interf": round(float(sif), 4),
            "sigma_instr": round(float(sin), 4),
            "SNR": round(float(snr), 2),
            "baseline_rms": round(float(brms), 4),
            "peak_height": round(float(ph), 4),
        }
    return out


def _sanitize_for_table(rows):
    """Return shallow copies of rows without non-primitive fields (e.g., FitCtx)."""
    if not isinstance(rows, list):
        return rows
    clean = []
    for r in rows:
        rc = dict(r)
        rc.pop("FitCtx", None)  # strip nested dict
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
        {"name": "Delta (obs-sim)", "id": "Delta", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
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

    # Define the boolean flags as normal columns (no `hidden` key here)
    hidden_flags = [
        {"name": "Include_SNR",    "id": "Include_SNR",    "type": "any"},
        {"name": "Include_Interf", "id": "Include_Interf", "type": "any"},
        {"name": "Include_Merge",  "id": "Include_Merge",  "type": "any"},
    ]

    return static_columns + qn_columns + toggle_display_cols + extras + hidden_flags


def recompute_peak_weights(assignments):
    if not assignments:
        return assignments
    by_obs = defaultdict(list)
    for i, r in enumerate(assignments):
        by_obs[r["obs"]].append(i)
    for obs, idxs in by_obs.items():
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
        for i, w in zip(idxs, weights):
            assignments[i]["Weight"] = round(w, 4)
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

        # NEW: numeric mirrors used by style filters
        r["Include_SNR_num"]    = 1 if r["Include_SNR"]    else 0
        r["Include_Interf_num"] = 1 if r["Include_Interf"] else 0
        r["Include_Merge_num"]  = 1 if r["Include_Merge"]  else 0
    return rows



# --- per-catalog sim-scale persistence ---
def _get_scale_cache_path():
    return os.path.join(tempfile.gettempdir(), "spectrum_assigner_scales.json")

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


def _get_cache_path():
    # a small temp JSON cache
    return os.path.join(tempfile.gettempdir(), "spectrum_assigner_fitcache.json")

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
    return f"cat{int(active_idx)}|obs{float(obs):.4f}|uid{int(sim_uid) if sim_uid is not None else -1}"

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
    for r in rows:
        if "FitCtx" not in r or not r["FitCtx"]:
            k = _cache_key(active_idx, r.get("obs"), r.get("SimUID"))
            if k in cache:
                r["FitCtx"] = cache[k]
                changed = True
    return rows, changed

def _recalc_uncertainties(rows, selection_range=None, delta_F=None):
    """
    Recompute uncertainty terms. Interference is computed once across ALL observed
    lines (so nearby obs can contribute), while SNR & Merge are still computed
    per-obs using that obs' saved selection range if available.
    """
    if not rows:
        return rows

    # --- NEW: compute interference ONCE across all obs ---
    # pick a global delta_F: first valid from any FitCtx, else provided, else 0.1
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


    # We only need sigma_interf from this call; use neutral selection width
    global_unc = compute_uncertainties(rows, [0.0, 0.0], delta_F=df_global)
    obs_to_interf = {float(o): u["sigma_interf"] for o, u in global_unc.items()}

    # --- Now compute SNR & Merge per-obs as before ---
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[r.get("obs")].append(i)

    for obs_val, idxs in grouped.items():
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
app = dash.Dash(__name__)
server = app.server

DEFAULT_XMIN = 5000.0
DEFAULT_XMAX = 18500.0

_initial_scales = _load_scale_cache()
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
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Label("Default X-axis Range (MHz):"),
        dcc.Input(id="default-xmin", type="number", value=DEFAULT_XMIN, step=100, style={"width": "120px"}),
        dcc.Input(id="default-xmax", type="number", value=DEFAULT_XMAX, step=100, style={"width": "120px", "marginLeft": "8px", "marginRight": "10px"}),
        html.Button("Apply X-range", id="apply-xrange", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    # Keyboard listener (added 's')
    Keyboard(id="keyboard", captureKeys=["q", "w", "e", "r", "a", "d", "s"] + [str(i) for i in range(0, 11)]),

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

    html.Button("Assign Region", id="assign-button", n_clicks=0, style={"marginBottom": "10px"}),

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

    html.Div([
        html.Label("Simulated Intensity Threshold (0–1):"),
        dcc.Input(id='intensity-threshold', type='number', min=0, max=1, step=0.0001, value=0.001, debounce=True)
    ], style={'marginBottom': '15px'}),

    html.Div([
        html.Label("Load .lin file from path:"),
        dcc.Input(id="int-file-path", type="text", placeholder="Enter path to .lin file",
                  style={"width": "70%"}),
        html.Button("Load .lin File", id="load-int-button", n_clicks=0)
    ], style={"marginBottom": "15px"}),

    html.H4("Assignments (Click row to delete; double-click Uncertainty to edit)"),
    html.Button("Save .lin file", id="save-lin-button", n_clicks=0, style={"marginTop": "8px"}),
    html.Div(id="save-lin-confirmation", style={"marginTop": "8px", "color": "green"}),
    html.Button("Recalculate Weights", id="recalc-weights-button", n_clicks=0, style={"marginTop": "8px"}),

    dash_table.DataTable(
        id='assignment-table',
        columns=[],  # set via callback
        data=[],
        hidden_columns=["Include_SNR", "Include_Interf", "Include_Merge"],  # <— add this
        row_selectable="single",
        selected_rows=[],
        style_table={'width': '95%'},
        style_cell={'textAlign': 'center','padding': '4px 6px','fontSize': 12},
        style_header={'fontSize': 12, 'fontWeight': 'bold'},
        virtualization=True,
        fixed_rows={'headers': True},
        editable=True,
        style_data_conditional=[
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
    dcc.Store(id="percat-assignments", data={}),           # {str(idx): [rows]}
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
])  # close app.layout



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
    prevent_initial_call=True
)
def handle_all_zoom_events(relayout, undo_clicks, zoom_in_clicks, zoom_out_clicks, y_in, y_out,
                           apply_xrange_clicks, xmin_val, xmax_val,
                           current_zoom, history, last_y):
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

    # Apply-xrange button
    if trigger_id == "apply-xrange":
        try:
            xmin = float(xmin_val); xmax = float(xmax_val)
            if xmax <= xmin:
                raise ValueError
        except Exception:
            xmin, xmax = DEFAULT_XMIN, DEFAULT_XMAX
        if isinstance(current_zoom, dict):
            history.append(current_zoom)
        return {"x": [xmin, xmax], "y": None}, history

    # Plotly relayouts (zoom, pan, double-click home, toolbar autorange)
    if trigger_id == "spectrum-plot" and relayout:
        # Any autorange flag → treat as reset unless explicit x range is provided
        if relayout.get("autosize") or relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
            if isinstance(current_zoom, dict):
                history.append(current_zoom)
            return {"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None}, history

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

        if x0 is not None and x1 is not None:
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
            return {"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None}, []
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

# =========================
# Mutate assignments (assign, delete, recalc, load, edits)
# =========================
@app.callback(
    Output("percat-assignments", "data"),
    Output("assignment-table", "selected_rows"),
    Input("assign-request", "data"),                 # <-- NEW Input (atomic trigger)
    Input("assignment-table", "selected_rows"),
    Input("recalc-weights-button", "n_clicks"),
    Input("assignment-table", "data_timestamp"),
    Input("load-int-button", "n_clicks"),
    State("percat-assignments", "data"),
    State("mode-selector", "value"),
    State("stored-region-selection", "data"),
    State("intensity-threshold", "value"),
    State("active-cat-idx", "data"),
    State("assignment-table", "data"),
    State("int-file-path", "value"),
    State("stored-fit-params", "data"),
    prevent_initial_call=True
)
def mutate_assignments(assign_req, selected_rows, recalc_clicks, data_ts, load_clicks,
                       percat, mode, selection, intensity_threshold, active_idx,
                       table_data, int_path, fit_params_state):

    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    percat = percat or {}
    key = str(int(active_idx or 0))
    current = list(percat.get(key, []))

    # A) Atomic assign: use the SNAPSHOT embedded in assign_req
    if trig.startswith("assign-request") and assign_req:
        req_idx = int(assign_req.get("active_idx", active_idx or 0))
        sel_range = assign_req.get("sel_range")
        thr = float(assign_req.get("thr", 0.01))
        sel_mu = assign_req.get("mu", None)

        # hard guardrails
        if sel_mu is None or not sel_range or len(sel_range) != 2:
            raise dash.exceptions.PreventUpdate

        x0, x1 = map(float, sel_range)
        key = str(req_idx)
        current = list((percat or {}).get(key, []))

        active = catalogs[req_idx]
        sim_df = active["df"]; qn_field_order = active["qn_order"]

        mask = (sim_df["Norm_Intensity"] >= thr) & (sim_df["Freq"] >= x0) & (sim_df["Freq"] <= x1)
        in_range = sim_df.loc[mask]

        fit_ctx = _build_fit_context({"range": {"x": sel_range}}, fit_params_state, delta_F=DELTA_F_DEFAULT)
        sel_mu = float(sel_mu)

        for _, row in in_range.iterrows():
            freq_full = float(row["Freq"])
            new_entry = {
                "obs": round(sel_mu, 4),
                "sim": freq_full,
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
        current = _recalc_uncertainties(current, selection_range=[0.0, 0.0], delta_F=DELTA_F_DEFAULT)
        percat[key] = current
        return percat, []



    # B) Delete selected rows
    if trig.startswith("assignment-table.selected_rows"):
        if selected_rows:
            drop = set(selected_rows)
            current = [row for i, row in enumerate(current) if i not in drop]
            for r in current:
                try:
                    r["Delta"] = round(float(r["obs"]) - float(r["sim"]), 4)
                except Exception:
                    r["Delta"] = None
            current = recompute_peak_weights(current)


            # Restore any missing FitCtx, then recompute using per-row saved settings
            current, _ = _restore_fitctx_if_missing(current, active_idx)
            current = _recalc_uncertainties(current, selection_range=None, delta_F=None)
            current = _decorate_display_flags(current)

            percat[key] = current
            return percat, []

        raise dash.exceptions.PreventUpdate

    # C) Recompute weights (button)
    if trig.startswith("recalc-weights-button"):
        for r in current:
            try:
                r["Delta"] = round(float(r["obs"]) - float(r["sim"]), 4)
            except Exception:
                r["Delta"] = None
        # Recompute uncertainties with a default width if no selection is present


        current = recompute_peak_weights(current)
        current, _ = _restore_fitctx_if_missing(current, active_idx)
        current = _recalc_uncertainties(current, selection_range=None, delta_F=None)
        current = _decorate_display_flags(current)

        percat[key] = current
        return percat, []




    # D) Inline edits via data_timestamp
    if trig.startswith("assignment-table.data_timestamp"):
        if not isinstance(table_data, list):
            raise dash.exceptions.PreventUpdate
        current_in_store = percat.get(key, [])
        if table_data == current_in_store:
            raise dash.exceptions.PreventUpdate
        for r in table_data:
            try:
                r["Delta"] = round(float(r["obs"]) - float(r["sim"]), 4)
            except Exception:
                r["Delta"] = None

        table_data, _ = _restore_fitctx_if_missing(table_data, active_idx)
        table_data = _recalc_uncertainties(table_data, selection_range=None, delta_F=None)
        table_data = _decorate_display_flags(table_data)

        percat[key] = table_data
        return percat, dash.no_update


    # E) Load .lin file
    if trig.startswith("load-int-button"):
        if not int_path or not os.path.isfile(int_path) or not int_path.endswith(".lin"):
            raise dash.exceptions.PreventUpdate

        sim_df = catalogs[int(active_idx or 0)]["df"]
        qn_field_order = catalogs[int(active_idx or 0)]["qn_order"]
        loaded = []
        try:
            with open(int_path, "r") as f:
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

    if r_idx is None or r_idx < 0 or r_idx >= len(ui_rows):
        raise dash.exceptions.PreventUpdate

    # Identify the row using stable fields present in the UI rows
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
    Input("stored-zoom", "data"),
    Input("mode-selector", "value"),
    Input("intensity-threshold", "value"),
    Input("selected-fit-mu", "data"),
    Input("flip-sim-checkbox", "value"),
    Input("percat-scales", "data"),
    Input("active-cat-idx", "data"),
)
def update_plot(fit_params, percat, zoom, mode, intensity_threshold, selected_mu,
                flip_checkbox, percat_scales, active_idx):

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

        if not isinstance(meas_freqs, np.ndarray) or not isinstance(meas_intensities, np.ndarray):
            raise ValueError("Measured arrays not initialized.")
        if meas_freqs.size == 0 or meas_intensities.size == 0:
            raise ValueError("Measured arrays are empty.")

        # measured subset by zoom + decimate
        if zoom and "x" in zoom and zoom["x"] is not None:
            x0z, x1z = zoom["x"]
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
            amp  = norm * float(scale if scale else 1.0)

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
                x0z, x1z = zoom["x"]
            else:
                x0z, x1z = -np.inf, np.inf

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
                active_traces.append(get_stick_trace(unassigned_lines, "red",  f"{cat['name']} (unassigned)", scale=cat_scale))
                active_traces.append(get_stick_trace(assigned_lines,   "blue", f"{cat['name']} (assigned)", dash="dash", scale=cat_scale))
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
            line=dict(color="#000000", width=1.6)
        ))



        fig = go.Figure(traces)

        # axes / zoom
        if zoom and "x" in zoom and zoom["x"] is not None:
            fig.update_xaxes(range=zoom["x"])
            if zoom.get("y") is not None:
                fig.update_yaxes(range=zoom["y"])
        else:
            fig.update_xaxes(range=[DEFAULT_XMIN, DEFAULT_XMAX])



        _apply_adaptive_xticks(fig, span)
        
        
        fig.update_layout(
            dragmode="select" if mode in ["select", "assign_all"] else "zoom",
            template="simple_white",
            height=600,
            xaxis_title="Frequency (MHz)",
            yaxis_title="Normalized Intensity" + (" (Sim Flipped)" if flip else ""),
            uirevision="zoom-lock",
            plot_bgcolor="#5A5A5A",
            paper_bgcolor="#5A5A5A",
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



        # fitted peaks (individual + sum + baseline)
        if fit_params and "multi" in fit_params and isinstance(fit_params["multi"], list):
            baseline = float(fit_params.get("baseline", 0.0))
            baseline_std = float(fit_params.get("baseline_std", 0.0))
            brange = fit_params.get("baseline_range", None)

            # individual Gaussians (offset by baseline)
            for p in fit_params["multi"]:
                mu = float(p["mu"]); sig = float(p["sigma"]); amp = float(p["amp"])
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
                    line=dict(color="blue", dash="dot", width=3.5),
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

        yr = fig.layout.yaxis.range
        last_y = list(yr) if yr else None
        return fig, last_y, measured_idx

    except Exception as e:
        print(f"[ERROR] update_plot crashed: {e}")
        try:
            fig = go.Figure([
                go.Scatter(x=meas_freqs, y=meas_intensities, mode="lines", name="Measured", line=dict(color="black"))
            ])
            fig.update_xaxes(range=[DEFAULT_XMIN, DEFAULT_XMAX])
            fig.update_layout(height=600, template="simple_white", plot_bgcolor="#5A5A5A",
                              paper_bgcolor="#5A5A5A", font_color="white")
            return fig, None, None
        except Exception as ee:
            print(f"[FATAL] Could not even draw measured trace: {ee}")
            return go.Figure(), None, None

# =========================
# Save .lin (active catalog)
# =========================
def generate_lin_file(assignments, qn_field_order):
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

@app.callback(
    Output("save-lin-confirmation", "children"),
    Input("save-lin-button", "n_clicks"),
    State("percat-assignments", "data"),
    State("active-cat-idx", "data"),
    prevent_initial_call=True
)
def save_lin_file(n_clicks, percat, active_idx):
    active_idx = int(active_idx or 0)
    key = str(active_idx)
    percat = percat or {}
    assignments = percat.get(key, [])
    if not assignments:
        return "❌ No assignments to save for this catalog."

    qn_field_order = catalogs[active_idx]["qn_order"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    assignments_dir = os.path.join(os.getcwd(), "assignments")
    os.makedirs(assignments_dir, exist_ok=True)

    cat_name = os.path.splitext(os.path.basename(catalogs[active_idx]["path"]))[0]
    filename = f"{cat_name}_assignments_{timestamp}.lin"
    filepath = os.path.join(assignments_dir, filename)

    content = generate_lin_file(assignments, qn_field_order)
    with open(filepath, "w") as f:
        f.write(content + "\n")

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
        data = recompute_peak_weights(data)
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
    prevent_initial_call=True
)
def handle_fit_mu_and_keyboard(fit_params, n_clicks_list, assign_n_clicks, n_keydowns, key_event, ids,
                               selected_mu, fit_params_state, active_idx,
                               selection_state, thr_state):
    trigger = ctx.triggered_id
    mode_value = dash.no_update
    undo_clicks = dash.no_update
    num_gauss = dash.no_update
    next_mu = selected_mu
    next_active_idx = dash.no_update

    assign_request = dash.no_update  

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
                "backgroundColor": "#007BFF" if is_selected else "#f0f0f0",
                "color": "white" if is_selected else "black",
            }
            buttons.append(html.Button(label, id={"type": "fit-mu-button", "index": mu}, n_clicks=0, style=style))

    return buttons, next_mu, mode_value, undo_clicks, num_gauss, next_active_idx, assign_request




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
    app.run(debug=True, port=8061)

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


# =========================
# Init & Config
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

qn_label_map = config.get("qn_labels", {})
cat_file_path = config["cat_file"]
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


sim_df, qn_field_order = parse_cat_file(cat_file_path)

# Build hover text for simulated sticks (includes QN info)
def generate_hover(row):
    parts = [
        f"<b>Freq:</b> {row['Freq']:.4f} MHz",
        f"<b>Intensity:</b> {row['Intensity']:.2e}",
        f"<b>Eu:</b> {row['Eu']:.2f} cmâ»Â¹"
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

# Precompute helpers for simulated lines
sim_df["Norm_Intensity"] = sim_df["Intensity"] / sim_df["Intensity"].max()
sim_df["RoundedFreq"] = sim_df["Freq"].round(4)
sim_df["StickX"] = sim_df["Freq"].apply(lambda f: [f, f, None])  # repeated x values to draw a stick
sim_df["Hover"] = sim_df.apply(generate_hover, axis=1)


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


# ---------- Uncertainty estimation (fast + vectorized) ----------
def compute_uncertainties(assignments, selection_range, delta_F=0.1):
    """
    assignments: list[dict] (current rows in the table)
    selection_range: [x0, x1] of the last selection (used for baseline width)
    delta_F: MHz (interference scale, default 0.1 = 100 kHz)
    returns: dict {obs: sigma_total}
    """
    if not assignments:
        return {}

    # --- group simulated freqs by observed freq (merge term)
    from collections import defaultdict
    sim_by_obs = defaultdict(list)
    for r in assignments:
        sim_by_obs[r["obs"]].append(r["sim"])

    obs = np.array(sorted(sim_by_obs.keys()), dtype=float)
    n = obs.size

    # Merge term: half the span of sim freqs for the same obs (0 if single)
    sigma_merge = np.zeros(n, dtype=float)
    for i, oi in enumerate(obs):
        sims = sim_by_obs[oi]
        if len(sims) > 1:
            smax = np.max(sims); smin = np.min(sims)
            sigma_merge[i] = 0.5 * (smax - smin)

    # Interference term: pairwise distances with smooth cutoff
    if n > 1:
        D = np.abs(obs[:, None] - obs[None, :])  # pairwise |oi-oj|
        np.fill_diagonal(D, 0.0)
        C = 0.5 * D * (1.0 - np.tanh(D / float(delta_F)))
        sigma_interf = np.sqrt(np.sum(C * C, axis=1))
    else:
        sigma_interf = np.zeros(1, dtype=float)

    # Instrument term: local SNR around each obs
    x0, x1 = selection_range
    baseline_width = float(x1 - x0)  # same idea as your old code
    if baseline_width <= 0.0:
        baseline_width = 10.0  # small fallback

    def nearest_idx(x):
        # argmin |meas_freqs - x| using searchsorted for speed
        j = np.searchsorted(meas_freqs, x)
        if j <= 0: return 0
        if j >= len(meas_freqs): return len(meas_freqs) - 1
        return j if (abs(meas_freqs[j] - x) < abs(meas_freqs[j-1] - x)) else (j-1)

    sigma_instr = np.zeros(n, dtype=float)
    for i, oi in enumerate(obs):
        j = nearest_idx(oi)
        peak_height = float(meas_intensities[j])

        left_mask  = (meas_freqs >= oi - 2*baseline_width) & (meas_freqs <  oi - baseline_width)
        right_mask = (meas_freqs >  oi + baseline_width)  & (meas_freqs <= oi + 2*baseline_width)
        baseline_vals = np.concatenate([meas_intensities[left_mask], meas_intensities[right_mask]])
        baseline_rms = float(np.std(baseline_vals)) if baseline_vals.size > 0 else 0.01

        SNR = peak_height / baseline_rms if baseline_rms > 0 else 1.0
        sigma_instr[i] = np.sqrt((0.0575 / SNR) ** 2 + (0.01) ** 2)

    # Combine (quadrature)
    sigma_total = np.sqrt(sigma_merge**2 + sigma_interf**2 + sigma_instr**2)
    # Round like before
    return {float(o): round(float(s), 4) for o, s in zip(obs, sigma_total)}


def decimate_xy(x, y, max_pts=35000):
    """Uniformly downsample to at most max_pts points to keep plotting responsive."""
    n = x.size
    if n <= max_pts:
        return x, y
    step = max(1, n // max_pts)
    return x[::step], y[::step]


@lru_cache(maxsize=64)
def _fit_sum_cached(amps, mus, sigmas, baseline, x_min, x_max, npts):
    """Cache the summed fit curve for the visible window."""
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


def build_assignment_columns(qn_field_order):
    static_columns = [
        {"name": "obs", "id": "obs", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "sim", "id": "sim", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "Delta (obs-sim)", "id": "Delta", "type": "numeric", "format": {"specifier": ".4f"}, "editable": False},
        {"name": "Eu", "id": "Eu", "type": "numeric", "format": {"specifier": ".2f"}, "editable": False},
        {"name": "logI", "id": "logI", "type": "numeric", "format": {"specifier": ".2f"}, "editable": False},
    ]
    qn_columns = [{"name": qn_label_map.get(col, col), "id": col, "editable": False} for col in qn_field_order]
    extra_columns = [
        # Editable by double click
        {"name": "Uncertainty", "id": "Uncertainty", "type": "numeric",
         "format": {"specifier": ".4f"}, "editable": True},
        {"name": "Weight", "id": "Weight", "type": "numeric", "format": {"specifier": ".2f"}, "editable": False},
    ]
    return static_columns + qn_columns + extra_columns


def recompute_peak_weights(assignments):
    """Recompute Weight per observed peak from simulated intensity within that observed peak."""
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


def parse_lin_line_flexible(line):
    """
    Parse one .lin line assuming the numeric tail uses fixed widths:
      freq:12.4f, unc:10.4f, wt:6.2f
    Returns (qns_list, freq, unc, wt).
    """
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

app.layout = html.Div([
    html.H2("Interactive Spectrum Assigner)"),

    # Intensity scaling & default X range controls
    html.Div([
        html.Label("Simulated Spectrum Intensity Scale:"),
        dcc.Input(id="sim-scale", type="number", value=1.0, step=0.1, style={"width": "120px", "marginRight": "10px"}),
        html.Button("Apply Scale", id="apply-scale", n_clicks=0),
    ], style={"marginBottom": "10px"}),

    html.Div([
        html.Label("Default X-axis Range (MHz):"),
        dcc.Input(id="default-xmin", type="number", value=DEFAULT_XMIN, step=100, style={"width": "120px"}),
        dcc.Input(id="default-xmax", type="number", value=DEFAULT_XMAX, step=100, style={"width": "120px", "marginLeft": "8px", "marginRight": "10px"}),
        html.Button("Apply X-range", id="apply-xrange", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    # Keyboard listener
    Keyboard(id="keyboard", captureKeys=["q", "w", "e", "r", "a", "d"] + [str(i) for i in range(0, 11)]),

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

    dcc.Graph(id='spectrum-plot', config={"modeBarButtonsToAdd": ["select2d", "zoom2d"]}),
    html.Div(id="fit-output", style={"marginBottom": 10}),

    # Zoom controls (X and Y)
    html.Div([
        html.Button("Undo Zoom", id="undo-zoom-button", n_clicks=0, style={"marginRight": "10px"}),

        html.Button("X+ (Zoom In)", id="x-zoom-in", n_clicks=0, style={"marginRight": "5px"}),
        html.Button("Xâ (Zoom Out)", id="x-zoom-out", n_clicks=0, style={"marginRight": "20px"}),

        html.Button("Y+ (Zoom In)", id="y-zoom-in", n_clicks=0, style={"marginRight": "5px"}),
        html.Button("Yâ (Zoom Out)", id="y-zoom-out", n_clicks=0),
    ], style={"marginBottom": "15px"}),

    html.Div([
        html.Label("Simulated Intensity Threshold (0â1):"),
        dcc.Input(id='intensity-threshold', type='number', min=0, max=1, step=0.01, value=0.01, debounce=True)
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
        columns=build_assignment_columns(qn_field_order),
        data=[],
        row_selectable="single",
        selected_rows=[],
        style_table={'width': '95%'},
        style_cell={'textAlign': 'center'},
        virtualization=True,
        fixed_rows={'headers': True},
        editable=True,  # Allow editing (Uncertainty)
    ),

    # Stores
    dcc.Store(id="selected-fit-mu"),
    dcc.Store(id="stored-fit-params"),
    dcc.Store(id="stored-assignments", data=[]),
    dcc.Store(id="stored-region-selection"),
    dcc.Store(id="stored-zoom", data={"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None}),  # default plotting range
    dcc.Store(id="zoom-history", data=[]),
    dcc.Store(id="sim-scale-store", data=1.0),  # intensity scaling factor
    dcc.Store(id="last-y-range", data=None),    # remember last y-range for Y+/Yâ
])


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
        return "â Too few points to fit.", dash.no_update

    x, y = meas_freqs[mask], meas_intensities[mask]
    # Light downsample for curve fitting only
    x, y = decimate_xy(x, y, max_pts=1200)

    window_width = x1 - x0

    # Estimate baseline from side regions
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

    # baseline parameter
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
        msg = "â Fitted peaks at: " + ", ".join(f"{p['mu']:.2f} MHz" for p in fits)
        msg += f"<br>Estimated baseline offset: {baseline:.4f} Â± {y_base_std:.4f}"
        return msg, {"multi": fits, "baseline": baseline, "baseline_range": [x0 - margin, x1 + margin]}
    except Exception as e:
        return f"â Fit failed: {str(e)}", dash.no_update


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
    State("stored-zoom", "data"),
    State("zoom-history", "data"),
    State("last-y-range", "data"),
    prevent_initial_call=True
)
def handle_all_zoom_events(relayout, undo_clicks, zoom_in_clicks, zoom_out_clicks, y_in, y_out,
                           current_zoom, history, last_y):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger = ctx.triggered_id
    history = history or []

    def default_y_range():
        if last_y and isinstance(last_y, list) and len(last_y) == 2:
            return last_y
        return [-0.1, 1.2]

    # Normal mouse/box zoom on the graph
    if trigger == "spectrum-plot" and relayout:
        if "autosize" in relayout or "xaxis.autorange" in relayout:
            if current_zoom:
                history.append(current_zoom)
            return {"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None}, history

        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")
        y0 = relayout.get("yaxis.range[0]")
        y1 = relayout.get("yaxis.range[1]")

        if x0 is not None and x1 is not None:
            new_zoom = {"x": [x0, x1], "y": [y0, y1] if (y0 is not None and y1 is not None) else None}
            if current_zoom:
                history.append(current_zoom)
            return new_zoom, history

    # Undo zoom
    elif trigger == "undo-zoom-button":
        if not history:
            return {"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None}, []
        last_zoom = history[-1]
        return last_zoom, history[:-1]

    # X-axis zoom in/out buttons
    elif trigger in ["x-zoom-in", "x-zoom-out"]:
        if not current_zoom or "x" not in current_zoom:
            raise dash.exceptions.PreventUpdate
        x0, x1 = current_zoom["x"]
        x_center = (x0 + x1) / 2
        x_width = x1 - x0
        zoom_factor = 0.3 if trigger == "x-zoom-in" else 2.5
        new_width = x_width * zoom_factor
        new_x0 = x_center - new_width / 2
        new_x1 = x_center + new_width / 2
        new_zoom = {"x": [new_x0, new_x1], "y": current_zoom.get("y")}
        history.append(current_zoom)
        return new_zoom, history

    # Y-axis zoom in/out buttons
    elif trigger in ["y-zoom-in", "y-zoom-out"]:
        yr = (current_zoom or {}).get("y") or default_y_range()
        y0, y1 = float(yr[0]), float(yr[1])
        y_center = (y0 + y1) / 2.0
        y_height = (y1 - y0)
        zoom_factor = 0.3 if trigger == "y-zoom-in" else 2.5
        new_height = y_height * zoom_factor
        new_y0 = y_center - new_height / 2.0
        new_y1 = y_center + new_height / 2.0
        new_zoom = {"x": (current_zoom or {}).get("x", [DEFAULT_XMIN, DEFAULT_XMAX]), "y": [new_y0, new_y1]}
        history.append(current_zoom or {"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None})
        return new_zoom, history

    raise dash.exceptions.PreventUpdate


# Apply default X-range from inputs
@app.callback(
    Output("stored-zoom", "data", allow_duplicate=True),
    Input("apply-xrange", "n_clicks"),
    State("default-xmin", "value"),
    State("default-xmax", "value"),
    prevent_initial_call=True
)
def apply_default_range(n, xmin, xmax):
    try:
        xmin = float(xmin); xmax = float(xmax)
        if xmax <= xmin:
            raise ValueError
        return {"x": [xmin, xmax], "y": None}
    except Exception:
        return {"x": [DEFAULT_XMIN, DEFAULT_XMAX], "y": None}


# Apply intensity scale from input
@app.callback(
    Output("sim-scale-store", "data"),
    Input("apply-scale", "n_clicks"),
    State("sim-scale", "value"),
    prevent_initial_call=True
)
def update_scale(n, scale_val):
    try:
        scale_val = float(scale_val)
        return scale_val if scale_val > 0 else 1.0
    except Exception:
        return 1.0


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
# Assign by region (with fast uncertainties)
# =========================
@app.callback(
    Output("stored-assignments", "data", allow_duplicate=True),
    Output("assignment-table", "data", allow_duplicate=True),
    Input("assign-button", "n_clicks"),
    State("stored-assignments", "data"),
    State("mode-selector", "value"),
    State("stored-region-selection", "data"),
    State("selected-fit-mu", "data"),
    State("intensity-threshold", "value"),
    prevent_initial_call=True
)
def assign_by_region(n_clicks, current_assignments, mode, selection, selected_mu, intensity_threshold):
    if mode != "assign_all" or not selection or selected_mu is None:
        raise dash.exceptions.PreventUpdate

    x0, x1 = selection["range"]["x"]
    thr = float(intensity_threshold or 0.01)

    # Filter simulated lines server-side
    mask = (sim_df["Norm_Intensity"] >= thr) & (sim_df["Freq"] >= x0) & (sim_df["Freq"] <= x1)
    in_range = sim_df.loc[mask]

    current_assignments = current_assignments or []

    # Add new assignments
    sel_mu = float(selected_mu)
    for _, row in in_range.iterrows():
        new_entry = {
            "obs": round(sel_mu, 4),
            "sim": round(float(row["Freq"]), 4),
            "Delta": round(sel_mu - float(row["Freq"]), 4),
            "Eu": round(float(row["Eu"]), 4),
            "logI": round(np.log10(float(row["Intensity"])), 4),
        }
        for key in qn_field_order:
            if key in row:
                new_entry[key] = int(row[key])
        if new_entry not in current_assignments:
            current_assignments.append(new_entry)

    # ---- FAST UNCERTAINTY COMPUTATION ----
    uncertainty_by_obs = compute_uncertainties(current_assignments, [x0, x1], delta_F=0.1)

    # Apply per-row
    for r in current_assignments:
        r["Uncertainty"] = uncertainty_by_obs[float(r["obs"])]

    # Recompute weights and return
    current_assignments = recompute_peak_weights(current_assignments)
    return current_assignments, current_assignments


# =========================
# Main plot (250 MHz grid + hover QNs for simulated)
# =========================
@app.callback(
    Output("spectrum-plot", "figure"),
    Output("last-y-range", "data"),
    Input("stored-fit-params", "data"),
    Input("stored-assignments", "data"),
    Input("stored-zoom", "data"),
    Input("mode-selector", "value"),
    Input("intensity-threshold", "value"),
    Input("selected-fit-mu", "data"),
    Input("flip-sim-checkbox", "value"),
    Input("sim-scale-store", "data")
)
def update_plot(fit_params, assignments, zoom, mode, intensity_threshold, selected_mu, flip_checkbox, sim_scale):
    flip_vals = flip_checkbox or []
    thr = intensity_threshold or 0.01
    sim_scale = float(sim_scale or 1.0)
    flip = ("flip" in flip_vals)

    # Measured subset by zoom + decimate to 35k (preserve overall shape while responsive)
    if zoom and "x" in zoom:
        x0z, x1z = zoom["x"]
        mask_meas = (meas_freqs >= x0z) & (meas_freqs <= x1z)
        x_meas = meas_freqs[mask_meas]
        y_meas = meas_intensities[mask_meas]
    else:
        x_meas = meas_freqs
        y_meas = meas_intensities
    x_meas, y_meas = decimate_xy(x_meas, y_meas, max_pts=35000)

    # Visible simulated lines
    if zoom and "x" in zoom:
        x0z, x1z = zoom["x"]
        mask = (sim_df["Norm_Intensity"] >= thr) & (sim_df["Freq"] >= x0z) & (sim_df["Freq"] <= x1z)
    else:
        mask = (sim_df["Norm_Intensity"] >= thr)
    sim_visible = sim_df.loc[mask]

    # For very wide windows, limit the number of sticks for speed
    span = None
    if zoom and "x" in zoom:
        span = float(zoom["x"][1] - zoom["x"][0])
    if span is None or span > 500:
        sim_visible = sim_visible.nlargest(1500, "Norm_Intensity")

    # Assigned vs unassigned via NumPy
    if assignments:
        assigned_arr = np.array([round(row["sim"], 4) for row in assignments], dtype=np.float64)
        rounded = np.round(sim_visible["Freq"].values, 4)
        is_assigned = np.isin(rounded, assigned_arr, assume_unique=False)
    else:
        is_assigned = np.zeros(sim_visible.shape[0], dtype=bool)

    assigned_lines = sim_visible.loc[is_assigned]
    unassigned_lines = sim_visible.loc[~is_assigned]

    def get_stick_trace(df, color, name):
        if df.empty:
            return go.Scattergl(x=[], y=[])
        # x from precomputed StickX
        x = [v for sub in df["StickX"] for v in sub]
        # y constructed on the fly: [0, +/-scale*Norm, None]
        sign = -1.0 if flip else 1.0
        y = []
        text = []
        for _, r in df.iterrows():
            y.extend([0.0, sign * sim_scale * float(r["Norm_Intensity"]), None])
            h = r.get("Hover", f"Freq {r['Freq']:.4f} MHz")
            text.extend([h, h, None])  # align with x/y
        return go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=color),
            name=name,
            hoverinfo="text",
            text=text,
            hovertemplate="%{text}<extra></extra>",
        )

    fig = go.Figure([
        get_stick_trace(unassigned_lines, "red", "Simulated (unassigned)"),
        get_stick_trace(assigned_lines, "blue", "Simulated (assigned)"),
        go.Scattergl(
            x=x_meas, y=y_meas,
            mode="lines",
            name="Measured",
            line=dict(color="black"),
            hoverinfo="skip"
        )
    ])

    # Axes / zoom ranges
    if zoom and "x" in zoom:
        fig.update_xaxes(range=zoom["x"])
        if zoom.get("y") is not None:
            fig.update_yaxes(range=zoom["y"])
    else:
        fig.update_xaxes(range=[DEFAULT_XMIN, DEFAULT_XMAX])

    # Add semi-transparent 250 MHz vertical grid lines
    xr = fig.layout.xaxis.range if fig.layout.xaxis.range else [DEFAULT_XMIN, DEFAULT_XMAX]
    x0v, x1v = float(xr[0]), float(xr[1])
    start = int(np.floor(x0v / 250.0) * 250)
    end = int(np.ceil(x1v / 250.0) * 250)
    for xline in range(start, end + 1, 250):
        fig.add_vline(
            x=xline,
            y0=0, y1=1,
            xref="x", yref="y",   # â tie height to the y-axis domain
            line_width=1,
            line_dash="dot",
            opacity=0.25,
            layer="below",
        )

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

    # Individual Gaussians from the fit
    if fit_params and "multi" in fit_params:
        baseline = float(fit_params.get("baseline", 0.0))
        for p in fit_params["multi"]:
            mu = float(p["mu"]); sig = float(p["sigma"]); amp = float(p["amp"])
            x_fit = np.linspace(mu - 4 * sig, mu + 4 * sig, 120)
            y_fit = gaussian(x_fit, amp, mu, sig) + baseline
            fig.add_trace(go.Scattergl(
                x=x_fit, y=y_fit, mode="lines",
                name=f"Î¼={mu:.2f}",
                line=dict(color="green", dash="dot"),
                hoverinfo="skip"
            ))

    # Sum curve (cached)
    if fit_params and "multi" in fit_params and len(fit_params["multi"]) > 0:
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
            baseline_val = float(fit_params.get("baseline", 0.0))
            x_grid, y_sum = _fit_sum_cached(amps, mus, sigmas, baseline_val, float(x_min), float(x_max), 800)
            fig.add_trace(go.Scattergl(
                x=x_grid, y=y_sum,
                mode="lines",
                name="Fit sum",
                line=dict(color="yellow", width=1, dash="dot"),
                opacity=0.5,
                hoverinfo="skip"
            ))

    # Observed peak vertical markers (when zoomed within ~100 MHz)
    if assignments:
        if not zoom or (zoom["x"][1] - zoom["x"][0] < 100):
            unique_obs = sorted(set(pair["obs"] for pair in assignments))
            obs_lines = go.Scatter(
                x=[mu for mu in unique_obs for _ in range(3)],
                y=[0, 1.02, None] * len(unique_obs),
                mode="lines",
                line=dict(color="blue", dash="dash", width=1.5),
                opacity=0.7,
                name="Assigned Obs",
                hoverinfo="skip",
                showlegend=False
            )
            fig.add_trace(obs_lines)

    # Baseline from the fit
    if fit_params and "baseline" in fit_params and "baseline_range" in fit_params:
        base = float(fit_params["baseline"]); x0b, x1b = fit_params["baseline_range"]
        fig.add_trace(go.Scattergl(
            x=[x0b, x1b], y=[base, base],
            mode="lines", name="Estimated Baseline",
            line=dict(color="orange", dash="dash"),
            hoverinfo="skip",
        ))

    # === Uncertainty bands (fast, capped, drawn only when zoomed) ===
    if assignments:
        # Collect per-obs uncertainty (first occurrence wins)
        obs_to_unc = {}
        for r in assignments:
            ov = r.get("obs"); uv = r.get("Uncertainty")
            if ov is None or uv is None:
                continue
            try:
                ov = float(ov); uv = float(uv)
            except (TypeError, ValueError):
                continue
            if ov not in obs_to_unc:
                obs_to_unc[ov] = uv

        # Heuristics to keep things snappy
        MAX_VRECTS = 80                # don't add more than this many bands
        MAX_SPAN_FOR_VRECTS = 150.0    # only draw when x-span < this (MHz)

        # Decide if we should draw now (based on zoom)
        draw_now = True
        x0z, x1z = (float("-inf"), float("inf"))
        if zoom and "x" in zoom and zoom["x"] is not None:
            x0z, x1z = zoom["x"]
            draw_now = (x1z - x0z) < MAX_SPAN_FOR_VRECTS

        if draw_now and obs_to_unc:
            # Limit to the visible obs and cap count
            visible_obs = [o for o in obs_to_unc.keys() if x0z <= o <= x1z]
            if visible_obs:
                # Prioritize bands near the center so the most relevant ones render first
                x_center = (x0z + x1z) / 2 if np.isfinite(x0z) and np.isfinite(x1z) else 0.0
                visible_obs.sort(key=lambda v: abs(x_center - v))
                visible_obs = visible_obs[:MAX_VRECTS]

                # Add vrects (paper yref draws full-height bands regardless of y-axis limits)
                for ov in visible_obs:
                    uv = obs_to_unc[ov]
                    fig.add_vrect(
                        x0=ov - uv, x1=ov + uv,
                        y0=0, y1=1,
                        xref="x", yref="y",   # â full y-axis height, not figure âpaperâ
                        fillcolor="lightblue",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    )







    # Remember y-range for Y+/Yâ controls
    yr = fig.layout.yaxis.range
    last_y = list(yr) if yr else None
    return fig, last_y


# =========================
# Load .lin file
# =========================
@app.callback(
    Output("stored-assignments", "data", allow_duplicate=True),
    Output("assignment-table", "data", allow_duplicate=True),
    Input("load-int-button", "n_clicks"),
    State("int-file-path", "value"),
    prevent_initial_call=True
)
def load_lin_file_from_path(n_clicks, filepath):
    if not filepath or not os.path.isfile(filepath) or not filepath.endswith(".lin"):
        return dash.no_update, dash.no_update

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except Exception:
        return dash.no_update, dash.no_update

    assignments = []
    with open(filepath, "r") as f:
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
                    "sim": round(float(sim_row["Freq"]), 4),
                    "Delta": round(float(freq) - float(sim_row["Freq"]), 4),
                    "Eu": round(float(sim_row["Eu"]), 4),
                    "logI": round(np.log10(float(sim_row["Intensity"])), 4),
                    "Uncertainty": round(float(unc), 4),
                    "Weight": round(float(wt), 4)
                }
                for field, value in zip(qn_fields, qn_values):
                    assignment[field] = value
                assignments.append(assignment)
            except Exception:
                continue

    return assignments, assignments


# =========================
# Update assignments table (delete / recompute + accept edits)
# =========================
@app.callback(
    Output("stored-assignments", "data"),
    Output("assignment-table", "data"),
    Output("assignment-table", "selected_rows"),
    Input("assignment-table", "selected_rows"),
    Input("recalc-weights-button", "n_clicks"),
    State("assignment-table", "data"),
    prevent_initial_call=True
)
def update_assignments_table(selected_rows, recalc_clicks, current_data):
    trig = callback_context.triggered_id
    if not current_data:
        raise dash.exceptions.PreventUpdate

    updated = current_data.copy()

    # Delete selected rows on click
    if trig == "assignment-table":
        if not selected_rows:
            raise dash.exceptions.PreventUpdate
        to_drop = set(selected_rows)
        updated = [row for i, row in enumerate(current_data) if i not in to_drop]

    # Recompute Delta & weights
    for r in updated:
        try:
            r["Delta"] = round(float(r["obs"]) - float(r["sim"]), 4)
        except Exception:
            r["Delta"] = None

    updated = recompute_peak_weights(updated)
    return updated, updated, []


# Accept table edits (mainly Uncertainty)
@app.callback(
    Output("stored-assignments", "data", allow_duplicate=True),
    Input("assignment-table", "data"),
    prevent_initial_call=True
)
def apply_table_edits(table_data):
    if not isinstance(table_data, list):
        raise dash.exceptions.PreventUpdate
    for r in table_data:
        try:
            r["Delta"] = round(float(r["obs"]) - float(r["sim"]), 4)
        except Exception:
            r["Delta"] = None
    return table_data


# =========================
# Save .lin
# =========================
def generate_lin_file(assignments):
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
    State("stored-assignments", "data"),
    prevent_initial_call=True
)
def save_lin_file(n_clicks, assignments):
    if not assignments:
        return "â No assignments to save."

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    assignments_dir = os.path.join(os.getcwd(), "assignments")
    os.makedirs(assignments_dir, exist_ok=True)

    filename = f"assignments_{timestamp}.lin"
    filepath = os.path.join(assignments_dir, filename)

    content = generate_lin_file(assignments)
    with open(filepath, "w") as f:
        f.write(content + "\n")

    return f"â Saved: {filename}"


# =========================
# Î¼ buttons + keyboard
# =========================
@app.callback(
    Output("fit-mu-button-container", "children"),
    Output("selected-fit-mu", "data"),
    Output("mode-selector", "value"),
    Output("undo-zoom-button", "n_clicks"),
    Output("num-gaussians", "value"),
    Output("assign-button", "n_clicks"),
    Input("stored-fit-params", "data"),
    Input({"type": "fit-mu-button", "index": ALL}, "n_clicks"),
    Input("keyboard", "n_keydowns"),
    Input("keyboard", "keydown"),
    State({"type": "fit-mu-button", "index": ALL}, "id"),
    State("selected-fit-mu", "data"),
    State("stored-fit-params", "data"),
    prevent_initial_call=True
)
def handle_fit_mu_and_keyboard(fit_params, n_clicks_list, n_keydowns, key_event, ids, selected_mu, fit_params_state):
    trigger = ctx.triggered_id
    mode_value = dash.no_update
    undo_clicks = dash.no_update
    num_gauss = dash.no_update
    assign_clicks = dash.no_update
    next_mu = selected_mu

    # Click on a Î¼ button
    if isinstance(trigger, dict) and trigger.get("type") == "fit-mu-button":
        next_mu = trigger["index"]

    # After a fit, select the lowest Î¼ by default
    elif trigger == "stored-fit-params" and fit_params and "multi" in fit_params:
        fits = sorted(fit_params["multi"], key=lambda p: p["mu"])
        next_mu = round(fits[0]["mu"], 4)

    # Keyboard shortcuts
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
            assign_clicks = int(time.time())
        elif key in [str(n) for n in range(10)]:
            num_gauss = 10 if key == "0" else int(key)
        elif key == "d" and fit_params_state and "multi" in fit_params_state:
            fits_sorted = sorted(fit_params_state["multi"], key=lambda p: p["mu"])
            mu_list = [round(p["mu"], 4) for p in fits_sorted]

            if not mu_list:
                next_mu = selected_mu
            elif selected_mu not in mu_list:
                next_mu = mu_list[0]
            else:
                current_idx = mu_list.index(selected_mu)
                next_mu = mu_list[(current_idx + 1) % len(mu_list)]

    # Render Î¼ selection buttons
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

    return buttons, next_mu, mode_value, undo_clicks, num_gauss, assign_clicks


if __name__ == "__main__":
    app.run(debug=False, port=8050)

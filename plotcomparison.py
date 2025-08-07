import dash
from dash import dcc, html, dash_table, Output, Input, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import dash.exceptions
import os
import datetime
from collections import defaultdict
from dash import ALL
from dash import ctx
from dash import callback_context
import re
import time
from dash_extensions import Keyboard
import json

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)
qn_label_map = config.get("qn_labels", {})
cat_file_path = config["cat_file"]
csv_file_path = config["csv_file"]


# === Load Simulated Spectrum ===
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

                entry = {
                    "Freq": freq,
                    "Intensity": 10 ** logI,
                    "Eu": E_low
                }

                # Save max countQN for use elsewhere
                max_qns = max(max_qns, countQN)

                label_map = {0: "J", 1: "Ka", 2: "Kc"}

                upper_start = 55
                lower_start = 67

                for i in range(countQN):
                    qn_name = label_map.get(i, f"Q{i - 2}") if i > 2 else label_map[i]
                    uq_label = f"Upper{qn_name}"
                    lq_label = f"Lower{qn_name}"

                    if upper_start + 2 <= len(line):
                        entry[uq_label] = int(line[upper_start:upper_start+2].strip())
                    if lower_start + 2 <= len(line):
                        entry[lq_label] = int(line[lower_start:lower_start+2].strip())

                    upper_start += 2
                    lower_start += 2

                sim_data.append(entry)

            except Exception as e:
                print(f"⚠️ Skipping line: {e}")
                continue

    df = pd.DataFrame(sim_data)

    # Store list of QN columns in correct order
    qn_order = []
    for i in range(max_qns):
        name = {0: "J", 1: "Ka", 2: "Kc"}.get(i, f"Q{i-2}" if i > 2 else f"Q{i}")
        qn_order.append(f"Upper{name}")
    for i in range(max_qns):
        name = {0: "J", 1: "Ka", 2: "Kc"}.get(i, f"Q{i-2}" if i > 2 else f"Q{i}")
        qn_order.append(f"Lower{name}")

    return df, qn_order



sim_df, qn_field_order = parse_cat_file(cat_file_path)


def generate_hover(row):
    hover = f"<b>Freq:</b> {row['Freq']:.2f} MHz<br>" \
            f"<b>Intensity:</b> {row['Intensity']:.2e}<br>" \
            f"<b>Eu:</b> {row['Eu']:.2f} cm⁻¹<br>"

    upper_qns = [
        f"{qn_label_map.get(k, k)}={row[k]}"
        for k in row.index if k.startswith("Upper")
    ]
    lower_qns = [
        f"{qn_label_map.get(k, k)}={row[k]}"
        for k in row.index if k.startswith("Lower")
    ]

    hover += "<b>Upper:</b> " + " ".join(upper_qns) + "<br>"
    hover += "<b>Lower:</b> " + " ".join(lower_qns)
    return hover


sim_df["Norm_Intensity"] = sim_df["Intensity"] / sim_df["Intensity"].max()
sim_df["RoundedFreq"] = sim_df["Freq"].round(4)
sim_df["StickX"] = sim_df["Freq"].apply(lambda f: [f, f, None])
sim_df["StickY"] = sim_df["Norm_Intensity"].apply(lambda y: [0, y, None])
sim_df["Hover"] = sim_df.apply(generate_hover, axis=1)

def build_assignment_columns(sim_df, qn_field_order):
    static_columns = [
        {"name": "obs", "id": "obs", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "sim", "id": "sim", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "Eu", "id": "Eu", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "logI", "id": "logI", "type": "numeric", "format": {"specifier": ".2f"}},
    ]

    qn_columns = [
        {"name": qn_label_map.get(col, col), "id": col}
        for col in qn_field_order
    ]

    extra_columns = [
        {"name": "Uncertainty", "id": "Uncertainty", "type": "numeric", "format": {"specifier": ".4f"}},
        {"name": "Weight", "id": "Weight", "type": "numeric", "format": {"specifier": ".2f"}}
    ]

    return static_columns + qn_columns + extra_columns




# === Load Measured Spectrum ===
meas_df = pd.read_csv(csv_file_path, sep=";")
meas_freqs = meas_df.iloc[:, 0].values
meas_intensities = meas_df.iloc[:, 1].values
meas_intensities = meas_intensities / np.max(meas_intensities)

# === Fit Functions ===

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def multi_gauss_with_offset(x, *params):
    offset = params[-1]
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):  # Exclude last param (offset)
        y += gaussian(x, params[i], params[i+1], params[i+2])
    return y + offset

# === Dash App Setup ===
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Interactive Spectrum Assigner"),

    # ✅ Keyboard listener here
    Keyboard(
        id="keyboard",
        captureKeys=["q", "w", "e", "r", "a" , "d"] + [str(i) for i in range(1, 8)]
    ),

    html.Div([
        html.Label("Number of Gaussians to Fit:"),
        dcc.RadioItems(
            id='num-gaussians',
            options=[{'label': f'{n}', 'value': n} for n in range(1, 8)],
            value=1,
            labelStyle={'display': 'inline-block', 'marginRight': '10px'},
            inputStyle={'marginRight': '5px'},
            style={'width': '350px'}
        )
    ], style={'marginBottom': '20px'}),

    html.Div([
        dcc.Checklist(
            id="flip-sim-checkbox",
            options=[{"label": "Flip Simulated Spectrum", "value": "flip"}],
            value=[],
            style={"color": "white"}
        )
    ], style={"marginBottom": "20px"}),


    html.Div([
        html.Label("Choose Fitted Peak to Assign:"),
        html.Div(id="fit-mu-button-container", style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginTop": "5px"})
    ], style={"marginBottom": "20px"}),


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
   
    # === Zoom Controls ===
    html.Div([
        html.Button("Undo Zoom", id="undo-zoom-button", n_clicks=0, style={"marginRight": "10px"}),
        html.Button("X+ (Zoom In)", id="x-zoom-in", n_clicks=0, style={"marginRight": "5px"}),
        html.Button("X– (Zoom Out)", id="x-zoom-out", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.Label("Simulated Intensity Threshold (0–1):"),
        dcc.Input(id='intensity-threshold', type='number', min=0, max=1, step=0.01, value=0.01, debounce=True)
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Label("Load .int file from path:"),
        dcc.Input(id="int-file-path", type="text", placeholder="Enter path to .int file", style={"width": "70%"}),
        html.Button("Load .int File", id="load-int-button", n_clicks=0)
    ], style={"marginBottom": "20px"}),
    html.H4("Assignments (Click row to delete)"),
    html.Button("Save .lin file", id="save-lin-button", n_clicks=0, style={"marginTop": "20px"}),
    html.Div(id="save-lin-confirmation", style={"marginTop": "10px", "color": "green"}),
    html.Button("Recalculate Weights", id="recalc-weights-button", n_clicks=0, style={"marginTop": "10px"}),
    dash_table.DataTable(
        id='assignment-table',
        columns=build_assignment_columns(sim_df, qn_field_order),
        data=[],
        row_selectable="single",
        selected_rows=[],
        style_table={'width': '95%'},
        style_cell={'textAlign': 'center'}
    ),

    dcc.Store(id="selected-fit-mu"),
    dcc.Store(id="stored-fit-params"),
    dcc.Store(id="stored-assignments", data=[]),
    dcc.Store(id="filtered-sim-data"),
    dcc.Store(id="stored-region-selection"),
    dcc.Store(id="stored-zoom", data=None),
    dcc.Store(id="zoom-history", data=[]),

])

# === Callback: Fit Peaks ===
@app.callback(
    Output("fit-output", "children"),
    Output("stored-fit-params", "data"),
    Input("spectrum-plot", "selectedData"),
    State("num-gaussians", "value"),
    State("mode-selector", "value"),
    prevent_initial_call=True
)
def fit_peak(selection, num_gauss, mode):
    #t0 = time.perf_counter()
    if mode != "select":
        raise dash.exceptions.PreventUpdate

    if not selection or "range" not in selection:
        return dash.no_update, dash.no_update

    x0, x1 = selection["range"]["x"]
    mask = (meas_freqs >= x0) & (meas_freqs <= x1)
    if np.sum(mask) < 5:
        return "❌ Too few points to fit.", dash.no_update

    x, y = meas_freqs[mask], meas_intensities[mask]
    window_width = x1 - x0

    # === Estimate baseline from side regions ===
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

        initial_p0 = []
        bounds_lower = []
        bounds_upper = []

        for mu in mu_guesses:
            amp_guess = y_max - y_base
            sigma_guess = max(window_width / (3 * num_gauss), 0.01)
            initial_p0 += [amp_guess, mu, sigma_guess]
            bounds_lower += [0, x0, 0.01]
            bounds_upper += [1.5 * amp_guess, x1, window_width]

        # Add baseline as final parameter
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

            fits = [{"amp": popt[i], "mu": popt[i+1], "sigma": popt[i+2]} for i in range(0, len(popt) - 1, 3)]
            baseline = popt[-1]

            msg = "✅ Fitted peaks at: " + ", ".join(f"{p['mu']:.2f} MHz" for p in fits)
            msg += f"<br>Estimated baseline offset: {baseline:.4f} ± {y_base_std:.4f}"
            #print(f"⏱ fit peak in {time.perf_counter() - t0:.3f} s")
            return msg, {"multi": fits, "baseline": baseline, "baseline_range": [x0 - margin, x1 + margin]}

        
        
        except Exception as e:
            return f"❌ Fit failed: {str(e)}", dash.no_update



@app.callback(
    Output("stored-zoom", "data"),
    Output("zoom-history", "data"),
    Input("spectrum-plot", "relayoutData"),
    Input("undo-zoom-button", "n_clicks"),
    Input("x-zoom-in", "n_clicks"),
    Input("x-zoom-out", "n_clicks"),
    State("stored-zoom", "data"),
    State("zoom-history", "data"),
    prevent_initial_call=True
)
def handle_all_zoom_events(relayout, undo_clicks, zoom_in_clicks, zoom_out_clicks, current_zoom, history):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger = ctx.triggered_id
    history = history or []

    # === Handle graph zoom
    if trigger == "spectrum-plot" and relayout:
        # Case 1: Autoscale (e.g., double-click to reset zoom)
        if "autosize" in relayout or "xaxis.autorange" in relayout:
            if current_zoom:
                history.append(current_zoom)
            return None, history

        # Case 2: Manual zoom
        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")
        y0 = relayout.get("yaxis.range[0]")
        y1 = relayout.get("yaxis.range[1]")

        if x0 is not None and x1 is not None and y0 is not None and y1 is not None:
            new_zoom = {"x": [x0, x1], "y": [y0, y1]}
            if current_zoom:
                history.append(current_zoom)
            return new_zoom, history

    # === Handle undo zoom
    elif trigger == "undo-zoom-button":
        if not history:
            return None, []
        last_zoom = history[-1]
        return last_zoom, history[:-1]

    # === Handle X+ / X– zoom
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


    raise dash.exceptions.PreventUpdate




@app.callback(
    Output("filtered-sim-data", "data"),
    Input("intensity-threshold", "value")
)
def filter_simulated_lines(threshold):
    #t0 = time.perf_counter()

    df = sim_df[sim_df["Norm_Intensity"] >= (threshold or 0.01)].copy()
    df["RoundedFreq"] = df["Freq"].round(4)
    #print(f"⏱ filter simulated lines  {time.perf_counter() - t0:.3f} s")
    return df.to_dict("records")

@app.callback(
    Output("stored-region-selection", "data"),
    Input("spectrum-plot", "selectedData"),
    prevent_initial_call=True
)
def store_selection(selection):
    #t0 = time.perf_counter()
    if selection and "range" in selection and "x" in selection["range"]:
        return selection
    #print(f"⏱ store selection  {time.perf_counter() - t0:.3f} s")
    return dash.no_update

@app.callback(
    Output("stored-assignments", "data", allow_duplicate=True),
    Output("assignment-table", "data", allow_duplicate=True),
    Input("assign-button", "n_clicks"),
    State("stored-assignments", "data"),
    State("filtered-sim-data", "data"),
    State("mode-selector", "value"),
    State("stored-region-selection", "data"),
    State("selected-fit-mu", "data"),
    prevent_initial_call=True
)
def assign_by_region(n_clicks, current_assignments, filtered_sim_records, mode, selection, selected_mu):
    #t0 = time.perf_counter()
    if mode != "assign_all" or not selection or selected_mu is None:
        raise dash.exceptions.PreventUpdate

    x0, x1 = selection["range"]["x"]
    sim_df_filtered = pd.DataFrame(filtered_sim_records)
    in_range = sim_df_filtered[(sim_df_filtered["Freq"] >= x0) & (sim_df_filtered["Freq"] <= x1)]

    # Ensure current_assignments is a list
    current_assignments = current_assignments or []

    for _, row in in_range.iterrows():
        new_entry = {
            "obs": round(selected_mu, 4),
            "sim": round(row["Freq"], 4),
            "Eu": round(row["Eu"], 4),
            "logI": round(np.log10(row["Intensity"]), 4),
            "Uncertainty": 0.01
        }

        # Add QNs dynamically
        for key in qn_field_order:
            if key in row:
                new_entry[key] = row[key]
        if new_entry not in current_assignments:
            current_assignments.append(new_entry)

    # Compute weights without removing data
    obs_counts = pd.Series([entry["obs"] for entry in current_assignments]).value_counts().to_dict()
    for entry in current_assignments:
        entry["Weight"] = round(1 / obs_counts[entry["obs"]], 4)
    
    #print(f"⏱ assign by region {time.perf_counter() - t0:.3f} s")
    return current_assignments, current_assignments

@app.callback(
    Output("spectrum-plot", "figure"),
    Input("stored-fit-params", "data"),
    Input("stored-assignments", "data"),
    Input("stored-zoom", "data"),
    Input("mode-selector", "value"),
    Input("filtered-sim-data", "data"),
    Input("selected-fit-mu", "data"),
    Input("flip-sim-checkbox", "value")  # 👈 ADD THIS LINE
)
def update_plot(fit_params, assignments, zoom, mode, filtered_sim_records, selected_mu, flip_checkbox):
    #print(f"📦 filtered-sim-data size: {len(filtered_sim_records)}")
    #print(f"📦 assignment count: {len(assignments)}")
    #t0 = time.perf_counter()

    sim_filtered_df = pd.DataFrame(filtered_sim_records)

    # === Step 1: Filter by zoom once
    if zoom and "x" in zoom:
        sim_visible_df = sim_filtered_df[
            (sim_filtered_df["Freq"] >= zoom["x"][0]) & (sim_filtered_df["Freq"] <= zoom["x"][1])
        ].copy()
    else:
        sim_visible_df = sim_filtered_df.copy()

    # === Step 2: Match assignments efficiently
    assigned_freqs = {round(row["sim"], 4) for row in assignments} if assignments else set()
    sim_visible_df["is_assigned"] = sim_visible_df["RoundedFreq"].isin(assigned_freqs)


    assigned_lines = sim_visible_df[sim_visible_df["is_assigned"]]
    unassigned_lines = sim_visible_df[~sim_visible_df["is_assigned"]]

    # === Step 3: Generate fast stick traces
    def get_stick_trace(df, color, name):
        if df.empty:
            return go.Scatter(x=[], y=[])
        x = [v for sub in df["StickX"] for v in sub]
        y = [(-v if v is not None and "flip" in flip_checkbox else v) for sub in df["StickY"] for v in sub]
        hover = [val for h in df["Hover"] for val in (h, h, None)]
        return go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=color),
            name=name,
            hoverinfo="text",
            text=hover
        )


    fig = go.Figure([
        get_stick_trace(unassigned_lines, "red", "Simulated (unassigned)"),
        get_stick_trace(assigned_lines, "blue", "Simulated (assigned)"),
        go.Scatter(
            x=meas_freqs, y=meas_intensities,
            mode="lines",
            name="Measured",
            line=dict(color="black"),
            hoverinfo="skip"
        )
    ])

    # === Zoom settings
    if zoom and "x" in zoom:
        fig.update_xaxes(range=zoom["x"])
        fig.update_yaxes(range=zoom["y"])
    else:
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)

    fig.update_layout(
        dragmode="select" if mode in ["select", "assign_all"] else "zoom",
        template="simple_white",
        height=600,
        xaxis_title="Frequency (MHz)",
        yaxis_title="Normalized Intensity (Sim Flipped)" if "flip" in flip_checkbox else "Normalized Intensity",
        uirevision="zoom-lock",
        plot_bgcolor="#5A5A5A",
        paper_bgcolor="#5A5A5A",
        font_color='white',
        margin=dict(t=20, b=40, l=60, r=20)
    )

    # === Fit overlays
    if fit_params and "multi" in fit_params:
        baseline = fit_params.get("baseline", 0)
        for p in fit_params["multi"]:
            x_fit = np.linspace(p["mu"] - 4 * p["sigma"], p["mu"] + 4 * p["sigma"], 100)
            y_fit = gaussian(x_fit, p["amp"], p["mu"], p["sigma"]) + baseline
            fig.add_trace(go.Scatter(
                x=x_fit, y=y_fit, mode="lines",
                name=f"μ={p['mu']:.2f}",
                line=dict(color="green", dash="dot")
            ))

    # === Assignment vertical lines (combined trace, much faster than add_shape)
    if assignments:
        # sim_lines = go.Scatter(
        #     x=[pair["sim"] for pair in assignments for _ in range(3)],
        #     y=[0, 1, None] * len(assignments),
        #     mode="lines",
        #     line=dict(color="gray", dash="dash"),
        #     name="Assigned Sims",
        #     hoverinfo="skip",
        #     showlegend=False
        # )
        # fig.add_trace(sim_lines)

        # Optional: draw obs lines only if zoomed-in (avoids hundreds of shapes)
        if not zoom or (zoom["x"][1] - zoom["x"][0] < 100):  # threshold in MHz
            obs_lines = go.Scatter(
                x=[mu for mu in set(pair["obs"] for pair in assignments) for _ in range(3)],
                y=[0, 1.02, None] * len(set(pair["obs"] for pair in assignments)),
                mode="lines",
                line=dict(color="blue", dash="dash", width=1.5),
                opacity=0.7,
                name="Assigned Obs",
                hoverinfo="skip",
                showlegend=False
            )
            fig.add_trace(obs_lines)

    # === Baseline line
    if fit_params and "baseline" in fit_params and "baseline_range" in fit_params:
        base = fit_params["baseline"]
        x0, x1 = fit_params["baseline_range"]
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[base, base],
            mode="lines",
            name="Estimated Baseline",
            line=dict(color="orange", dash="dash")
        ))

    #print(f"⏱ update_plot completed in {time.perf_counter() - t0:.3f} s")
    return fig



@app.callback(
    Output("stored-assignments", "data", allow_duplicate=True),
    Output("assignment-table", "data", allow_duplicate=True),
    Input("load-int-button", "n_clicks"),
    State("int-file-path", "value"),
    prevent_initial_call=True
)
def load_lin_file_from_path(n_clicks, filepath):
    #t0 = time.perf_counter()


    print(f"🔁 Callback triggered. Path: {filepath}")

    if not filepath or not os.path.isfile(filepath) or not filepath.endswith(".lin"):
        print("❌ Invalid path or file doesn't exist or is not a .lin file")
        return dash.no_update, dash.no_update

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return dash.no_update, dash.no_update

    assignments = []

    for line in lines:
        try:
            qns_raw = line[:24]
            qns = [int(qns_raw[i:i+3]) for i in range(0, len(qns_raw), 3)]

            freq = float(line[49:61].strip())
            unc = float(line[61:71].strip())
            wt = float(line[71:77].strip())

            # Get dynamic QN field names
            qn_fields = qn_field_order[:len(qns)]


            qn_values = qns[:len(qn_fields)]

            # Match simulation line based on quantum numbers
            match_df = sim_df.copy()
            for field, value in zip(qn_fields, qn_values):
                match_df = match_df[match_df[field] == value]

            if match_df.empty:
                print(f"⚠️ No match in sim_df for QNs: {qns}")
                continue

            sim_row = match_df.iloc[0]

            assignment = {
                "obs": round(freq, 4),
                "sim": round(sim_row["Freq"], 4),
                "Eu": round(sim_row["Eu"], 4),
                "logI": round(np.log10(sim_row["Intensity"]), 4),
                "Uncertainty": round(unc, 4),
                "Weight": round(wt, 4)
            }

            # Add dynamic QNs to assignment
            for field, value in zip(qn_fields, qn_values):
                assignment[field] = value

            assignments.append(assignment)

        except Exception as e:
            print(f"⚠️ Failed to parse or match line: {line.strip()} — {e}")
            continue


    print(f"✅ Loaded {len(assignments)} assignments from .lin file")
    #print(f"⏱ lin file read in {time.perf_counter() - t0:.3f} s")
    return assignments, assignments


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
    triggered_id = callback_context.triggered_id

    if not current_data:
        raise dash.exceptions.PreventUpdate

    updated_data = current_data.copy()

    # ✅ Inline helper to recompute weights
    def recompute_weights(assignments):
        from collections import Counter
        obs_counts = Counter(entry["obs"] for entry in assignments)
        for entry in assignments:
            entry["Weight"] = round(1 / obs_counts[entry["obs"]], 4)
        return assignments

    # If triggered by row selection, delete selected row
    if triggered_id == "assignment-table":
        if not selected_rows:
            raise dash.exceptions.PreventUpdate
        updated_data = [row for i, row in enumerate(current_data) if i not in selected_rows]

    # Recompute weights after either deletion or recalc button press
    updated_data = recompute_weights(updated_data)

    return updated_data, updated_data, []


# === Save .lin File Callback ===
def generate_lin_file(assignments):
    #t0 = time.perf_counter()
    """
    Generate .lin file content in the custom format:
    QNs: UpperJ UpperKa UpperKc UpperF LowerJ LowerKa LowerKc LowerF
    FREQ(12.4f), Uncertainty(10.4f), Weight(6.2f)
    """


    # Group sim frequencies by observed freq to compute spread
    sim_by_obs = defaultdict(list)
    for row in assignments:
        sim_by_obs[row["obs"]].append(row["sim"])

    # Uncertainty = spread + 0.01
    uncertainty_by_obs = {
        obs: round(max(sims) - min(sims) + 0.01, 4)
        for obs, sims in sim_by_obs.items()
    }

    lines = []
    for row in assignments:
        freq = float(row["obs"])
        unc = uncertainty_by_obs.get(row["obs"], 0.0100)
        wt = float(row.get("Weight", 1.00))

        qn_values = [int(row[k]) for k in qn_field_order if k in row]
        qn_str = "".join(f"{q:3d}" for q in qn_values)



        # Format: all 8 QNs (3 digits each), spacing, then freq, uncertainty, weight
        line = (
            f"{qn_str:<24s}" +  # will adjust size dynamically
            f"{'':25s}" +
            f"{freq:12.4f}{unc:10.4f}{wt:6.2f}"
        )
        lines.append(line)
    #print(f"⏱ safe lin file in {time.perf_counter() - t0:.3f} s")
    return "\n".join(lines)



@app.callback(
    Output("save-lin-confirmation", "children"),
    Input("save-lin-button", "n_clicks"),
    State("stored-assignments", "data"),
    prevent_initial_call=True
)
def save_lin_file(n_clicks, assignments):
    if not assignments:
        return "❌ No assignments to save."

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    assignments_dir = os.path.join(os.getcwd(), "assignments")
    os.makedirs(assignments_dir, exist_ok=True)  # ✅ Create directory if it doesn't exist

    filename = f"assignments_{timestamp}.lin"
    filepath = os.path.join(assignments_dir, filename)

    content = generate_lin_file(assignments)
    with open(filepath, "w") as f:
        f.write(content + "\n")

    return f"✅ Saved: {filename}"

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

    # === Case 1: μ Button clicked
    if isinstance(trigger, dict) and trigger.get("type") == "fit-mu-button":
        next_mu = trigger["index"]

    # === Case 2: Fit completed (select lowest μ)
    elif trigger == "stored-fit-params" and fit_params and "multi" in fit_params:
        fits = sorted(fit_params["multi"], key=lambda p: p["mu"])
        next_mu = round(fits[0]["mu"], 4)

    # === Case 3: Keyboard input
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
        elif key in [str(n) for n in range(1, 8)]:
            num_gauss = int(key)
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

    # === Build μ Buttons
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

            buttons.append(html.Button(
                label,
                id={"type": "fit-mu-button", "index": mu},
                n_clicks=0,
                style=style
            ))

    return buttons, next_mu, mode_value, undo_clicks, num_gauss, assign_clicks


if __name__ == "__main__":
    app.run(debug=True, port=8050)
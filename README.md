# Hyperfine Interactive Spectrum Assigner — v7

An interactive local workspace for comparing measured rotational spectra with SPCAT catalogs, assigning transitions, and preparing SPFIT line lists.

**Release candidate: 7.0.0-rc.1.** This branch extends the original project by Lukas Berger. The original `plotcomparison.py`, dependency files, license, and public molecular examples are retained. Start `plotcomparison_2026_7.py` for v7.

**[中文安装指南](docs/INSTALL_zh-CN.md)** · **[English installation](docs/INSTALL.md)** · **[Changes](CHANGELOG.md)** · **[Upgrading / publishing](docs/RELEASING.md)** · **[Original instructions](docs/LEGACY_README.md)**

## First start

Use **64-bit Python 3.11** for the reference installation. Python 3.12 is included in the CI matrix; see [validation status](docs/VALIDATION.md) for actual results. Do not use Python 3.13+ with this release's older numerical pins.
Download the whole branch ZIP and extract it to a writable folder, or clone this branch. Do not download only the main Python file.

Windows PowerShell, from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-v7.txt
.\.venv\Scripts\python.exe plotcomparison_2026_7.py
```

macOS / Linux, from the project folder (install Python 3.11 first):

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-v7.txt
./.venv/bin/python plotcomparison_2026_7.py
```

Open **http://127.0.0.1:8053/**. Select the supplied public example or use **New Configuration** to choose your spectrum and CAT files. If the port is already in use, append `--port 8054` and open that port. Stop the server with Ctrl+C.

SPFIT/SPCAT are **optional for viewing and assigning existing CAT files**. To run them in Fitting Space, install the separate native executables using the [step-by-step guide](docs/INSTALL.md#configure-spfit-and-spcat).

## What v7 adds

- Named spectrum configurations, each with one measured spectrum and multiple catalogs; create/edit dialogs with a local file browser.
- An integrated Fitting Space: text editing, draft preservation, backups, SPFIT/SPCAT execution, console output, and explicit CAT refresh.
- Catalog refresh matches full quantum numbers, preserving missing/ambiguous assignments for review rather than associating them with unrelated row numbers.
- Per-configuration autosaves, multi-catalog workflows, Loomis–Wood inspection, and a consistent dark interface inherited from the local development line.
- Simulation intensity references the strongest simulated line **inside the full measured frequency range**. A switch restores measured intensity values and scales predictions and fitted curves with them.

No new pip package is required by the v7 support modules: they use the existing Dash/Plotly/NumPy/Pandas/SciPy stack, Flask/Werkzeug, and Python's standard library. `requirements-v7.txt` pins the development stack; the upstream `requirements.txt` and `requirements_old.txt` remain unchanged.

## Files to keep together

```text
plotcomparison_2026_7.py
spectrum_workspace_v7.py
spectrum_config_v7.py
spectrum_intensity_v7.py
assets/                         # keep the entire directory
requirements-v7.txt
config.json                     # supplied public example
molecules/                      # supplied public example data
```

The `spectrum_*_v7.py` files are project source modules, not pip packages. Personal `config-*.json`, `autosave/`, `assignments/`, and backup directories are local data and are ignored by Git. Do not commit edits to the tracked example `config.json` if they contain personal paths or experimental data.

## Scientific and operational scope

The application is a single-user local tool listening on `127.0.0.1`. Use one editing tab per spectrum. It is not a remotely hosted multi-user service. The command field accepts SPFIT/SPCAT commands; it is not a general shell. Editor and job backups are stored beside the active molecular files in `.assigner-history/`, so those directories must be writable.

Catalog intensities are visualization references, not an absolute intensity calibration. Review fitted parameters, RMS, uncertainties and rejected lines; a successful executable exit is not evidence that a scientific model is valid. See [limits and troubleshooting](docs/INSTALL.md#troubleshooting).

## Tests and attribution

```bash
python -B -m unittest test_intensity_v7 test_workspace_v7 test_portability_v7
```

Run with the virtual environment's Python. Default tests use synthetic files; real Pickett integration is opt-in. See [validation](docs/VALIDATION.md).

Original project: [berger-lukas/Hyperfine-Spectrum-Analyzer](https://github.com/berger-lukas/Hyperfine-Spectrum-Analyzer). The original MIT license and copyright notice are retained. SPFIT/SPCAT are separate programs by H. M. Pickett and are not bundled with this release.

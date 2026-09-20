# Install and configure v7

[中文](INSTALL_zh-CN.md) · [Back to README](../README.md)

## 1. Choose an installation

Use a writable folder in your user account, a current browser, and 64-bit
**Python 3.11**. Python 3.12 is a CI target, not a claim of completed manual
testing. See [validation](VALIDATION.md). Python 3.13+ needs a different numerical
dependency combination and is outside this release's reference setup.

Download the **entire v7 branch** with GitHub's Code → Download ZIP, then extract
it. Alternatively, after the branch has been pushed:

```bash
git clone --branch v7-release --single-branch https://github.com/berger-lukas/Hyperfine-Spectrum-Analyzer.git
cd Hyperfine-Spectrum-Analyzer
```

If v7 is published in a fork, substitute that fork's URL. The default branch of
the original repository may still contain the old version.

Do not transfer a Windows virtual environment to a Mac (or between computers).
Create a new environment on the target machine. Run all commands below from the
extracted project folder.

### Windows (PowerShell)

Install 64-bit Python 3.11 from [python.org](https://www.python.org/downloads/).
Select the Python launcher / PATH options in the installer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-v7.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe plotcomparison_2026_7.py
```

Using the interpreter directly avoids PowerShell activation-policy problems.

### macOS (Terminal)

Install Python 3.11 from [python.org](https://www.python.org/downloads/macos/).
On an Apple Silicon Mac use a native/universal Python installation. Do not mix
an Intel Python environment with ARM-only packages. Check the Python process:

```bash
python3.11 -c "import platform; print(platform.python_version(), platform.machine())"
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-v7.txt
./.venv/bin/python -m pip check
./.venv/bin/python plotcomparison_2026_7.py
```

### Linux (terminal)

Install Python 3.11 and its `venv` support using your distribution's package
manager (package names/availability depend on the distribution). Do not replace
the system Python or install this application with `sudo pip`.

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-v7.txt
./.venv/bin/python -m pip check
./.venv/bin/python plotcomparison_2026_7.py
```

### Open the application

Open http://127.0.0.1:8053/ manually; keep the terminal running. Ctrl+C stops the
server. Use `--port 8054` if 8053 is occupied. In VS Code select the new `.venv`
interpreter and run **`plotcomparison_2026_7.py`**, not `plotcomparison.py`.

## 2. Configure a spectrum

The supplied `config.json` uses public relative-path examples. It demonstrates
viewing/assignment, not necessarily a complete executable SPFIT input set.

1. Click **New Configuration**, enter a descriptive name.
2. Browse to the experimental spectrum. Its first two columns must be numeric
   frequency in **MHz** and intensity, with a header row. New configurations
   detect the delimiter automatically. Existing configurations retain their
   `csv_separator` setting (legacy default: semicolon).
3. Add one or more CAT files. Different catalogs may live in different folders.
4. Save and open. Use **Edit Configuration** to change these choices later.

Relative data paths in a JSON are interpreted from that JSON's directory. Exact
letter case matters on case-sensitive filesystems. Windows `E:/...` paths are
not portable to macOS/Linux: select the files again on the target computer.
CAT and experimental frequencies must already be in MHz; there is no automatic
unit conversion. Avoid headerless files: add column names before loading.

Each CAT selects its own working folder and basename. For `sample.cat`, the
matching files are `sample.par`, `sample.var`, `sample.int`, `sample.lin`, etc.
The executable may be installed elsewhere; do not copy it into every molecule
folder. Keep working directories writable so backups and outputs can be saved.

## Configure SPFIT and SPCAT

These are separate native programs, not Python modules. You can use the
Assigner with existing CAT files before installing them. The repository does
not redistribute executables or install them automatically.

### 3. Download the correct programs

The [University of Cologne SPIN installation page](https://spin.astro.uni-koeln.de/chapter/Prerequisites/)
offers platform-labeled downloads for both programs. Select **both** SPFIT and
SPCAT for your platform:

| Computer | Download label |
|---|---|
| Windows | Windows |
| Apple Silicon Mac (M series) | macOS (ARM) |
| Intel Mac | macOS (Intel) |
| Compatible Ubuntu system | Ubuntu |

On Mac, Apple menu → About This Mac identifies the chip. On Linux, check `uname
-m`; do not assume an Ubuntu binary will run on every Linux distribution or ARM
device. If a binary is incompatible, obtain the sources from the linked
installation page or [CDMS](https://cdms.astro.uni-koeln.de/classic/pickett) and
build on the target machine with a C compiler and `make`. In the extracted
source directory, the documented targets are `make spfit` and `make spcat`.

Additional spectroscopy resources: [Kisiel's program collection](http://info.ifpan.edu.pl/~kisiel/asym/asym.htm#pickett).
Check the operating-system label of any downloaded binary; Windows `.exe`
files do not run natively on macOS/Linux.

### 4. Store and test the executables

Windows example: place `spfit.exe` and `spcat.exe` in `C:\Tools\Pickett`.
In PowerShell, run each separately:

```powershell
& 'C:\Tools\Pickett\spfit.exe'
& 'C:\Tools\Pickett\spcat.exe'
```

macOS/Linux example: put files named `spfit` and `spcat` in `~/Tools/Pickett`,
then grant execute permission and run each separately:

```bash
chmod +x "$HOME/Tools/Pickett/spfit" "$HOME/Tools/Pickett/spcat"
"$HOME/Tools/Pickett/spfit"
"$HOME/Tools/Pickett/spcat"
```

For each program, the filename prompt confirms startup. Press Enter to exit
before testing the other. This checks startup only, not a scientific fit.
On macOS, if the OS blocks a downloaded application, verify its source and use
the system's Privacy & Security approval for that specific program. Do not
disable macOS protection globally. `Bad CPU type` / `Exec format error` means
the executable needs a compatible architecture/OS or a native rebuild.

### 5. Tell the Assigner where they are

Choose one method. Resolution order is:

1. `spfit_path` / `spcat_path` in the active configuration JSON.
2. `SPFIT_PATH` / `SPCAT_PATH` in the server process environment.
3. The executable found on `PATH`.

An invalid higher-priority path produces an error; it does **not** silently fall
back. Remove/update old JSON paths when moving to a new computer. These variable
names are specific to this app, not Pyckett's `PYCKETT_*` variables.

#### Option A — environment variables (shared by all configurations)

Windows: search **Edit environment variables for your account**. Add user
variables `SPFIT_PATH` and `SPCAT_PATH` with full file paths, including `.exe`.
Restart VS Code and all terminals so they inherit the variables. For just the
current PowerShell session:

```powershell
$env:SPFIT_PATH = 'C:\Tools\Pickett\spfit.exe'
$env:SPCAT_PATH = 'C:\Tools\Pickett\spcat.exe'
.\.venv\Scripts\python.exe plotcomparison_2026_7.py
```

macOS/Linux: set these in the terminal used to launch the app:

```bash
export SPFIT_PATH="$HOME/Tools/Pickett/spfit"
export SPCAT_PATH="$HOME/Tools/Pickett/spcat"
./.venv/bin/python plotcomparison_2026_7.py
```

To persist them, add the two `export` lines to the startup file of your shell
(e.g. `~/.zshrc` for interactive zsh or `~/.bashrc` for interactive bash) and
open a new terminal. Applications launched through Finder/desktop icons may
not inherit shell variables. Launch from that terminal or use JSON paths.

#### Option B — explicit paths in each JSON

There is no executable-path field in the current configuration dialog. Open
the corresponding `config*.json` in a text editor and add these top-level
fields, retaining existing spectrum/catalog settings and valid JSON commas:

Windows:

```json
"spfit_path": "C:/Tools/Pickett/spfit.exe",
"spcat_path": "C:/Tools/Pickett/spcat.exe"
```

macOS (replace `yourname`; on Linux usually use `/home/yourname/...`):

```json
"spfit_path": "/Users/yourname/Tools/Pickett/spfit",
"spcat_path": "/Users/yourname/Tools/Pickett/spcat"
```

These are JSON fragments, not complete configs. Paths containing spaces work.
This release also expands `~` and interprets relative JSON executable paths
from the configuration's directory. Absolute paths are easiest to troubleshoot.
Restart the server after changing installation settings.

#### Option C — PATH

Add the **directory** containing the programs to PATH. Unlike the two dedicated
variables, PATH holds directories, not an executable filename. Verify discovery
with `Get-Command spfit,spcat` in PowerShell or `command -v spfit spcat` in a
POSIX shell. Shell aliases/functions alone are not sufficient: Python must be
able to find the real executable.

### 6. First run inside Fitting Space

1. Choose the intended **active catalog** and enter **FITTING SPACE**.
2. Check the displayed working directory and basename. Save assignments using
   **Write Assignments to Working LIN** if you want them used in this fit. The
   main page's timestamped LIN export does not replace the working LIN.
3. Inspect/edit the matching files. SPFIT requires `.par` and `.lin`; SPCAT
   requires `.var` and `.int`. Save drafts before running.
4. Run **SPFIT** first. Read the console and `.fit`: requested lines, Bad Line,
   rejected lines, RMS, fitted parameters and uncertainties.
5. Run **SPCAT** after reviewing the fit. **SPFIT → SPCAT** runs in sequence,
   stopping on execution failure, Bad Line, or missing FIT COMPLETE. It cannot
   judge the scientific quality of your model.
6. Click **Refresh CAT** (or Refresh All CATs) and return with **Back to Assigner
   Space**. Running SPCAT alone does not refresh the displayed predictions.

The built-in command field accepts `spfit`, `spcat`, or either followed by the
active basename. It does not accept arbitrary shell commands or interactive
stdin. Inputs and existing outputs are backed up to `.assigner-history/` before
execution; `run.log` is stored there too. These programs can rewrite PAR/VAR and
output files. Always review the result before continuing scientific work.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Old UI or missing v7 buttons | Run `_7.py`, stop the old server, verify the port, then hard-refresh (Ctrl+F5 / Cmd+Shift+R). |
| `No module named spectrum_workspace_v7` | Extract/copy all three support modules beside `_7.py`. Do not pip-install them. |
| Missing Dash/NumPy/etc. | Install `requirements-v7.txt` with the same Python interpreter used to run the app. |
| Installation tries to compile NumPy/SciPy | Check Python version/architecture. Use reference Python 3.11; do not reuse a foreign venv. |
| `spfit not found` | Use full executable paths; restart the server/IDE after setting variables. |
| Missing file despite valid environment variable | A stale JSON path takes precedence. Remove/update it. |
| Permission denied / WinError 5 | Check execute permission and write access to the molecular directory, including `.assigner-history`. Use a writable working copy; do not bypass backups. |
| Bad CPU type / Exec format / WinError 193 | Wrong OS/architecture or a downloaded HTML page instead of a binary; obtain the appropriate native program. |
| Missing `.var`, `.int`, `.par`, `.lin` | Match the active CAT directory and exact basename, including letter case. |
| Config cannot load | Verify all spectrum/CAT paths on this computer, numeric columns, delimiter, and valid CAT format. |
| Save refuses a file changed on disk | Preserve your draft separately; reload the disk version and reconcile the changes. |
| New CAT is not visible | Use Refresh CAT after SPCAT. A malformed/empty CAT is rejected and the previous display retained. |
| Refresh reports missing/ambiguous transitions | Review these assignments; export to working LIN is blocked until resolved. |
| Address already in use | Stop the previous server or choose another `--port`. |

### Known limits

- Header row expected; first two spectrum columns numeric, frequencies in MHz.
- The inherited CAT parser handles numeric fixed-width quantum-number fields;
  extended letter encodings are not promised. Do not silently discard parse errors.
- Working LIN import is intended for this app's exported format, not every
  externally produced LIN variant.
- Original-intensity mode restores stored numbers; it does not infer physical
  calibration. `intensity_unit` in JSON supplies an optional display label.
- Simulation normalization is independent per CAT, over the full measured
  frequency band. A catalog with no positive in-band line displays zero simulated
  intensity. Zooming does not change the reference. Loomis–Wood retains its
  own per-strip normalization.
- Stop may leave incomplete external outputs; restore from backups if needed.
- Config changes reload on next navigation. External changes to measured CSV
  require restarting the server; external CAT changes use Refresh CAT.
- One editing tab per spectrum; no remote multi-user deployment.

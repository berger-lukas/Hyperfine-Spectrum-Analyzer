# Changelog

## 7.0.0-rc.1 — release candidate

This candidate packages the local v7 development line for review against the
upstream `main` baseline `2c59ff9e5ce9988898fc29a9d5bda1a39c9824c8`.
The publication pass adds no new user-facing feature. It makes the existing
implementation distributable and improves portability, tests and documentation.

### Workspace and configuration

- Named configurations connect one measured spectrum with multiple catalogs.
  Create/edit dialogs include local browsing and expandable CAT lists.
- Configurations load on demand into separate Dash workspaces. Assignments,
  scale and view state are isolated per configuration; CAT reordering follows
  paths rather than array positions.
- Existing JSON scientific settings are preserved during GUI editing. Saves
  check file revisions and retain backups.
- The original `plotcomparison.py` remains available; v7 has its own entry point
  and three required local support modules.

### Fitting workflow

- Fitting Space edits PAR, VAR, INT, LIN, FIT and OUT files, retaining unsaved
  drafts across tabs and checking for external changes before saving.
- SPFIT, SPCAT and sequential SPFIT → SPCAT run in the active CAT directory,
  with console output, explicit status, cancellation and timestamped backups.
- Explicit working-LIN write/import connects assignments to the fitting files.
  Ordinary timestamped LIN exports remain separate.
- CAT refresh remaps by full upper/lower quantum numbers. Unmatched or ambiguous
  assignments remain visible for review and cannot silently become different
  transitions. Malformed refreshes retain the previous catalog.

### Display and inherited local improvements

- Consistent dark styling across configuration selection, Assigner, Fitting
  Space and dialogs, preserving the principal layout.
- Includes the local development line's Loomis–Wood and assignment refinements;
  not all differences from upstream originated in v7 itself.
- Per-catalog simulation normalization uses the strongest positive simulated
  line within the **full measured frequency range**, not the whole CAT or the
  current viewport. No-overlap catalogs do not use distant lines as references.
- Optional original-intensity display restores measured values and scales
  predictions, Gaussian fits and plot markers consistently. Default axis title
  is `Intensity`; units can be supplied through `intensity_unit`.
- Internal fitting arrays and uncertainty calculations retain their existing
  normalization. Loomis–Wood retains per-strip normalization.

### Publication fixes and packaging

- Resolve relative JSON executable paths from the configuration directory;
  expand home/environment notation and explain missing overrides/execute bits.
- Preserve configured delimiters when editing the same spectrum, including
  relative-path configs. Do not collapse case-distinct CAT selections in JS.
- Preserve the original runner error even when writing the error log fails.
  New text files use the host newline convention; existing files retain theirs.
- Portable synthetic regression data replace required personal molecular files.
  Real Pickett tests are opt-in and run on copied files only.
- Add `requirements-v7.txt` without changing upstream dependency files:
  NumPy 2.0.2 and SciPy 1.13.1 match the development environment; upstream's main
  requirements instead pin NumPy 2.2 and SciPy 1.15. Other listed pins match.
- Add Windows/macOS/Linux CI on Python 3.11/3.12, bilingual installation and
  external-program setup, migration, release guidance, and validation status.
- Exclude personal configs, autosaves, assignments, test outputs, environments
  and backup histories from the distributable source package.

### Before upgrading

Back up configs, autosaves and all scientific inputs/outputs. Download the whole
branch, recreate the environment, and run `_7.py`. Copy personal configurations
as `config-<name>.json`; check data and executable paths. Do not rename `_7.py`
over the upstream file while omitting support modules. See [migration](docs/RELEASING.md).

### Validation and remaining limits

See [the validation record](docs/VALIDATION.md): Windows/macOS/Linux × Python
3.11/3.12 all passed the isolated CI suite. Keep the release candidate designation
until native external-program and manual platform smoke tests are complete.
Known parsing, calibration and single-user limits are listed in the installation
guide. No new physical model or molecular constants are supplied by this release.

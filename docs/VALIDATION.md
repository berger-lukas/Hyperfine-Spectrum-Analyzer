# Validation record — 7.0.0-rc.1

Publication preparation: 2026-09-19. Upstream baseline:
`2c59ff9e5ce9988898fc29a9d5bda1a39c9824c8`.

## Completed locally

| Check | Result |
|---|---|
| Fresh Windows x64 Python 3.11.5 venv | `requirements-v7.txt` installed successfully |
| Dependency metadata | `python -m pip check`: no broken requirements |
| Portable default suite | 22 tests discovered; 20 passed, 2 intentionally skipped on Windows |
| Optional real SPFIT → SPCAT integration | Passed on copies; original input hashes unchanged |
| CLI startup | `plotcomparison_2026_7.py --help` passed |
| Public upstream example | Loaded both supplied catalogs and the measured spectrum |
| Browser smoke check | Configuration page, main plot, Fitting Space and Edit Configuration opened; dark styling present |
| Exported source ZIP | Extracted independently; same 22-test suite passed (2 Windows skips) |
| Original content | Original entry point, dependency files, license and public example files retained |

The two default skips are the POSIX execute-permission test on Windows and the
opt-in real-Pickett test. The latter was run separately with installed native
programs. Default tests require neither personal configs nor Pickett binaries.

Coverage includes intensity references and raw display/zoom, layout/callbacks,
editor drafts/conflicts/backups, CAT refresh/remapping/rollback, missing
transitions, LIN round trips, configuration creation/editing/conflicts,
delimiter preservation, path precedence, missing executables and log failures.

## Real executable check (private data not distributed)

A local user-supplied case was copied to the ignored test-output directory.
No scientific changes were made to its parameters or source inputs. The test
verified input hashes after the run. The result is a workflow regression, not
a new fit or a validation of the underlying assignments.

| Measure | Result |
|---|---:|
| LIN nonblank records | 916 |
| SPFIT LINES REQUESTED | 916 |
| Bad Line occurrences in FIT | 0 |
| Rejected lines | 10 |
| MICROWAVE RMS | 0.013638 MHz |
| RMS ERROR | 0.70931 |
| FIT COMPLETE | Present |
| SPFIT exit code | 0 |
| SPCAT rerun / exit code | Yes / 0 |

Scientific files and detailed logs are deliberately excluded from the public
package. Maintainers can reproduce the workflow with their own approved test
case as described below.

## Hosted cross-platform checks — passed

[GitHub Actions run 35487356545](https://github.com/berger-lukas/Hyperfine-Spectrum-Analyzer/actions/runs/35487356545)
passed all six jobs for code/CI commit
`7bca7403b1c639f78f6dfcc638bbc92dde072771`:

| Hosted runner | Python 3.11 | Python 3.12 |
|---|---|---|
| Windows | Passed | Passed |
| macOS | Passed | Passed |
| Ubuntu Linux | Passed | Passed |

Every job installed the release dependencies in a clean venv, ran `pip check`,
the portable suite and CLI startup. POSIX jobs also exercised execute-permission
validation. The opt-in external-binary test remained skipped on hosted runners.
These results cover the specific hosted images, not every OS version or chip.

The first hosted matrix exposed unrelated preinstalled `pipx` metadata in the
Windows 3.12 runner (`packaging>=26` versus the application's pinned 25.0).
CI now creates a clean virtual environment on every platform, matching the
documented installation instead of sharing the hosted system environment.

## Not yet claimed

- No local macOS or Linux machine was available for execution in this pass.
- The six hosted jobs above passed; that does not replace manual UI testing or
  real Pickett runs on macOS/Linux hardware.
- Native external-program integration on Mac/Linux still requires a platform
  smoke test. Hosted CI does not install or redistribute Pickett executables.
- Python 3.13+ and unlisted architectures are outside the reference setup.
- The publication changes do not certify all inherited scientific algorithms
  or all externally generated CAT/LIN encodings.

## Reproduce

From a fresh reference environment:

```bash
python -m pip install -r requirements-v7.txt
python -m pip check
python -B -m unittest test_intensity_v7 test_workspace_v7 test_portability_v7
python -B plotcomparison_2026_7.py --help
```

Use the environment's interpreter path in place of `python` if it is not
activated. All synthetic artifacts are written to ignored `.v7-test-output/`
and workspace state directories. Tests retain artifacts for diagnosis.

For a **separate opt-in real test**, configure SPFIT/SPCAT through environment
variables or PATH, then set `ASSIGNER_TEST_CAT` to a CAT with matching PAR, VAR,
INT and LIN files. The test copies the four inputs before running:

PowerShell:

```powershell
$env:ASSIGNER_TEST_CAT = 'C:/your-working-data/sample.cat'
.\.venv\Scripts\python.exe -B -m unittest test_workspace_v7.WorkspaceTests.test_07_real_pickett_in_copied_directory
```

POSIX shell:

```bash
ASSIGNER_TEST_CAT=/path/to/sample.cat ./.venv/bin/python -B -m unittest test_workspace_v7.WorkspaceTests.test_07_real_pickett_in_copied_directory
```

The optional test uses a 120-second limit. Logs and copied outputs remain under
`.v7-test-output/regression-*/`; they are not part of the source archive.

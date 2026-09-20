# Hyperfine Interactive Spectrum Assigner v7

Release candidate: **7.0.0-rc.1**. Start `plotcomparison_2026_7.py`.
The original `plotcomparison.py` is retained. The local `_6.py` development
version is not included in this release and has not been overwritten.

- [Installation and SPFIT/SPCAT configuration](docs/INSTALL.md)
- [Detailed release notes](CHANGELOG.md)
- [Migration, GitHub branches and release guidance](docs/RELEASING.md)
- [Test results and platform validation limits](docs/VALIDATION.md)

Download the complete project and install `requirements-v7.txt`; do not copy
only the main script. Python 3.11 is the reference version. Keep all three
`spectrum_*_v7.py` support modules and the entire `assets/` folder alongside
the entry point. Personal configurations, assignments, autosaves and private
molecular working files are excluded from the release archive.

Open http://127.0.0.1:8053/. If the port is occupied, start with `--port 8054`.
This publication pass packages, fixes and validates existing functionality;
it does not add a new executable-path settings dialog.

# Plot Comparison Tool

This project provides a Python-based tool for comparing a measured spectrum to a .cat, assign transitions and export a .lin file..  
It uses a configuration file for inputs and an external CSS file for visual theming.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/berger-lukas/Hyperfine-Spectrum-Analyzer
```
and navigate into the new created folder Hyperfine-Spectrum-Analyzer

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
# OR
venv\Scripts\activate         # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Project Structure

```
Hyperfine-Spectrum-Analyzer/
│
├── plot comparison.py       # Main script for running plot comparison
├── config.json              # Configuration file (input)
├── requirements.txt         # Dependencies needed to run the project
├── assets/
│   └── dark.css             # Optional CSS styling for visual output
├── molecules/
│   └── acenaphthene              
|          └──  acenaphthene.cat             # test .cat file to try
|          └── acenaphthene_measurement.csv             # test .csv file to try
│   └── ethanol              
|          └── ethanol.cat             # test .cat file to try
└── .venv/                   # Local virtual environment (excluded from Git)
```

---

## Configuration (`config.json`)

The `config.json` file contains runtime parameters like:

- File paths for input data

Make sure to update this file as needed before running the script.

---

## Styling (`assets/dark.css`)

This CSS file defines the theme for any HTML or GUI-based visual output.  
You can customize colors, fonts, or layout by editing `assets/dark.css`.

---

## Running the Script

> Ensure your virtual environment is activated before running the script.
(go into the Hyperfine-Spectrum-Analyzer and "run source venv/bin/activate")
After setting up the environment and configuration, run the tool with:

```bash
python plotcomparison.py
```

>
> Open your browser at http://127.0.0.1:8053

---


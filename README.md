# Plot Comparison Tool

This project provides a Python-based tool for generating and comparing visual plots.  
It uses a configuration file for inputs and an external CSS file for visual theming.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/berger-lukas/Hyperfine-Spectrum-Analyzer
```

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

## 🧾 Project Structure

```
plot-comparison/
│
├── plot comparison.py       # Main script for running plot comparison
├── config.json              # Configuration file (input/output settings, etc.)
├── requirements.txt         # Dependencies needed to run the project
├── assets/
│   └── dark.css             # Optional CSS styling for visual output
└── .venv/                   # Local virtual environment (excluded from Git)
```

---

## ⚙️ Configuration (`config.json`)

The `config.json` file contains runtime parameters like:

- File paths for input data
- Output plot preferences
- Any comparison logic or flags used by the script

Make sure to update this file as needed before running the script.

---

## 🎨 Styling (`assets/dark.css`)

This CSS file defines the theme for any HTML or GUI-based visual output.  
You can customize colors, fonts, or layout by editing `assets/dark.css`.

---

## ▶️ Running the Script

After setting up the environment and configuration, run the tool with:

```bash
python "plot comparison.py"
```

> Ensure your virtual environment is activated before running the script.

---

## 📄 License

> A license will be added before public release.  
> Until then, please do not reuse, redistribute, or share this project outside the approved team.

---

## 👥 Collaborators

This is a private, in-progress project intended for internal collaboration.  
Contact the repository owner for access or questions.

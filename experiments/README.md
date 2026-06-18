# UAIbotPy – SSDF Branch Experiment

This repository contains the code for the paper experiments using a CBF based control with our proposed distance.

---

## Prerequisites

- **Python** version `>=3.10` and `<=3.14`.
- **Git** (to clone the repository).
- **CMake** (required for building the pybind11 bindings).
- A C++ compiler (e.g., GCC, Clang, or MSVC) to compile the extensions.

---

## 1. Clone the Repository (Specific Branch)

Clone the `feat/ssdf` branch directly:

```bash
git clone --branch feat/ssdf https://github.com/UAIbot/UAIbotPy.git
cd UAIbotPy
```

---

## 2. Set Up a Python Virtual Environment (Recommended)

Create and activate a virtual environment with the required Python version:

```bash
# Create environment (replace with your preferred tool)
python3 -m venv venv
source venv/bin/activate   # On Linux/macOS
# or
venv\Scripts\activate      # On Windows
```

---

## 3. Install the Package and Dependencies

Install the package in the current directory. This will trigger CMake to build the C++ extensions via pybind11:

```bash
pip install .
```

**Note:** Make sure you are in the root directory (`UAIbotPy/`) when running this command. The build process requires CMake and a C++ compiler.

---

## 4. Install Additional Dependencies

The experiment script requires `matplotlib` for plotting. Install it manually:

```bash
pip install matplotlib
```

(Other dependencies are automatically installed via `pip install .`.)

---

## 5. Run the Experiment

**Important:** Always run the experiment script from the `experiments/` directory **without** adding the root folder to `sys.path`. This avoids accidentally importing the local `uaibot` source folder instead of the installed package (which contains the compiled `.so` bindings).

```bash
cd experiments
python paper_experiment.py
```

---

## Experiment Modes

Inside `paper_experiment.py`, you can specify the distance function to use by passing a command‑line argument:

| Mode | Description |
|------|-------------|
| `0`  | Use **Euclidean distance** (baseline). |
| `1`  | Use **our proposed distance function** (HD-SDF). |


---

## Outputs

After execution, the script will:

- Generate an interactive **HTML simulation** of the robot motion (saved in the `experiments/` folder).
- Produce a **plot** of the control input applied to the last joint.

---

## Troubleshooting

- **"cpp not found" / import error**: This usually means the C++ extension was not built. Ensure you have CMake and a compiler installed, and run `pip install .` again. Also, verify that you are not in the root directory while running the script.
- **Python version mismatch**: Make sure your environment uses Python 3.10–3.14. Check with `python --version`.
- **Missing matplotlib**: Run `pip install matplotlib` separately.

---

## Additional Notes

- The package uses **pybind11** for C++ bindings; the build process is automatic via `pip install .`.
- For any issues, feel free to contact by e-mail.

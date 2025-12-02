# Computer Graphics Lab (Lab2)

Short instructions to prepare the environment and run `Lab2.py` on Windows.

## ✅ Quick setup (Windows)

1. Install a supported Python (3.8+ recommended). Verify with:

```bash
python --version
```

2. Make sure your GPU drivers are installed and up to date. OpenGL comes from the graphics driver (NVIDIA / AMD / Intel). If you're not sure, use the vendor updater / "OpenGL Extensions Viewer" to check the supported OpenGL version.

3. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate   # on Windows (bash shell)
```

4. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Tip: if `PyOpenGL_accelerate` fails to build with pip, you can still run without it — it's optional but faster. You can also try installing a wheel from Christoph Gohlke's repository if necessary.

## ▶️ Run the project

```bash
python Lab2.py
```

### Table model (replaces cubes)

This lab now renders a simple 3D table in the scene (a tabletop + four legs) instead of the previous cubes. The table sits on the grid floor (the floor is at y = -1.0) and is centered at the origin.

Default table parameters used in the code:

- Width: 2.0 units
- Depth: 1.0 units
- Top thickness: 0.1 units
- Table top height above the floor: 1.0 units

You can edit `create_table_data()` in `Lab2.py` to change those dimensions or placement.

If the program opens a window and prints camera controls, the application started successfully.

## Troubleshooting

- If `import OpenGL` or `import glfw` fails: ensure you're installing into the same Python interpreter that you run the program with. Use `which python` / `python -m pip install ...`.
- If the program starts but you see a black screen or crashes, check your GPU drivers and ensure your system supports OpenGL 3.3 (the shaders use `#version 330 core`).

## 📋 Lab2 code summary

A concise summary table of the `Lab2.py` file (components, functions, purpose) has been created as `Lab2_table.md` — see that file for a quick reference.

You can view it directly with:

```bash
# on Windows (bash)
cat "Lab2_table.md"
```

### Lab report (PDF)

An official Lab 2 report PDF is included in the workspace:

```bash
# view or open
ls -l Lab2_report.pdf
```

Open `Lab2_report.pdf` for a full lab writeup (now includes Objective, Introduction, Environment, Implementation, Output and Conclusion sections).

If a Word document target was locked when the `.docx` update was run, you may also find a fallback file named `Lab2_report_new.docx` alongside `Lab2_report.docx`.

Additionally a Word document copy of the lab report is included:

```bash
# view or open
ls -l Lab2_report.docx
```

Open `Lab2_report.docx` in Word or a compatible editor.

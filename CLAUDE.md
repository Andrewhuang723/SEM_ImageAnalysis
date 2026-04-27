# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive web dashboard for analyzing Scanning Electron Microscopy (SEM) images. Users upload images, apply threshold-based segmentation, and extract particle/pore morphology metrics (size distributions, porosity, diameters).

## Deployment (Render)

The app is configured for [Render](https://render.com) via `render.yaml`. Push to GitHub, connect the repo in Render, and it will use:

- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn dashboard:server --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`

`server = app.server` at module level is the WSGI entry point gunicorn imports. Single worker is required because the app uses module-level global state (`image_data`, `analysis_log`, etc.) that is not shared across processes.

## Running the Application (Local)

```bash
python dashboard.py
```

Starts a Dash development server at `http://localhost:5050` (debug mode on). For WSGI production deployment, the app exposes `server = app.server`.

The Jupyter notebook (`Tutorial copy.ipynb`) shows post-analysis data workflows:

```bash
jupyter notebook "Tutorial copy.ipynb"
```

## Dependencies

No `requirements.txt` exists. Key packages (install via conda env `image` or pip):

- `dash`, `dash-bootstrap-components` — web UI
- `pillow`, `scikit-image`, `scipy` — image processing
- `pandas`, `numpy`, `plotly` — data and visualization

## Architecture

All application logic lives in `dashboard.py`. There are no submodules.

**Image processing pipeline (in order):**

1. `parse_contents()` — decode base64 upload → numpy array
2. `crop_image()` — optional rectangular crop (preserves `original_image_data` untouched)
3. `create_binary_image()` — threshold binarization; supports single or dual threshold (pixels inside range → dark)
4. Analysis splits into two modes:
   - **Pore mode** — `analyze_regions()`: connected-component labeling with morphological cleanup (disk r=3, min region 50px)
   - **Particle mode** — `anaylyze_particle_regions()`: watershed segmentation with distance transform (separation threshold 5px)
5. Metrics: porosity, region count, area stats, D10/D50/D90 diameters (with µm/pixel scale factor)
6. `handle_analysis_log()` — appends results to in-memory `analysis_log` list; CSV export

**Global mutable state (module-level):**

| Variable | Role |
|---|---|
| `original_image_data` | Raw upload, never modified |
| `image_data` | Working copy (may be cropped) |
| `binary_data` | Thresholded boolean array |
| `labeled_data` | Connected-component integer labels |
| `regions_data` | scikit-image `regionprops` output |
| `analysis_log` | In-memory list of result dicts (lost on restart) |

**UI layout (Bootstrap tabs):**
- Tab 1 — Analysis: upload, crop, threshold sliders, area histogram, analysis log table with CSV export
- Tab 2 — Diameter Distribution: particle diameter histogram (D10/D50/D90)

**Dash callbacks wire the UI to processing functions** — each callback reads/writes the global state variables above rather than passing data through Dash `dcc.Store`.

## Known Issues

- Function `anaylyze_particle_regions` has a typo in its name (missing 'l').
- `analysis_log` is not persisted; all records are lost when the server restarts.
- Hardcoded processing parameters (min region size: 50px, watershed threshold: 5px, morphological disk: r=3) — no config file.

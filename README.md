# Tissue section geometry explorer

Interactive 3D tool for a tissue block, a thin target region near the top face, and a cutting section. You can rotate the section, move its pivot, change sizes, and see what fraction of the section is tissue, target, or empty space.

The section is a cuboid. It stays centred on a pivot you choose. Rotations are applied about that pivot.

## Requirements

- Python 3.10 or newer
- The packages in `requirements.txt` (NumPy, SciPy, Plotly, Dash)

## Run locally

In a terminal, from this folder:

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

Leave the terminal open while you use the page. Closing it stops the app.

If `python` is not found, try `py` or `python3` instead.

### Optional: check the geometry

```bash
python test_geometry.py
```

## Default scene

| Object | Size (mm) | Placement |
| --- | --- | --- |
| Tissue block | 3 × 3 × 3 | Centred at the origin |
| Target region | 3 × 3 × 0.2 | Parallel to the top face, 0.1–0.3 mm below the top |
| Section | 6 × 6 × 0.1 | Centred on the pivot; thickness is local Z |

Coordinates: **+Z points toward the top of the block**. The cube centre is `(0, 0, 0)`.

- Angles `0°, 0°, 0°`: section parallel to the top face.
- `Rx = 90°` or `Ry = 90°`: perpendicular (vertical) cut.
- Pivot `(0, 0, 0)`: rotation about the cube centre.
- **Pivot on target** moves the pivot to the centre of the target slab.

## Volume percentages

Reported values are fractions of the **section cuboid’s own volume**:

- **Tissue**: section ∩ tissue block
- **Target region**: section ∩ target (the target lies inside the tissue)
- **Nothingness**: part of the section that lies outside the tissue

Tissue + nothingness = 100%. The display also shows how much of the target cuboid is captured.

## Project files

| File | Role |
| --- | --- |
| `app.py` | Browser interface (Dash + Plotly) |
| `geometry.py` | Rotations and intersection volumes |
| `test_geometry.py` | Basic volume checks |
| `requirements.txt` | Python dependencies |

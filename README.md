# Tissue section geometry explorer

Interactive 3D tool for a tissue block, a thin target region near the top face, and a vibratome-style cutting section.

A **section** is a knife slab of the chosen thickness, extended in the cutting plane until it leaves the tissue. That is what a vibratome actually yields: you keep the tissue you cut off, not empty space around the block. Display X and Y only draw a cuboid in the 3D view; they are **not** used in the volumes.

You can rotate the knife, move the mid-plane (pivot), and step to the previous or next **consecutive** cut. Consecutive cuts use the same orientation and move along the **knife normal** by one thickness.

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
| Knife thickness | 0.1 | Mid-plane at the pivot; the slab is infinite in-plane for volume |
| Display cuboid | 6 × 6 × 0.1 | Drawn around the pivot; visualization only |

Coordinates: **+Z points toward the top of the block**. The cube centre is `(0, 0, 0)`.

- Angles `0°, 0°, 0°`: cut parallel to the top face.
- `Rx = 90°` or `Ry = 90°`: perpendicular (vertical) cut.
- Pivot `(0, 0, 0)`: mid-plane through the cube centre.
- **Pivot on target** moves the mid-plane to the centre of the target slab.
- **Previous / Next cut** steps the mid-plane by one knife thickness along the knife normal.

## Volume percentages

The denominator is the **tissue spanned by the knife slab**, not the 6 × 6 × 0.1 display cuboid.

- **Tissue**: 100% of this cut (the section *is* that tissue)
- **Target region**: (slab ∩ target) / (slab ∩ cube), also shown as absolute mm³
- **Nothingness**: 0% of this cut (empty space outside the block is not collected)

The table of consecutive cuts lists absolute **target mm³** in each serial slab at the current orientation. That is the quantity to compare parallel vs perpendicular stacks.

Changing display X or Y must not change any of these numbers.

## Project files

| File | Role |
| --- | --- |
| `app.py` | Browser interface (Dash + Plotly) |
| `geometry.py` | Knife slab, rotations, intersection volumes, consecutive stack |
| `test_geometry.py` | Volume and consecutive-step checks |
| `argument_volumes.py` | Sweep of target volume vs tilt (parallel vs perpendicular argument) |
| `requirements.txt` | Python dependencies |

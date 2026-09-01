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

### Top surface

The side walls and bottom of the block stay axis-aligned. Only the **top** can change:

| Top surface | What moves |
| --- | --- |
| Flat | Original cube |
| Spherical convex | Centre of the top rises above the original top plane (a dome). The four top **corners stay at the original z**. |
| Spherical concave | Centre sinks below that plane (a bowl). Corners stay at the original z. |
| Cylindrical convex / concave | Same, but the surface bends in only one horizontal direction (choose the cylinder axis). |

**Apex / nadir offset** is how far that centre point sits above (convex) or below (concave) the original cube top. At offset 0 the block is the original cube.

### Target region

- **Cuboid**: the original box, parallel to XY, depth measured from the **original flat top**. A concave top can clip it (the dent eats into the box). A convex cap sits above it and does not change the cuboid.
- **Follow the top surface**: the target is a **parallel shell**. Depth and thickness are distances **along the surface normal** (inward from the top), not along Z. The shell therefore has constant normal thickness; its vertical thickness is larger where the top is steeper, and for a dome the shell sits on a slightly smaller concentric surface. Volume is no longer simply (XY area) × thickness.

The knife slab is unchanged: same thickness, still infinite in-plane, still clipped to whatever tissue solid you chose.

### Consecutive cuts on a curved top

Cuts still step along the knife normal by one thickness. The range grows if a convex cap makes the block taller.

On a **flat** block, parallel cuts miss the target until they hit the layer, then spike. If the target **follows** a curved top, a horizontal knife meets that parallel shell at different z in the middle and at the edges, so the same target is spread over **more consecutive cuts** and the peak mm³ in any one cut falls. Perpendicular stacks stay a thin strip; curvature mainly warps that strip in Z.

Curved volumes are integrated as XY columns (exact in z, numerical in x and y). A flat cuboid still uses the exact box–slab clip.

- Angles `0°, 0°, 0°`: cut parallel to the original top face.
- `Rx = 90°` or `Ry = 90°`: perpendicular (vertical) cut.
- Pivot `(0, 0, 0)`: mid-plane through the cube centre.
- **Pivot on target** moves the mid-plane to the centre of the target slab.
- **Previous / Next cut** steps the mid-plane by one knife thickness along the knife normal.

## Volume percentages

The denominator is the **tissue spanned by the knife slab**, not the 6 × 6 × 0.1 display cuboid.

- **Tissue**: 100% of this cut (the section *is* that tissue)
- **Target region**: (slab ∩ target) / (slab ∩ cube), also shown as absolute mm³
- **Nothingness**: 0% of this cut (empty space outside the block is not collected)

The table of consecutive cuts, and the graph under the 3D view, list every serial slab through the tissue at the current knife orientation. The graph Y-axis is **absolute target volume in that cut (mm³)**. Parallel stacks are near zero until they reach the target layer, then spike. Face-on perpendicular stacks are almost flat. A 45° rotation about Z makes a longer cut through the layer, so the middle of that stack is higher.

Changing display X or Y must not change any of these numbers.

## Project files

| File | Role |
| --- | --- |
| `app.py` | Browser interface (Dash + Plotly) |
| `geometry.py` | Knife slab, rotations, intersection volumes, consecutive stack |
| `test_geometry.py` | Volume and consecutive-step checks |
| `argument_volumes.py` | Sweep of target volume vs tilt (parallel vs perpendicular argument) |
| `requirements.txt` | Python dependencies |

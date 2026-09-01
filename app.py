"""Interactive 3D explorer for a tissue cube, a near-surface target cuboid,
and a rotatable section about a movable pivot.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:8050
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from scipy.spatial import ConvexHull, QhullError

from geometry import OrientedBox, compute_volumes, knife_normal

DEFAULTS = {
    "cube_x": 3.0,
    "cube_y": 3.0,
    "cube_z": 3.0,
    "target_x": 3.0,
    "target_y": 3.0,
    "target_z": 0.2,
    "target_depth": 0.1,
    "section_x": 6.0,
    "section_y": 6.0,
    "section_z": 0.1,
    "pivot_x": 0.0,
    "pivot_y": 0.0,
    "pivot_z": 0.0,
    "rx": 0.0,
    "ry": 0.0,
    "rz": 0.0,
}

# Vertex order from OrientedBox.vertices: x varies slowest, then y, then z.
BOX_TRIANGLES = np.array(
    [
        [0, 1, 3], [0, 3, 2],
        [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4],
        [1, 5, 7], [1, 7, 3],
    ],
    dtype=int,
)
BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CUBE_COLOR = "#4b5563"
TARGET_COLOR = "#dc2626"
SECTION_COLOR = "#2563eb"
TISSUE_CUT_COLOR = "#0f766e"
OVERLAP_COLOR = "#f59e0b"
PIVOT_COLOR = "#111827"

CONTROLS = [
    ("cube_x", "Cube X", "mm", 0.2, 15.0, 0.1, DEFAULTS["cube_x"]),
    ("cube_y", "Cube Y", "mm", 0.2, 15.0, 0.1, DEFAULTS["cube_y"]),
    ("cube_z", "Cube Z", "mm", 0.2, 15.0, 0.1, DEFAULTS["cube_z"]),
    ("target_x", "Target X", "mm", 0.1, 15.0, 0.1, DEFAULTS["target_x"]),
    ("target_y", "Target Y", "mm", 0.1, 15.0, 0.1, DEFAULTS["target_y"]),
    ("target_z", "Target thickness", "mm", 0.01, 5.0, 0.01, DEFAULTS["target_z"]),
    ("target_depth", "Target depth from top", "mm", 0.0, 10.0, 0.01, DEFAULTS["target_depth"]),
    ("section_x", "Display X (visualization only)", "mm", 0.1, 20.0, 0.1, DEFAULTS["section_x"]),
    ("section_y", "Display Y (visualization only)", "mm", 0.1, 20.0, 0.1, DEFAULTS["section_y"]),
    ("section_z", "Knife thickness", "mm", 0.01, 5.0, 0.01, DEFAULTS["section_z"]),
    ("pivot_x", "Pivot X", "mm", -10.0, 10.0, 0.01, DEFAULTS["pivot_x"]),
    ("pivot_y", "Pivot Y", "mm", -10.0, 10.0, 0.01, DEFAULTS["pivot_y"]),
    ("pivot_z", "Pivot Z", "mm", -10.0, 10.0, 0.01, DEFAULTS["pivot_z"]),
    ("rx", "Rotation about X", "deg", -180.0, 180.0, 0.5, DEFAULTS["rx"]),
    ("ry", "Rotation about Y", "deg", -180.0, 180.0, 0.5, DEFAULTS["ry"]),
    ("rz", "Rotation about Z", "deg", -180.0, 180.0, 0.5, DEFAULTS["rz"]),
]
CONTROL_BY_ID = {spec[0]: spec for spec in CONTROLS}


def _mesh3d(box: OrientedBox, color: str, name: str, opacity: float) -> go.Mesh3d:
    verts = box.vertices()
    i, j, k = BOX_TRIANGLES.T
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
        hovertemplate=f"{name}<extra></extra>",
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1),
    )


def _wireframe(box: OrientedBox, color: str, name: str, width: float = 3.0) -> go.Scatter3d:
    verts = box.vertices()
    xs, ys, zs = [], [], []
    for a, b in BOX_EDGES:
        xs += [verts[a, 0], verts[b, 0], None]
        ys += [verts[a, 1], verts[b, 1], None]
        zs += [verts[a, 2], verts[b, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def _poly_mesh(points: np.ndarray, color: str, name: str, opacity: float) -> go.Mesh3d | None:
    if points.shape[0] < 4:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    i, j, k = hull.simplices.T
    return go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,
        hovertemplate=f"{name}<extra></extra>",
        flatshading=True,
    )


def build_figure(tissue: OrientedBox, target: OrientedBox | None, section: OrientedBox,
                 overlap_pts: np.ndarray, tissue_cut_pts: np.ndarray, pivot: np.ndarray) -> go.Figure:
    traces: list = [
        _mesh3d(tissue, CUBE_COLOR, "Tissue cube", 0.08),
        _wireframe(tissue, CUBE_COLOR, "cube-edges", 4.0),
        _mesh3d(section, SECTION_COLOR, "Display cuboid (not used in volumes)", 0.08),
        _wireframe(section, SECTION_COLOR, "section-edges", 2.0),
    ]
    cut = _poly_mesh(tissue_cut_pts, TISSUE_CUT_COLOR, "Tissue in this cut", 0.55)
    if cut is not None:
        traces.append(cut)
    if target is not None:
        traces.append(_mesh3d(target, TARGET_COLOR, "Target region", 0.35))
        traces.append(_wireframe(target, TARGET_COLOR, "target-edges", 3.0))
    overlap = _poly_mesh(overlap_pts, OVERLAP_COLOR, "Target in this cut", 0.92)
    if overlap is not None:
        traces.append(overlap)
    pivot = np.asarray(pivot, dtype=float).reshape(3)
    traces.append(
        go.Scatter3d(
            x=[pivot[0]],
            y=[pivot[1]],
            z=[pivot[2]],
            mode="markers",
            marker=dict(size=7, color=PIVOT_COLOR, symbol="x"),
            name="Rotation pivot",
            hovertemplate=(
                "Rotation pivot<br>X=%{x:.3f} mm<br>Y=%{y:.3f} mm<br>Z=%{z:.3f} mm<extra></extra>"
            ),
        )
    )

    all_pts = [tissue.vertices(), section.vertices(), pivot.reshape(1, 3)]
    if target is not None:
        all_pts.append(target.vertices())
    pts = np.vstack(all_pts)
    span = float(np.max(np.abs(pts))) * 2.0
    span = max(span, 1.0)

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z up, top of block (mm)",
            aspectmode="data",
            xaxis=dict(backgroundcolor="#eef2ff", gridcolor="#cbd5e1", zerolinecolor="#94a3b8"),
            yaxis=dict(backgroundcolor="#eef2ff", gridcolor="#cbd5e1", zerolinecolor="#94a3b8"),
            zaxis=dict(backgroundcolor="#e2e8f0", gridcolor="#cbd5e1", zerolinecolor="#94a3b8"),
            camera=dict(eye=dict(x=1.7, y=1.45, z=1.15)),
        ),
        title=dict(text="Tissue block, target region, and section (true millimetre scale)", font=dict(size=15)),
    )
    half = span / 2.0 * 1.08
    fig.update_scenes(
        xaxis_range=[-half, half],
        yaxis_range=[-half, half],
        zaxis_range=[-half, half],
        aspectmode="data",
    )
    return fig


def build_stack_figure(stack) -> go.Figure:
    fig = go.Figure()
    if stack:
        xs = [c.number for c in stack]
        ys = [c.pct_of_target for c in stack]
        fig.add_bar(
            x=xs,
            y=ys,
            name="% of target cuboid",
            marker=dict(color="#93c5fd"),
            hovertemplate="Section %{x}<br>%{y:.3f}% of target<extra></extra>",
        )
        current = next((c for c in stack if c.is_current), None)
        if current is not None:
            fig.add_scatter(
                x=[current.number],
                y=[current.pct_of_target],
                mode="markers",
                marker=dict(size=11, color="#dc2626"),
                name="This cut",
                hovertemplate="This cut (section %{x})<br>%{y:.3f}% of target<extra></extra>",
            )
    fig.update_layout(
        margin=dict(l=48, r=16, t=40, b=44),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        title=dict(
            text="Target cuboid captured in each consecutive cut",
            font=dict(size=14),
        ),
        xaxis=dict(title="Consecutive section number (along knife normal)", dtick=1, gridcolor="#e2e8f0"),
        yaxis=dict(
            title="% of target cuboid in that cut",
            rangemode="tozero",
            ticksuffix="%",
            gridcolor="#e2e8f0",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        bargap=0.12,
    )
    return fig


def _slider_input(control_id: str, label: str, unit: str, vmin: float, vmax: float,
                  step: float, value: float) -> html.Div:
    return html.Div(
        className="control-row",
        children=[
            html.Label(f"{label} ({unit})", htmlFor=f"{control_id}-input", className="control-label"),
            html.Div(
                className="slider-input",
                children=[
                    dcc.Slider(
                        id=f"{control_id}-slider",
                        min=vmin,
                        max=vmax,
                        step=step,
                        value=value,
                        updatemode="drag",
                        included=True,
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="param-slider",
                    ),
                    dcc.Input(
                        id=f"{control_id}-input",
                        type="number",
                        value=value,
                        min=vmin,
                        max=vmax,
                        step=step,
                        className="param-input",
                    ),
                ],
            ),
        ],
    )


def _pct_card(title: str, value_id: str, detail_id: str, color: str) -> html.Div:
    return html.Div(
        className="pct-card",
        style={"borderTop": f"4px solid {color}"},
        children=[
            html.Div(title, className="pct-title"),
            html.Div(id=value_id, className="pct-value"),
            html.Div(id=detail_id, className="pct-detail"),
        ],
    )


app = Dash(__name__)
app.title = "Tissue section geometry"

app.layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="header",
            children=[
                html.H1("Tissue section geometry"),
                html.P(
                    "A vibratome section is a knife slab as thick as you set, extended in-plane until it leaves the "
                    "tissue. Display X/Y only draw a cuboid; they are not used in the volumes. "
                    "Consecutive cuts step along the knife normal (local Z) by one thickness. "
                    "(0, 0, 0) is the cube centre; +Z points toward the top face. "
                    "At 0°, 0°, 0° the cut is parallel to the top. Rx = 90° or Ry = 90° is perpendicular."
                ),
            ],
        ),
        html.Div(
            className="main",
            children=[
                html.Div(
                    className="scene-pane",
                    children=[
                        dcc.Graph(id="scene", figure=go.Figure(), style={"height": "58%"}),
                        dcc.Graph(id="stack-graph", figure=go.Figure(), style={"height": "42%"}),
                    ],
                ),
                html.Div(
                    className="side-pane",
                    children=[
                        html.Div(
                            className="presets",
                            children=[
                                html.Button("Reset defaults", id="preset-default", n_clicks=0),
                                html.Button("Perpendicular (Rx=90°)", id="preset-perp-x", n_clicks=0),
                                html.Button("Perpendicular (Ry=90°)", id="preset-perp-y", n_clicks=0),
                                html.Button("Perpendicular + 45° in Z", id="preset-perp-z45", n_clicks=0),
                                html.Button("Oblique 45°", id="preset-oblique", n_clicks=0),
                                html.Button("Pivot on target", id="preset-pivot-target", n_clicks=0),
                                html.Button("Pivot at cube centre", id="preset-pivot-centre", n_clicks=0),
                                html.Button("Previous cut (−thickness along normal)", id="step-normal-neg", n_clicks=0),
                                html.Button("Next cut (+thickness along normal)", id="step-normal-pos", n_clicks=0),
                            ],
                        ),
                        html.H2("Tissue cube"),
                        _slider_input(*CONTROL_BY_ID["cube_x"]),
                        _slider_input(*CONTROL_BY_ID["cube_y"]),
                        _slider_input(*CONTROL_BY_ID["cube_z"]),
                        html.H2("Target region"),
                        html.P("Parallel to the top face, centred in X and Y, measured down from the top.", className="hint"),
                        _slider_input(*CONTROL_BY_ID["target_x"]),
                        _slider_input(*CONTROL_BY_ID["target_y"]),
                        _slider_input(*CONTROL_BY_ID["target_z"]),
                        _slider_input(*CONTROL_BY_ID["target_depth"]),
                        html.H2("Section"),
                        html.P(
                            "Knife thickness is the only in-plane-independent size used for volumes. "
                            "Display X and Y are the drawn cuboid only.",
                            className="hint",
                        ),
                        _slider_input(*CONTROL_BY_ID["section_x"]),
                        _slider_input(*CONTROL_BY_ID["section_y"]),
                        _slider_input(*CONTROL_BY_ID["section_z"]),
                        html.H2("Rotation pivot"),
                        html.P(
                            "Mid-plane of this cut, in millimetres from the cube centre. "
                            "Only the component along the knife normal changes which tissue is cut. "
                            "Use Previous/Next cut to step a full thickness along that normal.",
                            className="hint",
                        ),
                        _slider_input(*CONTROL_BY_ID["pivot_x"]),
                        _slider_input(*CONTROL_BY_ID["pivot_y"]),
                        _slider_input(*CONTROL_BY_ID["pivot_z"]),
                        html.H2("Section rotation"),
                        _slider_input(*CONTROL_BY_ID["rx"]),
                        _slider_input(*CONTROL_BY_ID["ry"]),
                        _slider_input(*CONTROL_BY_ID["rz"]),
                        html.H2("Volume inside this cut"),
                        html.P(
                            "Percentages are of the tissue spanned by the knife slab (what a vibratome actually yields). "
                            "Nothingness is 0% because empty space outside the block is not part of the section. "
                            "Absolute target millimetres cubed are the quantity to compare across orientations.",
                            className="hint",
                        ),
                        html.Div(
                            className="pct-grid",
                            children=[
                                _pct_card("a) Tissue", "pct-tissue", "detail-tissue", CUBE_COLOR),
                                _pct_card("b) Target region", "pct-target", "detail-target", TARGET_COLOR),
                                _pct_card("c) Nothingness", "pct-empty", "detail-empty", SECTION_COLOR),
                            ],
                        ),
                        html.H2("Consecutive cuts along the knife normal"),
                        html.P(
                            "The graph under the 3D view shows every serial cut through the tissue at this orientation. "
                            "Y is the percentage of the whole target cuboid that lies in that tissue-clipped slab. "
                            "Parallel cuts stay near zero until they reach the target layer, then spike. "
                            "Face-on perpendicular cuts are almost flat. A 45° rotation about Z lengthens the cut "
                            "through the layer, so the middle of that stack is higher; it is not perfectly flat, "
                            "because the chord through the square is shorter near the corners.",
                            className="hint",
                        ),
                        html.Div(id="stack-table"),
                        html.Div(id="extra-metrics", className="extra-metrics"),
                    ],
                ),
            ],
        ),
    ],
)

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  :root {
    --bg: #e8eef6;
    --panel: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #dbe3ee;
    --accent: #1d4ed8;
  }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: "Segoe UI", system-ui, sans-serif; }
  .page { min-height: 100vh; display: flex; flex-direction: column; }
  .header { padding: 16px 22px 8px; }
  .header h1 { margin: 0 0 6px; font-size: 22px; font-weight: 650; }
  .header p { margin: 0; color: var(--muted); max-width: 1100px; line-height: 1.45; font-size: 14px; }
  .main { flex: 1; display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(340px, 0.9fr);
    gap: 14px; padding: 8px 16px 16px; min-height: 0; }
  .scene-pane { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    min-height: 72vh; overflow: hidden; display: flex; flex-direction: column; }
  #scene { flex: 1.3; min-height: 360px; }
  #stack-graph { flex: 0.95; min-height: 250px; border-top: 1px solid var(--line); }
  .side-pane { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 14px 18px; overflow: auto; max-height: calc(100vh - 110px); }
  .side-pane h2 { margin: 16px 0 8px; font-size: 13px; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--muted); }
  .hint { margin: -4px 0 8px; color: var(--muted); font-size: 12px; line-height: 1.35; }
  .control-row { margin-bottom: 8px; }
  .control-label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 2px; }
  .slider-input { display: grid; grid-template-columns: 1fr 84px; gap: 8px; align-items: center; }
  .param-input { width: 100%; box-sizing: border-box; padding: 6px 8px; border: 1px solid var(--line);
    border-radius: 6px; font-size: 13px; }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .presets button { border: 1px solid var(--line); background: #f8fafc; border-radius: 6px;
    padding: 6px 8px; cursor: pointer; font-size: 12px; }
  .presets button:hover { border-color: var(--accent); color: var(--accent); }
  .pct-grid { display: grid; grid-template-columns: 1fr; gap: 8px; }
  .pct-card { background: #f8fafc; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .pct-title { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .pct-value { font-size: 28px; font-weight: 700; line-height: 1.15; margin: 2px 0; }
  .pct-detail, .extra-metrics { font-size: 12px; color: var(--muted); line-height: 1.4; }
  .stack-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 4px; }
  .stack-table th, .stack-table td { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--line); }
  .stack-table th { color: var(--muted); font-weight: 600; }
  .stack-current td { font-weight: 700; color: var(--ink); background: #ecfdf5; }
  .js-plotly-plot .plotly .modebar { right: 8px !important; }
  @media (max-width: 980px) {
    .main { grid-template-columns: 1fr; }
    .scene-pane { min-height: 52vh; }
    .side-pane { max-height: none; }
  }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""


def _sync_callback(control_id: str):
    slider_id = f"{control_id}-slider"
    input_id = f"{control_id}-input"

    @app.callback(
        Output(input_id, "value", allow_duplicate=True),
        Output(slider_id, "value", allow_duplicate=True),
        Input(input_id, "value"),
        Input(slider_id, "value"),
        prevent_initial_call=True,
    )
    def sync(input_val, slider_val):
        triggered = callback_context.triggered_id
        value = input_val if triggered == input_id else slider_val
        if value is None:
            return no_update, no_update
        return value, value

    sync.__name__ = f"sync_{control_id}"
    return sync


for spec in CONTROLS:
    _sync_callback(spec[0])


PRESET_OUTPUTS = [
    Output(f"{c[0]}-slider", "value", allow_duplicate=True) for c in CONTROLS
] + [Output(f"{c[0]}-input", "value", allow_duplicate=True) for c in CONTROLS]


def _angle_outputs():
    return [
        Output("rx-slider", "value", allow_duplicate=True),
        Output("ry-slider", "value", allow_duplicate=True),
        Output("rz-slider", "value", allow_duplicate=True),
        Output("rx-input", "value", allow_duplicate=True),
        Output("ry-input", "value", allow_duplicate=True),
        Output("rz-input", "value", allow_duplicate=True),
    ]


def _pivot_outputs():
    return [
        Output("pivot_x-slider", "value", allow_duplicate=True),
        Output("pivot_y-slider", "value", allow_duplicate=True),
        Output("pivot_z-slider", "value", allow_duplicate=True),
        Output("pivot_x-input", "value", allow_duplicate=True),
        Output("pivot_y-input", "value", allow_duplicate=True),
        Output("pivot_z-input", "value", allow_duplicate=True),
    ]


@app.callback(
    PRESET_OUTPUTS,
    Input("preset-default", "n_clicks"),
    prevent_initial_call=True,
)
def apply_default_preset(_n):
    slider_vals = [DEFAULTS[c[0]] for c in CONTROLS]
    return slider_vals + slider_vals


@app.callback(
    _angle_outputs(),
    Input("preset-perp-x", "n_clicks"),
    Input("preset-perp-y", "n_clicks"),
    Input("preset-perp-z45", "n_clicks"),
    Input("preset-oblique", "n_clicks"),
    prevent_initial_call=True,
)
def apply_angle_preset(_x, _y, _z45, _o):
    triggered = callback_context.triggered_id
    rx, ry, rz = 0.0, 0.0, 0.0
    if triggered == "preset-perp-x":
        rx = 90.0
    elif triggered == "preset-perp-y":
        ry = 90.0
    elif triggered == "preset-perp-z45":
        rx = 90.0
        rz = 45.0
    elif triggered == "preset-oblique":
        rx = 45.0
    return rx, ry, rz, rx, ry, rz


@app.callback(
    _pivot_outputs(),
    Input("preset-pivot-centre", "n_clicks"),
    Input("preset-pivot-target", "n_clicks"),
    State("cube_z-slider", "value"),
    State("target_z-slider", "value"),
    State("target_depth-slider", "value"),
    prevent_initial_call=True,
)
def apply_pivot_preset(_centre, _target, cube_z, target_z, depth):
    triggered = callback_context.triggered_id
    px, py, pz = 0.0, 0.0, 0.0
    if triggered == "preset-pivot-target":
        cube_z = float(cube_z if cube_z is not None else DEFAULTS["cube_z"])
        target_z = float(target_z if target_z is not None else DEFAULTS["target_z"])
        depth = float(depth if depth is not None else DEFAULTS["target_depth"])
        pz = cube_z / 2.0 - depth - target_z / 2.0
    return px, py, pz, px, py, pz


@app.callback(
    _pivot_outputs(),
    Input("step-normal-neg", "n_clicks"),
    Input("step-normal-pos", "n_clicks"),
    State("pivot_x-slider", "value"),
    State("pivot_y-slider", "value"),
    State("pivot_z-slider", "value"),
    State("section_z-slider", "value"),
    State("rx-slider", "value"),
    State("ry-slider", "value"),
    State("rz-slider", "value"),
    prevent_initial_call=True,
)
def step_along_knife_normal(_neg, _pos, px, py, pz, thickness, rx, ry, rz):
    triggered = callback_context.triggered_id
    px = float(px if px is not None else 0.0)
    py = float(py if py is not None else 0.0)
    pz = float(pz if pz is not None else 0.0)
    thickness = float(thickness if thickness is not None else DEFAULTS["section_z"])
    rx = float(rx if rx is not None else 0.0)
    ry = float(ry if ry is not None else 0.0)
    rz = float(rz if rz is not None else 0.0)
    direction = -1.0 if triggered == "step-normal-neg" else 1.0
    delta = direction * thickness * knife_normal(rx, ry, rz)
    nx, ny, nz = px + delta[0], py + delta[1], pz + delta[2]
    return nx, ny, nz, nx, ny, nz


@app.callback(
    Output("scene", "figure"),
    Output("pct-tissue", "children"),
    Output("detail-tissue", "children"),
    Output("pct-target", "children"),
    Output("detail-target", "children"),
    Output("pct-empty", "children"),
    Output("detail-empty", "children"),
    Output("extra-metrics", "children"),
    Output("stack-table", "children"),
    Output("stack-graph", "figure"),
    [Input(f"{c[0]}-slider", "value") for c in CONTROLS],
)
def update_scene(*values):
    params = {spec[0]: float(val if val is not None else spec[6]) for spec, val in zip(CONTROLS, values)}
    volumes, tissue, target, section, overlap_pts, tissue_cut_pts, stack = compute_volumes(
        cube_size=(params["cube_x"], params["cube_y"], params["cube_z"]),
        target_size=(params["target_x"], params["target_y"], params["target_z"]),
        depth_from_top=params["target_depth"],
        section_size=(params["section_x"], params["section_y"], params["section_z"]),
        rx=params["rx"],
        ry=params["ry"],
        rz=params["rz"],
        pivot=(params["pivot_x"], params["pivot_y"], params["pivot_z"]),
    )
    pivot = np.array([params["pivot_x"], params["pivot_y"], params["pivot_z"]])
    fig = build_figure(tissue, target, section, overlap_pts, tissue_cut_pts, pivot)
    stack_fig = build_stack_figure(stack)

    tissue_detail = (
        f"{volumes.tissue_in_section:.4f} mm³ of tissue in this cut  ·  "
        f"{volumes.pct_tissue_captured:.3f}% of the {volumes.tissue_block:.4f} mm³ block"
    )
    target_detail = (
        f"{volumes.target_in_section:.4f} mm³ of target in this cut  ·  "
        f"{volumes.pct_target_captured:.3f}% of the {volumes.target_block:.4f} mm³ target"
    )
    empty_detail = (
        "0 mm³ of the real section is empty space. "
        f"The display cuboid is {volumes.display_volume:.4f} mm³ "
        f"({volumes.empty_in_display:.4f} mm³ of that drawing lies outside the tissue)."
    )
    extra = [
        html.Div(
            f"Other tissue in this cut (tissue minus target): {volumes.pct_other_tissue:.3f}% "
            f"({max(volumes.tissue_in_section - volumes.target_in_section, 0.0):.4f} mm³)"
        ),
        html.Div(
            f"Pivot (mm): X={params['pivot_x']:.3f}, Y={params['pivot_y']:.3f}, Z={params['pivot_z']:.3f}. "
            f"Angles (deg): Rx={params['rx']:.2f}, Ry={params['ry']:.2f}, Rz={params['rz']:.2f}."
        ),
    ]
    best = max(stack, key=lambda c: c.target_volume) if stack else None
    if best is not None:
        extra.append(
            html.Div(
                f"Largest target in this serial stack: {best.pct_of_target:.3f}% of the target cuboid "
                f"({best.target_volume:.4f} mm³) in consecutive section {best.number}."
            )
        )
        extra.append(
            html.Div(
                f"Sum over the stack: {sum(c.pct_of_target for c in stack):.2f}% of the target "
                "(should be 100% if the stack covers the whole target)."
            )
        )
    stack_table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("#"),
                        html.Th("% of target cuboid"),
                        html.Th("Target (mm³)"),
                        html.Th("Tissue in cut (mm³)"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(f"{c.number}" + (" (this cut)" if c.is_current else "")),
                            html.Td(f"{c.pct_of_target:.3f}%"),
                            html.Td(f"{c.target_volume:.4f}"),
                            html.Td(f"{c.tissue_volume:.4f}"),
                        ],
                        className="stack-current" if c.is_current else "",
                    )
                    for c in stack
                ]
            ),
        ],
        className="stack-table",
    )
    return (
        fig,
        f"{volumes.pct_tissue:.3f}%",
        tissue_detail,
        f"{volumes.pct_target:.3f}%",
        target_detail,
        f"{volumes.pct_nothingness:.3f}%",
        empty_detail,
        extra,
        stack_table,
        stack_fig,
    )


if __name__ == "__main__":
    print("Open http://127.0.0.1:8050 in your browser")
    app.run_server(debug=False, host="127.0.0.1", port=8050, use_reloader=False)

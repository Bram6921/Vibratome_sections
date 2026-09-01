"""Geometry for an axis-aligned tissue block, a near-surface target slab,
and a rotatable section cuboid about a movable pivot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

EPS = 1e-12


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Active rotation Rx then Ry then Rz about the world axes, in degrees."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


@dataclass
class OrientedBox:
    """Cuboid with centre, side lengths (x, y, z), and a local-to-world rotation."""

    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.size = np.asarray(self.size, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def vertices(self) -> np.ndarray:
        half = self.size / 2.0
        corners = np.array(
            [[x, y, z] for x in (-half[0], half[0])
             for y in (-half[1], half[1])
             for z in (-half[2], half[2])],
            dtype=float,
        )
        return corners @ self.rotation.T + self.center

    def halfspaces(self) -> np.ndarray:
        """12 inequalities as rows [nx, ny, nz, c] meaning n·x + c <= 0."""
        axes = self.rotation
        half = self.size / 2.0
        rows = []
        for i in range(3):
            normal = axes[:, i]
            rows.append(np.append(normal, -np.dot(normal, self.center) - half[i]))
            rows.append(np.append(-normal, np.dot(normal, self.center) - half[i]))
        return np.asarray(rows, dtype=float)

    def contains(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        local = (pts - self.center) @ self.rotation
        return np.all(np.abs(local) <= (self.size / 2.0) + tol, axis=1)


def aabb_box(xmin: float, xmax: float, ymin: float, ymax: float,
             zmin: float, zmax: float) -> OrientedBox:
    center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    return OrientedBox(center=center, size=size, rotation=np.eye(3))


def tissue_box(cube_size: np.ndarray) -> OrientedBox:
    return OrientedBox(center=np.zeros(3), size=np.asarray(cube_size, float), rotation=np.eye(3))


def target_box(cube_size: np.ndarray, target_size: np.ndarray,
               depth_from_top: float) -> OrientedBox | None:
    """Target slab parallel to the top face (z = +cube_z/2), clipped to the tissue."""
    cube_size = np.asarray(cube_size, dtype=float)
    target_size = np.asarray(target_size, dtype=float)
    half = cube_size / 2.0
    top_z = half[2]
    z_hi = top_z - depth_from_top
    z_lo = z_hi - target_size[2]

    x_half = min(target_size[0] / 2.0, half[0])
    y_half = min(target_size[1] / 2.0, half[1])
    z_lo = max(z_lo, -half[2])
    z_hi = min(z_hi, half[2])
    if z_hi - z_lo <= EPS or x_half <= EPS or y_half <= EPS:
        return None
    return aabb_box(-x_half, x_half, -y_half, y_half, z_lo, z_hi)


def section_box(
    section_size: np.ndarray,
    rx: float,
    ry: float,
    rz: float,
    pivot=(0.0, 0.0, 0.0),
) -> OrientedBox:
    return OrientedBox(
        center=np.asarray(pivot, dtype=float).reshape(3),
        size=np.asarray(section_size, dtype=float),
        rotation=rotation_matrix_xyz(rx, ry, rz),
    )


def _strictly_inside(point: np.ndarray, halfspaces: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.all(halfspaces[:, :3] @ point + halfspaces[:, 3] < -tol))


def _interior_point(halfspaces: np.ndarray, candidates=None) -> np.ndarray | None:
    points = []
    if candidates is not None:
        points.extend(np.asarray(p, dtype=float).reshape(3) for p in candidates)
    points.append(np.zeros(3))
    for point in points:
        if _strictly_inside(point, halfspaces):
            return point

    n_hs = halfspaces.shape[0]
    cost = np.array([0.0, 0.0, 0.0, -1.0])
    a_ub = np.hstack([halfspaces[:, :3], np.ones((n_hs, 1))])
    b_ub = -halfspaces[:, 3]
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]
    result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return None
    slack = float(result.x[3])
    if slack <= 1e-12:
        return None
    return result.x[:3]


def intersection_vertices(box_a: OrientedBox, box_b: OrientedBox) -> np.ndarray:
    halfspaces = np.vstack([box_a.halfspaces(), box_b.halfspaces()])
    midpoint = 0.5 * (box_a.center + box_b.center)
    interior = _interior_point(halfspaces, [box_a.center, box_b.center, midpoint])
    if interior is None:
        return np.empty((0, 3))
    try:
        hit = HalfspaceIntersection(halfspaces, interior, incremental=False)
    except (QhullError, ValueError):
        return np.empty((0, 3))
    return np.asarray(hit.intersections, dtype=float)


def convex_volume(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 4:
        return 0.0
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return 0.0
    return float(max(hull.volume, 0.0))


def intersection_volume(box_a: OrientedBox, box_b: OrientedBox) -> float:
    return convex_volume(intersection_vertices(box_a, box_b))


@dataclass
class SceneVolumes:
    section: float
    tissue_in_section: float
    target_in_section: float
    empty_in_section: float
    tissue_block: float
    target_block: float

    @property
    def pct_tissue(self) -> float:
        return 100.0 * self.tissue_in_section / self.section if self.section else 0.0

    @property
    def pct_target(self) -> float:
        return 100.0 * self.target_in_section / self.section if self.section else 0.0

    @property
    def pct_nothingness(self) -> float:
        return 100.0 * self.empty_in_section / self.section if self.section else 0.0

    @property
    def pct_other_tissue(self) -> float:
        other = max(self.tissue_in_section - self.target_in_section, 0.0)
        return 100.0 * other / self.section if self.section else 0.0

    @property
    def pct_target_captured(self) -> float:
        return 100.0 * self.target_in_section / self.target_block if self.target_block else 0.0

    @property
    def pct_tissue_captured(self) -> float:
        return 100.0 * self.tissue_in_section / self.tissue_block if self.tissue_block else 0.0


def compute_volumes(
    cube_size,
    target_size,
    depth_from_top: float,
    section_size,
    rx: float,
    ry: float,
    rz: float,
    pivot=(0.0, 0.0, 0.0),
) -> tuple[SceneVolumes, OrientedBox, OrientedBox | None, OrientedBox, np.ndarray]:
    tissue = tissue_box(cube_size)
    target = target_box(cube_size, target_size, depth_from_top)
    section = section_box(section_size, rx, ry, rz, pivot=pivot)

    v_section = section.volume
    v_tissue_cap = intersection_volume(section, tissue)
    v_target_cap = intersection_volume(section, target) if target is not None else 0.0
    v_target_cap = min(v_target_cap, v_tissue_cap)
    v_empty = max(v_section - v_tissue_cap, 0.0)
    v_target_block = target.volume if target is not None else 0.0

    volumes = SceneVolumes(
        section=v_section,
        tissue_in_section=v_tissue_cap,
        target_in_section=v_target_cap,
        empty_in_section=v_empty,
        tissue_block=tissue.volume,
        target_block=v_target_block,
    )
    overlap_pts = intersection_vertices(section, target) if target is not None else np.empty((0, 3))
    return volumes, tissue, target, section, overlap_pts

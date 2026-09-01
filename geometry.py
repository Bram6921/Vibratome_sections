"""Geometry for an axis-aligned tissue block, a near-surface target slab,
and a vibratome section: an infinite knife slab clipped to the tissue.
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
    """Finite cuboid used only for drawing. Volumes use calculation_slab()."""
    return OrientedBox(
        center=np.asarray(pivot, dtype=float).reshape(3),
        size=np.asarray(section_size, dtype=float),
        rotation=rotation_matrix_xyz(rx, ry, rz),
    )


# In-plane extent for the calculation slab: larger than any cube in the UI,
# so the knife is effectively infinite and X/Y of the display cuboid drop out.
SLAB_INPLANE = 80.0


def knife_normal(rx: float, ry: float, rz: float) -> np.ndarray:
    """Unit normal of the knife: local Z after Rx, Ry, Rz."""
    n = rotation_matrix_xyz(rx, ry, rz)[:, 2]
    return n / np.linalg.norm(n)


def calculation_slab(
    thickness: float,
    rx: float,
    ry: float,
    rz: float,
    pivot=(0.0, 0.0, 0.0),
) -> OrientedBox:
    """Infinite (numerically huge) slab of the given thickness, mid-plane at pivot."""
    return OrientedBox(
        center=np.asarray(pivot, dtype=float).reshape(3),
        size=np.array([SLAB_INPLANE, SLAB_INPLANE, float(thickness)]),
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


AABB_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def aabb_slab_points(box: OrientedBox, normal: np.ndarray, midplane: float,
                     thickness: float) -> np.ndarray:
    """Vertices of an axis-aligned box clipped to |x·n - midplane| <= thickness/2."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    verts = box.vertices()
    half = float(thickness) / 2.0
    d = verts @ n - midplane
    kept = [verts[i] for i in range(len(verts)) if abs(d[i]) <= half + 1e-12]
    for i, j in AABB_EDGES:
        da, db = d[i], d[j]
        a, b = verts[i], verts[j]
        for plane in (half, -half):
            if (da - plane) * (db - plane) < 0.0:
                w = (plane - da) / (db - da)
                kept.append(a + w * (b - a))
    if not kept:
        return np.empty((0, 3))
    pts = np.unique(np.round(np.asarray(kept, dtype=float), 10), axis=0)
    return pts


def aabb_slab_volume(box: OrientedBox, normal: np.ndarray, midplane: float,
                     thickness: float) -> float:
    return convex_volume(aabb_slab_points(box, normal, midplane, thickness))


TOP_FLAT = "flat"
TOP_SPHERE_CONVEX = "sphere_convex"
TOP_SPHERE_CONCAVE = "sphere_concave"
TOP_CYL_CONVEX = "cylinder_convex"
TOP_CYL_CONCAVE = "cylinder_concave"
TARGET_CUBOID = "cuboid"
TARGET_FOLLOW = "follow_surface"

CURVED_TOPS = {
    TOP_SPHERE_CONVEX,
    TOP_SPHERE_CONCAVE,
    TOP_CYL_CONVEX,
    TOP_CYL_CONCAVE,
}


@dataclass
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=int).reshape(-1, 3)

    def is_empty(self) -> bool:
        return self.vertices.shape[0] < 4 or self.faces.shape[0] == 0


@dataclass
class SceneDraw:
    """Meshes for the 3D view when the top is not a flat cube."""

    top_shape: str
    target_mode: str
    curve_offset: float
    tissue_top_xyz: tuple | None
    target_top_xyz: tuple | None
    target_bot_xyz: tuple | None
    cut_mesh: TriangleMesh | None
    overlap_mesh: TriangleMesh | None


@dataclass
class TissueBlock:
    """Axis-aligned block whose top face may be spherical or cylindrical."""

    size: np.ndarray
    top_shape: str = TOP_FLAT
    curve_offset: float = 0.0
    cylinder_axis: str = "y"

    def __post_init__(self) -> None:
        self.size = np.asarray(self.size, dtype=float).reshape(3)
        self.curve_offset = float(max(self.curve_offset, 0.0))
        self.top_shape = str(self.top_shape)
        self.cylinder_axis = "x" if str(self.cylinder_axis).lower().startswith("x") else "y"
        if self.curve_offset <= 1e-12 or self.top_shape not in CURVED_TOPS:
            self.top_shape = TOP_FLAT

    @property
    def is_flat(self) -> bool:
        return self.top_shape == TOP_FLAT

    @property
    def is_convex(self) -> bool:
        return "convex" in self.top_shape

    @property
    def z_ref(self) -> float:
        return 0.5 * float(self.size[2])

    @property
    def z_bottom(self) -> float:
        return -0.5 * float(self.size[2])

    def rho_max2(self) -> float:
        hx, hy = 0.5 * self.size[0], 0.5 * self.size[1]
        if self.top_shape.startswith("sphere"):
            return float(hx * hx + hy * hy)
        if self.cylinder_axis == "y":
            return float(hx * hx)
        return float(hy * hy)

    def z_surface(self, x, y):
        """Top surface z(x, y). Corners of the original top face stay at z_ref."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z_ref = self.z_ref
        if self.is_flat:
            return np.broadcast_arrays(x, y)[0] * 0.0 + z_ref
        h = self.curve_offset
        if self.top_shape.startswith("sphere"):
            rho2 = x * x + y * y
        elif self.cylinder_axis == "y":
            rho2 = x * x
        else:
            rho2 = y * y
        rho_max2 = self.rho_max2()
        radius = (rho_max2 + h * h) / (2.0 * h)
        rad = np.sqrt(np.maximum(radius * radius - rho2, 0.0))
        sag = rho2 / (radius + rad + 1e-30)
        if self.is_convex:
            z = z_ref + h - sag
        else:
            z = z_ref - h + sag
        return np.maximum(z, self.z_bottom)

    def rho2(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.is_flat:
            return x * 0.0
        if self.top_shape.startswith("sphere"):
            return x * x + y * y
        if self.cylinder_axis == "y":
            return x * x
        return y * y

    def generator_radius(self) -> float:
        h = self.curve_offset
        if self.is_flat or h <= 1e-12:
            return float("inf")
        return (self.rho_max2() + h * h) / (2.0 * h)

    def generator_center_z(self) -> float:
        radius = self.generator_radius()
        if self.is_convex:
            return self.z_ref + self.curve_offset - radius
        return self.z_ref - self.curve_offset + radius

    def outward_normal(self, x, y):
        """Unit normal pointing out of the tissue through the top."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = self.z_surface(x, y)
        shape = np.broadcast(x, y).shape
        nx = np.zeros(shape, dtype=float)
        ny = np.zeros(shape, dtype=float)
        nz = np.ones(shape, dtype=float)
        if self.is_flat:
            return nx, ny, nz
        radius = self.generator_radius()
        zc = self.generator_center_z()
        vz = z - zc
        sign = 1.0 if self.is_convex else -1.0
        if self.top_shape.startswith("sphere"):
            nx = sign * x / radius
            ny = sign * y / radius
            nz = sign * vz / radius
        elif self.cylinder_axis == "y":
            nx = sign * x / radius
            ny = np.zeros(shape, dtype=float)
            nz = sign * vz / radius
        else:
            nx = np.zeros(shape, dtype=float)
            ny = sign * y / radius
            nz = sign * vz / radius
        return nx, ny, nz


def tissue_aabb(block: TissueBlock) -> OrientedBox:
    """Bounding box of the solid, including a convex cap."""
    hx, hy = 0.5 * block.size[0], 0.5 * block.size[1]
    z_hi = block.z_ref + (block.curve_offset if block.is_convex and not block.is_flat else 0.0)
    return aabb_box(-hx, hx, -hy, hy, block.z_bottom, z_hi)


def parallel_shell_z_bounds(x, y, block: TissueBlock, depth: float, thickness: float):
    """Vertical [z_lo, z_hi] of the inward normal-offset shell at each (x, y).

    Depth and thickness are distances along the surface normal, not along Z.
    For a sphere/cylinder this is the region between two parallel surfaces.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if block.is_flat:
        z_surf = block.z_surface(x, y)
        z_hi = z_surf - float(depth)
        z_lo = z_hi - float(thickness)
        return z_lo, z_hi

    radius = block.generator_radius()
    zc = block.generator_center_z()
    rho2 = block.rho2(x, y)
    d = float(depth)
    t = float(thickness)
    if block.is_convex:
        r_near = radius - d
        r_far = radius - d - t
    else:
        r_near = radius + d
        r_far = radius + d + t

    empty = r_near <= 1e-12
    r_near2 = np.maximum(r_near, 0.0) ** 2
    r_far2 = np.maximum(r_far, 0.0) ** 2
    empty = empty | (rho2 > r_near2 + 1e-15)

    inner_span = np.sqrt(np.maximum(r_near2 - rho2, 0.0))
    far_ok = rho2 <= r_far2 + 1e-15
    far_span = np.sqrt(np.maximum(r_far2 - rho2, 0.0))

    if block.is_convex:
        z_hi = zc + inner_span
        z_lo = np.where(far_ok, zc + far_span, block.z_bottom)
    else:
        z_hi = zc - inner_span
        z_lo = np.where(far_ok, zc - far_span, block.z_bottom)

    z_lo = np.where(empty, np.nan, z_lo)
    z_hi = np.where(empty, np.nan, z_hi)
    return z_lo, z_hi


def target_z_bounds(x, y, block: TissueBlock, target_size, depth_from_top: float, mode: str):
    """Per-column [z_lo, z_hi] of the target, clipped to tissue. NaN where empty."""
    target_size = np.asarray(target_size, dtype=float)
    hx = min(0.5 * target_size[0], 0.5 * block.size[0])
    hy = min(0.5 * target_size[1], 0.5 * block.size[1])
    in_xy = (np.abs(x) <= hx + 1e-12) & (np.abs(y) <= hy + 1e-12)
    z_surf = block.z_surface(x, y)
    if mode == TARGET_FOLLOW:
        z_lo, z_hi = parallel_shell_z_bounds(x, y, block, depth_from_top, float(target_size[2]))
    else:
        z_hi = block.z_ref - float(depth_from_top)
        z_lo = z_hi - target_size[2]
    z_lo = np.maximum(z_lo, block.z_bottom)
    z_hi = np.minimum(z_hi, z_surf)
    empty = (~in_xy) | ~np.isfinite(z_lo) | ~np.isfinite(z_hi) | (z_hi - z_lo <= EPS)
    z_lo = np.where(empty, np.nan, z_lo)
    z_hi = np.where(empty, np.nan, z_hi)
    return z_lo, z_hi


@dataclass
class ColumnModel:
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dy: float
    z_bot: np.ndarray
    z_top: np.ndarray
    t_lo: np.ndarray
    t_hi: np.ndarray
    tissue_volume: float
    target_volume: float
    z_min: float
    z_max: float

    @property
    def dA(self) -> float:
        return self.dx * self.dy


def make_columns(
    block: TissueBlock,
    target_size,
    depth_from_top: float,
    target_mode: str,
    n: int = 96,
) -> ColumnModel:
    n = max(int(n), 8)
    dx = float(block.size[0]) / n
    dy = float(block.size[1]) / n
    hx, hy = 0.5 * block.size[0], 0.5 * block.size[1]
    x = -hx + dx * (np.arange(n) + 0.5)
    y = -hy + dy * (np.arange(n) + 0.5)
    X, Y = np.meshgrid(x, y, indexing="xy")
    z_top = block.z_surface(X, Y)
    z_bot = np.full_like(X, block.z_bottom)
    height = np.maximum(z_top - z_bot, 0.0)
    t_lo, t_hi = target_z_bounds(X, Y, block, target_size, depth_from_top, target_mode)
    dA = dx * dy
    tissue_volume = float(np.sum(height) * dA)
    t_h = np.nan_to_num(np.maximum(np.subtract(t_hi, t_lo), 0.0), nan=0.0)
    target_volume = float(np.sum(t_h) * dA)
    aabb = tissue_aabb(block)
    half = aabb.size / 2.0
    return ColumnModel(
        X=X,
        Y=Y,
        dx=dx,
        dy=dy,
        z_bot=z_bot,
        z_top=z_top,
        t_lo=t_lo,
        t_hi=t_hi,
        tissue_volume=tissue_volume,
        target_volume=target_volume,
        z_min=float(aabb.center[2] - half[2]),
        z_max=float(aabb.center[2] + half[2]),
    )


def _slab_z_interval(normal, midplane, thickness, X, Y):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    half = float(thickness) / 2.0
    p = float(midplane)
    s = n[0] * X + n[1] * Y
    if abs(n[2]) > 1e-10:
        za = (p - half - s) / n[2]
        zb = (p + half - s) / n[2]
        return np.minimum(za, zb), np.maximum(za, zb), None
    inside = np.abs(s - p) <= half + 1e-12
    return None, None, inside


def column_slab_volumes(cols: ColumnModel, normal, midplane, thickness) -> tuple[float, float]:
    """Tissue and target volume of one knife slab, by XY columns."""
    slo, shi, inside = _slab_z_interval(normal, midplane, thickness, cols.X, cols.Y)
    t_valid = np.isfinite(cols.t_lo) & np.isfinite(cols.t_hi)
    if inside is None:
        tiss_lo = np.maximum(cols.z_bot, slo)
        tiss_hi = np.minimum(cols.z_top, shi)
        tiss_h = np.maximum(tiss_hi - tiss_lo, 0.0)
        tgt_lo = np.maximum(cols.t_lo, slo)
        tgt_hi = np.minimum(cols.t_hi, shi)
        tgt_h = np.where(t_valid, np.maximum(tgt_hi - tgt_lo, 0.0), 0.0)
    else:
        tiss_h = np.where(inside, np.maximum(cols.z_top - cols.z_bot, 0.0), 0.0)
        tgt_h = np.where(
            inside & t_valid, np.maximum(cols.t_hi - cols.t_lo, 0.0), 0.0
        )
    dA = cols.dA
    v_tiss = float(np.sum(tiss_h) * dA)
    v_tgt = float(np.sum(tgt_h) * dA)
    return v_tiss, min(v_tgt, v_tiss)


def _column_cut_heights(cols: ColumnModel, normal, midplane, thickness, kind: str):
    slo, shi, inside = _slab_z_interval(normal, midplane, thickness, cols.X, cols.Y)
    if kind == "target":
        z0, z1 = cols.t_lo, cols.t_hi
        valid = np.isfinite(z0) & np.isfinite(z1)
    else:
        z0, z1 = cols.z_bot, cols.z_top
        valid = np.ones(z0.shape, dtype=bool)
    if inside is None:
        lo = np.maximum(z0, slo)
        hi = np.minimum(z1, shi)
        h = np.where(valid, np.maximum(hi - lo, 0.0), 0.0)
        return lo, hi, h
    h = np.where(valid & inside, np.maximum(z1 - z0, 0.0), 0.0)
    return z0, z1, h


def cut_prism_mesh(
    cols: ColumnModel, normal, midplane, thickness, kind: str = "tissue"
) -> TriangleMesh:
    lo, hi, height = _column_cut_heights(cols, normal, midplane, thickness, kind)
    mask = height > 1e-9
    iy, ix = np.where(mask)
    if iy.size == 0:
        return TriangleMesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    dx, dy = cols.dx, cols.dy
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    tris = (
        (0, 1, 2), (0, 2, 3),
        (4, 7, 6), (4, 6, 5),
        (0, 4, 5), (0, 5, 1),
        (3, 2, 6), (3, 6, 7),
        (0, 3, 7), (0, 7, 4),
        (1, 5, 6), (1, 6, 2),
    )
    for r, c in zip(iy, ix):
        x, y = float(cols.X[r, c]), float(cols.Y[r, c])
        z0, z1 = float(lo[r, c]), float(hi[r, c])
        x0, x1 = x - 0.5 * dx, x + 0.5 * dx
        y0, y1 = y - 0.5 * dy, y + 0.5 * dy
        base = len(verts)
        verts.extend(
            [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ]
        )
        faces.extend([[base + i, base + j, base + k] for i, j, k in tris])
    return TriangleMesh(np.asarray(verts, dtype=float), np.asarray(faces, dtype=int))


def surface_xyz(block: TissueBlock, n: int = 40, z_offset: float = 0.0):
    n = max(int(n), 4)
    hx, hy = 0.5 * block.size[0], 0.5 * block.size[1]
    x = np.linspace(-hx, hx, n)
    y = np.linspace(-hy, hy, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = block.z_surface(X, Y) + float(z_offset)
    return X, Y, Z


def offset_surface_xyz(block: TissueBlock, distance: float, n: int = 40):
    """Parallel surface displaced inward along the outward normal by `distance`."""
    X, Y, Z = surface_xyz(block, n=n, z_offset=0.0)
    nx, ny, nz = block.outward_normal(X, Y)
    d = float(distance)
    return X - d * nx, Y - d * ny, Z - d * nz


def intersection_volume(box_a: OrientedBox, box_b: OrientedBox) -> float:
    return convex_volume(intersection_vertices(box_a, box_b))


@dataclass
class SceneVolumes:
    """Volumes in mm³. `section` is the tissue spanned by the knife slab."""

    section: float
    tissue_in_section: float
    target_in_section: float
    empty_in_section: float
    tissue_block: float
    target_block: float
    display_volume: float
    empty_in_display: float

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


@dataclass
class ConsecutiveCut:
    """One slab in a serial stack stepped along the knife normal."""

    index: int
    number: int
    midplane: float
    tissue_volume: float
    target_volume: float
    pct_of_target: float
    is_current: bool


def consecutive_cuts(
    cube_size,
    target_size,
    depth_from_top: float,
    thickness: float,
    rx: float,
    ry: float,
    rz: float,
    pivot=(0.0, 0.0, 0.0),
    top_shape: str = TOP_FLAT,
    curve_offset: float = 0.0,
    cylinder_axis: str = "y",
    target_mode: str = TARGET_CUBOID,
    grid_n: int = 96,
    _columns: ColumnModel | None = None,
) -> list[ConsecutiveCut]:
    """Serial sections with pitch = thickness, along the knife normal, covering the tissue."""
    block = TissueBlock(cube_size, top_shape, curve_offset, cylinder_axis)
    curved = not block.is_flat
    tissue = tissue_aabb(block) if curved else tissue_box(cube_size)
    target = target_box(cube_size, target_size, depth_from_top)
    n = knife_normal(rx, ry, rz)
    t = float(thickness)
    pivot = np.asarray(pivot, dtype=float).reshape(3)
    proj = tissue.vertices() @ n
    lo, hi = float(proj.min()), float(proj.max())
    p0 = float(np.dot(pivot, n))
    k_lo = int(np.floor((lo - t / 2.0 - p0) / t))
    k_hi = int(np.ceil((hi + t / 2.0 - p0) / t))
    cols = _columns
    if curved and cols is None:
        cols = make_columns(block, target_size, depth_from_top, target_mode, n=grid_n)
    target_vol = (
        cols.target_volume if cols is not None else (target.volume if target is not None else 0.0)
    )
    raw: list[tuple[int, float, float, float]] = []
    for k in range(k_lo, k_hi + 1):
        p = p0 + k * t
        if p + t / 2.0 < lo - 1e-12 or p - t / 2.0 > hi + 1e-12:
            continue
        if curved:
            v_tissue, v_target = column_slab_volumes(cols, n, p, t)
        else:
            v_tissue = aabb_slab_volume(tissue, n, p, t)
            if target is None:
                v_target = 0.0
            else:
                tproj = target.vertices() @ n
                if tproj.max() < p - t / 2.0 - 1e-12 or tproj.min() > p + t / 2.0 + 1e-12:
                    v_target = 0.0
                else:
                    v_target = min(aabb_slab_volume(target, n, p, t), v_tissue)
        if v_tissue <= 1e-12 and k != 0:
            continue
        raw.append((k, p, v_tissue, v_target))
    raw.sort(key=lambda row: row[0])
    cuts: list[ConsecutiveCut] = []
    for number, (k, p, v_tissue, v_target) in enumerate(raw, start=1):
        cuts.append(
            ConsecutiveCut(
                index=k,
                number=number,
                midplane=p,
                tissue_volume=v_tissue,
                target_volume=v_target,
                pct_of_target=(100.0 * v_target / target_vol) if target_vol else 0.0,
                is_current=(k == 0),
            )
        )
    return cuts


def compute_volumes(
    cube_size,
    target_size,
    depth_from_top: float,
    section_size,
    rx: float,
    ry: float,
    rz: float,
    pivot=(0.0, 0.0, 0.0),
    top_shape: str = TOP_FLAT,
    curve_offset: float = 0.0,
    cylinder_axis: str = "y",
    target_mode: str = TARGET_CUBOID,
    grid_n: int = 96,
    vis_n: int = 36,
) -> tuple[SceneVolumes, OrientedBox, OrientedBox | None, OrientedBox, np.ndarray, np.ndarray, list[ConsecutiveCut], SceneDraw]:
    block = TissueBlock(cube_size, top_shape, curve_offset, cylinder_axis)
    curved = not block.is_flat
    cube = tissue_box(cube_size)
    tissue = tissue_aabb(block) if curved else cube
    target = target_box(cube_size, target_size, depth_from_top)
    section_size = np.asarray(section_size, dtype=float)
    thickness = float(section_size[2])
    display = section_box(section_size, rx, ry, rz, pivot=pivot)

    n = knife_normal(rx, ry, rz)
    p0 = float(np.dot(np.asarray(pivot, dtype=float).reshape(3), n))
    cols = make_columns(block, target_size, depth_from_top, target_mode, n=grid_n) if curved else None
    vis_cols = (
        make_columns(block, target_size, depth_from_top, target_mode, n=vis_n) if curved else None
    )

    if curved:
        v_section, v_target_cap = column_slab_volumes(cols, n, p0, thickness)
        v_target_block = cols.target_volume
        overlap_pts = np.empty((0, 3))
        tissue_cut_pts = np.empty((0, 3))
        cut_mesh = cut_prism_mesh(vis_cols, n, p0, thickness, "tissue")
        overlap_mesh = cut_prism_mesh(vis_cols, n, p0, thickness, "target")
        follow = target_mode == TARGET_FOLLOW
        tgt_top = offset_surface_xyz(block, float(depth_from_top), n=40) if follow else None
        tgt_bot = (
            offset_surface_xyz(
                block,
                float(depth_from_top) + float(np.asarray(target_size)[2]),
                n=40,
            )
            if follow
            else None
        )
        if follow and tgt_top is not None:
            hx = min(0.5 * float(np.asarray(target_size)[0]), 0.5 * block.size[0])
            hy = min(0.5 * float(np.asarray(target_size)[1]), 0.5 * block.size[1])
            Xs, Ys, _ = surface_xyz(block, n=40)
            mask = (np.abs(Xs) > hx + 1e-9) | (np.abs(Ys) > hy + 1e-9)
            Xt, Yt, Zt = (np.array(a, dtype=float, copy=True) for a in tgt_top)
            Xb, Yb, Zb = (np.array(a, dtype=float, copy=True) for a in tgt_bot)
            for arr in (Xt, Yt, Zt, Xb, Yb, Zb):
                arr[mask] = np.nan
            tgt_top = (Xt, Yt, Zt)
            tgt_bot = (Xb, Yb, Zb)
        draw = SceneDraw(
            top_shape=block.top_shape,
            target_mode=target_mode if follow else TARGET_CUBOID,
            curve_offset=block.curve_offset,
            tissue_top_xyz=surface_xyz(block, n=40),
            target_top_xyz=tgt_top,
            target_bot_xyz=tgt_bot,
            cut_mesh=cut_mesh,
            overlap_mesh=overlap_mesh,
        )
        v_target_cap = min(v_target_cap, v_section)
    else:
        v_section = aabb_slab_volume(tissue, n, p0, thickness)
        if target is None:
            v_target_cap = 0.0
            overlap_pts = np.empty((0, 3))
            v_target_block = 0.0
        else:
            v_target_cap = min(aabb_slab_volume(target, n, p0, thickness), v_section)
            overlap_pts = aabb_slab_points(target, n, p0, thickness)
            v_target_block = target.volume
        tissue_cut_pts = aabb_slab_points(tissue, n, p0, thickness)
        draw = SceneDraw(
            top_shape=TOP_FLAT,
            target_mode=TARGET_CUBOID,
            curve_offset=0.0,
            tissue_top_xyz=None,
            target_top_xyz=None,
            target_bot_xyz=None,
            cut_mesh=None,
            overlap_mesh=None,
        )

    v_display = display.volume
    v_display_tissue = intersection_volume(display, tissue)
    volumes = SceneVolumes(
        section=v_section,
        tissue_in_section=v_section,
        target_in_section=v_target_cap,
        empty_in_section=0.0,
        tissue_block=cols.tissue_volume if cols is not None else tissue.volume,
        target_block=v_target_block,
        display_volume=v_display,
        empty_in_display=max(v_display - v_display_tissue, 0.0),
    )
    stack = consecutive_cuts(
        cube_size,
        target_size,
        depth_from_top,
        thickness,
        rx,
        ry,
        rz,
        pivot,
        top_shape=block.top_shape,
        curve_offset=block.curve_offset,
        cylinder_axis=block.cylinder_axis,
        target_mode=target_mode if curved else TARGET_CUBOID,
        grid_n=grid_n,
        _columns=cols,
    )
    return volumes, tissue, target, display, overlap_pts, tissue_cut_pts, stack, draw

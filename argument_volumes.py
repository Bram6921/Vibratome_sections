"""Max target volume in one consecutive tissue-clipped slab, vs tilt.

A vibratome section is modelled as an infinite slab of thickness 0.1 mm
clipped to the tissue. In-plane cuboid size is not used in the volume.
"""

from __future__ import annotations

import numpy as np

from geometry import intersection_volume, target_box, tissue_box, OrientedBox

CUBE = np.array([3.0, 3.0, 3.0])
TARGET_SIZE = np.array([3.0, 3.0, 0.2])
DEPTH = 0.1
THICKNESS = 0.1
INPLANE = 80.0
VOX = 0.025  # mm grid inside the target


def rotation_from_normal(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    return np.column_stack([x_axis, y_axis, n])


def infinite_slab(normal: np.ndarray, offset: float) -> OrientedBox:
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return OrientedBox(
        center=n * offset,
        size=np.array([INPLANE, INPLANE, THICKNESS]),
        rotation=rotation_from_normal(n),
    )


def target_grid(target: OrientedBox) -> tuple[np.ndarray, float]:
    c, s = target.center, target.size
    xs = np.arange(c[0] - s[0] / 2 + VOX / 2, c[0] + s[0] / 2, VOX)
    ys = np.arange(c[1] - s[1] / 2 + VOX / 2, c[1] + s[1] / 2, VOX)
    zs = np.arange(c[2] - s[2] / 2 + VOX / 2, c[2] + s[2] / 2, VOX)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    voxel_vol = VOX ** 3
    return pts, voxel_vol


def sliding_window_max(s: np.ndarray, width: float, voxel_vol: float) -> float:
    """Maximum mass of 1D points inside any interval of given width."""
    s = np.sort(s)
    j = 0
    best = 0
    n = s.size
    for i in range(n):
        while j < n and s[j] <= s[i] + width + 1e-15:
            j += 1
        best = max(best, j - i)
    return best * voxel_vol


def consecutive_window_max(s: np.ndarray, width: float, voxel_vol: float, n_phase: int) -> tuple[float, float]:
    """Max mass in one bin of a partition of R into intervals of `width`.

    Returns (best phase, worst phase) among n_phase starting offsets.
    """
    s = np.sort(s)
    if s.size == 0:
        return 0.0, 0.0
    lo, hi = float(s[0]), float(s[-1])
    phase_bests = []
    for phase in np.linspace(0.0, width, n_phase, endpoint=False):
        origin = lo - phase
        bins = np.floor((s - origin) / width).astype(int)
        _, counts = np.unique(bins, return_counts=True)
        phase_bests.append(int(counts.max()) * voxel_vol)
    return max(phase_bests), min(phase_bests)


def normal_at(tilt_from_z_deg: float, azimuth_deg: float) -> np.ndarray:
    t = np.deg2rad(tilt_from_z_deg)
    a = np.deg2rad(azimuth_deg)
    return np.array([np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)])


def exact_best(normal, target, s_vals, voxel_vol) -> float:
    """Refine the voxel sliding-window centre with exact convex intersection."""
    s = np.sort(s_vals)
    j = 0
    best_i = 0
    best_c = 0
    n = s.size
    for i in range(n):
        while j < n and s[j] <= s[i] + THICKNESS + 1e-15:
            j += 1
        if j - i > best_c:
            best_c = j - i
            best_i = i
    off = s[best_i] + THICKNESS / 2.0
    return intersection_volume(infinite_slab(normal, off), target)


def main() -> None:
    tissue = tissue_box(CUBE)
    target = target_box(CUBE, TARGET_SIZE, DEPTH)
    assert target is not None
    pts, voxel_vol = target_grid(target)
    print("voxels", pts.shape[0], "voxel_vol", voxel_vol, "target", target.volume)
    print("tissue", tissue.volume)

    print("tilt  az   best_voxel  consec_best  consec_worst")
    rows = []
    tilts = list(range(0, 91, 5))
    for az in (0.0, 45.0):
        for tilt in tilts:
            n = normal_at(float(tilt), az)
            s = pts @ n
            best = sliding_window_max(s, THICKNESS, voxel_vol)
            cbest, cworst = consecutive_window_max(s, THICKNESS, voxel_vol, 20)
            rows.append((tilt, az, best, cbest, cworst))
            print(f"{tilt:4d} {az:4.0f}  {best:10.5f}  {cbest:11.5f}  {cworst:12.5f}", flush=True)

    print("\nexact check at key angles (best pivot)", flush=True)
    for tilt, az in [(0, 0), (15, 0), (15, 45), (75, 0), (75, 45), (90, 0), (90, 45)]:
        n = normal_at(float(tilt), float(az))
        s = pts @ n
        exact = exact_best(n, target, s, voxel_vol)
        voxel = sliding_window_max(s, THICKNESS, voxel_vol)
        print(f"tilt {tilt} az {az}: voxel {voxel:.5f}  exact {exact:.5f}", flush=True)

    par = [r for r in rows if r[0] <= 15]
    perp = [r for r in rows if r[0] >= 75]
    print("\nmin parallel 0-15 best", min(r[2] for r in par))
    print("min parallel 0-15 worst consec", min(r[4] for r in par))
    print("max perp 75-90 best", max(r[2] for r in perp))
    print("max perp 75-90 best consec", max(r[3] for r in perp))

    print("\nazimuth sweep", flush=True)
    for tilt in (0, 15, 75, 90):
        bests, worsts = [], []
        for az in np.linspace(0, 90, 10):
            n = normal_at(float(tilt), float(az))
            s = pts @ n
            bests.append(sliding_window_max(s, THICKNESS, voxel_vol))
            _, w = consecutive_window_max(s, THICKNESS, voxel_vol, 16)
            worsts.append(w)
        print(
            f"tilt {tilt}: best-pivot [{min(bests):.5f}, {max(bests):.5f}]  "
            f"worst-consec [{min(worsts):.5f}, {max(worsts):.5f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()

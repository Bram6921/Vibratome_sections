"""Sanity checks for tissue-clipped knife-slab volumes."""

from geometry import compute_volumes, consecutive_cuts, knife_normal
import numpy as np


def nearly(a: float, b: float, tol: float = 1e-4) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} (tol={tol})")


def test_horizontal_through_centre() -> None:
    vols, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0,
        ry=0,
        rz=0,
    )
    nearly(vols.section, 0.9)
    nearly(vols.tissue_in_section, 0.9)
    nearly(vols.target_in_section, 0.0)
    nearly(vols.empty_in_section, 0.0)
    nearly(vols.pct_tissue, 100.0)
    nearly(vols.pct_nothingness, 0.0)
    nearly(vols.pct_target, 0.0)


def test_display_xy_does_not_change_volumes() -> None:
    kwargs = dict(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        rx=0,
        ry=0,
        rz=0,
        pivot=(0.0, 0.0, 1.3),
    )
    a, *_ = compute_volumes(section_size=(6, 6, 0.1), **kwargs)
    b, *_ = compute_volumes(section_size=(2, 2, 0.1), **kwargs)
    nearly(a.section, b.section)
    nearly(a.target_in_section, b.target_in_section)
    nearly(a.section, 0.9)
    nearly(a.target_in_section, 0.9)


def test_perpendicular_rx() -> None:
    vols, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=90,
        ry=0,
        rz=0,
    )
    nearly(vols.section, 0.9)
    nearly(vols.target_in_section, 0.06)
    nearly(vols.pct_target, 100.0 * 0.06 / 0.9)
    nearly(vols.pct_target_captured, 100.0 * 0.06 / 1.8)
    nearly(vols.pct_tissue, 100.0)
    nearly(vols.pct_nothingness, 0.0)


def test_perpendicular_ry() -> None:
    vols, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0,
        ry=90,
        rz=0,
    )
    nearly(vols.section, 0.9)
    nearly(vols.target_in_section, 0.06)


def test_horizontal_on_target() -> None:
    vols, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0,
        ry=0,
        rz=0,
        pivot=(0.0, 0.0, 1.3),
    )
    nearly(vols.section, 0.9)
    nearly(vols.target_in_section, 0.9)
    nearly(vols.pct_target, 100.0)
    nearly(vols.pct_target_captured, 50.0)


def test_consecutive_steps_along_knife_normal() -> None:
    n = knife_normal(90, 0, 0)
    t = 0.1
    cuts = consecutive_cuts(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        thickness=t,
        rx=90,
        ry=0,
        rz=0,
        pivot=(0.0, 0.0, 0.0),
    )
    current = next(c for c in cuts if c.is_current)
    neighbor = next(c for c in cuts if c.index == 1)
    nearly(neighbor.midplane - current.midplane, t)
    # Stepping the pivot by t along n must reproduce the neighbour cut.
    moved = np.array([0.0, 0.0, 0.0]) + t * n
    vols, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=90,
        ry=0,
        rz=0,
        pivot=moved,
    )
    nearly(vols.target_in_section, neighbor.target_volume)


if __name__ == "__main__":
    test_horizontal_through_centre()
    test_display_xy_does_not_change_volumes()
    test_perpendicular_rx()
    test_perpendicular_ry()
    test_horizontal_on_target()
    test_consecutive_steps_along_knife_normal()
    print("geometry checks passed")

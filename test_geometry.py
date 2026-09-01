"""Sanity checks for tissue-clipped knife-slab volumes."""

from geometry import (
    TissueBlock,
    compute_volumes,
    consecutive_cuts,
    knife_normal,
    parallel_shell_z_bounds,
)
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


def test_parallel_stack_is_zero_then_spike() -> None:
    cuts = consecutive_cuts(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        thickness=0.1,
        rx=0,
        ry=0,
        rz=0,
        pivot=(0.0, 0.0, 0.0),
    )
    vols = [c.target_volume for c in cuts]
    assert len(vols) >= 20
    assert vols[0] < 0.02
    assert max(vols) > 0.4
    assert vols.index(max(vols)) > 5


def test_diagonal_perpendicular_captures_more_than_face() -> None:
    face = consecutive_cuts(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        thickness=0.1,
        rx=90,
        ry=0,
        rz=0,
        pivot=(0.0, 0.0, 0.0),
    )
    diag = consecutive_cuts(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        thickness=0.1,
        rx=90,
        ry=0,
        rz=45,
        pivot=(0.0, 0.0, 0.0),
    )
    assert max(c.target_volume for c in diag) > max(c.target_volume for c in face) + 0.01


def test_zero_curve_offset_matches_flat() -> None:
    kwargs = dict(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0,
        ry=0,
        rz=0,
        pivot=(0.0, 0.0, 1.3),
    )
    flat, *_ = compute_volumes(**kwargs)
    curved, *_ = compute_volumes(
        top_shape="sphere_convex", curve_offset=0.0, target_mode="follow_surface", **kwargs
    )
    nearly(flat.section, curved.section)
    nearly(flat.target_in_section, curved.target_in_section)
    nearly(flat.tissue_block, curved.tissue_block)


def test_sphere_corners_stay_at_cube_top() -> None:
    block = TissueBlock((3, 3, 3), "sphere_convex", 0.4)
    z_ref = 1.5
    nearly(float(block.z_surface(1.5, 1.5)), z_ref, tol=1e-6)
    nearly(float(block.z_surface(0.0, 0.0)), z_ref + 0.4, tol=1e-6)
    dent = TissueBlock((3, 3, 3), "sphere_concave", 0.4)
    nearly(float(dent.z_surface(1.5, 1.5)), z_ref, tol=1e-6)
    nearly(float(dent.z_surface(0.0, 0.0)), z_ref - 0.4, tol=1e-6)


def test_convex_adds_tissue_concave_removes() -> None:
    cube = 3.0 * 3.0 * 3.0
    conv, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0, ry=0, rz=0,
        top_shape="sphere_convex",
        curve_offset=0.5,
    )
    conc, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0, ry=0, rz=0,
        top_shape="sphere_concave",
        curve_offset=0.5,
    )
    assert conv.tissue_block > cube + 0.3
    assert conc.tissue_block < cube - 0.3


def test_follow_surface_is_parallel_shell_along_normal() -> None:
    block = TissueBlock((3, 3, 3), "sphere_convex", 0.5)
    depth, thick = 0.1, 0.2
    z_lo, z_hi = parallel_shell_z_bounds(0.0, 0.0, block, depth, thick)
    z_surf = float(block.z_surface(0.0, 0.0))
    nearly(float(z_hi), z_surf - depth, tol=1e-6)
    nearly(float(z_hi - z_lo), thick, tol=1e-6)

    p = np.array([1.0, 0.0, float(block.z_surface(1.0, 0.0))])
    nx, ny, nz = block.outward_normal(1.0, 0.0)
    n = np.array([float(nx), float(ny), float(nz)])
    nearly(float(np.linalg.norm(n)), 1.0, tol=1e-6)
    p_off = p - depth * n
    centre = np.array([0.0, 0.0, block.generator_center_z()])
    nearly(float(np.linalg.norm(p_off - centre)), block.generator_radius() - depth, tol=1e-5)

    vols, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0, ry=0, rz=0,
        top_shape="sphere_convex",
        curve_offset=0.5,
        target_mode="follow_surface",
        grid_n=120,
    )
    assert abs(vols.target_block - 1.8) > 0.02


def test_concave_cuboid_target_is_clipped() -> None:
    vols, *_ = compute_volumes(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        section_size=(6, 6, 0.1),
        rx=0, ry=0, rz=0,
        top_shape="sphere_concave",
        curve_offset=0.5,
        target_mode="cuboid",
        grid_n=120,
    )
    assert vols.target_block < 1.5


def test_follow_parallel_spreads_target_across_more_cuts() -> None:
    cuboid = consecutive_cuts(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        thickness=0.1,
        rx=0, ry=0, rz=0,
        top_shape="sphere_convex",
        curve_offset=0.5,
        target_mode="cuboid",
    )
    follow = consecutive_cuts(
        cube_size=(3, 3, 3),
        target_size=(3, 3, 0.2),
        depth_from_top=0.1,
        thickness=0.1,
        rx=0, ry=0, rz=0,
        top_shape="sphere_convex",
        curve_offset=0.5,
        target_mode="follow_surface",
    )
    assert max(c.target_volume for c in cuboid) > 0.7
    assert max(c.target_volume for c in follow) < 0.7
    n_cuboid = sum(c.target_volume > 0.02 for c in cuboid)
    n_follow = sum(c.target_volume > 0.02 for c in follow)
    assert n_follow > n_cuboid


def test_cylinder_bends_only_in_one_direction() -> None:
    block = TissueBlock((3, 3, 3), "cylinder_convex", 0.4, cylinder_axis="y")
    z_ref = 1.5
    nearly(float(block.z_surface(1.5, 0.0)), z_ref, tol=1e-6)
    nearly(float(block.z_surface(0.0, 0.0)), z_ref + 0.4, tol=1e-6)
    nearly(float(block.z_surface(0.0, 1.5)), z_ref + 0.4, tol=1e-6)


if __name__ == "__main__":
    test_horizontal_through_centre()
    test_display_xy_does_not_change_volumes()
    test_perpendicular_rx()
    test_perpendicular_ry()
    test_horizontal_on_target()
    test_consecutive_steps_along_knife_normal()
    test_parallel_stack_is_zero_then_spike()
    test_diagonal_perpendicular_captures_more_than_face()
    test_zero_curve_offset_matches_flat()
    test_sphere_corners_stay_at_cube_top()
    test_convex_adds_tissue_concave_removes()
    test_follow_surface_is_parallel_shell_along_normal()
    test_concave_cuboid_target_is_clipped()
    test_follow_parallel_spreads_target_across_more_cuts()
    test_cylinder_bends_only_in_one_direction()
    print("geometry checks passed")

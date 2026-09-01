"""Sanity checks for intersection volumes used by the explorer."""

from geometry import compute_volumes


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
    nearly(vols.section, 3.6)
    nearly(vols.tissue_in_section, 0.9)
    nearly(vols.target_in_section, 0.0)
    nearly(vols.empty_in_section, 2.7)
    nearly(vols.pct_tissue, 25.0)
    nearly(vols.pct_nothingness, 75.0)


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
    nearly(vols.tissue_in_section, 0.9)
    nearly(vols.target_in_section, 0.06)
    nearly(vols.pct_target_captured, 100.0 * 0.06 / 1.8)


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
    nearly(vols.tissue_in_section, 0.9)
    nearly(vols.target_in_section, 0.06)


def test_horizontal_on_target() -> None:
    """A 0.1 mm horizontal section centred in the 0.2 mm target slab."""
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
    nearly(vols.tissue_in_section, 0.9)
    nearly(vols.target_in_section, 0.9)
    nearly(vols.pct_target_captured, 50.0)


if __name__ == "__main__":
    test_horizontal_through_centre()
    test_perpendicular_rx()
    test_perpendicular_ry()
    test_horizontal_on_target()
    print("geometry checks passed")

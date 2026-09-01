"""Publication figures: parallel vs perpendicular vibratome sectioning.

Saves PNG, PDF, and CSV to D:\\Vibratome_sections
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from geometry import consecutive_cuts, knife_normal

OUT = Path(r"D:\Vibratome_sections")
CUBE = (3.0, 3.0, 3.0)
TARGET = (3.0, 3.0, 0.2)
DEPTH = 0.1
THICKNESS = 0.1
DELTA = 15.0
CSV_PATH = OUT / "max_target_vs_tilt.csv"

FACE_COLOR = "#0072B2"
DIAG_COLOR = "#E69F00"
PAR_BAND = "#009E73"
PERP_BAND = "#CC79A7"
PAR0 = "#0072B2"
PAR15 = "#56B4E9"
PERP90 = "#D55E00"
PERP45 = "#882255"

OLD_COMBINED = (
    OUT / "parallel_vs_perpendicular_target_volume.png",
    OUT / "parallel_vs_perpendicular_target_volume.pdf",
)


def stack_kwargs(rx, ry, rz, pivot=(0.0, 0.0, 0.0)):
    return dict(
        cube_size=CUBE,
        target_size=TARGET,
        depth_from_top=DEPTH,
        thickness=THICKNESS,
        rx=rx,
        ry=ry,
        rz=rz,
        pivot=pivot,
    )


def max_target_in_stack(rx, ry, rz, n_phase: int = 8) -> float:
    n = knife_normal(rx, ry, rz)
    best = 0.0
    for phase in np.linspace(0.0, THICKNESS, n_phase, endpoint=False):
        pivot = phase * n
        cuts = consecutive_cuts(**stack_kwargs(rx, ry, rz, pivot))
        if cuts:
            best = max(best, max(c.target_volume for c in cuts))
    return best


def tilt_curve(tilts, rz: float) -> np.ndarray:
    return np.array([max_target_in_stack(float(th), 0.0, rz) for th in tilts])


def load_or_compute_curves(tilts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if CSV_PATH.exists():
        data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
        if data.shape[0] == len(tilts) and np.allclose(data[:, 0], tilts):
            print(f"Loaded {CSV_PATH}", flush=True)
            return data[:, 1], data[:, 2]
    print("Computing face-aligned curve...", flush=True)
    face = tilt_curve(tilts, rz=0.0)
    print("Computing diagonal curve...", flush=True)
    diag = tilt_curve(tilts, rz=45.0)
    np.savetxt(
        CSV_PATH,
        np.column_stack([tilts, face, diag]),
        delimiter=",",
        header="tilt_deg,face_aligned_mm3,diagonal_mm3",
        comments="",
    )
    print("wrote", CSV_PATH, flush=True)
    return face, diag


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def hide_top_right(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, stem: str) -> None:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    for path in (png, pdf):
        if path.exists():
            path.unlink()
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    print("wrote", png)
    print("wrote", pdf)


def make_figure_a(tilts, face, diag, par_min, perp_max) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    ax.axvspan(0, DELTA, color=PAR_BAND, alpha=0.18, lw=0, zorder=0)
    ax.axvspan(90 - DELTA, 90, color=PERP_BAND, alpha=0.18, lw=0, zorder=0)
    ax.plot(tilts, face, color=FACE_COLOR, lw=2.4, label="Face-aligned tilt (about X)")
    ax.plot(tilts, diag, color=DIAG_COLOR, lw=2.4, label="Diagonal tilt (45° in Z)")
    ax.axhline(par_min, color=PAR_BAND, ls=":", lw=1.6)
    ax.axhline(perp_max, color=PERP_BAND, ls=":", lw=1.6)

    ax.text(
        7.5,
        0.93,
        "Meant parallel\n(±15°)",
        ha="center",
        va="top",
        fontsize=9,
        color="#14532d",
    )
    ax.text(
        82.5,
        0.93,
        "Meant perpendicular\n(±15°)",
        ha="center",
        va="top",
        fontsize=9,
        color="#6b21a8",
    )
    ax.text(
        88,
        par_min + 0.025,
        f"Worst near-parallel band  {par_min:.3f} mm³",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=PAR_BAND,
    )
    ax.text(
        88,
        perp_max + 0.025,
        f"Best near-perpendicular band  {perp_max:.3f} mm³",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=PERP_BAND,
    )

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Tilt of the section plane from the top face (degrees)")
    ax.set_ylabel("Target volume in the best serial 0.1 mm cut (mm³)")
    ax.legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(0.98, 0.62))
    hide_top_right(ax)

    fig.suptitle(
        "Best serial cut still favours parallel sectioning under 15° error",
        fontsize=13,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.915,
        f"Worst cut in the 0–15° band still holds {par_min / perp_max:.1f}× more target "
        f"({par_min:.3f} mm³) than the best cut in the 75–90° band ({perp_max:.3f} mm³).",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#334155",
    )
    fig.text(
        0.08,
        0.02,
        "Model: 3×3×3 mm cube, 3×3×0.2 mm target at 0.1–0.3 mm depth, 0.1 mm knife slabs "
        "clipped to the tissue. Y is the maximum target volume in any one consecutive cut "
        f"(pitch = 0.1 mm). Shaded bands: ±{DELTA:.0f}° blade uncertainty. Diagonal tilt is "
        "the most favourable perpendicular geometry (longer chord through the square).",
        fontsize=7.5,
        color="#334155",
        wrap=True,
        va="bottom",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.20)
    save_fig(fig, "best_cut_vs_tilt")
    plt.close(fig)


def _histogram(ax, xs, ys, color) -> None:
    ax.bar(xs, ys, width=0.82, align="center", color=color, edgecolor="white", linewidth=0.35)


def make_figure_b(stacks, par_min, perp_max) -> None:
    max_n = max(int(xs.max()) for _, xs, _, _ in stacks)
    xlim = (0.5, max_n + 0.5)

    fig = plt.figure(figsize=(9.6, 9.2))
    gs = fig.add_gridspec(
        2, 2, hspace=0.72, wspace=0.28, left=0.10, right=0.97, top=0.76, bottom=0.12
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for i, (ax, (label, xs, ys, color)) in enumerate(zip(axes, stacks)):
        _histogram(ax, xs, ys, color)
        ax.axhline(par_min, color=PAR_BAND, ls=":", lw=1.3, zorder=1)
        ax.axhline(perp_max, color=PERP_BAND, ls=":", lw=1.3, zorder=1)
        n_hit = int(np.sum(ys > 1e-6))
        ax.set_title(
            f"{label}\nmax {ys.max():.3f} mm³   ·   {n_hit} cuts contain target",
            loc="left",
            fontsize=10.5,
            pad=10,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(0, max_n + 1, 5))
        hide_top_right(ax)

        if i >= 2:
            axins = ax.inset_axes([0.12, 0.38, 0.58, 0.48])
            _histogram(axins, xs, ys, color)
            axins.set_xlim(*xlim)
            axins.set_ylim(0, 0.12)
            axins.set_xticks([])
            axins.set_yticks([0.00, 0.06, 0.12])
            axins.tick_params(labelsize=7)
            axins.set_title("inset  0–0.12 mm³", fontsize=7.5, pad=2)
            axins.axhline(perp_max, color=PERP_BAND, ls=":", lw=1.0)
            for spine in axins.spines.values():
                spine.set_color("#64748b")
                spine.set_linewidth(0.8)

    axes[0].set_ylabel("Target volume in that cut (mm³)")
    axes[2].set_ylabel("Target volume in that cut (mm³)")
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].set_xlabel("Consecutive section number (along the knife normal)")
    axes[3].set_xlabel("Consecutive section number (along the knife normal)")

    fig.legend(
        handles=[
            Line2D([0], [0], color=PAR_BAND, ls=":", lw=1.6,
                   label=f"Worst parallel-band cut ({par_min:.3f} mm³)"),
            Line2D([0], [0], color=PERP_BAND, ls=":", lw=1.6,
                   label=f"Best perpendicular-band cut ({perp_max:.3f} mm³)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Consecutive 0.1 mm cuts: target is a late spike (parallel)\n"
        "or a thin ribbon spread across many sections (perpendicular)",
        fontsize=13,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.08,
        0.02,
        "Each bar is one 0.1 mm knife slab. All four histograms share the same y-scale, so "
        "heights are comparable: parallel 0° puts 0.90 mm³ in two or three cuts; 15° off still "
        "reaches 0.23 mm³; perpendicular stacks stay at 0.060 mm³ (face) or 0.084 mm³ (diagonal). "
        "Insets enlarge only the perpendicular profiles (0–0.12 mm³) so the flat vs peaked shape "
        "is visible. Cube 3×3×3 mm, target 3×3×0.2 mm at 0.1–0.3 mm depth.",
        fontsize=7.5,
        color="#334155",
        wrap=True,
        va="bottom",
    )
    save_fig(fig, "consecutive_stacks")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OLD_COMBINED:
        if path.exists():
            path.unlink()
            print("deleted", path, flush=True)

    apply_style()
    tilts = np.arange(0, 91, 1)
    face, diag = load_or_compute_curves(tilts)

    par_mask = tilts <= DELTA
    perp_mask = tilts >= (90.0 - DELTA)
    par_min = float(min(face[par_mask].min(), diag[par_mask].min()))
    perp_max = float(max(face[perp_mask].max(), diag[perp_mask].max()))

    profiles = [
        ("Parallel 0°", 0, 0, 0, PAR0),
        ("Parallel, 15° off", 15, 0, 0, PAR15),
        ("Perpendicular 90°", 90, 0, 0, PERP90),
        ("Perpendicular 90° + 45° in Z", 90, 0, 45, PERP45),
    ]
    stacks = []
    for label, rx, ry, rz, color in profiles:
        cuts = consecutive_cuts(**stack_kwargs(rx, ry, rz))
        xs = np.array([c.number for c in cuts], dtype=float)
        ys = np.array([c.target_volume for c in cuts], dtype=float)
        stacks.append((label, xs, ys, color))
        print(f"{label}: n={len(cuts)} max={max(ys):.4f} mm³", flush=True)

    make_figure_a(tilts, face, diag, par_min, perp_max)
    make_figure_b(stacks, par_min, perp_max)
    print(
        f"parallel-band min {par_min:.4f}  "
        f"perpendicular-band max {perp_max:.4f}  "
        f"ratio {par_min / perp_max:.2f}"
    )


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def tapered_pore(center, width, height, color, alpha=0.9, edge='k'):
    x0, y0 = center
    rx = width / 2
    ry = height / 2

    k = 0.5522847498  # ellipse Bezier constant

    verts = [
        # Start at left
        (x0 - rx, y0),

        # Left → top cusp
        (x0 - rx, y0 + k * ry),
        (x0, y0 + ry),        # control collapsed
        (x0, y0 + ry),        # cusp point

        # Top cusp → right
        (x0, y0 + ry),        # control collapsed
        (x0 + rx, y0 + k * ry),
        (x0 + rx, y0),

        # Right → bottom cusp
        (x0 + rx, y0 - k * ry),
        (x0, y0 - ry),        # control collapsed
        (x0, y0 - ry),        # cusp point

        # Bottom cusp → left
        (x0, y0 - ry),        # control collapsed
        (x0 - rx, y0 - k * ry),
        (x0 - rx, y0),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
    ]

    return PathPatch(
        Path(verts, codes),
        facecolor=color,
        edgecolor=edge,
        linewidth=1.0,
        alpha=alpha
    )

def run_plot_fig5():
    # Conceptual plot with ovals showing final pore geometry (shape) vs pressure sensitivity
    nx, ny = 6, 5
    x = np.linspace(0.08, 0.92, nx)  # pressure requirement (low -> high)
    y = np.linspace(0.12, 0.88, ny)  # final pore shape (eccentric -> circular)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = plt.Normalize(vmin=x.min(), vmax=x.max())
    cmap = cm.Blues

    base_height = 0.20          # tallest pore length (bottom rows)
    base_width = 0.16           # baseline width before oval adjustment
    min_width_ratio = 0.45      # very slit-like
    max_width_ratio = 0.95      # nearly circular
    min_height_scale = 0.6      # shortest pore length at highest y

    for xi, yi in zip(X.ravel(), Y.ravel()):
        width_ratio = min_width_ratio + (max_width_ratio - min_width_ratio) * yi
        height_scale = min_height_scale + (1 - yi) * (1 - min_height_scale)
        ellipse =tapered_pore(
        (xi, yi),
        width=base_width * width_ratio,
        height=base_height * height_scale,
        color=cmap(norm(xi)),
        edge='k',
        alpha=0.9,
        )
        ax.add_patch(ellipse)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low', 'High'], fontsize=15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Low', 'High'], fontsize=15)
    ax.set_xlabel('Midsection aspect ratio', fontsize=15)
    ax.set_ylabel('Material anisotropy', fontsize=15)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('../Figures/Fig5/concept_ovals.png', dpi=300, bbox_inches='tight')
    plt.show()
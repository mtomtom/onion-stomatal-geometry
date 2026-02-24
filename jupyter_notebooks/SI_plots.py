import matplotlib.pyplot as plt
import cross_section_helpers as csh
import os
import sys
from pathlib import Path
import pandas as pd
from IPython.display import display
import re

src_path = str(Path.cwd().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def _parse_mesh_metadata(mesh_entry):
    stem = Path(mesh_entry).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse pressure token from {stem}")
    pressure_token = parts[-1]
    try:
        pressure = float(pressure_token)
    except ValueError as exc:
        raise ValueError(f"Could not convert pressure '{pressure_token}' in {stem}") from exc
    id_match = re.search(r"\d+_\d+", stem)
    if id_match:
        mesh_id = id_match.group(0)
    else:
        mesh_id = parts[0]
    return mesh_id, pressure

def get_width_height(mesh_paths, selected_meshes):
    records = []
    results = csh.batch_midsection_width_height(mesh_paths, guard_cell="both")
    for row in results:
        if "error" in row:
            print(f"Skipping {row['mesh']}: {row['error']}")
            continue
        try:
            mesh_id, pressure = _parse_mesh_metadata(row["mesh"])
        except ValueError as exc:
            print(exc)
            continue
        if mesh_id not in selected_meshes:
            continue
        records.append({
            "mesh_id": mesh_id,
            "pressure": pressure,
            "left_width": row.get("left_width", 0.0),
            "left_height": row.get("left_height", 0.0),
            "right_width": row.get("right_width", 0.0),
            "right_height": row.get("right_height", 0.0),
        })

    if not records:
        print("No valid midsection measurements available for selected meshes. Re-run the previous cell?")
    else:
        df_midsections = pd.DataFrame(records)
        df_midsections["width_mean"] = (df_midsections["left_width"] + df_midsections["right_width"]) / 2
        df_midsections["height_mean"] = (df_midsections["left_height"] + df_midsections["right_height"]) / 2
        df_midsections = df_midsections.sort_values(["mesh_id", "pressure"]).reset_index(drop=True)
        display(df_midsections)
    return df_midsections

def plot_SI7(df_midsections, filename=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        df_midsections["width_mean"],
        df_midsections["height_mean"],
        c=df_midsections["pressure"],
        cmap="viridis",
        s=60,
        edgecolor="k",
        zorder=3
     )

    line_color = "#555555"
    for mesh_id, subset in df_midsections.groupby("mesh_id"):
        subset = subset.sort_values("pressure")
        if len(subset) < 2:
            continue
        ax.plot(
            subset["width_mean"],
            subset["height_mean"],
            color=line_color,
            linewidth=1.3,
            alpha=0.8,
            zorder=2
        )

    ax.set_xlabel("Midsection width (μm)")
    ax.set_ylabel("Midsection height (μm)")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, label="Pressure (MPa)")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.savefig("../Figures/SI/SI_figure_7_midsection_width_height.png")
    plt.show()
import numpy as np
from skimage.draw import polygon
import cross_section_helpers as csh
import trimesh
import pandas as pd

def process_mesh(file):
    mesh_id = "_".join(file.stem.split("_")[2:4])
    cross_section_type = "confocal"
    pressure = float(file.stem.split("_")[-1])
    pressure = round(pressure, 2)
    section_points_right, section_points_left, section_traces_left, section_traces_right = csh.analyze_stomata_mesh(
        file, num_sections=20, n_points=40, visualize=False
    )
    mid_index = len(section_points_left) // 2
    tip_index = -1
    left_tip = section_points_left[tip_index]
    right_tip = section_points_right[tip_index]
    lr_tip, major_length_l_tip, minor_length_l_tip = csh.calculate_cross_section_aspect_ratios_and_lengths(left_tip)
    rr_tip, major_length_r_tip, minor_length_r_tip = csh.calculate_cross_section_aspect_ratios_and_lengths(right_tip)
    left_midsection = section_points_left[mid_index]
    right_midsection = section_points_right[mid_index]
    lr, major_length_l, minor_length_l = csh.calculate_cross_section_aspect_ratios_and_lengths(left_midsection)
    rr, major_length_r, minor_length_r = csh.calculate_cross_section_aspect_ratios_and_lengths(right_midsection)
    return {
        "Mesh ID": mesh_id,
        "Cross-section type": cross_section_type,
        "Pressure": pressure,
        "Midsection AR left": lr[0],
        "Midsection AR right": rr[0],
        "Midsection Points Left": left_midsection,
        "Midsection Points Right": right_midsection,
        "Tip AR left": lr_tip[0],
        "Tip AR right": rr_tip[0],
        "Tip Points Left": left_tip,
        "Tip Points Right": right_tip,
        "Major length left": major_length_l[0],
        "Major length right": major_length_r[0],
        "Minor length left": minor_length_l[0],
        "Minor length right": minor_length_r[0],
    }

def process_idealised_mesh(file):
    mesh_id = "_".join(file.stem.split("_")[2:4])
    cross_section_type = file.stem.split("_")[4]
    pressure = float(file.stem.split("_")[-1])
    pressure = round(pressure, 2)

    mesh = trimesh.load(file, process=True)
    x_min, y_min, z_min = mesh.bounds[0]
    x_max, y_max, z_max = mesh.bounds[1]
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    tol = 0.4
    midsection_mask = np.abs(mesh.vertices[:, 0] - x_mid) < tol
    midsection_points = mesh.vertices[midsection_mask]
    one_side_mask = midsection_points[:, 1] > y_mid
    one_guard_cell_points = midsection_points[one_side_mask]

    aspect_ratio = csh.calculate_cross_section_aspect_ratios(one_guard_cell_points)
    pore_area = fast_pore_area(mesh.vertices, mesh.faces, step=0.01)

    return {
        "Mesh ID": mesh_id,
        "Cross-section type": cross_section_type,
        "Pressure (MPa)": pressure,
        "Aspect Ratio": aspect_ratio,
        "Pore Area (um^2)": pore_area
    }

def fast_pore_area(vertices, faces, step=0.01):
    verts_2d = vertices[:, :2]
    bb_min = verts_2d.min(axis=0)
    bb_max = verts_2d.max(axis=0)
    size = ((bb_max - bb_min) / step).astype(int) + 2
    raster = np.zeros((size[0], size[1]), dtype=np.uint8)
    origin = bb_min - step

    def to_raster(pt):
        return ((pt - origin) / step).astype(int)

    # Rasterize triangles efficiently
    for tri in faces:
        tri_2d = verts_2d[tri]
        rr, cc = polygon(
            [to_raster(tri_2d[0])[0], to_raster(tri_2d[1])[0], to_raster(tri_2d[2])[0]],
            [to_raster(tri_2d[0])[1], to_raster(tri_2d[1])[1], to_raster(tri_2d[2])[1]],
            raster.shape
        )
        raster[rr, cc] = 1

    # Flood fill from border (same as before)
    from collections import deque
    queue = deque()
    for i in range(size[0]):
        queue.append((i, 0))
        queue.append((i, size[1]-1))
    for j in range(size[1]):
        queue.append((0, j))
        queue.append((size[0]-1, j))
    while queue:
        x, y = queue.popleft()
        if 0 <= x < size[0] and 0 <= y < size[1] and raster[x, y] == 0:
            raster[x, y] = 2
            queue.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])

    pore_pix = np.sum(raster == 0)
    pore_area = pore_pix * step * step
    return pore_area

def get_pore_area_and_volume(df, results_file, mesh_files, pressures):
    mesh_id = "_".join(results_file.stem.split("_")[4:6])
    confocal_data = pd.read_csv(results_file, sep=r'\s+', header=None, names=["Pore Area"])
    confocal_data["Pressure"] = pressures

    for index, row in df[df["Mesh ID"] == mesh_id].iterrows():
        pressure = row["Pressure"]
        pore_area = confocal_data.loc[
            np.isclose(confocal_data["Pressure"], pressure, atol=1e-2),
            "Pore Area"
        ].values
        df.at[index, "Pore Area"] = pore_area[0] if len(pore_area) > 0 else None

    ## Add in the volume information
    for f in mesh_files:
        # Extract mesh_id and pressure from filename
        parts = f.stem.split("_")
        mesh_id = "_".join(parts[2:4])
        # Extract pressure (handles e.g. '1.2' or '2.0')
        pressure_str = parts[4]
        try:
            pressure = float(pressure_str)
        except ValueError:
            # If pressure is like '1.2.obj', remove extension
            pressure = float(pressure_str.split(".")[0] + "." + pressure_str.split(".")[1])

        # Find the matching row
        mask = (
            (df["Mesh ID"] == mesh_id) &
            (df["Pressure"] == pressure)
        )

        mesh = trimesh.load(f)
        vol = mesh.volume
        df.loc[mask, "Volume"] = vol
    return df
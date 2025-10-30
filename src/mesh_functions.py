import numpy as np
from skimage.draw import polygon
import importlib
import cross_section_helpers
importlib.reload(cross_section_helpers)
import cross_section_helpers as csh
import trimesh
import pandas as pd

def get_pore_area_and_volume(mesh_id, pressure, confocal_results_files, confocal_mesh_files):
    # Find pore area from results files
    for f in confocal_results_files:
        file_mesh_id = "_".join(f.stem.split("_")[4:6])
        if file_mesh_id == mesh_id:
            pressures = np.round(np.arange(0, 2.1, 0.1), 2)
            confocal_data = pd.read_csv(f, sep=r'\s+', header=None, names=["Pore Area (um^2)"], engine='python')            
            confocal_data["Pressure (MPa)"] = pressures
            pore_area = confocal_data.loc[
                np.isclose(confocal_data["Pressure (MPa)"], pressure, atol=1e-2),
                "Pore Area (um^2)"
            ].values
            pore_area_val = pore_area[0] if len(pore_area) > 0 else None
            break
    else:
        pore_area_val = None

    # Find volume from mesh files
    for f in confocal_mesh_files:
        parts = f.stem.split("_")
        file_mesh_id = "_".join(parts[2:4])
        pressure_str = parts[4]
        try:
            file_pressure = float(pressure_str)
        except ValueError:
            file_pressure = float(pressure_str.split(".")[0] + "." + pressure_str.split(".")[1])
        if file_mesh_id == mesh_id and np.isclose(file_pressure, pressure, atol=1e-2):
            mesh = trimesh.load(f)
            vol = mesh.volume
            break
    else:
        vol = None

    return pore_area_val, vol

def process_mesh(file, confocal_results_files, confocal_mesh_files, mid_area_left_0 = None, mid_area_right_0 = None):
    mesh_id = "_".join(file.stem.split("_")[2:4])
    cross_section_type = "confocal"
    pressure = float(file.stem.split("_")[-1])
    pressure = round(pressure, 2)

    try:
        section_points_right, section_points_left, section_traces_left, section_traces_right, [spline_x, spline_y, spline_z] = csh.analyze_stomata_mesh(
            file, num_sections=20, n_points=40, visualize=False, mid_area_left_0=mid_area_left_0, mid_area_right_0=mid_area_right_0
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
        pore_area, volume = get_pore_area_and_volume(mesh_id, pressure, confocal_results_files, confocal_mesh_files)
    except Exception as e:
        print(f"Error processing {mesh_id}: {e}")
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
        'Pore Area' :pore_area, 
        'Volume': volume,
        'Spline length': csh.curve_length(spline_x, spline_y, spline_z)
    }

import numpy as np
import trimesh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def process_idealised_mesh(file, debug=False):
    # --- Parse file name metadata ---
    mesh_id = "_".join(file.stem.split("_")[2:4])
    cross_section_type = file.stem.split("_")[4]
    pressure = float(file.stem.split("_")[-1])
    pressure = round(pressure, 2)

    # --- Load mesh ---
    mesh = trimesh.load(file, process=False)

    # --- Slice through midplane perpendicular to Y ---
    # (so we cut the torus vertically through the hole)
    y_mid = mesh.bounds[:, 1].mean()
    tol = 0.2  # thickness of slice
    midsection_mask = np.abs(mesh.vertices[:, 1] - y_mid) < tol
    midsection_points = mesh.vertices[midsection_mask]

    if len(midsection_points) == 0:
        raise ValueError("No vertices found within midsection tolerance.")

    # --- Cluster into two halves (the two guard cells) ---
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(midsection_points)
    labels = kmeans.labels_

    group1 = midsection_points[labels == 0]
    group2 = midsection_points[labels == 1]

    # Pick upper cell (higher z mean)
    if group1[:, 2].mean() > group2[:, 2].mean():
        one_guard_cell_points = group1
    else:
        one_guard_cell_points = group2

    # Ensure 3D shape
    if one_guard_cell_points.ndim != 2 or one_guard_cell_points.shape[1] != 3:
        raise ValueError(
            f"Expected shape (N,3), got {one_guard_cell_points.shape}"
        )

    # --- Calculate aspect ratio ---
    aspect_ratio = csh.calculate_cross_section_aspect_ratios(one_guard_cell_points)

    # --- Optional plot for debugging ---
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(one_guard_cell_points[:, 0], one_guard_cell_points[:, 2], s=10)
        plt.gca().set_aspect('equal')
        plt.title("Selected guard cell cross-section (X vs Z)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()

    # --- Calculate pore area ---
    pore_area = fast_pore_area(mesh.vertices, mesh.faces, step=0.01)

    return {
        "Mesh ID": mesh_id,
        "Cross-section type": cross_section_type,
        "Pressure (MPa)": pressure,
        "Aspect Ratio": aspect_ratio,
        "Pore Area (um^2)": pore_area,
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

def get_pore_area_and_volume_old(df, results_file, mesh_files, pressures):
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

def curve_length(x, y, z):
    points = np.column_stack((x, y, z))
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return segment_lengths.sum()

def process_mesh_pressure(args):
    mesh, p , mid_area_left_0, mid_area_right_0 = args
    test_mesh = f"../Meshes/Onion meshes/pressure_results/Ac_DA_{mesh}_{p:.1f}.obj"
    _, _, _, _, [spline_x, spline_y, spline_z] = csh.analyze_stomata_mesh(test_mesh, mid_area_left_0 = mid_area_left_0, mid_area_right_0 = mid_area_right_0)
    return curve_length(spline_x, spline_y, spline_z)
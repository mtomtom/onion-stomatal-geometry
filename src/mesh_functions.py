import numpy as np
from skimage.draw import polygon
import importlib
import cross_section_helpers
importlib.reload(cross_section_helpers)
import cross_section_helpers as csh
import trimesh
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_fill_holes

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

def process_idealised_mesh_old(file, debug=False):
    # --- Parse file name metadata ---
    parts = file.stem.split("_")
    mesh_id = "_".join(parts[3:5])          # '2_6a'
    cross_section_type = parts[5]           # 'circular'
    pressure = round(float(parts[-1]), 2)

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
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(midsection_points)
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

import numpy as np
import trimesh
from sklearn.cluster import MiniBatchKMeans

def process_idealised_mesh(file, debug=False):
    # --- Parse file name metadata ---
    parts = file.stem.split("_")
    mesh_id = "_".join(parts[3:5])          # '2_6a'
    cross_section_type = parts[5]           # 'circular'
    pressure = round(float(parts[-1]), 2)   # 0.8


    # --- Load mesh ---
    mesh = trimesh.load(file, process=False)

    # --- Slice through midplane perpendicular to Y ---
    y_mid = mesh.bounds[:, 1].mean()
    tol = 0.2  # thickness of slice

    # Efficient slicing using searchsorted
    y_sorted_idx = np.argsort(mesh.vertices[:,1])
    y_sorted = mesh.vertices[y_sorted_idx,1]
    low, high = y_mid - tol, y_mid + tol
    start = np.searchsorted(y_sorted, low, side='left')
    end   = np.searchsorted(y_sorted, high, side='right')
    midsection_points = mesh.vertices[y_sorted_idx[start:end]]

    if len(midsection_points) == 0:
        raise ValueError("No vertices found within midsection tolerance.")

    # --- Cluster into two halves ---
    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=100, n_init=1, random_state=0)
    labels = kmeans.fit_predict(midsection_points)

    group1 = midsection_points[labels == 0]
    group2 = midsection_points[labels == 1]

    # Pick upper cell (higher z mean)
    one_guard_cell_points = group1 if group1[:,2].mean() > group2[:,2].mean() else group2

    # Ensure 3D shape
    if one_guard_cell_points.ndim != 2 or one_guard_cell_points.shape[1] != 3:
        raise ValueError(f"Expected shape (N,3), got {one_guard_cell_points.shape}")

    # --- Calculate aspect ratio ---
    aspect_ratio = csh.calculate_cross_section_aspect_ratios(one_guard_cell_points)

    # --- Optional plot for debugging (random subset to speed up) ---
    if debug:
        import matplotlib.pyplot as plt
        subset = one_guard_cell_points
        if len(one_guard_cell_points) > 2000:
            idx = np.random.choice(len(one_guard_cell_points), 2000, replace=False)
            subset = one_guard_cell_points[idx]
        plt.figure(figsize=(6,6))
        plt.scatter(subset[:,0], subset[:,2], s=10)
        plt.gca().set_aspect('equal')
        plt.title("Selected guard cell cross-section (X vs Z)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()

    # --- Calculate pore area ---
    pore_area = fast_pore_area_fast(mesh, step=0.01)

    return {
        "Mesh ID": mesh_id,
        "Cross-section type": cross_section_type,
        "Pressure (MPa)": pressure,
        "Aspect Ratio": aspect_ratio,
        "Pore Area (um^2)": pore_area,
    }

import numpy as np
from scipy import ndimage
from matplotlib.path import Path

import numpy as np
from scipy.spatial import ConvexHull

def fast_pore_area_fast(mesh, step=0.01):
    """
    Faster raster-based pore area estimate for planar (x,y) mesh.
    Uses vectorized rasterization and ndimage filling for speed.
    No file I/O overhead—mesh object must be pre-loaded.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Loaded mesh object
    step : float
        Rasterization step size (default 0.01)
        
    Returns
    -------
    float
        Pore area in mesh units squared
    """
    vertices = mesh.vertices
    faces = mesh.faces
    verts_2d = vertices[:, :2]
    bb_min = verts_2d.min(axis=0)
    bb_max = verts_2d.max(axis=0)
    size = np.ceil((bb_max - bb_min) / step).astype(int) + 3
    raster = np.zeros((size[0], size[1]), dtype=bool)
    origin = bb_min - step

    # Convert all vertices to raster once
    verts_pix = ((verts_2d - origin) / step).astype(int)

    # Vectorized polygon fill for all triangles
    for tri in faces:
        pts = verts_pix[tri]
        rr, cc = polygon(pts[:, 0], pts[:, 1], raster.shape)
        raster[rr, cc] = True

    # Fill holes (flood fill background implicitly)
    filled = binary_fill_holes(raster)

    # pore = filled XOR original = interior holes
    pore_mask = filled ^ raster

    pore_area = np.sum(pore_mask) * step * step
    return pore_area

def fast_pore_area_fast_ar(mesh, step=0.01):
    """
    Faster raster-based pore area estimate with aspect ratio for planar (x,y) mesh.
    Uses vectorized rasterization and ndimage filling for speed.
    No file I/O overhead—mesh object must be pre-loaded.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Loaded mesh object
    step : float
        Rasterization step size (default 0.01)
        
    Returns
    -------
    tuple
        (pore_area, aspect_ratio) where aspect_ratio = width/height of pore region
    """
    vertices = mesh.vertices
    faces = mesh.faces
    verts_2d = vertices[:, :2]
    bb_min = verts_2d.min(axis=0)
    bb_max = verts_2d.max(axis=0)
    size = np.ceil((bb_max - bb_min) / step).astype(int) + 3
    raster = np.zeros((size[0], size[1]), dtype=bool)
    origin = bb_min - step

    # Convert all vertices to raster once
    verts_pix = ((verts_2d - origin) / step).astype(int)

    # Vectorized polygon fill for all triangles
    for tri in faces:
        pts = verts_pix[tri]
        rr, cc = polygon(pts[:, 0], pts[:, 1], raster.shape)
        raster[rr, cc] = True

    # Fill holes (flood fill background implicitly)
    filled = binary_fill_holes(raster)

    # pore = filled XOR original = interior holes
    pore_mask = filled ^ raster

    pore_area = np.sum(pore_mask) * step * step

    # --- Aspect ratio calculation ---
    pore_indices = np.argwhere(pore_mask)
    if pore_indices.shape[0] > 0:
        y_coords, x_coords = pore_indices[:, 0], pore_indices[:, 1]
        width = (x_coords.max() - x_coords.min() + 1) * step
        height = (y_coords.max() - y_coords.min() + 1) * step
        aspect_ratio = width / height if height > 0 else float('nan')
    else:
        aspect_ratio = float('nan')

    return pore_area, aspect_ratio

def fast_pore_area_fast_from_file(mesh_file, step=0.01):
    """
    Wrapper for fast_pore_area_fast that loads mesh from file path.
    Safe to use with ProcessPoolExecutor (only serializes file path, not mesh object).
    
    Parameters
    ----------
    mesh_file : Path or str
        Path to mesh file
    step : float
        Rasterization step size (default 0.01)
        
    Returns
    -------
    dict
        Dictionary with Mesh ID, cross-section type, pressure, and pore area
    """
    parts = mesh_file.stem.split("_")
    mesh_id = "_".join(parts[3:5])        
    cross_section_type = parts[5]      
    pressure = round(float(parts[-1]), 2)

    mesh = trimesh.load(mesh_file, process=False)
    pore_area = fast_pore_area_fast(mesh, step=step)
    
    return {
        "Mesh ID": mesh_id,
        "Cross-section type": cross_section_type,
        "Pressure": pressure,
        "Pore Area": pore_area,
    }

def cross_section_points_and_aspect(mesh_path, tol=0.5, side="left", visualize=False):
    """
    Extract cross-section points for one guard cell from a torus-like mesh.
    Assumes mesh is centered at (0,0,0) and symmetric along Y and X.
    
    Parameters
    ----------
    mesh_path : str
        Path to OBJ mesh.
    tol : float
        Half-thickness of slice around Y=0.
    side : str
        "left" for x < 0 (default), "right" for x > 0.
    visualize : bool
        If True, plots the 2D cross-section.
    
    Returns
    -------
    points_2D : (N, 2) ndarray
        Cross-section coordinates in the XZ plane.
    aspect_ratio : float
        Width / height of the cross-section.
    """

    parts = mesh_path.stem.split("_")

    try:
        mesh_id = "_".join(parts[3:5])          # e.g. "1_2"
        cross_section_type = parts[5]           # e.g. "circular"
        pressure = round(float(parts[-1]), 2)   # e.g. 0.0
    except (IndexError, ValueError):
        raise ValueError(f"Unexpected filename format: {mesh_path.name}")

    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    # --- Slice points near the Y midplane ---
    y_mid = 0.0
    low, high = y_mid - tol, y_mid + tol
    mask = (mesh.vertices[:, 1] > low) & (mesh.vertices[:, 1] < high)
    midsection_points = mesh.vertices[mask]

    if midsection_points.shape[0] == 0:
        raise ValueError("No vertices found near Y=0 midplane.")

    # --- Take only one half (left or right) ---
    if side == "left":
        half_points = midsection_points[midsection_points[:, 0] < 0]
    elif side == "right":
        half_points = midsection_points[midsection_points[:, 0] > 0]
    else:
        raise ValueError("side must be 'left' or 'right'.")

    if half_points.shape[0] == 0:
        raise ValueError(f"No vertices found on {side} half.")

    # --- Project to XZ plane ---
    points_2D = half_points[:, [0, 2]]

    # --- Compute aspect ratio ---
    min_pt = points_2D.min(axis=0)
    max_pt = points_2D.max(axis=0)
    width, height = max_pt - min_pt
    aspect_ratio = width / height if height > 0 else np.nan

    # --- Optional visualization ---
    if visualize:
        plt.figure(figsize=(5, 5))
        plt.scatter(points_2D[:, 0], points_2D[:, 1], s=2, alpha=0.6)
        plt.gca().set_aspect("equal")
        plt.title(f"{side.capitalize()} cross-section (aspect = {aspect_ratio:.3f})")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()

    return {
            "Mesh ID": mesh_id,
            "Cross-section type": cross_section_type,
            "Pressure": pressure,
            "Cross section": points_2D,
            "Aspect Ratio": aspect_ratio,
        }


def fast_pore_area_old_old(vertices, faces, step=0.02):
    """
    Faster pore area calculation by vectorizing triangle rasterization.
    """
    # 1. Take X/Y coordinates
    verts_2d = vertices[:, :2]
    bb_min = verts_2d.min(axis=0)
    bb_max = verts_2d.max(axis=0)

    # 2. Create raster grid
    size = np.ceil((bb_max - bb_min) / step).astype(int) + 2
    raster = np.zeros((size[1], size[0]), dtype=np.uint8)
    origin = bb_min - step

    # 3. Transform vertices to raster coordinates
    verts_px = ((verts_2d - origin) / step).astype(int)

    # 4. Vectorized triangle rasterization
    for tri in faces:
        tri_px = verts_px[tri]
        # Compute bounding box of the triangle in pixel coordinates
        xmin = tri_px[:, 0].min()
        xmax = tri_px[:, 0].max() + 1
        ymin = tri_px[:, 1].min()
        ymax = tri_px[:, 1].max() + 1

        # Generate all pixel coordinates in bounding box
        x_grid, y_grid = np.meshgrid(np.arange(xmin, xmax),
                                     np.arange(ymin, ymax))
        pts = np.vstack((x_grid.ravel(), y_grid.ravel())).T

        # Use matplotlib.path to check which points are inside triangle
        path = Path(tri_px)
        mask = path.contains_points(pts).reshape(x_grid.shape)
        raster[y_grid, x_grid] |= mask.astype(np.uint8)

    # 5. Fill holes
    filled = ndimage.binary_fill_holes(raster)

    # 6. Count pore pixels (holes)
    pore_pix = np.sum(filled & ~raster)

    # 7. Convert to area
    pore_area = pore_pix * step * step
    return pore_area



def fast_pore_area_old(vertices, faces, step=0.02):
    """
    Vectorized and faster pore area calculation.
    """
    verts_2d = vertices[:, :2]
    bb_min = verts_2d.min(axis=0)
    bb_max = verts_2d.max(axis=0)
    size = np.ceil((bb_max - bb_min) / step).astype(int) + 2
    raster = np.zeros((size[1], size[0]), dtype=np.uint8)
    origin = bb_min - step

    # --- Vectorize coordinate transform ---
    verts_px = ((verts_2d - origin) / step).astype(int)

    # --- Rasterize all triangles ---
    # Group into triplets of x/y coords
    for tri in faces:
        tri_px = verts_px[tri]
        rr, cc = polygon(tri_px[:, 1], tri_px[:, 0], raster.shape)
        raster[rr, cc] = 1

    # --- Flood fill (use fast C-optimized SciPy) ---
    filled = ndimage.binary_fill_holes(raster)
    pore_pix = np.sum(filled & ~raster)

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
    mesh, p , mid_area_left_0, mid_area_right_0, stiffening = args
    if stiffening == "isotropic":
        test_mesh = f"../Meshes/Onion meshes/pressure_results/Ac_DA_{mesh}_{p:.1f}.obj"
    if stiffening == "anisotropic":
        test_mesh = f"../Meshes/Onion meshes anisotropy/pressure_results/Ac_DA_{mesh}_{p:.1f}.obj"
    _, _, _, _, [spline_x, spline_y, spline_z] = csh.analyze_stomata_mesh(test_mesh, mid_area_left_0 = mid_area_left_0, mid_area_right_0 = mid_area_right_0)
    return curve_length(spline_x, spline_y, spline_z)
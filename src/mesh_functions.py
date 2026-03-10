import numpy as np
from skimage.draw import polygon
import importlib
import cross_section_helpers
importlib.reload(cross_section_helpers)
import cross_section_helpers as csh
import trimesh
import pandas as pd
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_midsection_area_lookup(csv_path="../output/midsection_areas.csv"):
    """Create mesh_id -> (area_left, area_right) lookup dictionary."""
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        mesh_id = "_".join(Path(row['mesh_file']).stem.split("_")[2:4])
        lookup[mesh_id] = (row['midsection_area1'], row['midsection_area2'])
    return lookup

def process_mesh_batch(mesh_files, results_files, 
                       mid_area_dict, output_csv, 
                       description="Processing meshes"):
    """
    Process a batch of meshes in parallel with progress reporting.
    
    Parameters
    ----------
    mesh_files : list of Path
        Mesh files to process
    results_files : list of Path
        Corresponding results files
    mid_area_dict : dict
        mesh_id -> (area_left, area_right) lookup
    output_csv : str
        Output CSV filename
    description : str
        Description for progress messages
    
    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    print(f"{description}: {len(mesh_files)} files...")
    
    results = []
    missing_ids = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for file in mesh_files:
            mesh_id = "_".join(file.stem.split("_")[2:4])
            if mesh_id not in mid_area_dict:
                missing_ids.append(mesh_id)
                continue
            
            futures.append(
                executor.submit(
                    process_mesh, file, results_files, mesh_files,
                    mid_area_left_0=mid_area_dict[mesh_id][0],
                    mid_area_right_0=mid_area_dict[mesh_id][1]
                )
            )
        
        for future in as_completed(futures):
            results.append(future.result())
    
    if missing_ids:
        print(f"  ⚠ Skipped {len(set(missing_ids))} meshes: {sorted(set(missing_ids))}")
    
    df = pd.DataFrame(results)
    if "Mesh ID" in df.columns:
        df.sort_values(by=["Mesh ID", "Pressure"], inplace=True)
        print(f"  ✓ {len(df)} measurements from {len(df['Mesh ID'].unique())} meshes")
    else:
        print("  ⚠ No results returned")
    
    df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved to {output_csv}")
    
    return df

def get_pore_area_and_volume(mesh_id, pressure, confocal_results_files, confocal_mesh_files):
    """Extract pore area and volume for a specific mesh at a given pressure.
    
    Searches through confocal results and mesh files to find matching data
    for the specified mesh ID and pressure.
    
    Parameters
    ----------
    mesh_id : str
        Mesh identifier (e.g., "1_2").
    pressure : float
        Pressure value in MPa.
    confocal_results_files : list of Path
        List of file paths to confocal pore area result files.
    confocal_mesh_files : list of Path
        List of file paths to confocal mesh files.
    
    Returns
    -------
    tuple of (float or None, float or None)
        (pore_area, volume) for the specified mesh and pressure.
        Returns None for values if not found.
    """
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
    """Process a confocal stomata mesh to extract geometric measurements.
    
    Analyzes a stomata mesh to extract cross-section aspect ratios, dimensions,
    pore area, volume, and spline length at both midsection and tip locations
    for left and right guard cells.
    
    Parameters
    ----------
    file : Path
        Path to mesh file.
    confocal_results_files : list of Path
        List of confocal pore area result files.
    confocal_mesh_files : list of Path
        List of confocal mesh files for volume calculation.
    mid_area_left_0 : float, optional
        Reference mid-area for left guard cell at zero pressure.
    mid_area_right_0 : float, optional
        Reference mid-area for right guard cell at zero pressure.
    
    Returns
    -------
    dict
        Dictionary containing mesh ID, pressure, aspect ratios, dimensions,
        cross-section points, pore area, volume, and spline length.
    """
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


def process_idealised_mesh(file, debug=False):
    """Process an idealised stomata mesh to extract geometric properties.
    
    Analyzes an idealised mesh by slicing through the Y-midplane, clustering
    vertices into guard cells, and calculating aspect ratio and pore area.
    
    Parameters
    ----------
    file : Path
        Path to idealised mesh file. Filename must follow convention:
        'idealised_final_{mesh_id}_{cross_section_type}_{pressure}.obj'
    debug : bool, optional
        If True, displays debug visualization of selected guard cell (default: False).
    
    Returns
    -------
    dict
        Dictionary containing mesh ID, cross-section type, pressure,
        aspect ratio, and pore area.
    
    Raises
    ------
    ValueError
        If no vertices found within midsection tolerance or if vertex array
        has unexpected shape.
    """
    parts = file.stem.split("_")
    mesh_id = "_".join(parts[3:5])          # '2_6a'
    cross_section_type = parts[5]           # 'circular'
    pressure = round(float(parts[-1]), 2)   # 0.8


    # --- Load mesh ---
    mesh = trimesh.load(file, process=False)

    # --- Slice through midplane perpendicular to Y ---
    y_mid = mesh.bounds[:, 1].mean()
    tol = 0.25  # thickness of slice

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
    pore_area = fast_pore_area(mesh, step=0.05)

    return {
        "Mesh ID": mesh_id,
        "Cross-section type": cross_section_type,
        "Pressure (MPa)": pressure,
        "Aspect Ratio": aspect_ratio,
        "Pore Area (um^2)": pore_area,
    }


def fast_pore_area(mesh, step=0.05):
    """Calculate pore area using fast raster-based method.
    
    Rasterizes the mesh onto a 2D grid, fills holes using binary morphology,
    and computes pore area as the XOR of filled and original rasters.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Loaded mesh object.
    step : float, optional
        Rasterization step size in mesh units (default: 0.05).
        Smaller values increase accuracy but slow computation.
    
    Returns
    -------
    float
        Pore area in mesh units squared.
    
    Notes
    -----
    Uses vectorized polygon rasterization and scipy's binary_fill_holes
    for efficient computation. The pore area is calculated as the area
    of interior holes (filled XOR original).
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

def cross_section_points_and_aspect(mesh_path, tol=0.5, side="left", visualize=False):
    """Extract cross-section points and aspect ratio for one guard cell.
    
    Slices a torus-like stomata mesh near the Y-midplane and extracts points
    for one guard cell (left or right half). Assumes mesh is centered at origin
    and symmetric along Y and X axes.
    
    Parameters
    ----------
    mesh_path : Path
        Path to OBJ mesh file.
    tol : float, optional
        Half-thickness of slice around Y=0 (default: 0.5).
    side : {'left', 'right'}, optional
        Which guard cell to extract: 'left' for x < 0, 'right' for x > 0 (default: 'left').
    visualize : bool, optional
        If True, displays 2D scatter plot of the cross-section (default: False).
    
    Returns
    -------
    dict
        Dictionary containing mesh ID, cross-section type, pressure,
        cross-section points (N, 2) array in XZ plane, and aspect ratio.
    
    Raises
    ------
    ValueError
        If filename format is unexpected, no vertices found near Y=0,
        no vertices on specified side, or invalid side parameter.
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


def curve_length(x, y, z):
    """Calculate total arc length of a 3D curve.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of curve points.
    y : array_like
        Y-coordinates of curve points.
    z : array_like
        Z-coordinates of curve points.
    
    Returns
    -------
    float
        Total arc length computed as sum of segment lengths.
    """
    points = np.column_stack((x, y, z))
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return segment_lengths.sum()

def process_mesh_pressure(args):
    """Process a pressure simulation mesh to calculate spline length.
    
    Loads a mesh from a pressure simulation, analyzes its stomata geometry,
    and computes the centreline spline length.
    
    Parameters
    ----------
    args : tuple
        (mesh, p, mid_area_left_0, mid_area_right_0, stiffening) where:
        - mesh : str - Mesh identifier
        - p : float - Pressure value
        - mid_area_left_0 : float - Reference left guard cell mid-area
        - mid_area_right_0 : float - Reference right guard cell mid-area
        - stiffening : {'isotropic', 'anisotropic'} - Material model type
    
    Returns
    -------
    float
        Spline length of the stomata centreline.
    """
    mesh, p , mid_area_left_0, mid_area_right_0, stiffening = args
    if stiffening == "isotropic":
        test_mesh = f"../Meshes/Onion meshes/pressure_results/Ac_DA_{mesh}_{p:.1f}.obj"
    if stiffening == "anisotropic":
        test_mesh = f"../Meshes/Onion meshes anisotropy/pressure_results/Ac_DA_{mesh}_{p:.1f}.obj"
    _, _, _, _, [spline_x, spline_y, spline_z] = csh.analyze_stomata_mesh(test_mesh, mid_area_left_0 = mid_area_left_0, mid_area_right_0 = mid_area_right_0)
    return curve_length(spline_x, spline_y, spline_z)
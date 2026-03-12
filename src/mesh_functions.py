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
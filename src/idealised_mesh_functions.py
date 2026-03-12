import cross_section_helpers as csh
import generate_idealised_mesh_new as gim
import numpy as np
from skimage.draw import polygon
import trimesh
from mesh_functions import fast_pore_area

def orient_top_view(tri: trimesh.Trimesh) -> trimesh.Trimesh:
    """Orient a mesh to top view using principal component analysis.
    
    Aligns the mesh so that the smallest-variance axis points along the Z-axis,
    effectively providing a top-down view of the mesh.
    
    Parameters
    ----------
    tri : trimesh.Trimesh
        Input mesh to be oriented.
    
    Returns
    -------
    trimesh.Trimesh
        Reoriented mesh with smallest-variance axis aligned to Z.
    """
    V = tri.vertices - tri.vertices.mean(axis=0)
    # SVD on covariance
    U, S, VT = np.linalg.svd(np.cov(V.T))
    axes = VT  # principal axes
    # Align smallest-variance axis to Z (torus plane normal)
    z_target = np.array([0.0, 0.0, 1.0])
    z_src = axes[-1]
    v = np.cross(z_src, z_target)
    c = float(np.dot(z_src, z_target))
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    tri_oriented = tri.copy()
    tri_oriented.apply_translation(-tri_oriented.vertices.mean(axis=0))
    tri_oriented.apply_transform(np.block([[R, np.zeros((3,1))],[np.zeros((1,3)), 1]]))
    return tri_oriented

def align_inplane_to_Y(tri: trimesh.Trimesh) -> trimesh.Trimesh:
    """Align mesh in-plane so the major XY axis points along +Y (vertical).
    
    Rotates the mesh around the Z-axis to align its major in-plane axis with
    the Y-axis, and ensures the minor axis points toward +X.
    
    Parameters
    ----------
    tri : trimesh.Trimesh
        Input mesh to be aligned.
    
    Returns
    -------
    trimesh.Trimesh
        Mesh with major axis aligned to +Y and minor axis to +X.
    """
    verts = tri.vertices.copy()
    xy = verts[:, :2] - verts[:, :2].mean(axis=0)
    C = np.cov(xy.T)
    w, v = np.linalg.eig(C)  # columns of v are eigenvectors
    idx = int(np.argmax(w))
    major = v[:, idx]
    # Rotation around Z by delta so that major -> +Y
    alpha = float(np.arctan2(major[1], major[0]))  # current angle of major vs +X
    delta = np.pi/2 - alpha  # rotate so it points to +Y
    ca, sa = np.cos(delta), np.sin(delta)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    tri_rot = tri.copy()
    tri_rot.apply_transform(np.block([[Rz, np.zeros((3,1))],[np.zeros((1,3)), 1]]))
    # Optional: ensure a consistent left/right (minor axis pointing +X)
    minor = v[:, 1-idx]  # the other eigenvector
    minor_rot = Rz[:2, :2] @ minor
    if minor_rot[0] < 0:  # flip 180° if pointing to -X
        Rz_flip = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        tri_rot.apply_transform(np.block([[Rz_flip, np.zeros((3,1))],[np.zeros((1,3)), 1]]))
    return tri_rot





def get_major_minor_stomata(mesh):
    """Calculate major and minor axis lengths of a stomata mesh.
    
    Orients the mesh to top view, aligns it to the Y-axis, and calculates
    the lengths along the major and minor principal axes.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input stomata mesh.
    
    Returns
    -------
    tuple of (float, float)
        Major and minor axis lengths of the stomata.
    """
    mesh = orient_top_view(mesh)
    mesh = align_inplane_to_Y(mesh) 
    verts = mesh.vertices
    xy = verts[:, :2] - verts[:, :2].mean(axis=0)
    C = np.cov(xy.T)
    w, v = np.linalg.eig(C)  # columns of v are eigenvectors
    idx = int(np.argmax(w))
    major = v[:, idx]
    minor = v[:, 1-idx]
    # Project vertices onto major and minor axes to get lengths
    proj_major = xy @ major
    proj_minor = xy @ minor
    length_major = proj_major.max() - proj_major.min()
    length_minor = proj_minor.max() - proj_minor.min()
    return length_major, length_minor


def run_idealised_mesh_creation(selected_meshes, df, major_segments=100, minor_segments=30, ar = "oval"):
    """Create idealised stomata meshes using empirical data.
    
    Generates idealised elliptical torus meshes matching target pore areas and
    dimensions from experimental data. Iteratively refines mesh parameters to
    achieve target pore area within tolerance.
    
    Parameters
    ----------
    selected_meshes : list
        List of mesh IDs to process.
    df : pandas.DataFrame
        DataFrame containing mesh data with columns for pressure, aspect ratios,
        lengths, and pore areas.
    major_segments : int, optional
        Number of segments for major radius (default: 100).
    minor_segments : int, optional
        Number of segments for minor radius (default: 30).
    ar : str, optional
        Shape type, either "oval" or "circular" (default: "oval").
    
    Notes
    -----
    Saves final meshes to ../Meshes/Idealised/ directory.
    Uses iterative refinement with up to 10 iterations to match target pore area.
    """
    import trimesh
    import tempfile
    from pathlib import Path
    
    for mesh_id in selected_meshes:
        ## Create a temporary folder to store the intermediate meshes
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Processing mesh: {mesh_id}")
            
            
            # Get baseline data (pressure = 0.0) from dataframe
            baseline_data = df[(df["Mesh ID"] == mesh_id) & (df["Pressure"] == 0.0)]
            
            if baseline_data.empty:
                print(f"No baseline data found for mesh {mesh_id}, skipping...")
                continue
                
            # Extract target parameters from dataframe
            target_pore_area = baseline_data["Pore Area"].values[0]
            print(f"Target pore area: {target_pore_area}")
            
            # Handle numpy arrays in the dataframe columns (stored as strings)
            import ast
            
            def parse_dataframe_array(value):
                """Parse string representations of numpy arrays from dataframe.
                
                Parameters
                ----------
                value : str or array-like
                    String representation of array or actual numeric value.
                
                Returns
                -------
                float
                    Parsed numeric value.
                """
                if isinstance(value, str):
                    # Remove numpy type prefixes and convert to list
                    clean_str = value.replace('np.float64(', '').replace(')', '')
                    if clean_str.startswith('[') and clean_str.endswith(']'):
                        # It's a list representation
                        parsed = ast.literal_eval(clean_str)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            return float(parsed[0])
                        else:
                            return float(parsed)
                    else:
                        return float(clean_str)
                else:
                    # Handle actual arrays/numbers
                    if hasattr(value, '__len__') and len(value) > 0:
                        return float(value[0])
                    else:
                        return float(value)
            
            midsection_ar_left = baseline_data["Midsection AR left"].values[0]
            major_length_left = baseline_data["Measured major length"].values[0]  
            minor_length_left = baseline_data["Measured minor length"].values[0]
            
            # Parse the values correctly
            target_midsection_aspect_ratio = parse_dataframe_array(midsection_ar_left)
            target_length = parse_dataframe_array(major_length_left)
            target_width = parse_dataframe_array(minor_length_left)

            margin = 0.01 * target_width 

            # Cap the minor radius so it can NEVER overlap the center
            max_allowable_minor_radius = (target_width / 4) - margin
            
            print(f"Target midsection aspect ratio: {target_midsection_aspect_ratio}")
            print(f"Target length: {target_length}")
            print(f"Target width: {target_width}")
            
            # Initialize radii - estimate minor radii based on aspect ratio and dimensions
            # This is an initial guess that will be refined in the iteration loop
            major_length_left_str = baseline_data["Major length left"].values[0]
            minor_length_left_str = baseline_data["Minor length left"].values[0]

            if ar == "circular":
                minor_length_left_str = baseline_data["Major length left"].values[0]

            # Use your parsing function to convert string to float
            major_length_left = parse_dataframe_array(major_length_left_str)
            minor_length_left = parse_dataframe_array(minor_length_left_str)

            minor_radius_a = major_length_left / 2
            minor_radius_b = minor_length_left / 2

            minor_radius_a = min(minor_radius_a, max_allowable_minor_radius)
            minor_radius_b = min(minor_radius_b, max_allowable_minor_radius)
            
            # Calculate major radii to maintain target dimensions
            major_radius_a = (target_width - 2 * minor_radius_a) / 2
            major_radius_b = (target_length - 2 * minor_radius_a) / 2
            
            print(f"Initial minor radii: a={minor_radius_a:.4f}, b={minor_radius_b:.4f}")
            print(f"Initial major radii: a={major_radius_a:.4f}, b={major_radius_b:.4f}")
            
            # Iterative refinement to match target pore area
            for iteration in range(10):
                print(f"\nAttempt {iteration+1}:")
            
                # Create a scene with both halves — keeps them as separate parts
                try:
                    left_mesh, right_mesh = gim.create_elliptical_torus_with_shared_wall(
                        major_radius_a, major_radius_b,
                        minor_radius_a, minor_radius_b,
                        major_segments, minor_segments,
                         wall_thickness=0.0
                    )

                    # Scene keeps both halves separate (no merge)
                    scene = trimesh.Scene({'left': left_mesh, 'right': right_mesh})

                except Exception as e:
                    print(f"Error creating mesh: {e}")
                    break

                # --- Export the newly created scene for inspection ---
                mesh_filename = Path(tmpdir) / f"idealised_attempt_{mesh_id}_{iteration+1}.obj"
                scene.export(mesh_filename)
                print(f"Exported scene for inspection: {mesh_filename}")

                # Check the pore area
                try:
                    ideal_mesh = trimesh.load(mesh_filename, force='mesh')
                    pore_area = fast_pore_area(ideal_mesh, step=0.05)
                    print(f"Central pore area: {pore_area:.2f}")
                except Exception as e:
                    print(f"Error calculating pore area: {e}")
                    break
                
                # Calculate difference from target
                diff = target_pore_area - pore_area
                print(f"Difference from target pore area: {diff:.2f}")
                
                # Check if we're close enough
                if abs(diff) < 0.5:
                    print("Pore area within acceptable range. Stopping iterations.")
                    final_filename = f'../Meshes/Idealised/idealised_final_{mesh_id}_oval.obj'
                    if ar == "circular":
                        final_filename = f'../Meshes/Idealised/idealised_final_{mesh_id}_circular.obj'

                    ## Reload the mesh as a scene, so we can save the parts
                    #final_scene = trimesh.load(mesh_filename, force='scene')
                    #final_scene.export(final_filename)
                    #print(f"Final mesh saved as: {final_filename}")
                    import shutil
                    shutil.copy(mesh_filename, final_filename)
                    print(f"Final mesh saved as: {final_filename}")
                    break
                    
                # Adjust minor radii to correct pore area
                adjustment_factor = 0.1 * (diff / target_pore_area)
                adjustment = adjustment_factor * minor_radius_a
                
                if diff > 0:  # Need larger pore area, reduce minor radii
                    minor_radius_a -= abs(adjustment)
                    minor_radius_b -= abs(adjustment)
                else:  # Need smaller pore area, increase minor radii
                    minor_radius_a += abs(adjustment)
                    minor_radius_b += abs(adjustment)
                    
                # Ensure radii don't become negative or too small
                #minor_radius_a = max(minor_radius_a, 0.1)
                #minor_radius_b = max(minor_radius_b, 0.1)
                minor_radius_a = min(minor_radius_a, max_allowable_minor_radius)
                minor_radius_b = min(minor_radius_b, max_allowable_minor_radius)
                
                print(f"Adjusting minor radii by {adjustment:.4f} to a={minor_radius_a:.4f}, b={minor_radius_b:.4f}")
                
                # Recalculate major radii to maintain target dimensions
                major_radius_a = (target_width - 2 * minor_radius_a) / 2
                major_radius_b = (target_length - 2 * minor_radius_a) / 2
                
                # Ensure major radii are positive
                if major_radius_a <= 0 or major_radius_b <= 0:
                    print("Major radii became negative or zero. Cannot continue with current parameters.")
                    break
                    
                print(f"New major radii: a={major_radius_a:.4f}, b={major_radius_b:.4f}")
                
            else:
                print("Maximum iterations reached without convergence.")
                
            print(f"Completed processing mesh {mesh_id}\n" + "="*50)


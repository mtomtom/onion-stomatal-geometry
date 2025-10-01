import cross_section_helpers as csh
import generate_idealised_mesh_new as gim
import numpy as np
from skimage.draw import polygon
import trimesh

def orient_top_view(tri: trimesh.Trimesh) -> trimesh.Trimesh:
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

# Helper: align in-plane so the major XY axis points along +Y (vertical)
def align_inplane_to_Y(tri: trimesh.Trimesh) -> trimesh.Trimesh:
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

def get_cross_sections(mesh_list, meshdir_path):
    section_right = []
    section_left = []
    for sm in mesh_list:
        mesh_path = meshdir_path + "Ac_DA_" + sm + ".obj"
        section_points_right, section_points_left, _, _ = csh.analyze_stomata_mesh(mesh_path, num_sections=20, n_points=40, visualize=False)
        section_right.append(section_points_right)
        section_left.append(section_points_left)
    return section_right, section_left

def get_aspect_ratios(section_right, section_left):
    cross_section_ratios = []
    major_lengths = []
    minor_lengths = []
    for right, left in zip(section_right, section_left):
        lr, major_length_l, minor_length_l = csh.calculate_cross_section_aspect_ratios_and_lengths(left)
        rr, major_length_r, minor_length_r = csh.calculate_cross_section_aspect_ratios_and_lengths(right)
        cross_section_ratios.append((lr, rr))
        major_lengths.append((major_length_l, major_length_r))
        minor_lengths.append((minor_length_l, minor_length_r))
    return cross_section_ratios, major_lengths, minor_lengths

def get_midsection_and_tip_data(cross_section_ratios, major_lengths, minor_lengths):

    left_midsection_ar = []
    right_midsection_ar = []
    left_tip_ar = []
    right_tip_ar = []
    left_midsection_major = []
    right_midsection_major = []
    left_midsection_minor = []
    right_midsection_minor = []

    for r, major, minor in zip(cross_section_ratios, major_lengths, minor_lengths):
        ## Get the midsection cross section for each guard cell
        mid_left = r[0][len(r[0]) // 2]
        mid_right = r[1][len(r[1]) // 2]
        
        left_midsection_ar.append(mid_left)
        right_midsection_ar.append(mid_right)
        ## Get the tip cross section for each guard cell
        tip_left = r[0][-1]
        tip_right = r[1][-1]
        
        left_tip_ar.append(tip_left)
        right_tip_ar.append(tip_right)
        ## Get the major lengths for each guard cell - get the midsection values
        major_left = major[0][len(r[0]) // 2]
        major_right = major[1][len(r[1]) // 2]
        
        left_midsection_major.append(major_left)
        right_midsection_major.append(major_right)
        ## Get the minor lengths for each guard cell
        minor_left = minor[0][len(r[0]) // 2]
        minor_right = minor[1][len(r[0]) // 2]
        
        left_midsection_minor.append(minor_left)
        right_midsection_minor.append(minor_right)
    return left_midsection_ar, right_midsection_ar, left_tip_ar, right_tip_ar, left_midsection_major, right_midsection_major, left_midsection_minor, right_midsection_minor

def get_major_minor_stomata(mesh):
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
    """
    Create idealised meshes using data from the dataframe.
    
    Parameters:
    - selected_meshes: List of mesh IDs to process
    - df: DataFrame containing mesh data with columns for pressure, aspect ratios, lengths, and pore areas
    - major_segments: Number of segments for major radius (default: 100)
    - minor_segments: Number of segments for minor radius (default: 30)
    """
    import trimesh
    
    for mesh_id in selected_meshes:
        print(f"Processing mesh: {mesh_id}")
        
        # Load the original mesh for reference
        try:
            mesh = trimesh.load(f"../Meshes/Onion_OBJ/Ac_DA_{mesh_id}.obj", force='mesh')
        except:
            print(f"Could not load mesh file for {mesh_id}, skipping...")
            continue
        
        # Get baseline data (pressure = 0.0) from dataframe
        baseline_data = df[(df["Mesh ID"] == mesh_id) & (df["Pressure (MPa)"] == 0.0)]
        
        if baseline_data.empty:
            print(f"No baseline data found for mesh {mesh_id}, skipping...")
            continue
            
        # Extract target parameters from dataframe
        target_pore_area = baseline_data["Pore Area (um²)"].values[0]
        print(f"Target pore area: {target_pore_area}")
        
        # Handle numpy arrays in the dataframe columns (stored as strings)
        import ast
        
        def parse_dataframe_array(value):
            """Parse string representations of numpy arrays from dataframe"""
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
        
        # Calculate major radii to maintain target dimensions
        major_radius_a = (target_width - 2 * minor_radius_a) / 2
        major_radius_b = (target_length - 2 * minor_radius_a) / 2
        
        print(f"Initial minor radii: a={minor_radius_a:.4f}, b={minor_radius_b:.4f}")
        print(f"Initial major radii: a={major_radius_a:.4f}, b={major_radius_b:.4f}")
        
        # Iterative refinement to match target pore area
        for iteration in range(10):
            print(f"\nAttempt {iteration+1}:")
            
            # Create elliptical torus mesh
            try:
                mesh = gim.create_elliptical_torus(
                    major_radius_a, major_radius_b, 
                    minor_radius_a, minor_radius_b, 
                    major_segments, minor_segments
                )
            except Exception as e:
                print(f"Error creating mesh: {e}")
                break
                
            # Export the mesh for inspection
            mesh_filename = f'idealised_attempt_{mesh_id}_{iteration+1}.ply'
            mesh.export(mesh_filename)
            
            # Check the pore area
            try:
                ideal_mesh = trimesh.load(mesh_filename, force='mesh')
                pore_area = fast_pore_area(ideal_mesh.vertices, ideal_mesh.faces, step=0.01)
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
                final_filename = f'idealised_final_{mesh_id}_oval.ply'
                if ar == "circular":
                    final_filename = f'idealised_final_{mesh_id}_circular.ply'
                mesh.export(final_filename)
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
            minor_radius_a = max(minor_radius_a, 0.1)
            minor_radius_b = max(minor_radius_b, 0.1)
            
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
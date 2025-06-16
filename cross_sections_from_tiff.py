import tifffile
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import os
import trimesh

# Attempt to import necessary functions from other project files
try:
    # Assuming analyze_centerline_sections returns a dict that includes
    # 'raw_section_data_items' (list of dicts, each with 'position_3d', 'tangent_3d', 'points_2d', 'transform', 'valid_geometry')
    # and potentially 'minor_radius' at the top level.
    from full_length_AR_analysis import analyze_centerline_sections
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'analyze_centerline_sections' from 'full_length_AR_analysis.py': {e}")
    print("Please ensure 'full_length_AR_analysis.py' is in the Python path and has this function.")
    # Define a dummy function to allow the script to be parsed, but it will fail at runtime.
    def analyze_centerline_sections(*args, **kwargs):
        raise ImportError("analyze_centerline_sections is not available.")

try:
    from helper_functions import order_points
except ImportError:
    print("Warning: 'order_points' from 'helper_functions.py' not found. Overlay visualizations might be affected.")
    def order_points(points, method="angular", center=None): # Basic fallback
        if points is None or len(points) < 1: return np.array([])
        if method == "angular" and len(points) > 1:
            if center is None: center = np.mean(points, axis=0)
            angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
            return points[np.argsort(angles)]
        return points # Fallback for other cases or if too few points

def visualize_full_mesh_alignment(
    mesh_file_path,
    tiff_stack,
    mesh_to_tiff_world_transform,
    tiff_origin_world_xyz,
    tiff_voxel_size_xyz,
    output_path,
    view_axis='Z', # Axis to view along ('X', 'Y', or 'Z')
    slice_index_vox=None # Specific voxel index for the TIFF slice, None for mid-slice
):
    """
    Visualizes the full mesh (as a point cloud) overlaid on a slice of the TIFF
    to help with alignment.
    """
    print(f"  Generating full mesh alignment view: {output_path}")
    try:
        mesh = trimesh.load_mesh(mesh_file_path, process=False)
        mesh_vertices_local = mesh.vertices
    except Exception as e:
        print(f"    Error loading mesh {mesh_file_path} for alignment view: {e}")
        return

    if tiff_stack is None:
        print("    TIFF stack is None, cannot generate alignment view.")
        return

    # Transform mesh vertices from local to world space
    local_vertices_h = np.hstack((mesh_vertices_local, np.ones((mesh_vertices_local.shape[0], 1))))
    world_vertices_h = (mesh_to_tiff_world_transform @ local_vertices_h.T).T
    mesh_vertices_world = world_vertices_h[:, :3] / world_vertices_h[:, 3, np.newaxis]

    # Convert world mesh vertices to voxel coordinates
    mesh_vertices_voxel = get_voxel_coords(mesh_vertices_world, tiff_origin_world_xyz, tiff_voxel_size_xyz)

    # Prepare TIFF slice
    if slice_index_vox is None:
        if view_axis == 'Z': slice_index_vox = tiff_stack.shape[0] // 2
        elif view_axis == 'Y': slice_index_vox = tiff_stack.shape[1] // 2
        elif view_axis == 'X': slice_index_vox = tiff_stack.shape[2] // 2
        else: slice_index_vox = tiff_stack.shape[0] // 2 # Default to Z mid-slice
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_mesh_points_x_idx, plot_mesh_points_y_idx = 0, 1 # Default for Z-view (X,Y)
    xlabel, ylabel = "Voxel X", "Voxel Y"
    aspect_ratio = tiff_voxel_size_xyz[1] / tiff_voxel_size_xyz[0] # Y/X for Z-view

    if view_axis == 'Z':
        tiff_slice_2d = tiff_stack[slice_index_vox, :, :]
        plot_mesh_points_x_idx, plot_mesh_points_y_idx = 0, 1 # Plot Voxel X, Voxel Y
        xlabel, ylabel = f"Voxel X (size: {tiff_voxel_size_xyz[0]:.3f})", f"Voxel Y (size: {tiff_voxel_size_xyz[1]:.3f})"
        aspect_ratio = tiff_voxel_size_xyz[1] / tiff_voxel_size_xyz[0]
    elif view_axis == 'Y':
        tiff_slice_2d = tiff_stack[:, slice_index_vox, :]
        plot_mesh_points_x_idx, plot_mesh_points_y_idx = 0, 2 # Plot Voxel X, Voxel Z
        xlabel, ylabel = f"Voxel X (size: {tiff_voxel_size_xyz[0]:.3f})", f"Voxel Z (size: {tiff_voxel_size_xyz[2]:.3f})"
        aspect_ratio = tiff_voxel_size_xyz[2] / tiff_voxel_size_xyz[0]
    elif view_axis == 'X':
        tiff_slice_2d = tiff_stack[:, :, slice_index_vox]
        plot_mesh_points_x_idx, plot_mesh_points_y_idx = 1, 2 # Plot Voxel Y, Voxel Z
        xlabel, ylabel = f"Voxel Y (size: {tiff_voxel_size_xyz[1]:.3f})", f"Voxel Z (size: {tiff_voxel_size_xyz[2]:.3f})"
        aspect_ratio = tiff_voxel_size_xyz[2] / tiff_voxel_size_xyz[1]
    else:
        print(f"    Unknown view_axis: {view_axis}. Defaulting to Z-view.")
        tiff_slice_2d = tiff_stack[slice_index_vox, :, :]

    ax.imshow(tiff_slice_2d, cmap='gray', origin='lower', aspect=aspect_ratio)
    ax.scatter(mesh_vertices_voxel[:, plot_mesh_points_x_idx], 
               mesh_vertices_voxel[:, plot_mesh_points_y_idx], 
               s=1, c='red', alpha=0.3)
    
    ax.set_title(f"Full Mesh Alignment on TIFF (View along {view_axis}, Slice: {slice_index_vox})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"    Saved full mesh alignment view: {output_path}")

# --- TIFF Helper Functions (from your existing file) ---
def load_tiff_stack(tiff_path):
    """Loads a TIFF stack."""
    try:
        stack = tifffile.imread(tiff_path)
        print(f"  Loaded TIFF stack '{os.path.basename(tiff_path)}' with shape: {stack.shape} (likely Z,Y,X)")
        return stack
    except Exception as e:
        print(f"  Error loading TIFF {tiff_path}: {e}")
        return None

def get_voxel_coords(world_coords, tiff_origin_world_xyz, voxel_size_world_xyz):
    """
    Converts world coordinates (X,Y,Z) to voxel coordinates (Xv,Yv,Zv).
    Assumes tiff_origin_world_xyz and voxel_size_world_xyz are [X,Y,Z] ordered.
    """
    if voxel_size_world_xyz is None or np.any(np.array(voxel_size_world_xyz) == 0):
        print("  Warning: Voxel size is None or contains zero. Assuming 1-to-1 world-to-voxel mapping for now.")
        return world_coords - tiff_origin_world_xyz
    return (world_coords - tiff_origin_world_xyz) / voxel_size_world_xyz

def resample_slice_from_tiff(
    tiff_stack, plane_origin_world, plane_normal_world,
    plane_x_axis_world, plane_y_axis_world,
    tiff_origin_world_xyz, voxel_size_world_xyz, # These are expected in X,Y,Z world order
    slice_dims_pixels=(100, 100), slice_extent_world=(10.0, 10.0)
):
    """
    Resamples a 2D slice from a 3D TIFF stack on a given plane.
    IMPORTANT: Assumes tiff_stack is ordered (Z,Y,X).
    """
    if tiff_stack is None: return None

    px_coords_y = np.linspace(-slice_extent_world[0] / 2, slice_extent_world[0] / 2, slice_dims_pixels[0])
    px_coords_x = np.linspace(-slice_extent_world[1] / 2, slice_extent_world[1] / 2, slice_dims_pixels[1])
    yy, xx = np.meshgrid(px_coords_y, px_coords_x, indexing='ij')

    points_on_plane_world = plane_origin_world + \
                            xx.ravel()[:, np.newaxis] * plane_x_axis_world + \
                            yy.ravel()[:, np.newaxis] * plane_y_axis_world

    # points_on_plane_voxel will be (N, [Xv, Yv, Zv]) if inputs to get_voxel_coords are XYZ
    points_on_plane_voxel = get_voxel_coords(points_on_plane_world, tiff_origin_world_xyz, voxel_size_world_xyz)

    valid_indices = np.ones(points_on_plane_voxel.shape[0], dtype=bool)
    # Check bounds against tiff_stack dimensions (assumed Z,Y,X)
    # points_on_plane_voxel columns are X,Y,Z. tiff_stack.shape is Z,Y,X.
    # So, points_on_plane_voxel[:,2] is Z_vox, points_on_plane_voxel[:,1] is Y_vox, points_on_plane_voxel[:,0] is X_vox
    if points_on_plane_voxel.shape[1] == 3: # Ensure it's 3D
        valid_indices &= (points_on_plane_voxel[:, 2] >= 0) & (points_on_plane_voxel[:, 2] < tiff_stack.shape[0] - 1) # Z
        valid_indices &= (points_on_plane_voxel[:, 1] >= 0) & (points_on_plane_voxel[:, 1] < tiff_stack.shape[1] - 1) # Y
        valid_indices &= (points_on_plane_voxel[:, 0] >= 0) & (points_on_plane_voxel[:, 0] < tiff_stack.shape[2] - 1) # X
    else:
        print("  Error: points_on_plane_voxel are not 3D.")
        return None

    resampled_values_flat = np.full(points_on_plane_voxel.shape[0], np.nan)

    if np.any(valid_indices):
        # coords_for_map_xyz has rows: [Xv_coords...], [Yv_coords...], [Zv_coords...]
        coords_for_map_xyz = points_on_plane_voxel[valid_indices, :].T

        # map_coordinates expects coordinates in the order of tiff_stack dimensions (Z,Y,X)
        # So, we need to provide [[Zv_coords...], [Yv_coords...], [Xv_coords...]]
        # This maps the Z-voxel coordinates to the 0th dim of stack, Y-voxel to 1st, X-voxel to 2nd.
        mapped_coords_zyx = np.array([
            coords_for_map_xyz[2], # Z voxel coordinates
            coords_for_map_xyz[1], # Y voxel coordinates
            coords_for_map_xyz[0]  # X voxel coordinates
        ])
        
        # Your original code had:
        # # This depends on your convention. Assuming points_on_plane_voxel[:,0] is Z, [:,1] is Y, [:,2] is X:
        # mapped_coords = np.array([coords_for_map[0], coords_for_map[1], coords_for_map[2]])
        # If the above assumption holds true for your `get_voxel_coords` output, then your original line is correct.
        # The `mapped_coords_zyx` version assumes `get_voxel_coords` produces standard X,Y,Z voxel coordinates.
        # CHOOSE THE VERSION THAT MATCHES YOUR `get_voxel_coords` OUTPUT AND `tiff_stack` DIMENSION ORDER.
        # Using the standard interpretation here (XYZ from get_voxel_coords, ZYX for map_coordinates):

        resampled_values_flat[valid_indices] = scipy.ndimage.map_coordinates(
            tiff_stack,
            mapped_coords_zyx, # Use the ZYX ordered coordinates
            order=1,
            mode='constant', cval=np.nan
        )
    
    resampled_slice_2d = resampled_values_flat.reshape(slice_dims_pixels)
    return resampled_slice_2d

# --- New Orthogonal Visualization Function ---
def visualize_mesh_sections_on_orthogonal_tiff_slices(
    mesh_sections_definitions, # These should have their 'position_3d' and 'transform' in TIFF world space
    tiff_stack,
    tiff_origin_world_xyz,    # The true origin of the TIFF in world coordinates
    tiff_voxel_size_xyz,
    output_dir,
    base_name
):
    """
    Visualizes all mesh cross-sections overlaid on representative orthogonal
    slices of the TIFF stack (mid-X, mid-Y, mid-Z).
    Uses the true tiff_origin_world_xyz for plotting.
    """
    if not mesh_sections_definitions or tiff_stack is None:
        print("  Skipping orthogonal TIFF visualization: Missing mesh definitions or TIFF stack.")
        return

    print("  Generating orthogonal TIFF slice visualizations with mesh overlays...")
    os.makedirs(output_dir, exist_ok=True)

    # Determine mid-slice indices for the TIFF stack (Z, Y, X)
    mid_z_vox = tiff_stack.shape[0] // 2
    mid_y_vox = tiff_stack.shape[1] // 2
    mid_x_vox = tiff_stack.shape[2] // 2
    
    slice_info = [
        {'axis_label': 'Z', 'slice_idx_vox': mid_z_vox, 'tiff_dim_idx': 0, 'world_dims_xy_indices': (0, 1), 'voxel_sizes_for_aspect': (tiff_voxel_size_xyz[0], tiff_voxel_size_xyz[1])},
        {'axis_label': 'Y', 'slice_idx_vox': mid_y_vox, 'tiff_dim_idx': 1, 'world_dims_xy_indices': (0, 2), 'voxel_sizes_for_aspect': (tiff_voxel_size_xyz[0], tiff_voxel_size_xyz[2])},
        {'axis_label': 'X', 'slice_idx_vox': mid_x_vox, 'tiff_dim_idx': 2, 'world_dims_xy_indices': (1, 2), 'voxel_sizes_for_aspect': (tiff_voxel_size_xyz[1], tiff_voxel_size_xyz[2])}
    ]

    all_mesh_cs_3d_world_points = []
    for i, section_def in enumerate(mesh_sections_definitions):
        if not section_def.get('valid_geometry', False) or section_def.get('points_2d') is None or section_def.get('transform') is None:
            continue
        
        points_2d_local = section_def['points_2d'] # These are in the section's 2D plane
        transform_matrix = section_def['transform'] # This maps section 2D -> TIFF world 3D
        
        if points_2d_local.shape[0] < 3: continue

        points_2d_h = np.hstack((points_2d_local, 
                                 np.zeros((points_2d_local.shape[0], 1)), 
                                 np.ones((points_2d_local.shape[0], 1))))
        points_3d_world_h = (transform_matrix @ points_2d_h.T).T
        mesh_cs_3d_world = points_3d_world_h[:, :3] # Outline points in TIFF world space
        all_mesh_cs_3d_world_points.append(mesh_cs_3d_world)

    if not all_mesh_cs_3d_world_points:
        print("  No valid 3D mesh sections to visualize on orthogonal slices.")
        return

    for s_info in slice_info:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # ... (code to get tiff_slice_2d, labels, aspect_ratio based on s_info['axis_label']) ...
        if s_info['axis_label'] == 'Z':
            tiff_slice_2d = tiff_stack[s_info['slice_idx_vox'], :, :] 
            ax.set_xlabel(f"Voxel X (World X, size: {tiff_voxel_size_xyz[0]:.3f})")
            ax.set_ylabel(f"Voxel Y (World Y, size: {tiff_voxel_size_xyz[1]:.3f})")
            aspect_ratio = s_info['voxel_sizes_for_aspect'][1] / s_info['voxel_sizes_for_aspect'][0] 
        elif s_info['axis_label'] == 'Y':
            tiff_slice_2d = tiff_stack[:, s_info['slice_idx_vox'], :] 
            ax.set_xlabel(f"Voxel X (World X, size: {tiff_voxel_size_xyz[0]:.3f})")
            ax.set_ylabel(f"Voxel Z (World Z, size: {tiff_voxel_size_xyz[2]:.3f})")
            aspect_ratio = s_info['voxel_sizes_for_aspect'][1] / s_info['voxel_sizes_for_aspect'][0] 
        elif s_info['axis_label'] == 'X':
            tiff_slice_2d = tiff_stack[:, :, s_info['slice_idx_vox']] 
            ax.set_xlabel(f"Voxel Y (World Y, size: {tiff_voxel_size_xyz[1]:.3f})")
            ax.set_ylabel(f"Voxel Z (World Z, size: {tiff_voxel_size_xyz[2]:.3f})")
            aspect_ratio = s_info['voxel_sizes_for_aspect'][1] / s_info['voxel_sizes_for_aspect'][0] 
        else:
            continue
        ax.imshow(tiff_slice_2d, cmap='gray', origin='lower', aspect=aspect_ratio)

        for mesh_cs_3d_world in all_mesh_cs_3d_world_points:
            # Project 3D world points to voxel coordinates USING THE TRUE TIFF ORIGIN
            mesh_cs_voxel_coords = get_voxel_coords(
                mesh_cs_3d_world, 
                tiff_origin_world_xyz, # <<< THIS IS THE CORRECTED LINE
                tiff_voxel_size_xyz
            )
            
            plot_coords_pixel = mesh_cs_voxel_coords[:, s_info['world_dims_xy_indices']]
            
            try:
                ordered_plot_coords = order_points(plot_coords_pixel, method="angular")
                if ordered_plot_coords.shape[0] >=3:
                     ax.plot(np.append(ordered_plot_coords[:, 0], ordered_plot_coords[0, 0]),
                            np.append(ordered_plot_coords[:, 1], ordered_plot_coords[0, 1]),
                            linewidth=0.7, alpha=0.6, color='cyan') # Changed color for distinction
            except Exception as e_order:
                print(f"    Could not order/plot points for a mesh section on {s_info['axis_label']}-slice: {e_order}")

        ax.set_title(f"Mesh Sections on Orthogonal TIFF Slice (Axis: {s_info['axis_label']}, Voxel Index: {s_info['slice_idx_vox']})")
        
        plot_filename = os.path.join(output_dir, f"{base_name}_ortho_slice_{s_info['axis_label']}_{s_info['slice_idx_vox']}.png")
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)
        print(f"    Saved orthogonal slice visualization: {plot_filename}")

# --- New Orchestration and Visualization Functions ---

def run_tiff_cross_section_analysis(
    mesh_file_path,
    tiff_file_path,
    tiff_voxel_size_xyz,
    tiff_origin_world_xyz=np.array([0.0, 0.0, 0.0]),
    mesh_to_tiff_world_transform=None,
    num_mesh_sections=15,
    output_dir="tiff_cs_results",
    visualize_mesh_analysis_output=False,
    visualize_tiff_sections=True,
    tiff_z_resampling_factor=None
):
    """
    Orchestrates mesh analysis to get cross-sections, then extracts
    corresponding cross-sections from a TIFF stack.
    """
    print(f"\n--- Starting Combined Mesh and TIFF Cross-Section Analysis ---")
    if mesh_to_tiff_world_transform is None:
        print("  No mesh_to_tiff_world_transform provided. Assuming mesh OBJ is already in TIFF world coordinates.")
        mesh_to_tiff_world_transform = np.eye(4) 
    else:
        print(f"  Using provided mesh_to_tiff_world_transform for alignment.")
        if not isinstance(mesh_to_tiff_world_transform, np.ndarray) or mesh_to_tiff_world_transform.shape != (4,4):
            print("ERROR: mesh_to_tiff_world_transform must be a 4x4 numpy array.")
            return None
        
    print(f"Mesh: {mesh_file_path}")
    print(f"TIFF: {tiff_file_path}")

    base_name = os.path.splitext(os.path.basename(mesh_file_path))[0]
    current_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(current_output_dir, exist_ok=True)

    if tiff_z_resampling_factor is not None and tiff_z_resampling_factor != 1.0:
        print(f"TIFF Z-Resampling Factor: {tiff_z_resampling_factor}")

    # --- Load TIFF early for alignment visualization ---
    print("\nLoading TIFF stack for alignment check and analysis...")
    tiff_stack = load_tiff_stack(tiff_file_path)
    if tiff_stack is None:
        print("  Failed to load TIFF stack. Cannot proceed.")
        return None

    effective_tiff_voxel_size_xyz = list(tiff_voxel_size_xyz) # Start with a mutable copy

    if tiff_z_resampling_factor is not None and tiff_z_resampling_factor != 1.0:
        if tiff_z_resampling_factor <= 0:
            print("Error: tiff_z_resampling_factor must be positive. Skipping resampling.")
        else:
            print(f"Applying Z-resampling to TIFF stack with factor: {tiff_z_resampling_factor}...")
            # Assuming tiff_stack is ZYX ordered (axis 0 is Z)
            original_z_shape = tiff_stack.shape[0]
            
            # Use scipy.ndimage.zoom for resampling. order=1 is linear interpolation.
            # Zoom factors are (zoom_z, zoom_y, zoom_x)
            tiff_stack = scipy.ndimage.zoom(tiff_stack, (tiff_z_resampling_factor, 1, 1), order=1)
            
            # Adjust the Z-component of the voxel size.
            # If you stretch the TIFF (more Z slices, factor > 1), each new Z voxel covers a smaller physical distance.
            effective_tiff_voxel_size_xyz[2] = effective_tiff_voxel_size_xyz[2] / tiff_z_resampling_factor
            
            print(f"  Original TIFF Z-shape: {original_z_shape}, New Z-shape after resampling: {tiff_stack.shape[0]}")
            print(f"  Original Z voxel size: {tiff_voxel_size_xyz[2]:.4f}")
            print(f"  Effective Z voxel size after resampling: {effective_tiff_voxel_size_xyz[2]:.4f}")

    # --- Generate Full Mesh Alignment Visualization (for Z, Y, and X views) ---
    alignment_view_output_dir = os.path.join(current_output_dir, "alignment_visualization")
    os.makedirs(alignment_view_output_dir, exist_ok=True)
    # ... (calls to visualize_full_mesh_alignment for Z, Y, X views as before) ...
    alignment_image_path_z = os.path.join(alignment_view_output_dir, f"{base_name}_alignment_Z_view.png")
    visualize_full_mesh_alignment(mesh_file_path, tiff_stack, mesh_to_tiff_world_transform, tiff_origin_world_xyz, effective_tiff_voxel_size_xyz, alignment_image_path_z, view_axis='Z') # USE EFFECTIVE
    alignment_image_path_y = os.path.join(alignment_view_output_dir, f"{base_name}_alignment_Y_view.png")
    visualize_full_mesh_alignment(mesh_file_path, tiff_stack, mesh_to_tiff_world_transform, tiff_origin_world_xyz, effective_tiff_voxel_size_xyz, alignment_image_path_y, view_axis='Y') # USE EFFECTIVE
    alignment_image_path_x = os.path.join(alignment_view_output_dir, f"{base_name}_alignment_X_view.png")
    visualize_full_mesh_alignment(mesh_file_path, tiff_stack, mesh_to_tiff_world_transform, tiff_origin_world_xyz, effective_tiff_voxel_size_xyz, alignment_image_path_x, view_axis='X') # USE EFFECTIVE
    
    print(f"  Full mesh alignment views (Z, Y, X) saved in: {alignment_view_output_dir}")
    # Optional: input("  Press Enter to continue after checking alignment views...")

    # --- Step 1: Analyze mesh (in its local coordinate system) ---
    print("\nStep 1: Analyzing mesh to define cross-section planes (in mesh local coordinates)...")
    mesh_analysis_output_subdir = os.path.join(current_output_dir, "mesh_analysis_plots")
    if visualize_mesh_analysis_output:
        os.makedirs(mesh_analysis_output_subdir, exist_ok=True)

    mesh_results = analyze_centerline_sections(
        mesh_file_path,
        num_sections=num_mesh_sections,
        visualize=visualize_mesh_analysis_output,
        output_dir=mesh_analysis_output_subdir if visualize_mesh_analysis_output else None
    )

    if not mesh_results:
        print(f"  Mesh analysis failed for {mesh_file_path}. Cannot proceed.")
        return None
    
    local_mesh_sections_definitions = mesh_results.get('raw_section_data_items')
    global_minor_radius = mesh_results.get('minor_radius') # GET global_minor_radius HERE

    if not local_mesh_sections_definitions:
        print(f"  'raw_section_data_items' not found in mesh analysis results for {mesh_file_path}.")
        return None
    print(f"  Successfully obtained {len(local_mesh_sections_definitions)} local section definitions from mesh analysis.")
    if global_minor_radius is None:
        print("  Warning: 'minor_radius' not found in mesh analysis results. Using default for TIFF slice extent.")
        global_minor_radius = 20.0 # Default if not found

    # --- Step 2: Transform mesh section definitions from mesh local to TIFF world space ---
    print("\nStep 2: Transforming mesh section definitions to TIFF world space...")
    world_mesh_sections_definitions = [] # This will be the correctly transformed list
    for i, section_def in enumerate(local_mesh_sections_definitions):
        if not section_def.get('valid_geometry', False):
            world_mesh_sections_definitions.append(section_def) 
            continue

        local_pos = section_def['position_3d']
        local_tangent = section_def['tangent_3d']
        local_transform_2d_to_3d_local = section_def['transform'] 

        local_pos_h = np.append(local_pos, 1.0) 
        world_pos_h = mesh_to_tiff_world_transform @ local_pos_h
        world_pos = world_pos_h[:3] / world_pos_h[3] 

        upper_3x3_transform = mesh_to_tiff_world_transform[:3,:3]
        world_tangent = upper_3x3_transform @ local_tangent
        world_tangent /= np.linalg.norm(world_tangent) 

        world_transform_2d_to_3d_world = mesh_to_tiff_world_transform @ local_transform_2d_to_3d_local
        
        transformed_def = section_def.copy() 
        transformed_def['position_3d'] = world_pos
        transformed_def['tangent_3d'] = world_tangent
        transformed_def['transform'] = world_transform_2d_to_3d_world
        world_mesh_sections_definitions.append(transformed_def)
    
    # Now, use 'world_mesh_sections_definitions' for subsequent steps
    # And 'global_minor_radius' is also defined.

    # --- TIFF Information (uses the tiff_stack already loaded) ---
    tiff_shape_zyx = np.array(tiff_stack.shape) 
    # ... (rest of TIFF info printing) ...
    geometric_tiff_center_voxel_zyx = tiff_shape_zyx / 2.0
    geometric_tiff_center_voxel_xyz = np.array([geometric_tiff_center_voxel_zyx[2], geometric_tiff_center_voxel_zyx[1], geometric_tiff_center_voxel_zyx[0]])
    geometric_tiff_content_center_world_xyz = tiff_origin_world_xyz + (geometric_tiff_center_voxel_xyz * tiff_voxel_size_xyz)
    print(f"  TIFF's geometric center (world XYZ, for info only): {geometric_tiff_content_center_world_xyz}")
    print(f"  Using provided TIFF origin (world XYZ): {tiff_origin_world_xyz}")
    print(f"  Using provided TIFF voxel size (world XYZ): {tiff_voxel_size_xyz}")


    # --- Call to Orthogonal Visualization Function ---
    # Pass the 'world_mesh_sections_definitions'
    if world_mesh_sections_definitions and tiff_stack is not None:
        ortho_vis_output_dir = os.path.join(current_output_dir, "orthogonal_tiff_views")
        visualize_mesh_sections_on_orthogonal_tiff_slices(
            world_mesh_sections_definitions, # Use the transformed definitions
            tiff_stack,
            tiff_origin_world_xyz, 
            effective_tiff_voxel_size_xyz, # USE EFFECTIVE
            ortho_vis_output_dir,
            base_name
        )
    
    # --- Step 3: Resample TIFF at each mesh section plane ---
    print("\nStep 3: Resampling TIFF stack at mesh section planes (using transformed mesh coordinates)...")
    resampled_tiff_slices_list = []
    # global_minor_radius is now defined from mesh_results
    slice_extent_for_resampling = global_minor_radius * 2.5 
    if slice_extent_for_resampling <= 0: slice_extent_for_resampling = 20.0 # Fallback

    # Iterate over 'world_mesh_sections_definitions'
    for i, section_def in enumerate(world_mesh_sections_definitions): 
        if not section_def.get('valid_geometry', False):
            print(f"  Skipping TIFF resampling for mesh section {i+1} (original mesh section was invalid).")
            resampled_tiff_slices_list.append(None)
            continue
        # ... (rest of the resampling loop using section_def from world_mesh_sections_definitions) ...
        plane_orig_for_resampling_world = section_def['position_3d'] 
        plane_norm_world = section_def['tangent_3d'] 
        # ... (logic for plane_x_w, plane_y_w) ...
        temp_v = np.array([1.0, 0.0, 0.0]) 
        if np.allclose(np.abs(np.dot(temp_v, plane_norm_world)), 1.0): 
            temp_v = np.array([0.0, 1.0, 0.0]) 
        
        plane_x_w = np.cross(plane_norm_world, temp_v)
        if np.linalg.norm(plane_x_w) < 1e-6: 
            temp_v = np.array([0.0,0.0,1.0]) 
            plane_x_w = np.cross(plane_norm_world, temp_v)
            if np.linalg.norm(plane_x_w) < 1e-6: 
                 print(f"  Error: Could not define orthogonal plane_x_axis for section {i+1}. Using arbitrary default.")
                 dominant_axis = np.argmax(np.abs(plane_norm_world))
                 if dominant_axis == 0: plane_x_w = np.array([0,1,0])
                 elif dominant_axis == 1: plane_x_w = np.array([1,0,0])
                 else: plane_x_w = np.array([1,0,0])

        plane_x_w /= np.linalg.norm(plane_x_w)
        plane_y_w = np.cross(plane_norm_world, plane_x_w) 
        plane_y_w /= np.linalg.norm(plane_y_w)
        
        print(f"    Resampling TIFF for mesh section {i+1} (NormPos: {section_def.get('norm_pos', -1):.2f}) at transformed world origin: {plane_orig_for_resampling_world}")
        resampled_img = resample_slice_from_tiff(
            tiff_stack,
            plane_origin_world=plane_orig_for_resampling_world, 
            plane_normal_world=plane_norm_world,
            plane_x_axis_world=plane_x_w,
            plane_y_axis_world=plane_y_w,
            tiff_origin_world_xyz=tiff_origin_world_xyz, 
            voxel_size_world_xyz=effective_tiff_voxel_size_xyz, # USE EFFECTIVE
            slice_dims_pixels=(128, 128), 
            slice_extent_world=(slice_extent_for_resampling, slice_extent_for_resampling)
        )
        resampled_tiff_slices_list.append(resampled_img)
        if resampled_img is not None:
            print(f"      Successfully resampled. Output shape: {resampled_img.shape}")
        else:
            print(f"      Failed to resample.")

    # --- Step 4: Visualize resampled TIFF sections ---
    if visualize_tiff_sections and any(s is not None for s in resampled_tiff_slices_list):
        print("\nStep 4: Creating visualizations for resampled TIFF sections...")
        tiff_montage_filename = os.path.join(current_output_dir, f"{base_name}_tiff_resampled_montage.png")
        create_tiff_section_montage(
            resampled_tiff_slices_list,
            world_mesh_sections_definitions, # Pass transformed for titles
            tiff_montage_filename
        )
        
        mesh_outlines_2d_list = [s_def.get('points_2d') for s_def in world_mesh_sections_definitions]
        
        tiff_overlay_montage_filename = os.path.join(current_output_dir, f"{base_name}_tiff_overlay_montage.png")
        create_tiff_overlay_montage(
            resampled_tiff_slices_list,
            mesh_outlines_2d_list, 
            world_mesh_sections_definitions, 
            slice_extent_world_pair=(slice_extent_for_resampling, slice_extent_for_resampling),
            output_path=tiff_overlay_montage_filename
        )

    print(f"\n--- Combined Analysis Complete for {mesh_file_path} ---")
    return {
        'mesh_file': mesh_file_path,
        'tiff_file': tiff_file_path,
        'mesh_analysis_results': mesh_results, 
        'transformed_mesh_section_definitions': world_mesh_sections_definitions, 
        'resampled_tiff_slices': resampled_tiff_slices_list,
        'output_directory': current_output_dir
    }

def create_tiff_section_montage(tiff_slices, mesh_section_metadata, output_path):
    """Creates a montage plot of resampled TIFF sections."""
    valid_items = [{'image': img, 'meta': mesh_section_metadata[i]}
                   for i, img in enumerate(tiff_slices) if img is not None]

    if not valid_items:
        print("  No valid TIFF slices to create montage.")
        return

    n_valid = len(valid_items)
    cols = min(5, n_valid)
    rows = (n_valid + cols - 1) // cols

    fig, axes_flat = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2), squeeze=False)
    axes_flat = axes_flat.flatten() # Ensure it's always a flat array
    
    fig.suptitle("Resampled TIFF Cross-Sections", fontsize=16)

    for i, item_dict in enumerate(valid_items):
        ax = axes_flat[i]
        img = item_dict['image']
        meta = item_dict['meta']
        
        vmin, vmax = (np.nanmin(img), np.nanmax(img)) if not np.all(np.isnan(img)) else (0,1)
        if vmin == vmax: vmin, vmax = (vmin - 0.5 if vmin is not None else 0), (vmax + 0.5 if vmax is not None else 1)
        
        ax.imshow(img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        title = f"Sec {i+1}"
        if 'norm_pos' in meta and meta['norm_pos'] is not None: title += f"\nPos:{meta['norm_pos']:.2f}"
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    for i in range(n_valid, len(axes_flat)): # Hide unused subplots
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved TIFF section montage: {output_path}")


def create_tiff_overlay_montage(
    tiff_slices, mesh_outlines_2d_list, mesh_section_metadata,
    slice_extent_world_pair, output_path
):
    """
    Creates a montage of resampled TIFF sections with mesh outlines overlaid.
    slice_extent_world_pair: (height_world, width_world) used for resampling.
    """
    valid_items = []
    for i, slice_img in enumerate(tiff_slices):
        if slice_img is not None:
            outline = mesh_outlines_2d_list[i] if i < len(mesh_outlines_2d_list) and \
                                                 mesh_outlines_2d_list[i] is not None and \
                                                 len(mesh_outlines_2d_list[i]) >= 3 else None
            valid_items.append({
                'image': slice_img,
                'meta': mesh_section_metadata[i] if i < len(mesh_section_metadata) else {},
                'outline': outline
            })

    if not valid_items:
        print("  No valid TIFF slices or outlines for overlay montage.")
        return

    n_valid = len(valid_items)
    cols = min(5, n_valid)
    rows = (n_valid + cols - 1) // cols

    fig, axes_flat = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.7), squeeze=False)
    axes_flat = axes_flat.flatten()
    fig.suptitle("Resampled TIFF Sections with Mesh Outlines", fontsize=16)

    img_disp_extent = [
        -slice_extent_world_pair[1] / 2, slice_extent_world_pair[1] / 2, # x_min, x_max for imshow
        -slice_extent_world_pair[0] / 2, slice_extent_world_pair[0] / 2  # y_min, y_max for imshow
    ]

    for i, item_dict in enumerate(valid_items):
        ax = axes_flat[i]
        img = item_dict['image']
        meta = item_dict['meta']
        outline_pts = item_dict['outline']

        vmin, vmax = (np.nanmin(img), np.nanmax(img)) if not np.all(np.isnan(img)) else (0,1)
        if vmin == vmax: vmin, vmax = (vmin - 0.5 if vmin is not None else 0), (vmax + 0.5 if vmax is not None else 1)
        
        ax.imshow(img, cmap='gray', origin='lower', extent=img_disp_extent, vmin=vmin, vmax=vmax)
        
        if outline_pts is not None:
            # The outline_pts are in their own 2D local coordinate system from section.to_2D().
            # For direct overlay, these points need to be centered around (0,0) if the
            # resampled TIFF image is also centered at (0,0) in its local plane.
            # The `resample_slice_from_tiff` creates the image grid from -extent/2 to +extent/2.
            # So, if outline_pts are also centered, they should overlay correctly if orientations match.
            centered_outline = outline_pts - np.mean(outline_pts, axis=0)
            ordered_outline = order_points(outline_pts, method="angular", center=np.array([0.0, 0.0]))
            
            ax.plot(np.append(ordered_outline[:, 0], ordered_outline[0, 0]),
                    np.append(ordered_outline[:, 1], ordered_outline[0, 1]),
                    'r-', alpha=0.6, linewidth=1.2)
        
        title = f"Sec {i+1}"
        if 'norm_pos' in meta and meta['norm_pos'] is not None: title += f"\nPos:{meta['norm_pos']:.2f}"
        # You might want to add AR from mesh_section_metadata if it's calculated and stored there
        # e.g., if meta contains an 'aspect_ratio' key from the mesh analysis.
        # if 'aspect_ratio' in meta and meta['aspect_ratio'] is not None: title += f"\nARm:{meta['aspect_ratio']:.2f}"

        ax.set_title(title, fontsize=8)
        ax.set_xlim(img_disp_extent[0], img_disp_extent[1])
        ax.set_ylim(img_disp_extent[2], img_disp_extent[3])
        ax.axis('off')

    for i in range(n_valid, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved TIFF overlay montage: {output_path}")


if __name__ == '__main__':
    print("Executing example: TIFF cross-section processing.")
    
    # --- Parameters for the example run ---
    # Find a .obj file in Meshes/Onion_OBJ for testing
    test_obj_dir = "Meshes/Onion_OBJ" # Relative to where script is run
    obj_files_in_dir = []
    if os.path.isdir(test_obj_dir):
        obj_files_in_dir = [f for f in os.listdir(test_obj_dir) if f.lower().endswith('.obj')]
    
    if not obj_files_in_dir:
        print(f"ERROR: No .obj files found in '{os.path.abspath(test_obj_dir)}'. Cannot run example.")
        print("Please add a mesh file to this directory or update 'test_obj_dir'.")
        exit()
        
    example_mesh_file = os.path.join(test_obj_dir, obj_files_in_dir[0])
    example_mesh_file = "Meshes/Onion_OBJ_aligned_with_tiff/Ac_DA_1_2.obj" # For testing
    print(f"Using example mesh: {example_mesh_file}")

    # Create or use a dummy TIFF for testing
    example_tiff_file = "/home/tomkinsm/stomata-air-mattress/Meshes/Onion meshes/Ac_DA_1_2/Ac_DA_1_2_2023_09_11__13_18_06_Out.tif"
    if not os.path.exists(example_tiff_file):
        print(f"Creating dummy TIFF for example: {example_tiff_file}")
        dummy_data = np.random.randint(0, 255, size=(50, 128, 128), dtype=np.uint8) # Z, Y, X
        tifffile.imwrite(example_tiff_file, dummy_data, imagej=True, metadata={'spacing': 0.5, 'unit': 'um'})
    
    # Define voxel sizes (world units per voxel/pixel for X, Y, Z world axes)
    # IMPORTANT: This order (X,Y,Z) must match your mesh's world coordinate system.
    example_tiff_voxel_size_xyz = np.array([0.207566, 0.207566, 0.3])  # e.g., X=0.5um/vox, Y=0.5um/vox, Z=1.0um/slice

    # Define the world coordinate (X,Y,Z) that corresponds to TIFF voxel (0,0,0)
    example_tiff_origin_world_xyz = np.array([0.0, 0.0, 0.0]) # Assume TIFF origin aligns with world origin

    # Example: If the OBJ's (0,0,0) point needs to be at world (tx, ty, tz) micrometers
    # relative to the TIFF's origin, and there's no rotation or scaling.
    tx = 50.0  # Replace with the actual X translation in micrometers
    ty = 62.0  # Replace with the actual Y translation in micrometers
    tz = 10.0   # Replace with the actual Z translation in micrometers

     # --- Define Rotation around X-axis ---
    angle_x_degrees = -5.0  # Set your desired rotation angle in degrees
                           # Positive angle typically follows the right-hand rule
                           # (e.g., if X points right, positive rotation is "downward" for Y, "forward" for Z)
                           # Try values like 10, -10, 45, 90 to see the effect.
    angle_x_radians = np.deg2rad(angle_x_degrees)
    
    cos_a = np.cos(angle_x_radians)
    sin_a = np.sin(angle_x_radians)

    # X-axis rotation matrix (3x3)
    rotation_matrix_x = np.array([
        [1,  0,     0    ],
        [0,  cos_a, -sin_a],
        [0,  sin_a,  cos_a]
    ])

    # --- Combine Rotation and Translation into the 4x4 Transformation Matrix ---
    # Start with an identity 4x4 matrix
    example_mesh_to_tiff_transform = np.eye(4) 
    
    # Set the rotation part (upper-left 3x3)
    example_mesh_to_tiff_transform[:3, :3] = rotation_matrix_x
    
    # Set the translation part (last column, first three rows)
    example_mesh_to_tiff_transform[:3, 3] = [tx, ty, tz]

    example_main_output_dir = "example_tiff_analysis_results"
    # --- End Parameters ---

    if not os.path.exists(example_mesh_file):
        print(f"ERROR: Example mesh file not found: {example_mesh_file}")
    else:
        run_tiff_cross_section_analysis(
            mesh_file_path=example_mesh_file,
            tiff_file_path=example_tiff_file,
            tiff_voxel_size_xyz=example_tiff_voxel_size_xyz,
            tiff_origin_world_xyz=example_tiff_origin_world_xyz, # Pass your defined origin
            mesh_to_tiff_world_transform=example_mesh_to_tiff_transform,
            num_mesh_sections=10, 
            output_dir=example_main_output_dir,
            visualize_mesh_analysis_output=True, 
            visualize_tiff_sections=True,
            tiff_z_resampling_factor=1.55  # Set to None or 1.0 to skip resampling
            # Removed threshold_for_com if it was here
        )
        print(f"\nExample run finished. Results are in '{example_main_output_dir}'.")
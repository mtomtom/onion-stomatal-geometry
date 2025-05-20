import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import trimesh
from sklearn.decomposition import PCA

# Import necessary functions from existing files
from test_functions import get_radial_dimensions, filter_section_points
from helper_functions import (_smooth_centerline_savgol, 
                             _project_plane_origin_to_2d, 
                             _calculate_pca_metrics,
                             order_points, _determine_midpoint_plane, _determine_tip_plane_v2)
import edge_detection as ed

def analyze_centerline_sections(mesh_file, 
                               num_sections=20, 
                               visualize=True,
                               output_dir=None):
    """
    Analyzes cross-sections along the full centerline of a stomata guard cell
    and measures how aspect ratio changes along its length.
    
    Args:
        mesh_file (str): Path to the mesh file (OBJ format)
        num_sections (int): Number of cross-sections to analyze along the centerline
        visualize (bool): Whether to generate visualizations
        output_dir (str): Directory to save visualizations and results
        
    Returns:
        dict: Contains:
            - 'positions': Normalized positions along the centerline (0 to 1)
            - 'aspect_ratios': Aspect ratios at each position
            - 'widths': PCA minor widths at each position
            - 'section_points': List of 2D cross-section points at each position
            - 'centerline': The smoothed centerline used for analysis
    """
    # Create output directory if needed
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process mesh and extract key features
    print(f"\nProcessing mesh: {mesh_file}")
    ed_output = ed.find_seam_by_raycasting(mesh_file, visualize=False)
    
    if ed_output is None or 'mesh_object' not in ed_output or ed_output['mesh_object'] is None:
        print(f"  Error: Edge detection failed for {mesh_file}")
        return None
    
    mesh = ed_output['mesh_object']
    shared_wall_points = ed_output.get('shared_wall_points')
    pore_center = ed_output.get('pore_center_coords')
    ray_origin_for_radial_cast = pore_center 
    print(f"  Performing radial ray casting from origin: {ray_origin_for_radial_cast.round(3) if ray_origin_for_radial_cast is not None else 'mesh centroid (default)'}")
    ray_count = 45 # Example, adjust as needed
    inner_points, outer_points, raw_centerline_points_from_radial, minor_radius = get_radial_dimensions(
        mesh, center=ray_origin_for_radial_cast, ray_count=ray_count
    ) 
    smoothed_centerline = _smooth_centerline_savgol(raw_centerline_points_from_radial)
    if smoothed_centerline is None:
        smoothed_centerline = raw_centerline_points_from_radial
    
    if smoothed_centerline is None or len(smoothed_centerline) < 2: # Need at least 2 points for a segment
        print("  Error: Not enough points in initial smoothed centerline for analysis.")
        return None

    # Get ED centerline for _determine_tip_plane_v2 if available
    estimated_centerline_3d_from_ed = ed_output.get('estimated_centerline_points')

    # --- Determine Tip and Midpoint Plane Origins using Helper Functions ---
    print("  Determining robust tip and midpoint plane origins...")
    midpoint_plane_origin_3d, _ = _determine_midpoint_plane(
        smoothed_centerline, 
        pore_center # This is detected_pore_center_3d_ed from ed_output
    )

    tip_plane_origin_3d, _, _ = _determine_tip_plane_v2(
        smoothed_centerline,
        pore_center, # detected_pore_center_3d_ed
        shared_wall_points,
        minor_radius,
        inner_points, # inner_points_for_refinement
        min_tip_distance=0.05, # Adjustable: min distance to slide from initial tip
        estimated_centerline_3d_from_ed=estimated_centerline_3d_from_ed
    )

    if tip_plane_origin_3d is None or midpoint_plane_origin_3d is None:
        print("  Error: Could not determine robust tip or midpoint plane origins. Cannot proceed.")
        return None
    
    print(f"  Robust Tip Plane Origin: {tip_plane_origin_3d.round(3)}")
    print(f"  Robust Midpoint Plane Origin: {midpoint_plane_origin_3d.round(3)}")

    # --- Find closest points on the (radial) smoothed_centerline to these origins ---
    actual_tip_cl_idx = np.argmin(np.linalg.norm(smoothed_centerline - tip_plane_origin_3d, axis=1))
    actual_mid_cl_idx = np.argmin(np.linalg.norm(smoothed_centerline - midpoint_plane_origin_3d, axis=1))

    if actual_tip_cl_idx == actual_mid_cl_idx:
        print("  Error: Tip and Midpoint plane origins map to the same point on the centerline. Cannot form a segment.")
        # Fallback or error: For now, let's try to use a small segment if this happens,
        # or you might want to return None.
        # As a minimal fallback, take a few points around the tip.
        # This part might need more sophisticated handling based on your data.
        # For now, let's assume they are distinct enough. If not, the path logic below might be short.
        if len(smoothed_centerline) > 1:
             # Attempt to take a very short segment if they are identical
            _start_idx = actual_tip_cl_idx
            _end_idx = (actual_tip_cl_idx + 1) % len(smoothed_centerline)
            if _start_idx == _end_idx and len(smoothed_centerline) > 1: # single point centerline
                 print("  Single point centerline after tip/midpoint mapping. Cannot proceed.")
                 return None
            elif _start_idx == _end_idx: # still same
                 actual_mid_cl_idx = _end_idx # try to force a segment of 1
            else: # if distinct after +1
                 actual_mid_cl_idx = _end_idx


    print(f"  Closest CL idx to Tip Origin: {actual_tip_cl_idx} (Point: {smoothed_centerline[actual_tip_cl_idx].round(3)})")
    print(f"  Closest CL idx to Mid Origin: {actual_mid_cl_idx} (Point: {smoothed_centerline[actual_mid_cl_idx].round(3)})")

    # --- Extract the shortest path segment on smoothed_centerline (loop) ---
    N_cl = len(smoothed_centerline)
    
    path1_indices = []
    curr = actual_tip_cl_idx
    count1 = 0
    while True:
        path1_indices.append(curr)
        count1 += 1
        if curr == actual_mid_cl_idx: break
        if count1 > N_cl : # Safety break if loop doesn't terminate
            print("  Warning: Path 1 extraction exceeded centerline length.")
            path1_indices = list(range(N_cl)) # Fallback to full centerline
            break 
        curr = (curr + 1) % N_cl

    path2_indices = []
    curr = actual_tip_cl_idx
    count2 = 0
    while True:
        path2_indices.append(curr)
        count2 += 1
        if curr == actual_mid_cl_idx: break
        if count2 > N_cl: # Safety break
            print("  Warning: Path 2 extraction exceeded centerline length.")
            path2_indices = list(range(N_cl)) # Fallback to full centerline
            break
        curr = (curr - 1 + N_cl) % N_cl # Ensure positive index before modulo

    final_path_indices = []
    if len(path1_indices) <= len(path2_indices):
        final_path_indices = path1_indices
        print(f"  Selected Path 1 (length {len(path1_indices)}) for centerline segment.")
    else:
        final_path_indices = path2_indices
        print(f"  Selected Path 2 (length {len(path2_indices)}) for centerline segment.")

    if not final_path_indices: # Should not happen if N_cl >=1
        print("  Error: Could not determine a valid path between tip and midpoint.")
        return None

    centerline_segment_for_analysis = smoothed_centerline[final_path_indices]
    
    if len(centerline_segment_for_analysis) < 2:
        print(f"  Error: Extracted centerline segment has too few points ({len(centerline_segment_for_analysis)}). Cannot proceed.")
        return None
        
    print(f"  Extracted new centerline segment for analysis with {len(centerline_segment_for_analysis)} points.")
    
    # Replace the original smoothed_centerline with this new segment for all subsequent analysis
    smoothed_centerline = centerline_segment_for_analysis
    
    # --- Step 4: Get total centerline segment length for normalization (this part remains similar) ---
    # Ensure smoothed_centerline now refers to the extracted segment
    centerline_segments_diff = np.diff(smoothed_centerline, axis=0)
    segment_lengths = np.linalg.norm(centerline_segments_diff, axis=1)
    
    # Handle cases where segment_lengths might be empty (if smoothed_centerline has < 2 points)
    if len(segment_lengths) == 0: # Implies smoothed_centerline has 0 or 1 point
        if len(smoothed_centerline) == 1: # Single point segment
            normalized_distances = np.array([0.0]) # Or handle as an error
            print("  Warning: Centerline segment is a single point. Normalization may be trivial.")
        else: # Zero points
            print("  Error: Centerline segment is empty after path extraction.")
            return None
    else:
        # total_length = np.sum(segment_lengths) # Not strictly needed for normalized_distances
        cumulative_distances = np.zeros(len(smoothed_centerline))
        cumulative_distances[1:] = np.cumsum(segment_lengths) # Start cumsum from the first segment

        if cumulative_distances[-1] == 0: # Avoid division by zero if total length is 0 (e.g. all points identical)
            if len(smoothed_centerline) > 0:
                normalized_distances = np.zeros(len(smoothed_centerline))
                print("  Warning: Total length of centerline segment is zero. Normalized distances set to 0.")
            else: # Should have been caught earlier
                print("  Error: Centerline segment is empty and has zero length.")
                return None
        else:
            normalized_distances = cumulative_distances / cumulative_distances[-1]

    total_points = len(smoothed_centerline) # This is now the length of the new segment
    
    # Step 5: Sample positions along the centerline segment
    sampled_positions = []
    sampled_tangents = []
    normalized_positions = []
    
    # Create evenly spaced PHYSICAL positions
    target_positions = np.linspace(0, 1.0, num_sections)
    sampled_indices = []

    # For each target position, find the closest actual point
    for target in target_positions:
        idx = np.argmin(np.abs(normalized_distances - target))
        sampled_indices.append(idx)

    # Use these indices to get positions and calculate tangents
    for i in sampled_indices:
        position = smoothed_centerline[i]
        sampled_positions.append(position)
        
        # TANGENT CALCULATION - Use existing approach from helper_functions.py
        if i == 0:  # First point
            tangent_vec = smoothed_centerline[1] - smoothed_centerline[0]
        elif i == total_points - 1:  # Last point
            tangent_vec = smoothed_centerline[-1] - smoothed_centerline[-2]
        else:  # Internal point - use central difference
            tangent_vec = smoothed_centerline[i + 1] - smoothed_centerline[i - 1]
        
        # Normalize the tangent
        tangent_norm = np.linalg.norm(tangent_vec)
        tangent = tangent_vec / tangent_norm if tangent_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        
        sampled_tangents.append(tangent)
        normalized_positions.append(normalized_distances[i])
    # Create a COMPLETELY different visualization approach
    if visualize and output_dir:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # ADD MESH VISUALIZATION
        # Convert trimesh to vertices and faces for matplotlib
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create a Poly3DCollection to render the mesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        mesh_triangles = vertices[faces]
        mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.5, edgecolor='gray', 
                                          linewidth=0.1, facecolor='lightgray')
        ax.add_collection3d(mesh_collection)
        
        # 1. Plot centerline with clear black line
        ax.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], 
                'k-', linewidth=2, label='Centerline')
        
        # 2. Add colorful section planes - use more saturated colors to stand out against mesh
        for idx, (pos, tang) in enumerate(zip(sampled_positions, sampled_tangents)):
            # Calculate perpendicular vectors
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(ref, tang)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(ref, tang)) > 0.9:
                    ref = np.array([0.0, 0.0, 1.0])
            
            v1 = np.cross(tang, ref)
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(tang, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Create a small plane (square) perpendicular to the tangent
            # to represent the cross-section
            size = 0.3  # Size of the plane
            plane_corners = [
                pos + size * v1 + size * v2,
                pos + size * v1 - size * v2,
                pos - size * v1 - size * v2,
                pos - size * v1 + size * v2,
                pos + size * v1 + size * v2,  # Close the loop
            ]
            
            # Extract x, y, z coordinates of the plane
            x = [p[0] for p in plane_corners]
            y = [p[1] for p in plane_corners]
            z = [p[2] for p in plane_corners]
            
            # Plot the plane with a partially transparent colored face
            cmap = plt.cm.viridis
            color = cmap(idx / len(sampled_positions))
            
            ax.plot(x, y, z, 'r-', linewidth=1)  # Draw the outline
            ax.plot_surface(
                np.array([x[:-1]]),  # Last point is repeated to close loop
                np.array([y[:-1]]), 
                np.array([z[:-1]]),
                color=color, alpha=0.7  # Increase alpha to stand out against mesh
            )
            
            # Draw a short tangent arrow in black for clarity
            ax.quiver(pos[0], pos[1], pos[2], 
                    tang[0]*0.2, tang[1]*0.2, tang[2]*0.2,
                    color='black', arrow_length_ratio=0.3, linewidth=1.5)
            
        # Add sampled points
        ax.scatter([p[0] for p in sampled_positions], 
                  [p[1] for p in sampled_positions], 
                  [p[2] for p in sampled_positions], 
                  c='blue', marker='o', s=80)
                  
        # ZOOM IN MORE - Calculate bounding box more tightly around the centerline segment
        # Use only the smoothed_centerline segment (tip to midpoint) for tighter focus
        center = np.mean(smoothed_centerline, axis=0)
        
        # Use a smaller scaling factor for tighter zoom (0.7 instead of 1.0)
        max_range = 0.7 * np.max([
            np.ptp(smoothed_centerline[:, 0]),
            np.ptp(smoothed_centerline[:, 1]),
            np.ptp(smoothed_centerline[:, 2])
        ])
        
        # Set view limits based on center point and reduced max_range
        #ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        #ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        #ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        
        # Adjust the view angle for better visualization
        ax.view_init(elev=90, azim=90)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Centerline with Cross-Section Planes')
        
        # Save diagnostic plot
        base_name = os.path.splitext(os.path.basename(mesh_file))[0]
        plt.savefig(os.path.join(output_dir, f"{base_name}_tangent_vectors.png"), dpi=200)
        plt.close(fig)
    
    # Step 6: Analyze cross-sections at each point
    aspect_ratios = []
    widths = []
    section_points_list = []
    section_data_3d = []

    print("DEBUG: >>> Entering MAIN SECTIONING LOOP <<<")
    print(f"DEBUG: sampled_positions length: {len(sampled_positions)}")
    print(f"DEBUG: sampled_tangents length: {len(sampled_tangents)}")
    print(f"DEBUG: normalized_positions length: {len(normalized_positions)}")
    
    for idx, (position, tangent, norm_pos) in enumerate(zip(sampled_positions, sampled_tangents, normalized_positions)):
        print(f"  Analyzing section {idx+1}/{len(sampled_positions)} at position {norm_pos:.2f}")

        # Add this reference vector calculation EARLY
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, tangent)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(ref, tangent)) > 0.9:
                ref = np.array([0.0, 0.0, 1.0])
        
        # IMPROVED SECTION CREATION
        # Ensure tangent is non-zero
        if np.all(tangent == 0):
            print(f"  Error: Zero tangent at section {idx+1}")
            aspect_ratios.append(None)
            widths.append(None)
            section_points_list.append(None)
            continue
            
        # Create section by cutting mesh with plane
        try:
            # First attempt with direct tangent
            section = mesh.section(plane_origin=position, plane_normal=tangent)
            
            # Check if section is valid
            if section is None or not hasattr(section, 'entities') or len(section.entities) == 0:
                # Try a different approach - slightly perturb position
                for attempt in range(3):
                    # Small random offset perpendicular to tangent
                    perp1 = np.array([1, 0, 0])
                    if np.abs(np.dot(perp1, tangent)) > 0.9:  # Too parallel
                        perp1 = np.array([0, 1, 0])
                    perp1 = perp1 - np.dot(perp1, tangent) * tangent
                    perp1 = perp1 / np.linalg.norm(perp1)
                    
                    perp2 = np.cross(tangent, perp1)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    offset = 0.01 * (attempt + 1)  # Increasing offset
                    shift = offset * (perp1 + perp2)  # Diagonal shift
                    
                    new_position = position + shift
                    section = mesh.section(plane_origin=new_position, plane_normal=tangent)
                    
                    if section is not None and hasattr(section, 'entities') and len(section.entities) > 0:
                        print(f"  Successfully created section {idx+1} with offset {offset}")
                        break
                
        except Exception as e:
            print(f"  Error creating section {idx+1}: {e}")
            section = None
            
        # Get the section in 2D
        try:
            path_2D, transform_2d_to_3d = section.to_2D()
            points_2D = path_2D.vertices
            original_section_points_3d = section.vertices.copy()
        except Exception as e:
            print(f"  Error converting section {idx+1} to 2D: {e}")
            aspect_ratios.append(None)
            widths.append(None)
            section_points_list.append(None)
            continue

        # After section creation
        if section is not None:
            print(f"  Section {idx+1}: Created successfully with {len(section.entities)} entities")

        # After 2D conversion
        print(f"  Section {idx+1}: 2D projection has {len(points_2D)} points")

        
            
        # Get plane origin in 2D for filtering
        plane_origin_2d, transform_3d_to_2d = _project_plane_origin_to_2d(position, transform_2d_to_3d, points_2D) # CORRECTED LINE

        # Get inner/outer points for this section (needed for minor_radius)
        # _, _, _, section_minor_radius = get_radial_dimensions(mesh, center=position_val) # OLD: per-section calculation

        # NEW: USE THE GLOBALLY ESTIMATED MINOR RADIUS FOR FILTERING
        # The global 'minor_radius' is already defined earlier in the function.
        if minor_radius is None: # This checks the global minor_radius calculated earlier
            print(f"  [CRITICAL FAIL] Global minor_radius is None. Cannot proceed with section {idx+1}.")
            aspect_ratios.append(None)
            widths.append(None)
            section_points_list.append(None)
            continue
        else:
            # Use the global minor_radius for filtering this section
            current_minor_radius_for_filtering = minor_radius
            print(f"    Using global minor_radius for filtering section {idx+1}: {current_minor_radius_for_filtering:.3f}")
            
        # Filter the section points to get clean outline (around line 416)
        # Make sure to pass current_minor_radius_for_filtering
        filtered_points_2D, filter_mask = filter_section_points( # Capture filter_mask
            points_2D, 
            current_minor_radius_for_filtering, # Use the global minor_radius
            plane_origin_2d,
            eps_factor=0.15, # Original value, can be tuned
            min_samples=3    # Original value, can be tuned
        )
        # print(f"    Filter mask sum for section {idx+1}: {np.sum(filter_mask)}") # Optional debug

        # After filtering (around line 423)
        print(f"  Section {idx+1}: After filtering: {len(filtered_points_2D)} points from {len(points_2D)}")
        
        if len(filtered_points_2D) < 3:
            print(f"  [FAIL] Section {idx+1} has too few points ({len(filtered_points_2D)}) after filtering. Skipping section.")
            aspect_ratios.append(None)
            widths.append(None)
            section_points_list.append(None)
            continue
            
        # Calculate aspect ratio and width 
        aspect_ratio, pca_width = _calculate_pca_metrics(filtered_points_2D, "section")

        print(f"  Section {idx+1}: position {norm_pos:.2f}, " 
              f"valid={aspect_ratio is not None}, "
              f"AR={(f'{aspect_ratio:.2f}' if aspect_ratio is not None else 'None')}")


        # Store the values
        aspect_ratios.append(aspect_ratio)
        widths.append(pca_width)
        section_points_list.append(filtered_points_2D)
        
        # After calculating aspect_ratio and width, add:
        if aspect_ratio <= 30.0:
            # Store section data for 3D visualization
            # Transform 2D points back to 3D for visualization
            v1 = np.cross(tangent, ref)
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(tangent, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Transform 2D points back to 3D
            section_points_3d = []
            for pt in filtered_points_2D:
                p3d = position + pt[0] * v1 + pt[1] * v2
                section_points_3d.append(p3d)
                
            section_data_3d.append({
                'points_2d': filtered_points_2D,
                'position': position,
                'tangent': tangent,
                'norm_pos': norm_pos,
                'aspect_ratio': aspect_ratio,
                'transform': transform_2d_to_3d  # Add this line to store the transform
            })
        
    # Step 7: Create visualizations
    if visualize and output_dir:
        # Plot aspect ratio vs position - MODIFIED with filter
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_indices = [i for i, ar in enumerate(aspect_ratios) if ar is not None and ar <= 1.6]
        valid_positions = [normalized_positions[i] for i in valid_indices]
        valid_ratios = [aspect_ratios[i] for i in valid_indices]
        
        ax.plot(valid_positions, valid_ratios, 'o-', linewidth=2)
        ax.set_xlabel('Normalized Position Along Centerline')
        ax.set_ylabel('Aspect Ratio')
        ax.set_title(f'Aspect Ratio Along Centerline\n{os.path.basename(mesh_file)}')
        ax.grid(True)
        
        # Save aspect ratio plot
        base_name = os.path.splitext(os.path.basename(mesh_file))[0]
        aspect_plot_path = os.path.join(output_dir, f"{base_name}_aspect_ratio_curve.png")
        plt.savefig(aspect_plot_path, dpi=150)
        plt.close(fig)
        
        # Plot width vs position
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_width_indices = [i for i, w in enumerate(widths) if w is not None]
        valid_width_positions = [normalized_positions[i] for i in valid_width_indices]
        valid_widths = [widths[i] for i in valid_width_indices]
        
        ax.plot(valid_width_positions, valid_widths, 'o-', color='green', linewidth=2)
        ax.set_xlabel('Normalized Position Along Centerline')
        ax.set_ylabel('Width (PCA Minor Axis)')
        ax.set_title(f'Width Along Centerline\n{os.path.basename(mesh_file)}')
        ax.grid(True)
        
        width_plot_path = os.path.join(output_dir, f"{base_name}_width_curve.png")
        plt.savefig(width_plot_path, dpi=150)
        plt.close(fig)
        
        # Create multi-section visualization
        create_section_montage(
            section_points_list, 
            normalized_positions,
            aspect_ratios,
            os.path.join(output_dir, f"{base_name}_section_montage.png")
        )
        
        if section_data_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh at low opacity
            vertices = mesh.vertices
            faces = mesh.faces
            
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh_triangles = vertices[faces]
            mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.5, edgecolor='gray', 
                                            linewidth=0.05, facecolor='lightgray')
            ax.add_collection3d(mesh_collection)
            
            # Plot centerline
            ax.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], 
                    'k-', linewidth=2, label='Centerline')
            
            # Plot cross-sections
            cmap = plt.cm.plasma
            for idx, section in enumerate(section_data_3d):
                points_2d = section['points_2d']  # Get the 2D points
                transform = section['transform']  # Get the transform matrix
                norm_pos = section['norm_pos']
                
                # Create color based on position
                color = cmap(norm_pos)
                
                # Order points for clean visualization
                ordered_points_2d = order_points(points_2d, method="angular")
                
                # Transform 2D points back to 3D using the stored transform matrix
                points_3d = []
                for pt_2d in ordered_points_2d:
                    # Create homogeneous coordinates
                    pt_2d_h = np.array([pt_2d[0], pt_2d[1], 0.0, 1.0])  # Add z=0 and w=1
                    # Apply transformation
                    pt_3d_h = transform.dot(pt_2d_h)
                    # Convert back from homogeneous
                    points_3d.append(pt_3d_h[:3])
                
                points_3d = np.array(points_3d)
                
                # Close the loop for plotting
                pts_closed = np.vstack([points_3d, points_3d[0]])
                
                # Draw outline
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2], 
                        color=color, linewidth=2, alpha=0.9)
                
                # Create filled polygon
                verts = [list(zip(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2]))]
                poly = Poly3DCollection(verts, alpha=0.6, facecolor=color)
                ax.add_collection3d(poly)
                
                # Add small label with position
                pos = section['position']
                ax.text(pos[0], pos[1], pos[2], f"{norm_pos:.2f}", 
                    fontsize=8, color='white', 
                    horizontalalignment='center', verticalalignment='center')
            
            # Add sampled points
            positions = np.array([s['position'] for s in section_data_3d])
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c='blue', marker='o', s=30, alpha=0.7)
            
            # Focus the view
            center = np.mean(smoothed_centerline, axis=0)
            max_range = 0.7 * np.max([
                np.ptp(smoothed_centerline[:, 0]),
                np.ptp(smoothed_centerline[:, 1]),
                np.ptp(smoothed_centerline[:, 2])
            ])
            
            #ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
            #ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
            #ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
            
            # Set good viewing angle
            ax.view_init(elev=60, azim=90)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Cross-Sections Along Centerline\n{os.path.basename(mesh_file)}')
            
            # Save 3D cross-section visualization
            in_situ_path = os.path.join(output_dir, f"{base_name}_3d_sections.png")
            plt.savefig(in_situ_path, dpi=200)
            plt.close(fig)
            print(f"  Created 3D cross-section visualization: {in_situ_path}")

        # --- START: New Plotly 3D HTML Plot ---
            try:
                import plotly.graph_objects as go

                plotly_traces = []
                
                # Add Mesh
                plotly_traces.append(go.Mesh3d(
                    x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                    i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                    opacity=0.4, color='lightgrey', name='Mesh' 
                ))

                # Add Centerline
                plotly_traces.append(go.Scatter3d(
                    x=smoothed_centerline[:, 0], y=smoothed_centerline[:, 1], z=smoothed_centerline[:, 2],
                    mode='lines+markers', line=dict(color='black', width=5), marker=dict(size=3, color='black'),
                    name='Centerline'
                ))

                # Add Cross-Sections
                cmap_plotly = matplotlib.colormaps['plasma'] # Use the same colormap
                
                section_annotations = []

                for idx, section_data in enumerate(section_data_3d):
                    points_2d = section_data['points_2d']
                    transform_matrix = section_data['transform']
                    norm_pos = section_data['norm_pos']
                    section_position_3d = section_data['position'] # Get the 3D position of the section

                    # Use the same color mapping as Matplotlib
                    # Convert RGBA from plt.cm.plasma to a Plotly compatible string 'rgb(r,g,b)' or 'rgba(r,g,b,a)'
                    rgba_color = cmap_plotly(norm_pos) 
                    plotly_color_str = f'rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})'


                    ordered_points_2d = order_points(points_2d, method="angular")
                    
                    points_3d_list = []
                    for pt_2d in ordered_points_2d:
                        pt_2d_h = np.array([pt_2d[0], pt_2d[1], 0.0, 1.0])
                        pt_3d_h = transform_matrix.dot(pt_2d_h)
                        points_3d_list.append(pt_3d_h[:3])
                    
                    if not points_3d_list:
                        continue

                    points_3d_array = np.array(points_3d_list)
                    
                    # Close the loop for outline
                    outline_3d_closed = np.vstack([points_3d_array, points_3d_array[0]])
                    
                    plotly_traces.append(go.Scatter3d(
                        x=outline_3d_closed[:, 0], y=outline_3d_closed[:, 1], z=outline_3d_closed[:, 2],
                        mode='lines', line=dict(color=plotly_color_str, width=4),
                        name=f'Section {idx} (Pos: {norm_pos:.2f})'
                    ))
                    
                    # Add text label for the section position
                    section_annotations.append(dict(
                        showarrow=False,
                        x=section_position_3d[0],
                        y=section_position_3d[1],
                        z=section_position_3d[2],
                        text=f"{norm_pos:.2f}",
                        font=dict(color="white", size=10),
                        bgcolor="rgba(0,0,0,0.5)" # Semi-transparent background for better visibility
                    ))

                # Add sampled points (positions on centerline where sections are taken)
                sampled_plot_positions = np.array([s['position'] for s in section_data_3d])
                if len(sampled_plot_positions) > 0:
                    plotly_traces.append(go.Scatter3d(
                        x=sampled_plot_positions[:, 0], y=sampled_plot_positions[:, 1], z=sampled_plot_positions[:, 2],
                        mode='markers', marker=dict(size=5, color='blue', symbol='circle'),
                        name='Sampled Positions'
                    ))

                fig_plotly = go.Figure(data=plotly_traces)
                
                fig_plotly.update_layout(
                    title=f'Interactive 3D Cross-Sections - {os.path.basename(mesh_file)}',
                    scene=dict(
                        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                        aspectmode='data', # Ensures correct aspect ratio
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5) # Adjust camera for a good initial view
                        ),
                        annotations=section_annotations
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                
                plotly_html_path = os.path.join(output_dir, f"{base_name}_3d_sections_interactive.html")
                fig_plotly.write_html(plotly_html_path)
                print(f"  Created interactive 3D cross-section visualization (HTML): {plotly_html_path}")

            except ImportError:
                print("  Plotly is not installed. Skipping interactive 3D HTML plot.")
            except Exception as e_plotly:
                print(f"  Error creating Plotly 3D HTML plot: {e_plotly}")
            # --- END: New Plotly 3D HTML Plot ---
    
    # Return comprehensive results
    return {
        'positions': normalized_positions,
        'aspect_ratios': aspect_ratios,
        'widths': widths,
        'section_points': section_points_list,
        'centerline': smoothed_centerline
    }

def create_section_montage(section_points_list, positions, aspect_ratios, output_path):
    # Determine how many valid sections we have
    valid_sections = [(i, points) for i, points in enumerate(section_points_list) 
                     if points is not None and aspect_ratios[i] is not None 
                     and aspect_ratios[i] <= 30.0]
    
    if len(valid_sections) == 0:
        print("  No valid sections to create montage")
        return
        
    # Create grid layout
    n_sections = len(valid_sections)
    cols = min(5, n_sections)
    rows = (n_sections + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle("Cross-Sections Along Centerline", fontsize=16)
    
    # Flatten axes if needed
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    # Calculate the maximum extent across all valid sections for consistent scaling
    all_points = np.vstack([points for _, points in valid_sections])
    max_extent = np.max(np.abs(all_points)) * 1.1
    
    for i, (idx, points) in enumerate(valid_sections):
        if i >= len(axes.flatten()):
            break
            
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 and cols > 1 else axes[i]
        
        if points is not None and len(points) >= 3:
            # Order points for a clean polygon
            ordered_points = order_points(points, method="angular")
            
            # Plot outline and close the loop
            ax.plot(
                np.append(ordered_points[:, 0], ordered_points[0, 0]),
                np.append(ordered_points[:, 1], ordered_points[0, 1]),
                'b-', linewidth=1.5
            )
            
            # Fill the polygon
            ax.fill(ordered_points[:, 0], ordered_points[:, 1], alpha=0.2, color='blue')
            
            # Title with position and aspect ratio
            ax.set_title(f"Pos: {positions[idx]:.2f}\nAR: {aspect_ratios[idx]:.2f}")
        else:
            ax.text(0.5, 0.5, "No valid section", ha='center', va='center', transform=ax.transAxes)
            
        ax.set_aspect('equal')
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.grid(True, alpha=0.3)
        
    # Hide empty subplots
    for i in range(len(valid_sections), rows*cols):
        row = i // cols
        col = i % cols
        if rows > 1 and cols > 1:
            axes[row, col].axis('off')
        elif i < len(axes):
            axes[i].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make space for suptitle
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    files_to_process = [
        "Meshes/Onion_OBJ/Ac_DA_1_3.obj", "Meshes/Onion_OBJ/Ac_DA_1_2.obj", "Meshes/Onion_OBJ/Ac_DA_1_5.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_7.obj", "Meshes/Onion_OBJ/Ac_DA_3_6.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_3.obj", "Meshes/Onion_OBJ/Ac_DA_3_2.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_1.obj", "Meshes/Onion_OBJ/Ac_DA_2_7.obj",
         "Meshes/Onion_OBJ/Ac_DA_2_4.obj", "Meshes/Onion_OBJ/Ac_DA_2_3.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_8_mesh.obj", "Meshes/Onion_OBJ/Ac_DA_1_6.obj"
    ]

    ## Test with a single file
    #files_to_process = ["Meshes/Onion_OBJ/Ac_DA_1_2.obj"]
    output_dir = "centerline_results"
    
    # Initialize storage for collected results
    all_results = {}

    for files in files_to_process:
        mesh_file = files
        print(f"\nProcessing file: {mesh_file}")
        mesh_name = mesh_file.split("/")[-1]
        
        # Call the analysis function
        results = analyze_centerline_sections(
            mesh_file,
            num_sections=15,
            visualize=True,
            output_dir=output_dir+"/"+mesh_name
        )
        
        if results:
            all_results[mesh_name] = results
            print("\nAnalysis complete. Results summary:")
            valid_ars = [ar for ar in results['aspect_ratios'] if ar is not None]
            print(f"  - Number of valid sections: {len(valid_ars)} / {len(results['positions'])}")
            
            if valid_ars:  # Add this check
                print(f"  - Aspect ratio range: {min(valid_ars):.2f} - {max(valid_ars):.2f}")
            else:
                print("  - No valid aspect ratios found. Section analysis failed.")
    
    # After processing all files, create a combined aspect ratio plot with average
    if all_results:
        # Create combined aspect ratio plot
        plt.figure(figsize=(12, 8))
        
        # Use a colormap to distinguish different lines
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
        
        # Prepare data for average calculation
        all_positions = []
        all_ratios = []
        
        # Plot each mesh's aspect ratio curve with lower opacity
        for idx, (mesh_name, result) in enumerate(all_results.items()):
            # Get valid data points - MODIFIED with filter
            valid_indices = [i for i, ar in enumerate(result['aspect_ratios']) if ar is not None and ar <= 1.6]
            valid_positions = [result['positions'][i] for i in valid_indices]
            valid_ratios = [result['aspect_ratios'][i] for i in valid_indices]
            
            # Sort by position to ensure correct line drawing
            sorted_indices = np.argsort(valid_positions)
            valid_positions = [valid_positions[i] for i in sorted_indices]
            valid_ratios = [valid_ratios[i] for i in sorted_indices]
            
            # Store for average calculation
            all_positions.append(valid_positions)
            all_ratios.append(valid_ratios)
            
            # Plot with a distinct color but low opacity
            plt.plot(valid_positions, valid_ratios, 'o-', linewidth=1.0, color=colors[idx], 
                     label=mesh_name.replace('.obj', ''), alpha=0.3)
        
        # Calculate average curve using interpolation
        # Create a common set of position values
        common_positions = np.linspace(0.0, 1.0, 50)
        interpolated_ratios = []
        
        for positions, ratios in zip(all_positions, all_ratios):
            # Skip if too few points
            if len(positions) < 3:
                continue
                
            # Use numpy's interp function to get values at common positions
            try:
                # Only interpolate within the data range (avoid extrapolation)
                min_pos = max(common_positions[0], min(positions))
                max_pos = min(common_positions[-1], max(positions))
                valid_common = [p for p in common_positions if min_pos <= p <= max_pos]
                
                if valid_common:
                    interp_values = np.interp(valid_common, positions, ratios)
                    # Store with position for later averaging
                    interpolated_ratios.append(list(zip(valid_common, interp_values)))
            except:
                print(f"  Warning: Could not interpolate curve for averaging")
        
        # Combine all interpolated values
        if interpolated_ratios:
            # Group by position
            position_groups = {}
            for curve in interpolated_ratios:
                for pos, ratio in curve:
                    pos_rounded = round(pos, 3)  # Round to group nearby positions
                    if pos_rounded not in position_groups:
                        position_groups[pos_rounded] = []
                    position_groups[pos_rounded].append(ratio)
            
            # Calculate averages for each position
            avg_positions = []
            avg_ratios = []
            for pos in sorted(position_groups.keys()):
                if position_groups[pos]:  # Check if we have values
                    avg_positions.append(pos)
                    avg_ratios.append(np.median(position_groups[pos]))
            
            # Plot the average curve with high opacity and thicker line
            if avg_positions:
                plt.plot(avg_positions, avg_ratios, 'o-', linewidth=3.0, color='red', 
                         label='Average', alpha=1.0, markersize=8)
        
        # Add plot elements
        plt.xlabel('Normalized Position Along Centerline (0=Tip, 1=Midpoint)', fontsize=12)
        plt.ylabel('Aspect Ratio', fontsize=12)
        plt.title('Aspect Ratio Along Centerline - All Samples', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add legend with smaller font to accommodate multiple entries
        plt.legend(loc='best', fontsize=8, ncol=2)
        
        # Save combined plot
        plt.tight_layout()
        combined_plot_path = os.path.join(output_dir, "combined_aspect_ratio_curves.png")
        plt.savefig(combined_plot_path, dpi=200)
        plt.close()
        
        print(f"\nCreated combined aspect ratio plot: {combined_plot_path}")
        
        # Also create a width comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot each mesh's width curve
        for idx, (mesh_name, result) in enumerate(all_results.items()):
            # Get valid data points
            valid_indices = [i for i, w in enumerate(result['widths']) if w is not None]
            valid_positions = [result['positions'][i] for i in valid_indices]
            valid_widths = [result['widths'][i] for i in valid_indices]
            
            # Sort by position
            sorted_indices = np.argsort(valid_positions)
            valid_positions = [valid_positions[i] for i in sorted_indices]
            valid_widths = [valid_widths[i] for i in sorted_indices]
            
            # Plot with a distinct color
            plt.plot(valid_positions, valid_widths, 'o-', linewidth=1.5, color=colors[idx], 
                     label=mesh_name.replace('.obj', ''))
        
        # Add plot elements
        plt.xlabel('Normalized Position Along Centerline (0=Tip, 1=Midpoint)', fontsize=12)
        plt.ylabel('Width (PCA Minor Axis)', fontsize=12)
        plt.title('Width Along Centerline - All Samples', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='best', fontsize=8, ncol=2)
        
        # Save combined width plot
        plt.tight_layout()
        combined_width_path = os.path.join(output_dir, "combined_width_curves.png")
        plt.savefig(combined_width_path, dpi=200)
        plt.close()
        
        print(f"Created combined width plot: {combined_width_path}")
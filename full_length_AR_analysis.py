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
                             order_points, _determine_midpoint_plane, _determine_tip_plane_v2,
                             fit_ellipse_robust, generate_ellipse_points) # ADDED ellipse functions
import edge_detection as ed

def analyze_centerline_sections(mesh_file, 
                               num_sections=20, 
                               visualize=True,
                               output_dir=None):
    """
    Analyzes cross-sections along the full centerline of a stomata guard cell
    and measures how aspect ratio changes along its length.
    Aspect ratios are calculated relative to the orientation of the midpoint section.
    
    Args:
        mesh_file (str): Path to the mesh file (OBJ format)
        num_sections (int): Number of cross-sections to analyze along the centerline
        visualize (bool): Whether to generate visualizations
        output_dir (str): Directory to save visualizations and results
        
    Returns:
        dict: Contains:
            - 'positions': Normalized positions along the centerline (0 to 1)
            - 'aspect_ratios': Aspect ratios at each position (relative to midpoint orientation)
            - 'widths': PCA widths at each position (consistent with midpoint orientation)
            - 'section_points': List of 2D cross-section points at each position
            - 'centerline': The smoothed centerline used for analysis
    """
    # Create output directory if needed
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    base_name_for_outputs = os.path.splitext(os.path.basename(mesh_file))[0]

    # Step 1: Process mesh and extract key features
    print(f"\nProcessing mesh: {mesh_file}")
    ed_output = ed.find_seam_by_raycasting(mesh_file, visualize=False)
    
    if ed_output is None or 'mesh_object' not in ed_output or ed_output['mesh_object'] is None:
        print(f"  Error: Edge detection failed for {mesh_file}")
        return None
    
    mesh = ed_output['mesh_object']
    shared_wall_points = ed_output.get('shared_wall_points')
    pore_center = ed_output.get('pore_center_coords') # This is detected_pore_center_3d_ed
    estimated_centerline_3d_from_ed = ed_output.get('estimated_centerline_points')

    # Step 2: Get radial dimensions
    ray_origin_for_radial_cast = pore_center 
    print(f"  Performing radial ray casting from origin: {ray_origin_for_radial_cast.round(3) if ray_origin_for_radial_cast is not None else 'mesh centroid (default)'}")
    ray_count = 45 
    inner_points, outer_points, raw_centerline_points_from_radial, minor_radius = get_radial_dimensions(
        mesh, center=ray_origin_for_radial_cast, ray_count=ray_count
    ) 
    if raw_centerline_points_from_radial is None or minor_radius is None:
        print(f"  Error: Radial dimensions could not be determined for {mesh_file}")
        return None

    # Step 3: Smooth the raw centerline from radial dimensions
    smoothed_centerline_initial = _smooth_centerline_savgol(raw_centerline_points_from_radial)
    if smoothed_centerline_initial is None:
        smoothed_centerline_initial = raw_centerline_points_from_radial
    
    if smoothed_centerline_initial is None or len(smoothed_centerline_initial) < 2:
        print("  Error: Not enough points in initial smoothed centerline for analysis.")
        return None

    # Step 4: Determine Tip and Midpoint Plane Origins & Extract Centerline Segment
    print("  Determining robust tip and midpoint plane origins...")
    midpoint_plane_origin_3d, _ = _determine_midpoint_plane(
        smoothed_centerline_initial, 
        pore_center 
    )
    tip_plane_origin_3d, _, _ = _determine_tip_plane_v2(
        smoothed_centerline_initial, pore_center, shared_wall_points,
        minor_radius, inner_points, 
        min_tip_distance=0.05, 
        estimated_centerline_3d_from_ed=estimated_centerline_3d_from_ed
    )

    if tip_plane_origin_3d is None or midpoint_plane_origin_3d is None:
        print("  Error: Could not determine robust tip or midpoint plane origins. Cannot proceed.")
        return None
    
    print(f"  Robust Tip Plane Origin: {tip_plane_origin_3d.round(3)}")
    print(f"  Robust Midpoint Plane Origin: {midpoint_plane_origin_3d.round(3)}")

    actual_tip_cl_idx = np.argmin(np.linalg.norm(smoothed_centerline_initial - tip_plane_origin_3d, axis=1))
    actual_mid_cl_idx = np.argmin(np.linalg.norm(smoothed_centerline_initial - midpoint_plane_origin_3d, axis=1))

    if actual_tip_cl_idx == actual_mid_cl_idx:
        print("  Error: Tip and Midpoint plane origins map to the same point on the centerline. Adjusting for a minimal segment.")
        if len(smoothed_centerline_initial) > 1:
            actual_mid_cl_idx = (actual_tip_cl_idx + 1) % len(smoothed_centerline_initial)
            if actual_tip_cl_idx == actual_mid_cl_idx : # Still same (e.g. 1 point CL)
                 print("  Cannot form a segment even with adjustment. Centerline too short.")
                 return None
        else:
            print("  Cannot form a segment. Centerline too short.")
            return None

    print(f"  Closest CL idx to Tip Origin: {actual_tip_cl_idx} (Point: {smoothed_centerline_initial[actual_tip_cl_idx].round(3)})")
    print(f"  Closest CL idx to Mid Origin: {actual_mid_cl_idx} (Point: {smoothed_centerline_initial[actual_mid_cl_idx].round(3)})")

    N_cl = len(smoothed_centerline_initial)
    path1_indices = []
    curr = actual_tip_cl_idx; count1 = 0
    while True:
        path1_indices.append(curr); count1 += 1
        if curr == actual_mid_cl_idx or count1 > N_cl: break
        curr = (curr + 1) % N_cl
    if curr != actual_mid_cl_idx and count1 > N_cl: path1_indices = list(range(N_cl)) # Safety

    path2_indices = []
    curr = actual_tip_cl_idx; count2 = 0
    while True:
        path2_indices.append(curr); count2 += 1
        if curr == actual_mid_cl_idx or count2 > N_cl: break
        curr = (curr - 1 + N_cl) % N_cl
    if curr != actual_mid_cl_idx and count2 > N_cl: path2_indices = list(range(N_cl)) # Safety
        
    final_path_indices = path1_indices if len(path1_indices) <= len(path2_indices) else path2_indices
    print(f"  Selected Path (length {len(final_path_indices)}) for centerline segment.")

    if not final_path_indices:
        print("  Error: Could not determine a valid path between tip and midpoint.")
        return None

    centerline_segment_for_analysis = smoothed_centerline_initial[final_path_indices]
    if len(centerline_segment_for_analysis) < 2:
        print(f"  Error: Extracted centerline segment has too few points ({len(centerline_segment_for_analysis)}).")
        return None
    print(f"  Extracted new centerline segment for analysis with {len(centerline_segment_for_analysis)} points.")
    
    smoothed_centerline = centerline_segment_for_analysis # This is the final CL segment to use

    # Step 4b: Normalize distances along the new segment
    centerline_segments_diff = np.diff(smoothed_centerline, axis=0)
    segment_lengths = np.linalg.norm(centerline_segments_diff, axis=1)
    if len(segment_lengths) == 0:
        normalized_distances = np.array([0.0]) if len(smoothed_centerline) == 1 else np.array([])
    else:
        cumulative_distances = np.zeros(len(smoothed_centerline))
        cumulative_distances[1:] = np.cumsum(segment_lengths)
        normalized_distances = cumulative_distances / cumulative_distances[-1] if cumulative_distances[-1] > 0 else np.zeros(len(smoothed_centerline))
    
    total_points_segment = len(smoothed_centerline)

    # Step 5: Sample positions along the centerline segment
    sampled_positions = []
    sampled_tangents = []
    final_normalized_positions = [] # Renamed to avoid conflict with loop var
    
    target_norm_positions = np.linspace(0, 1.0, num_sections)
    sampled_indices = [np.argmin(np.abs(normalized_distances - target)) for target in target_norm_positions]

    for i in sampled_indices:
        position = smoothed_centerline[i]
        sampled_positions.append(position)
        
        if total_points_segment < 2: tangent_vec = np.array([0.0,1.0,0.0]) # Should not happen due to earlier check
        elif i == 0: tangent_vec = smoothed_centerline[1] - smoothed_centerline[0]
        elif i == total_points_segment - 1: tangent_vec = smoothed_centerline[-1] - smoothed_centerline[-2]
        else: tangent_vec = smoothed_centerline[i + 1] - smoothed_centerline[i - 1]
        
        tangent_norm_val = np.linalg.norm(tangent_vec)
        tangent = tangent_vec / tangent_norm_val if tangent_norm_val > 1e-6 else np.array([0.0, 1.0, 0.0])
        
        sampled_tangents.append(tangent)
        final_normalized_positions.append(normalized_distances[i]) # Use the actual normalized distance of the sampled point

    # --- Stage 1: Initial Section Processing & Geometry Extraction ---
    raw_section_data_list = []
    print("DEBUG: >>> Entering INITIAL SECTIONING LOOP (Geometry Extraction) <<<")
    for idx, (s_pos, s_tan, s_norm_pos) in enumerate(zip(sampled_positions, sampled_tangents, final_normalized_positions)):
        print(f"  Processing geometry for section {idx+1}/{len(sampled_positions)} at norm_pos {s_norm_pos:.2f}")
        
        current_section_data_item = {
            'points_2d': None, 'transform': None, 'position_3d': s_pos, 
            'tangent_3d': s_tan, 'norm_pos': s_norm_pos, 'valid_geometry': False
        }

        if np.all(s_tan == 0):
            print(f"  Error: Zero tangent at section {idx+1}")
            raw_section_data_list.append(current_section_data_item)
            continue
            
        section = None
        try:
            section = mesh.section(plane_origin=s_pos, plane_normal=s_tan)
            if section is None or not hasattr(section, 'entities') or len(section.entities) == 0:
                print(f"  Section {idx+1} failed or empty. Attempting perturbation...")
                for attempt in range(3):
                    perp1 = np.array([1,0,0]) - np.dot(np.array([1,0,0]), s_tan) * s_tan
                    if np.linalg.norm(perp1) < 1e-6: perp1 = np.array([0,1,0]) - np.dot(np.array([0,1,0]), s_tan) * s_tan
                    perp1 /= np.linalg.norm(perp1)
                    perp2 = np.cross(s_tan, perp1); perp2 /= np.linalg.norm(perp2)
                    offset = 0.01 * (attempt + 1); new_position = s_pos + offset * (perp1 + perp2)
                    section = mesh.section(plane_origin=new_position, plane_normal=s_tan)
                    if section is not None and hasattr(section, 'entities') and len(section.entities) > 0:
                        print(f"  Successfully created section {idx+1} with offset {offset}")
                        current_section_data_item['position_3d'] = new_position # Update position if perturbed
                        break
                if section is None or not hasattr(section, 'entities') or len(section.entities) == 0:
                     print(f"  [FAIL] Section {idx+1} could not be created even with perturbation.")
                     raw_section_data_list.append(current_section_data_item)
                     continue
        except Exception as e_sec:
            print(f"  Error creating section {idx+1}: {e_sec}")
            raw_section_data_list.append(current_section_data_item)
            continue
        
        print(f"  Section {idx+1}: Created successfully with {len(section.entities)} entities")
        
        try:
            path_2D, transform_2d_to_3d_geom = section.to_2D()
            points_2D_geom = path_2D.vertices
        except Exception as e_to_2d:
            print(f"  Error converting section {idx+1} to 2D: {e_to_2d}")
            raw_section_data_list.append(current_section_data_item)
            continue
        print(f"  Section {idx+1}: 2D projection has {len(points_2D_geom)} points")

        plane_origin_2d_geom, _ = _project_plane_origin_to_2d(current_section_data_item['position_3d'], transform_2d_to_3d_geom, points_2D_geom)
        
        if minor_radius is None: # Global minor_radius
            print(f"  [CRITICAL FAIL] Global minor_radius is None. Cannot filter section {idx+1}.")
            raw_section_data_list.append(current_section_data_item)
            continue
        
        print(f"    Using global minor_radius for filtering section {idx+1}: {minor_radius:.3f}")
        filtered_points_2D_geom, _ = filter_section_points(
            points_2D_geom, minor_radius, plane_origin_2d_geom,
            eps_factor=0.15, min_samples=3
        )
        print(f"  Section {idx+1}: After filtering: {len(filtered_points_2D_geom)} points from {len(points_2D_geom)}")

        if len(filtered_points_2D_geom) >= 3:
            current_section_data_item['points_2d'] = filtered_points_2D_geom
            current_section_data_item['transform'] = transform_2d_to_3d_geom
            current_section_data_item['valid_geometry'] = True
        else:
            print(f"  [FAIL] Section {idx+1} has too few points ({len(filtered_points_2D_geom)}) after filtering geometry.")
        
        raw_section_data_list.append(current_section_data_item)

    # --- Stage 2: Determine Midpoint Reference Orientation ---
    reference_long_axis_vector = None
    angle_ref_rad = None
    midpoint_section_idx_for_ref = -1

    if raw_section_data_list:
        valid_geom_indices = [i for i, data in enumerate(raw_section_data_list) if data['valid_geometry']]
        if valid_geom_indices:
            # Find section closest to norm_pos = 1.0 among those with valid geometry
            midpoint_candidate_norm_pos = [raw_section_data_list[i]['norm_pos'] for i in valid_geom_indices]
            closest_idx_in_valid_list = np.argmin(np.abs(np.array(midpoint_candidate_norm_pos) - 1.0))
            midpoint_section_idx_for_ref = valid_geom_indices[closest_idx_in_valid_list]
            
            midpoint_data_for_ref = raw_section_data_list[midpoint_section_idx_for_ref]
            if midpoint_data_for_ref['points_2d'] is not None and len(midpoint_data_for_ref['points_2d']) >= 3:
                try:
                    print(f"  Calculating reference orientation from section {midpoint_section_idx_for_ref+1} (norm_pos: {midpoint_data_for_ref['norm_pos']:.2f})")
                    midpoint_pca = PCA(n_components=2).fit(midpoint_data_for_ref['points_2d'])
                    reference_long_axis_vector = midpoint_pca.components_[0]
                    print(f"  Reference long axis vector from midpoint: {reference_long_axis_vector.round(3)}")
                except Exception as e_mid_pca:
                    print(f"  Error calculating PCA for midpoint reference: {e_mid_pca}")
            else:
                print("  Midpoint section selected for reference has insufficient points for PCA.")
        else:
            print("  No valid sections with geometry to determine midpoint reference.")

        if reference_long_axis_vector is not None:
            # Calculate the angle of the reference_long_axis_vector and normalize to [0, pi)
            angle_ref_rad_raw = np.arctan2(reference_long_axis_vector[1], reference_long_axis_vector[0])
            angle_ref_rad = (angle_ref_rad_raw + np.pi) % np.pi # Maps to [0, pi)
            print(f"  Reference orientation angle (radians): {angle_ref_rad:.3f}(degrees: {np.degrees(angle_ref_rad):.1f})")
    if reference_long_axis_vector is None:
        print("  Warning: Could not determine reference orientation. Aspect ratios will be max/min (standard PCA).")

    # --- Stage 3: Recalculate Aspect Ratios and Widths with Reference ---
    final_pca_aspect_ratios = [] # Renamed
    final_pca_widths = []        # Renamed
    final_ellipse_aspect_ratios = [] # New
    final_ellipse_widths = []        # New
    final_ellipse_points_for_plot = [] # New, for montage
    final_ellipse_inlier_ratios = []
    final_ellipse_relative_orientations_deg = []

    final_section_points_list = [] 
    section_data_3d = [] 

    print("DEBUG: >>> Entering FINAL ASPECT RATIO AND ELLIPSE FITTING LOOP <<<")
    for idx, data_item in enumerate(raw_section_data_list):
        pca_ar, pca_w = None, None
        ellipse_ar, ellipse_w = None, None
        ellipse_plot_pts = None
        inlier_ratio_val = 0.0 # NEW: Initialize inlier ratio for the section
        current_points_2d = data_item['points_2d']
        relative_orientation_deg_val = None
        
        final_section_points_list.append(current_points_2d) 

        if data_item['valid_geometry'] and current_points_2d is not None and len(current_points_2d) >= 3:
            # PCA-based metrics ...
            pca_ar, pca_w = _calculate_pca_metrics(
                current_points_2d, 
                f"section {idx+1} PCA", 
                reference_orientation_vector=reference_long_axis_vector
            )
            
            # Ellipse fitting
            try:
                # --- Reverted RANSAC Threshold Calculation ---
                # Set a threshold for RANSAC. This is a critical parameter.
                # It's the maximum distance for a data point to be classified as an inlier.
                # You might need to tune the factor (e.g., 0.05, 0.1) or the absolute fallback (e.g., 0.01, 0.02).
                ransac_threshold_factor = 0.15 # TUNABLE: Multiplier for global minor_radius
                absolute_fallback_threshold = 0.15 # TUNABLE: Absolute threshold if minor_radius is unavailable

                if minor_radius is not None and minor_radius > 1e-6:
                    threshold_to_use = ransac_threshold_factor * minor_radius
                else:
                    threshold_to_use = absolute_fallback_threshold
                
                # Ensure the threshold is not excessively small if calculated from a tiny minor_radius
                threshold_to_use = max(threshold_to_use, 1e-4) 

                # Optional print for tuning:
                # print(f"    Section {idx+1}: Using RANSAC threshold: {threshold_to_use:.4f} (minor_radius: {minor_radius:.3f if minor_radius is not None else 'N/A'})")

                # Assuming fit_ellipse_robust now returns: ellipse_params_tuple, current_inlier_ratio
                # If you also had it returning current_inlier_mask_from_fit, adjust the call accordingly.
                ellipse_params_tuple, current_inlier_ratio = fit_ellipse_robust(
                    current_points_2d, 
                    residual_threshold=threshold_to_use, 
                    max_trials=500 # Keep increased max_trials if it was helpful
                )
                inlier_ratio_val = current_inlier_ratio

                if ellipse_params_tuple:
                    el_center, el_minor_len, el_major_len, el_angle_rad = ellipse_params_tuple
                    
                    if el_minor_len > 1e-6: 
                        ellipse_ar = el_major_len / el_minor_len
                    else:
                        ellipse_ar = np.inf if el_major_len > 1e-6 else 0
                    
                    ellipse_w = el_minor_len / 2.0 
                    ellipse_plot_pts = generate_ellipse_points(el_center, el_minor_len, el_major_len, el_angle_rad)

                    if angle_ref_rad is not None:
                        delta_orientation_rad = el_angle_rad - angle_ref_rad
                        # Normalize to [-pi/2, pi/2)
                        normalized_relative_orientation_rad = (delta_orientation_rad + np.pi/2) % np.pi - np.pi/2
                        relative_orientation_deg_val = np.degrees(normalized_relative_orientation_rad)
                    else:
                        # If no reference, we could store the raw angle or None. Let's use None.
                        relative_orientation_deg_val = None
                else:
                    pass # Ellipse fitting failed, ellipse_ar/w remain None

            except Exception as e_ellipse:
                print(f"  Error during ellipse fitting for section {idx+1}: {e_ellipse}")
                inlier_ratio_val = 0.0 # Ensure it's 0 on exception too

            # Populate section_data_3d for 3D visualization (uses original cross-section)
            # The prompt specified 3D plots use original cross-section, so PCA metrics might be more relevant if an AR filter is applied here.
            # For now, let's assume the AR filter for 3D plot is on PCA_AR or remove it if not critical.
            # Using PCA AR for the 30.0 filter as it was the original logic.
            temp_ar_for_3d_filter = pca_ar if pca_ar is not None else ellipse_ar # Prioritize PCA for existing filter
            if temp_ar_for_3d_filter is not None and pca_w is not None and np.isfinite(temp_ar_for_3d_filter) and np.isfinite(pca_w) and abs(temp_ar_for_3d_filter) <= 30.0 : 
                section_data_3d.append({
                    'points_2d': current_points_2d, # Original points for 3D plot
                    'position': data_item['position_3d'],
                    'tangent': data_item['tangent_3d'],
                    'norm_pos': data_item['norm_pos'],
                    'aspect_ratio': temp_ar_for_3d_filter, # AR used for filtering this entry
                    'width': pca_w, # PCA width for this entry
                    'transform': data_item['transform'] 
                })
        
        final_pca_aspect_ratios.append(pca_ar)
        final_pca_widths.append(pca_w)
        final_ellipse_aspect_ratios.append(ellipse_ar)
        final_ellipse_widths.append(ellipse_w)
        final_ellipse_points_for_plot.append(ellipse_plot_pts)
        final_ellipse_inlier_ratios.append(inlier_ratio_val)
        final_ellipse_relative_orientations_deg.append(relative_orientation_deg_val) # NEW: Append relative orientation
    
    # Use ellipse data for the main 'aspect_ratios' and 'widths' going forward for plots
    # Keep PCA versions if needed for other comparisons later, but plots will use ellipse.
    aspect_ratios_for_plots = final_ellipse_aspect_ratios
    widths_for_plots = final_ellipse_widths
    #section_points_list = final_section_points_list 

    # Step 7: Create visualizations
    if visualize and output_dir:
        # Plot aspect ratio (ellipse) vs position
        fig_ar, ax_ar = plt.subplots(figsize=(10, 6))
        valid_ar_indices = [i for i, ar_val in enumerate(aspect_ratios_for_plots) if ar_val is not None]
        valid_ar_positions = [final_normalized_positions[i] for i in valid_ar_indices]
        valid_ar_ratios = [aspect_ratios_for_plots[i] for i in valid_ar_indices]
        
        if valid_ar_positions and valid_ar_ratios:
            sorted_plot_indices = np.argsort(valid_ar_positions)
            plot_positions_ar = np.array(valid_ar_positions)[sorted_plot_indices]
            plot_ratios_ar = np.array(valid_ar_ratios)[sorted_plot_indices]
            ax_ar.plot(plot_positions_ar, plot_ratios_ar, 'o-', linewidth=2)
        
        ax_ar.set_xlabel('Normalized Position Along Centerline')
        ax_ar.set_ylabel('Aspect Ratio (Ellipse)') # UPDATED LABEL
        ax_ar.set_title(f'Aspect Ratio (Ellipse) Along Centerline\n{base_name_for_outputs}') # UPDATED TITLE
        ax_ar.grid(True)
        aspect_plot_path = os.path.join(output_dir, f"{base_name_for_outputs}_aspect_ratio_ellipse_curve.png") # UPDATED FILENAME
        plt.savefig(aspect_plot_path, dpi=150); plt.close(fig_ar)
        
        # Plot width (ellipse semi-minor) vs position
        fig_w, ax_w = plt.subplots(figsize=(10, 6))
        valid_w_indices = [i for i, w_val in enumerate(widths_for_plots) if w_val is not None]
        valid_w_positions = [final_normalized_positions[i] for i in valid_w_indices]
        valid_w_values = [widths_for_plots[i] for i in valid_w_indices]

        if valid_w_positions and valid_w_values:
            sorted_plot_indices_w = np.argsort(valid_w_positions)
            plot_positions_w = np.array(valid_w_positions)[sorted_plot_indices_w]
            plot_values_w = np.array(valid_w_values)[sorted_plot_indices_w]
            ax_w.plot(plot_positions_w, plot_values_w, 'o-', color='green', linewidth=2)

        ax_w.set_xlabel('Normalized Position Along Centerline')
        ax_w.set_ylabel('Width (Ellipse Semi-Minor Axis)') # UPDATED LABEL
        ax_w.set_title(f'Width (Ellipse) Along Centerline\n{base_name_for_outputs}') # UPDATED TITLE
        ax_w.grid(True)
        width_plot_path = os.path.join(output_dir, f"{base_name_for_outputs}_width_ellipse_curve.png") # UPDATED FILENAME
        plt.savefig(width_plot_path, dpi=150); plt.close(fig_w)
        
        # Create section montage (will show original and fitted ellipse)
        create_section_montage(
            final_section_points_list, # Original 2D points
            final_ellipse_points_for_plot, # Points for drawing ellipses
            final_normalized_positions, 
            final_ellipse_aspect_ratios, # AR from ellipse
            os.path.join(output_dir, f"{base_name_for_outputs}_section_montage_with_ellipse.png") # UPDATED FILENAME
        )

        # NEW: Plot Inlier Ratio (ellipse) vs position
        fig_ir, ax_ir = plt.subplots(figsize=(10, 6))
        valid_ir_indices = [i for i, ir_val in enumerate(final_ellipse_inlier_ratios) if ir_val is not None] # Should always be a float
        valid_ir_positions = [final_normalized_positions[i] for i in valid_ir_indices]
        valid_ir_ratios = [final_ellipse_inlier_ratios[i] for i in valid_ir_indices]
        
        if valid_ir_positions and valid_ir_ratios:
            sorted_plot_indices_ir = np.argsort(valid_ir_positions)
            plot_positions_ir = np.array(valid_ir_positions)[sorted_plot_indices_ir]
            plot_ratios_ir = np.array(valid_ir_ratios)[sorted_plot_indices_ir]
            ax_ir.plot(plot_positions_ir, plot_ratios_ir, 'o-', color='purple', linewidth=2)
        
        ax_ir.set_xlabel('Normalized Position Along Centerline')
        ax_ir.set_ylabel('Ellipse Inlier Ratio (RANSAC)')
        ax_ir.set_title(f'Ellipse Fit Regularity (Inlier Ratio) Along Centerline\n{base_name_for_outputs}')
        ax_ir.grid(True)
        ax_ir.set_ylim(0, 1.05) # Inlier ratio is between 0 and 1
        inlier_ratio_plot_path = os.path.join(output_dir, f"{base_name_for_outputs}_ellipse_inlier_ratio_curve.png")
        plt.savefig(inlier_ratio_plot_path, dpi=150); plt.close(fig_ir)
        print(f"  Created ellipse inlier ratio plot: {inlier_ratio_plot_path}")

        # NEW: Plot Relative Orientation (ellipse major axis) vs position
        fig_orient, ax_orient = plt.subplots(figsize=(10, 6))
        valid_orient_indices = [i for i, orient_val in enumerate(final_ellipse_relative_orientations_deg) if orient_val is not None and np.isfinite(orient_val)]
        valid_orient_positions = [final_normalized_positions[i] for i in valid_orient_indices]
        valid_orient_values_deg = [final_ellipse_relative_orientations_deg[i] for i in valid_orient_indices]
        
        if valid_orient_positions and valid_orient_values_deg:
            sorted_plot_indices_orient = np.argsort(valid_orient_positions)
            plot_positions_orient = np.array(valid_orient_positions)[sorted_plot_indices_orient]
            plot_values_orient_deg = np.array(valid_orient_values_deg)[sorted_plot_indices_orient]
            ax_orient.plot(plot_positions_orient, plot_values_orient_deg, 'o-', color='cyan', linewidth=2)
        
        ax_orient.set_xlabel('Normalized Position Along Centerline')
        ax_orient.set_ylabel('Relative Orientation of Major Axis (degrees)')
        ax_orient.set_title(f'Ellipse Major Axis Orientation (Relative to Midpoint)\n{base_name_for_outputs}')
        ax_orient.grid(True)
        ax_orient.set_ylim(-95, 95) # Degrees, -90 to +90
        orientation_plot_path = os.path.join(output_dir, f"{base_name_for_outputs}_ellipse_relative_orientation_curve.png")
        plt.savefig(orientation_plot_path, dpi=150); plt.close(fig_orient)
        print(f"  Created ellipse relative orientation plot: {orientation_plot_path}")
        
        # Matplotlib 3D plot (uses section_data_3d populated in Stage 3)
        if section_data_3d: # Check if there's data to plot
            fig_3d_mpl = plt.figure(figsize=(12, 10))
            ax_3d_mpl = fig_3d_mpl.add_subplot(111, projection='3d')
            
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh_triangles = mesh.vertices[mesh.faces]
            mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.3, edgecolor='gray', linewidth=0.05, facecolor='lightgray')
            ax_3d_mpl.add_collection3d(mesh_collection)
            ax_3d_mpl.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], 'k-', linewidth=2, label='Centerline')
            
            cmap_mpl = matplotlib.colormaps['plasma']
            for sd_item in section_data_3d:
                points_2d_vis = sd_item['points_2d']
                transform_vis = sd_item['transform']
                norm_pos_vis = sd_item['norm_pos']
                color_vis = cmap_mpl(norm_pos_vis)
                
                ordered_points_2d_vis = order_points(points_2d_vis, method="angular")
                points_3d_vis_list = []
                for pt_2d_vis in ordered_points_2d_vis:
                    pt_2d_h_vis = np.array([pt_2d_vis[0], pt_2d_vis[1], 0.0, 1.0])
                    pt_3d_h_vis = transform_vis.dot(pt_2d_h_vis)
                    points_3d_vis_list.append(pt_3d_h_vis[:3])
                
                if not points_3d_vis_list: continue
                points_3d_array_vis = np.array(points_3d_vis_list)
                pts_closed_vis = np.vstack([points_3d_array_vis, points_3d_array_vis[0]])
                ax_3d_mpl.plot(pts_closed_vis[:, 0], pts_closed_vis[:, 1], pts_closed_vis[:, 2], color=color_vis, linewidth=2, alpha=0.9)
                
                verts_vis = [list(zip(pts_closed_vis[:, 0], pts_closed_vis[:, 1], pts_closed_vis[:, 2]))]
                poly_vis = Poly3DCollection(verts_vis, alpha=0.6, facecolor=color_vis)
                ax_3d_mpl.add_collection3d(poly_vis)
            
            sampled_positions_plot = np.array([s['position'] for s in section_data_3d])
            if len(sampled_positions_plot) > 0:
                 ax_3d_mpl.scatter(sampled_positions_plot[:, 0], sampled_positions_plot[:, 1], sampled_positions_plot[:, 2], c='blue', marker='o', s=30, alpha=0.7, depthshade=False)

            ax_3d_mpl.set_xlabel('X'); ax_3d_mpl.set_ylabel('Y'); ax_3d_mpl.set_zlabel('Z')
            ax_3d_mpl.set_title(f'3D Cross-Sections - {base_name_for_outputs}')
            ax_3d_mpl.view_init(elev=30, azim=45) # Adjust view for better visibility
            # Auto-scale axes
            max_val = np.max(mesh.bounds)
            min_val = np.min(mesh.bounds)
            ax_3d_mpl.auto_scale_xyz([min_val, max_val], [min_val, max_val], [min_val, max_val])


            in_situ_path = os.path.join(output_dir, f"{base_name_for_outputs}_3d_sections.png")
            plt.savefig(in_situ_path, dpi=200); plt.close(fig_3d_mpl)
            print(f"  Created 3D cross-section visualization (Matplotlib): {in_situ_path}")

        # Plotly 3D HTML Plot (uses section_data_3d populated in Stage 3)
            try:
                import plotly.graph_objects as go
                plotly_traces = []
                plotly_traces.append(go.Mesh3d(
                    x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                    i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                    opacity=0.3, color='lightgrey', name='Mesh' 
                ))
                plotly_traces.append(go.Scatter3d(
                    x=smoothed_centerline[:, 0], y=smoothed_centerline[:, 1], z=smoothed_centerline[:, 2],
                    mode='lines+markers', line=dict(color='black', width=6), marker=dict(size=3.5, color='black'),
                    name='Centerline Segment'
                ))

                cmap_plotly = matplotlib.colormaps['plasma']
                section_annotations_plotly = []
                for idx_plotly, sd_item_plotly in enumerate(section_data_3d):
                    points_2d_plotly = sd_item_plotly['points_2d']
                    transform_plotly = sd_item_plotly['transform']
                    norm_pos_plotly = sd_item_plotly['norm_pos']
                    section_pos_3d_plotly = sd_item_plotly['position']
                    rgba_color_plotly = cmap_plotly(norm_pos_plotly) 
                    plotly_color_str = f'rgba({int(rgba_color_plotly[0]*255)}, {int(rgba_color_plotly[1]*255)}, {int(rgba_color_plotly[2]*255)}, {rgba_color_plotly[3]})'

                    ordered_points_2d_plotly = order_points(points_2d_plotly, method="angular")
                    points_3d_list_plotly = []
                    for pt_2d_pl in ordered_points_2d_plotly:
                        pt_2d_h_pl = np.array([pt_2d_pl[0], pt_2d_pl[1], 0.0, 1.0])
                        pt_3d_h_pl = transform_plotly.dot(pt_2d_h_pl)
                        points_3d_list_plotly.append(pt_3d_h_pl[:3])
                    
                    if not points_3d_list_plotly: continue
                    points_3d_array_plotly = np.array(points_3d_list_plotly)
                    outline_3d_closed_plotly = np.vstack([points_3d_array_plotly, points_3d_array_plotly[0]])
                    
                    plotly_traces.append(go.Scatter3d(
                        x=outline_3d_closed_plotly[:, 0], y=outline_3d_closed_plotly[:, 1], z=outline_3d_closed_plotly[:, 2],
                        mode='lines', line=dict(color=plotly_color_str, width=4),
                        name=f'Section {idx_plotly} (Pos: {norm_pos_plotly:.2f}, AR: {sd_item_plotly["aspect_ratio"]:.2f})'
                    ))
                    section_annotations_plotly.append(dict(
                        showarrow=False, x=section_pos_3d_plotly[0], y=section_pos_3d_plotly[1], z=section_pos_3d_plotly[2],
                        text=f"{norm_pos_plotly:.2f}", font=dict(color="white", size=10), bgcolor="rgba(0,0,0,0.5)"
                    ))

                sampled_positions_plotly_plot = np.array([s['position'] for s in section_data_3d])
                if len(sampled_positions_plotly_plot) > 0:
                    plotly_traces.append(go.Scatter3d(
                        x=sampled_positions_plotly_plot[:, 0], y=sampled_positions_plotly_plot[:, 1], z=sampled_positions_plotly_plot[:, 2],
                        mode='markers', marker=dict(size=5, color='blue', symbol='circle'), name='Sampled Positions'
                    ))

                fig_plotly = go.Figure(data=plotly_traces)
                fig_plotly.update_layout(
                    title=f'Interactive 3D Cross-Sections - {base_name_for_outputs}',
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data', 
                               camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)), annotations=section_annotations_plotly),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                plotly_html_path = os.path.join(output_dir, f"{base_name_for_outputs}_3d_sections_interactive.html")
                fig_plotly.write_html(plotly_html_path)
                print(f"  Created interactive 3D cross-section visualization (HTML): {plotly_html_path}")
            except ImportError: print("  Plotly is not installed. Skipping interactive 3D HTML plot.")
            except Exception as e_plotly: print(f"  Error creating Plotly 3D HTML plot: {e_plotly}")
    
    return {
        'positions': final_normalized_positions,
        'pca_aspect_ratios': final_pca_aspect_ratios,    
        'pca_widths': final_pca_widths,                  
        'ellipse_aspect_ratios': final_ellipse_aspect_ratios, 
        'ellipse_widths': final_ellipse_widths,              
        'ellipse_inlier_ratios': final_ellipse_inlier_ratios, # NEW: Add to return
        'ellipse_relative_orientations_deg': final_ellipse_relative_orientations_deg, # NEW: Add to return
        'section_points': final_section_points_list,         
        'ellipse_points_for_plot': final_ellipse_points_for_plot, 
        'centerline': smoothed_centerline
    }

def create_section_montage(section_points_list, ellipse_points_for_plot_list, 
                           positions, aspect_ratios_ellipse, output_path): # MODIFIED SIGNATURE
    valid_sections_info = []
    for i, (orig_pts, ellipse_pts) in enumerate(zip(section_points_list, ellipse_points_for_plot_list)):
        if orig_pts is not None and aspect_ratios_ellipse[i] is not None and \
           not (hasattr(aspect_ratios_ellipse[i], '__iter__') and len(aspect_ratios_ellipse[i])==0) and \
           np.isfinite(aspect_ratios_ellipse[i]) and \
           abs(aspect_ratios_ellipse[i]) <= 30.0: # Filter based on ellipse AR
            valid_sections_info.append({
                'original_points': orig_pts,
                'ellipse_points': ellipse_pts, # Can be None if fitting failed
                'position': positions[i],
                'aspect_ratio': aspect_ratios_ellipse[i],
                'original_index': i
            })
    
    if not valid_sections_info:
        print("  No valid sections with ellipse data to create montage")
        return
        
    n_sections = len(valid_sections_info)
    cols = min(5, n_sections)
    rows = (n_sections + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5)) # Slightly larger subplots
    fig.suptitle("Cross-Sections (Original and Fitted Ellipse)", fontsize=16) # UPDATED TITLE
    
    if rows == 1 and cols == 1: axes = np.array([axes])
    elif rows == 1 or cols == 1: axes = axes.flatten()
    
    all_orig_points_for_scale = np.vstack([info['original_points'] for info in valid_sections_info if info['original_points'] is not None])
    max_extent = np.max(np.abs(all_orig_points_for_scale)) * 1.1 if len(all_orig_points_for_scale) > 0 else 1.0
    
    for i, section_info in enumerate(valid_sections_info):
        if i >= len(axes.flatten()): break
            
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx, col_idx] if rows > 1 and cols > 1 else axes[i]
        
        orig_pts = section_info['original_points']
        ellipse_pts = section_info['ellipse_points']
        
        if orig_pts is not None and len(orig_pts) >= 3:
            ordered_orig_points = order_points(orig_pts, method="angular")
            ax.plot(np.append(ordered_orig_points[:, 0], ordered_orig_points[0, 0]),
                    np.append(ordered_orig_points[:, 1], ordered_orig_points[0, 1]),
                    'b-', linewidth=1.5, label='Original Section')
            ax.fill(ordered_orig_points[:, 0], ordered_orig_points[:, 1], alpha=0.2, color='blue')

            if ellipse_pts is not None and len(ellipse_pts) > 0:
                ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], 'r--', linewidth=1.2, label='Fitted Ellipse')
            
            ax.set_title(f"Pos: {section_info['position']:.2f}\nAR (Ell): {section_info['aspect_ratio']:.2f}")
        else:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            
        ax.set_aspect('equal')
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize='small', loc='upper right') # Add legend to first plot
        
    for i in range(n_sections, rows*cols):
        row_idx = i // cols
        col_idx = i % cols
        if rows > 1 and cols > 1: axes[row_idx, col_idx].axis('off')
        elif i < len(axes.flatten()): axes.flatten()[i].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    # plt.subplots_adjust(top=0.92) # Already handled by tight_layout rect
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Created section montage with ellipses: {output_path}")

# Example usage
if __name__ == "__main__":
    files_to_process = [
        "Meshes/Onion_OBJ/Ac_DA_1_3.obj", "Meshes/Onion_OBJ/Ac_DA_1_2.obj", "Meshes/Onion_OBJ/Ac_DA_1_5.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_7.obj", "Meshes/Onion_OBJ/Ac_DA_3_6.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_4.obj",  "Meshes/Onion_OBJ/Ac_DA_3_2.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_1.obj", 
        "Meshes/Onion_OBJ/Ac_DA_2_3.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_8_mesh.obj", "Meshes/Onion_OBJ/Ac_DA_1_6.obj"
    ]

    ## Test with a single file
    #files_to_process = ["Meshes/Onion_OBJ/Ac_DA_1_2.obj"]
    main_output_directory = "centerline_results" # Renamed for clarity
    
    # Initialize storage for collected results
    all_results = {}

    for mesh_file_path in files_to_process: # Renamed loop variable for clarity
        print(f"\nProcessing file: {mesh_file_path}")
        base_name = os.path.splitext(os.path.basename(mesh_file_path))[0] # Used for subfolder and dict key
        
        # Define output directory for this specific mesh's detailed plots
        mesh_specific_output_dir = os.path.join(main_output_directory, base_name)
        # The analyze_centerline_sections function will create this if visualize=True

        results = analyze_centerline_sections(
            mesh_file_path,
            num_sections=15,
            visualize=True,
            output_dir=mesh_specific_output_dir # Pass the specific dir for this mesh
        )
        
        if results:
            all_results[base_name] = results # Use base_name as key
            print("\nAnalysis complete. Results summary for this mesh:")
            # This summary block was problematic, removed the first part.
            # Now directly summarizing ellipse aspect ratios as intended.
            valid_ellipse_ars = [ar for ar in results['ellipse_aspect_ratios'] if ar is not None and np.isfinite(ar)]
            print(f"  - Number of valid sections (ellipse fit): {len(valid_ellipse_ars)} / {len(results['positions'])}")
            if valid_ellipse_ars:
                print(f"  - Ellipse Aspect Ratio range: {min(valid_ellipse_ars):.2f} - {max(valid_ellipse_ars):.2f}")
            else:
                print("  - No valid ellipse aspect ratios found for this mesh.")

        if results and 'ellipse_inlier_ratios' in results:
            valid_ellipse_irs = [ir for ir in results['ellipse_inlier_ratios'] if ir is not None and np.isfinite(ir)]
            if valid_ellipse_irs:
                print(f"  - Ellipse Inlier Ratio range: {min(valid_ellipse_irs):.2f} - {max(valid_ellipse_irs):.2f}")
            else:
                print("  - No valid ellipse inlier ratios found for this mesh.")

        if results and 'ellipse_relative_orientations_deg' in results:
            valid_ellipse_orients = [o for o in results['ellipse_relative_orientations_deg'] if o is not None and np.isfinite(o)]
            if valid_ellipse_orients:
                print(f"  - Ellipse Relative Orientation range (deg): {min(valid_ellipse_orients):.1f} - {max(valid_ellipse_orients):.1f}")
            else:
                print("  - No valid ellipse relative orientations found for this mesh.")
    
    # After processing all files, create combined plots
    if all_results:
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
        all_positions_for_avg = [] 
        all_ratios_for_avg = []    
        
        for idx, (mesh_key, result_data) in enumerate(all_results.items()): # Use clearer var names
            current_aspect_ratios = result_data['ellipse_aspect_ratios']
            current_positions = result_data['positions']
            
            valid_indices = [i for i, ar in enumerate(current_aspect_ratios) 
                             if ar is not None and np.isfinite(ar) and ar <= 10.0] 
            valid_positions = [current_positions[i] for i in valid_indices]
            valid_ratios = [current_aspect_ratios[i] for i in valid_indices]
            
            if not valid_positions: continue

            sorted_indices = np.argsort(valid_positions)
            valid_positions_sorted = np.array(valid_positions)[sorted_indices] # New var name
            valid_ratios_sorted = np.array(valid_ratios)[sorted_indices]       # New var name
            
            all_positions_for_avg.append(valid_positions_sorted.tolist())
            all_ratios_for_avg.append(valid_ratios_sorted.tolist())
            
            plt.plot(valid_positions_sorted, valid_ratios_sorted, 'o-', linewidth=1.0, color=colors[idx], 
                     label=mesh_key.replace('_mesh',''), alpha=0.3) 
        
        common_positions = np.linspace(0.0, 1.0, 50)
        interpolated_ratios_all_curves = [] 
        
        for p_list, r_list in zip(all_positions_for_avg, all_ratios_for_avg):
            if len(p_list) < 2: continue 
            try:
                min_pos_data = min(p_list)
                max_pos_data = max(p_list)
                interp_range_common_pos = common_positions[(common_positions >= min_pos_data) & (common_positions <= max_pos_data)]
                if len(interp_range_common_pos) > 0:
                    interp_values = np.interp(interp_range_common_pos, p_list, r_list)
                    interpolated_ratios_all_curves.append(list(zip(interp_range_common_pos, interp_values)))
            except Exception as e_interp:
                print(f"  Warning: Could not interpolate AR curve for averaging: {e_interp}")
        
        if interpolated_ratios_all_curves:
            position_groups = {}
            for curve_data in interpolated_ratios_all_curves:
                for pos, ratio_val in curve_data:
                    pos_rounded = round(pos, 3)
                    position_groups.setdefault(pos_rounded, []).append(ratio_val)
            
            avg_positions = sorted(position_groups.keys())
            avg_ratios = [np.median(position_groups[pos_val]) for pos_val in avg_positions if position_groups[pos_val]]
            
            if avg_positions and avg_ratios:
                plt.plot(avg_positions, avg_ratios, 'o-', linewidth=3.0, color='red', 
                         label='Median (Ellipse AR)', alpha=1.0, markersize=8) 
        
        plt.xlabel('Normalized Position Along Centerline (0=Tip, 1=Midpoint)', fontsize=12)
        plt.ylabel('Aspect Ratio (Ellipse-Based)') 
        plt.title('Aspect Ratio (Ellipse) Along Centerline - All Samples', fontsize=14) 
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.tight_layout()
        
        # Simplified path for combined plot - saves to main_output_directory
        combined_ar_plot_path = os.path.join(main_output_directory, "combined_aspect_ratio_ellipse_curves.png")
        plt.savefig(combined_ar_plot_path, dpi=200); plt.close()
        print(f"\nCreated combined aspect ratio (ellipse) plot: {combined_ar_plot_path}")
        
        # Combined width plot (ellipse-based)
        plt.figure(figsize=(12, 8))
        # all_widths_for_avg = [] # Already defined if you want to add median width
        
        for idx, (mesh_key, result_data) in enumerate(all_results.items()):
            current_widths = result_data['ellipse_widths'] 
            current_positions = result_data['positions']
            valid_indices = [i for i, w in enumerate(current_widths) if w is not None and np.isfinite(w)]
            valid_positions = [current_positions[i] for i in valid_indices]
            valid_widths = [current_widths[i] for i in valid_indices]

            if not valid_positions: continue

            sorted_indices = np.argsort(valid_positions)
            valid_positions_sorted_w = np.array(valid_positions)[sorted_indices] # New var name
            valid_widths_sorted_w = np.array(valid_widths)[sorted_indices]       # New var name
            # all_widths_for_avg.append(valid_widths_sorted_w.tolist()) # Uncomment if adding median width

            plt.plot(valid_positions_sorted_w, valid_widths_sorted_w, 'o-', linewidth=1.5, color=colors[idx], 
                     label=mesh_key.replace('_mesh',''))
        
        plt.xlabel('Normalized Position Along Centerline (0=Tip, 1=Midpoint)', fontsize=12)
        plt.ylabel('Width (Ellipse Semi-Minor Axis)') 
        plt.title('Width (Ellipse) Along Centerline - All Samples', fontsize=14) 
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.tight_layout()

        # Simplified path for combined plot - saves to main_output_directory
        combined_width_plot_path = os.path.join(main_output_directory, "combined_width_ellipse_curves.png")
        plt.savefig(combined_width_plot_path, dpi=200); plt.close()
        print(f"Created combined width (ellipse) plot: {combined_width_plot_path}")

        # NEW: Combined Inlier Ratio plot
        plt.figure(figsize=(12, 8))
        # colors are already defined from previous plots
        all_positions_for_avg_ir = [] 
        all_ratios_for_avg_ir = []    
        
        for idx, (mesh_key, result_data) in enumerate(all_results.items()):
            current_inlier_ratios = result_data.get('ellipse_inlier_ratios', []) # Use .get for safety
            current_positions = result_data['positions']
            
            valid_indices_ir = [i for i, ir_val in enumerate(current_inlier_ratios) 
                                if ir_val is not None and np.isfinite(ir_val)] 
            valid_positions_ir = [current_positions[i] for i in valid_indices_ir]
            valid_ratios_ir = [current_inlier_ratios[i] for i in valid_indices_ir]
            
            if not valid_positions_ir: continue

            sorted_indices_ir = np.argsort(valid_positions_ir)
            valid_positions_sorted_ir = np.array(valid_positions_ir)[sorted_indices_ir]
            valid_ratios_sorted_ir = np.array(valid_ratios_ir)[sorted_indices_ir]      
            
            all_positions_for_avg_ir.append(valid_positions_sorted_ir.tolist())
            all_ratios_for_avg_ir.append(valid_ratios_sorted_ir.tolist())
            
            plt.plot(valid_positions_sorted_ir, valid_ratios_sorted_ir, 'o-', linewidth=1.0, color=colors[idx], 
                     label=mesh_key.replace('_mesh',''), alpha=0.3) 
        
        common_positions_ir = np.linspace(0.0, 1.0, 50) # Can reuse common_positions
        interpolated_ratios_all_curves_ir = [] 
        
        for p_list_ir, r_list_ir in zip(all_positions_for_avg_ir, all_ratios_for_avg_ir):
            if len(p_list_ir) < 2: continue 
            try:
                min_pos_data_ir = min(p_list_ir)
                max_pos_data_ir = max(p_list_ir)
                interp_range_common_pos_ir = common_positions_ir[
                    (common_positions_ir >= min_pos_data_ir) & (common_positions_ir <= max_pos_data_ir)
                ]
                if len(interp_range_common_pos_ir) > 0:
                    interp_values_ir = np.interp(interp_range_common_pos_ir, p_list_ir, r_list_ir)
                    interpolated_ratios_all_curves_ir.append(list(zip(interp_range_common_pos_ir, interp_values_ir)))
            except Exception as e_interp_ir:
                print(f"  Warning: Could not interpolate Inlier Ratio curve for averaging: {e_interp_ir}")
        
        if interpolated_ratios_all_curves_ir:
            position_groups_ir = {}
            for curve_data_ir in interpolated_ratios_all_curves_ir:
                for pos_ir, ratio_val_ir in curve_data_ir:
                    pos_rounded_ir = round(pos_ir, 3)
                    position_groups_ir.setdefault(pos_rounded_ir, []).append(ratio_val_ir)
            
            avg_positions_ir = sorted(position_groups_ir.keys())
            avg_ratios_ir = [np.median(position_groups_ir[pos_val_ir]) for pos_val_ir in avg_positions_ir if position_groups_ir[pos_val_ir]]
            
            if avg_positions_ir and avg_ratios_ir:
                plt.plot(avg_positions_ir, avg_ratios_ir, 'o-', linewidth=3.0, color='purple', # Median line color
                         label='Median (Inlier Ratio)', alpha=1.0, markersize=8) 
        
        plt.xlabel('Normalized Position Along Centerline (0=Tip, 1=Midpoint)', fontsize=12)
        plt.ylabel('Ellipse Inlier Ratio (RANSAC)') 
        plt.title('Ellipse Fit Regularity (Inlier Ratio) - All Samples', fontsize=14) 
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05) # Inlier ratio is 0-1
        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.tight_layout()

        # NEW: Combined Relative Orientation plot
        plt.figure(figsize=(12, 8))
        # colors are already defined
        all_positions_for_avg_orient = [] 
        all_values_for_avg_orient = []    
        
        for idx, (mesh_key, result_data) in enumerate(all_results.items()):
            current_orientations_deg = result_data.get('ellipse_relative_orientations_deg', [])
            current_positions = result_data['positions']
            
            valid_indices_orient = [i for i, orient_val in enumerate(current_orientations_deg) 
                                    if orient_val is not None and np.isfinite(orient_val)] 
            valid_positions_orient = [current_positions[i] for i in valid_indices_orient]
            valid_values_orient = [current_orientations_deg[i] for i in valid_indices_orient]
            
            if not valid_positions_orient: continue

            sorted_indices_orient = np.argsort(valid_positions_orient)
            valid_positions_sorted_orient = np.array(valid_positions_orient)[sorted_indices_orient]
            valid_values_sorted_orient = np.array(valid_values_orient)[sorted_indices_orient]      
            
            all_positions_for_avg_orient.append(valid_positions_sorted_orient.tolist())
            all_values_for_avg_orient.append(valid_values_sorted_orient.tolist())
            
            plt.plot(valid_positions_sorted_orient, valid_values_sorted_orient, 'o-', linewidth=1.0, color=colors[idx], 
                     label=mesh_key.replace('_mesh',''), alpha=0.3) 
        
        common_positions_orient = np.linspace(0.0, 1.0, 50)
        interpolated_values_all_curves_orient = [] 
        
        for p_list_orient, v_list_orient in zip(all_positions_for_avg_orient, all_values_for_avg_orient):
            if len(p_list_orient) < 2: continue 
            try:
                min_pos_data_orient = min(p_list_orient)
                max_pos_data_orient = max(p_list_orient)
                interp_range_common_pos_orient = common_positions_orient[
                    (common_positions_orient >= min_pos_data_orient) & (common_positions_orient <= max_pos_data_orient)
                ]
                if len(interp_range_common_pos_orient) > 0:
                    interp_values_orient = np.interp(interp_range_common_pos_orient, p_list_orient, v_list_orient)
                    interpolated_values_all_curves_orient.append(list(zip(interp_range_common_pos_orient, interp_values_orient)))
            except Exception as e_interp_orient:
                print(f"  Warning: Could not interpolate Relative Orientation curve for averaging: {e_interp_orient}")
        
        if interpolated_values_all_curves_orient:
            position_groups_orient = {}
            for curve_data_orient in interpolated_values_all_curves_orient:
                for pos_orient, val_orient in curve_data_orient:
                    pos_rounded_orient = round(pos_orient, 3)
                    position_groups_orient.setdefault(pos_rounded_orient, []).append(val_orient)
            
            avg_positions_orient = sorted(position_groups_orient.keys())
            avg_values_orient = [np.median(position_groups_orient[pos_val_orient]) for pos_val_orient in avg_positions_orient if position_groups_orient[pos_val_orient]]
            
            if avg_positions_orient and avg_values_orient:
                plt.plot(avg_positions_orient, avg_values_orient, 'o-', linewidth=3.0, color='cyan', # Median line color
                         label='Median (Relative Orientation)', alpha=1.0, markersize=8) 
        
        plt.xlabel('Normalized Position Along Centerline (0=Tip, 1=Midpoint)', fontsize=12)
        plt.ylabel('Relative Orientation of Major Axis (degrees)') 
        plt.title('Ellipse Major Axis Orientation (Relative to Midpoint) - All Samples', fontsize=14) 
        plt.grid(True, alpha=0.3)
        plt.ylim(-95, 95) # Degrees
        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.tight_layout()
        
        combined_orient_plot_path = os.path.join(main_output_directory, "combined_ellipse_relative_orientation_curves.png")
        plt.savefig(combined_orient_plot_path, dpi=200); plt.close()
        print(f"\nCreated combined ellipse relative orientation plot: {combined_orient_plot_path}")
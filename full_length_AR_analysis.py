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
    if reference_long_axis_vector is None:
        print("  Warning: Could not determine reference orientation. Aspect ratios will be max/min (standard PCA).")

    # --- Stage 3: Recalculate Aspect Ratios and Widths with Reference ---
    final_aspect_ratios = []
    final_widths = []
    final_section_points_list = [] 
    section_data_3d = [] # For 3D plots

    print("DEBUG: >>> Entering FINAL ASPECT RATIO CALCULATION LOOP <<<")
    for idx, data_item in enumerate(raw_section_data_list):
        ar, w = None, None
        current_points_2d = data_item['points_2d']
        final_section_points_list.append(current_points_2d) 

        if data_item['valid_geometry'] and current_points_2d is not None and len(current_points_2d) >= 3:
            ar, w = _calculate_pca_metrics(
                current_points_2d, 
                f"section {idx+1}", 
                reference_orientation_vector=reference_long_axis_vector
            )
            print(f"  Final AR for section {idx+1} (pos {data_item['norm_pos']:.2f}): {(f'{ar:.2f}' if ar is not None else 'None')}, Width: {(f'{w:.2f}' if w is not None else 'None')}")
            
            if ar is not None and w is not None and np.isfinite(ar) and np.isfinite(w) and abs(ar) <= 30.0 : 
                section_data_3d.append({
                    'points_2d': current_points_2d,
                    'position': data_item['position_3d'],
                    'tangent': data_item['tangent_3d'],
                    'norm_pos': data_item['norm_pos'],
                    'aspect_ratio': ar, 
                    'width': w,        
                    'transform': data_item['transform'] 
                })
        final_aspect_ratios.append(ar)
        final_widths.append(w)
    
    aspect_ratios = final_aspect_ratios
    widths = final_widths
    section_points_list = final_section_points_list 

    # Step 7: Create visualizations
    if visualize and output_dir:
        # Plot aspect ratio vs position
        fig_ar, ax_ar = plt.subplots(figsize=(10, 6))
        valid_ar_indices = [i for i, ar_val in enumerate(aspect_ratios) if ar_val is not None] # Allow AR < 1
        valid_ar_positions = [final_normalized_positions[i] for i in valid_ar_indices]
        valid_ar_ratios = [aspect_ratios[i] for i in valid_ar_indices]
        
        if valid_ar_positions and valid_ar_ratios: # Check if lists are not empty
            # Sort by position before plotting for a clean line
            sorted_plot_indices = np.argsort(valid_ar_positions)
            plot_positions_ar = np.array(valid_ar_positions)[sorted_plot_indices]
            plot_ratios_ar = np.array(valid_ar_ratios)[sorted_plot_indices]
            ax_ar.plot(plot_positions_ar, plot_ratios_ar, 'o-', linewidth=2)
        
        ax_ar.set_xlabel('Normalized Position Along Centerline')
        ax_ar.set_ylabel('Aspect Ratio (Midpoint Ref)')
        ax_ar.set_title(f'Aspect Ratio Along Centerline\n{base_name_for_outputs}')
        ax_ar.grid(True)
        aspect_plot_path = os.path.join(output_dir, f"{base_name_for_outputs}_aspect_ratio_curve.png")
        plt.savefig(aspect_plot_path, dpi=150); plt.close(fig_ar)
        
        # Plot width vs position
        fig_w, ax_w = plt.subplots(figsize=(10, 6))
        valid_w_indices = [i for i, w_val in enumerate(widths) if w_val is not None]
        valid_w_positions = [final_normalized_positions[i] for i in valid_w_indices]
        valid_w_values = [widths[i] for i in valid_w_indices]

        if valid_w_positions and valid_w_values: # Check if lists are not empty
            sorted_plot_indices_w = np.argsort(valid_w_positions)
            plot_positions_w = np.array(valid_w_positions)[sorted_plot_indices_w]
            plot_values_w = np.array(valid_w_values)[sorted_plot_indices_w]
            ax_w.plot(plot_positions_w, plot_values_w, 'o-', color='green', linewidth=2)

        ax_w.set_xlabel('Normalized Position Along Centerline')
        ax_w.set_ylabel('Width (Midpoint Ref)')
        ax_w.set_title(f'Width Along Centerline\n{base_name_for_outputs}')
        ax_w.grid(True)
        width_plot_path = os.path.join(output_dir, f"{base_name_for_outputs}_width_curve.png")
        plt.savefig(width_plot_path, dpi=150); plt.close(fig_w)
        
        create_section_montage(
            raw_section_data_list,
            final_normalized_positions, # Use the consistent normalized positions
            aspect_ratios, # Use the final aspect ratios
            os.path.join(output_dir, f"{base_name_for_outputs}_section_montage.png"),
            pore_center_for_orientation=pore_center
        )
        
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
        'aspect_ratios': aspect_ratios,
        'widths': widths,
        'section_points': section_points_list, 
        'centerline': smoothed_centerline,
        'raw_section_data_items': raw_section_data_list, # ADD THIS LINE
        'minor_radius': minor_radius # ADD THIS LINE (use the actual variable name for average minor radius)
    }

def create_section_montage(all_section_data_items, positions, aspect_ratios, output_path, pore_center_for_orientation=None):
    # Filter for valid sections that have points_2d, transform, and a valid aspect_ratio
    valid_section_tuples = [] # Will store (original_index, section_data_dict)
    for original_idx, data_item in enumerate(all_section_data_items):
        if (data_item is not None and
            data_item.get('points_2d') is not None and
            len(data_item['points_2d']) >= 3 and # Need at least 3 points for a polygon
            data_item.get('transform') is not None and # Need transform for 3D Z-point
            aspect_ratios[original_idx] is not None and
            abs(aspect_ratios[original_idx]) <= 30.0): # Ensure AR is reasonable
            valid_section_tuples.append((original_idx, data_item))
    
    if not valid_section_tuples:
        print("  No valid sections with necessary data to create montage.")
        return
        
    n_valid_sections = len(valid_section_tuples)
    cols = min(5, n_valid_sections)
    rows = (n_valid_sections + cols - 1) // cols # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2)) # Adjusted height slightly
    fig.suptitle("Cross-Sections Along Centerline (Oriented by Highest Z-Point)", fontsize=16)
    
    if rows == 1 and cols == 1: axes = np.array([axes]) # Ensure axes is always iterable
    elif rows == 1 or cols == 1: axes = axes.flatten() # Flatten if 1D array of axes
    
    all_rotated_points_for_scaling = []

    for original_idx, section_item_data in valid_section_tuples:
        current_points_2d = section_item_data['points_2d']
        transform_2d_to_3d = section_item_data['transform']
        section_pos_3d = section_item_data['position_3d'] # Origin of the section plane
        section_normal_3d = section_item_data['tangent_3d'] # Normal to the section plane
        
        # Convert 2D points to 3D to find the highest Z point
        points_3d_list = []
        for pt_2d in current_points_2d:
            pt_2d_h = np.array([pt_2d[0], pt_2d[1], 0.0, 1.0]) 
            pt_3d_h = transform_2d_to_3d.dot(pt_2d_h)
            points_3d_list.append(pt_3d_h[:3]) 
        section_points_3d = np.array(points_3d_list)

        if not section_points_3d.size:
            all_rotated_points_for_scaling.append(None)
            print(f"  Warning: Section {original_idx} resulted in no 3D points for montage.")
            continue
            
        # Find the 2D point corresponding to the highest Z in 3D
        highest_z_idx_in_section = np.argmax(section_points_3d[:, 2])
        landmark_2d_for_rotation = current_points_2d[highest_z_idx_in_section] 
        
        mean_2d = np.mean(current_points_2d, axis=0)
        centered_section_points_2d = current_points_2d - mean_2d
        centered_landmark_2d = landmark_2d_for_rotation - mean_2d
        
        angle_of_highest_pt_vec = np.arctan2(centered_landmark_2d[1], centered_landmark_2d[0])
        target_angle_up = np.pi / 2  
        rotation_rad_primary = target_angle_up - angle_of_highest_pt_vec
        
        cos_r_p, sin_r_p = np.cos(rotation_rad_primary), np.sin(rotation_rad_primary)
        primary_rotation_matrix = np.array([[cos_r_p, -sin_r_p], [sin_r_p, cos_r_p]])
        rotated_centered_pts_2d = (primary_rotation_matrix @ centered_section_points_2d.T).T
        
        # --- Secondary Orientation (Flipping based on pore_center) ---
        if pore_center_for_orientation is not None:
            # 1. Calculate 3D radial vector from pore to section origin, project to plane
            radial_vector_3d = section_pos_3d - pore_center_for_orientation
            radial_vector_on_plane_3d = radial_vector_3d - np.dot(radial_vector_3d, section_normal_3d) * section_normal_3d
            norm_rvop = np.linalg.norm(radial_vector_on_plane_3d)

            if norm_rvop > 1e-6:
                radial_vector_on_plane_3d /= norm_rvop

                # 2. Transform this 3D plane vector to original 2D coords of the section
                x_axis_2d_in_3d = transform_2d_to_3d[:3, 0]
                y_axis_2d_in_3d = transform_2d_to_3d[:3, 1]
                
                comp_x_orig_2d = np.dot(radial_vector_on_plane_3d, x_axis_2d_in_3d)
                comp_y_orig_2d = np.dot(radial_vector_on_plane_3d, y_axis_2d_in_3d)
                ref_vec_orig_2d = np.array([comp_x_orig_2d, comp_y_orig_2d])
                
                norm_ref_vec_orig_2d = np.linalg.norm(ref_vec_orig_2d)
                if norm_ref_vec_orig_2d > 1e-6:
                    ref_vec_orig_2d /= norm_ref_vec_orig_2d

                    # 3. Apply the primary rotation to this 2D reference vector
                    ref_vec_after_primary_rotation = (primary_rotation_matrix @ ref_vec_orig_2d.T).T

                    # 4. Check if the X component is negative, if so, flip horizontally
                    # We want the radial vector (pointing "out" from pore) to generally point to +X (right)
                    if ref_vec_after_primary_rotation[0] < -1e-5: # Small tolerance for zero
                        rotated_centered_pts_2d[:, 0] *= -1
                        # print(f"    Section {original_idx}: Flipped horizontally for consistent side orientation.")
            # else:
                # print(f"    Section {original_idx}: Radial vector for side orientation is zero or too small. Skipping flip.")
        # else:
            # print(f"    Pore center not provided. Skipping secondary (side) orientation for section {original_idx}.")

        all_rotated_points_for_scaling.append(rotated_centered_pts_2d)

    # Calculate the maximum extent across all valid rotated sections for consistent scaling
    valid_rotated_points_for_extent = [pts for pts in all_rotated_points_for_scaling if pts is not None and len(pts) > 0]
    if not valid_rotated_points_for_extent:
        print("  No valid rotated points to determine montage scaling.")
        if 'fig' in locals() and fig: plt.close(fig) # Clean up figure if created
        return
        
    all_points_for_extent_calc = np.vstack(valid_rotated_points_for_extent)
    max_extent = np.max(np.abs(all_points_for_extent_calc)) * 1.1 if len(all_points_for_extent_calc) > 0 else 1.0
    
    # Plotting loop
    for i, (original_idx, _) in enumerate(valid_section_tuples): # i is index in valid_section_tuples
        plot_points = all_rotated_points_for_scaling[i] # These are the rotated and centered points

        if i >= len(axes.flatten()): break # Should not happen if rows/cols calculated correctly
            
        current_ax = axes[i // cols, i % cols] if rows > 1 and cols > 1 else axes[i]
        
        if plot_points is not None and len(plot_points) >= 3:
            # Order points for a clean polygon (already centered and rotated)
            ordered_plot_points = order_points(plot_points, method="angular") # from helper_functions
            
            current_ax.plot(
                np.append(ordered_plot_points[:, 0], ordered_plot_points[0, 0]),
                np.append(ordered_plot_points[:, 1], ordered_plot_points[0, 1]),
                'b-', linewidth=1.5
            )
            current_ax.fill(ordered_plot_points[:, 0], ordered_plot_points[:, 1], alpha=0.2, color='blue')
            # Use original_idx to access the correct positions and aspect_ratios
            current_ax.set_title(f"Pos: {positions[original_idx]:.2f}\nAR: {aspect_ratios[original_idx]:.2f}", fontsize=9)
        else:
            current_ax.text(0.5, 0.5, "No valid section", ha='center', va='center', transform=current_ax.transAxes, fontsize=9)
            
        current_ax.set_aspect('equal')
        current_ax.set_xlim(-max_extent, max_extent)
        current_ax.set_ylim(-max_extent, max_extent)
        current_ax.grid(True, alpha=0.3)
        current_ax.tick_params(axis='both', which='major', labelsize=8)
        
    # Hide empty subplots
    for i in range(n_valid_sections, rows * cols):
        ax_to_hide = axes[i // cols, i % cols] if rows > 1 and cols > 1 else axes[i]
        ax_to_hide.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Created section montage (oriented by highest Z): {output_path}")

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

    files_to_process = [
        "Meshes/Onion_OBJ/Ac_DA_1_3.obj", "Meshes/Onion_OBJ/Ac_DA_1_2.obj", "Meshes/Onion_OBJ/Ac_DA_1_5.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_7.obj", "Meshes/Onion_OBJ/Ac_DA_3_6.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_4.obj",  "Meshes/Onion_OBJ/Ac_DA_3_2.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_1.obj", 
        "Meshes/Onion_OBJ/Ac_DA_2_3.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_8_mesh.obj", "Meshes/Onion_OBJ/Ac_DA_1_6.obj"
    ]

    ## Test with a single file
    files_to_process = ["results/full_stomata_Ac_DA_1_3_bulge_not_preserved.ply"]
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
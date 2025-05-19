import numpy as np
import matplotlib.pyplot as plt
import os
import trimesh
from sklearn.decomposition import PCA

# Import necessary functions from existing files
from test_functions import get_radial_dimensions, filter_section_points
from helper_functions import (_smooth_centerline_savgol, 
                             _project_plane_origin_to_2d, 
                             _calculate_pca_metrics,
                             order_points)
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
    ray_origin_for_radial_cast = np.array([0.0, 0.0, 0.0])  # Default
    
    if pore_center is not None:
        ray_origin_for_radial_cast = pore_center
        print(f"  Using pore center from ED as ray origin: {ray_origin_for_radial_cast.round(3)}")
    else:
        print("  Warning: Pore center not available. Defaulting to [0,0,0].")
    
    # Perform radial ray casting
    print(f"  Performing radial ray casting from origin: {ray_origin_for_radial_cast.round(3)}")
    ray_count = 45  # Number of rays for radial casting
    inner_points, outer_points, raw_centerline_points_from_radial, minor_radius = get_radial_dimensions(
        mesh, center=ray_origin_for_radial_cast, ray_count=ray_count
    )
    
    if (inner_points is None or outer_points is None or 
        raw_centerline_points_from_radial is None or minor_radius is None):
        print("  Error: Could not determine dimensions via radial ray casting.")
        return None
    
    print(f"  Estimated Minor Radius: {minor_radius:.3f}")
    
    # Smooth the centerline from radial ray casting
    smoothed_centerline = _smooth_centerline_savgol(raw_centerline_points_from_radial)
    if smoothed_centerline is None:
        smoothed_centerline = raw_centerline_points_from_radial
    
    # Step 2.5: Find tip and midpoint positions along centerline
    print("  Finding tip and midpoint positions...")
    
    # Find the tip (minimum Y-coordinate)
    y_coordinates = smoothed_centerline[:, 1]
    tip_idx = np.argmin(y_coordinates)
    tip_position = smoothed_centerline[tip_idx]
    print(f"  Initial tip position (min Y): {tip_position.round(3)}")
    
    # Find the midpoint (closest to pore center's Y-coordinate)
    midpoint_idx = None
    if pore_center is not None:
        target_y_for_midpoint = pore_center[1]
        midpoint_idx = np.argmin(np.abs(y_coordinates - target_y_for_midpoint))
        midpoint_position = smoothed_centerline[midpoint_idx]
        print(f"  Midpoint position (at pore Y): {midpoint_position.round(3)}")
    else:
        # If no pore center, use maximum Y as an approximation
        midpoint_idx = np.argmax(y_coordinates)
        midpoint_position = smoothed_centerline[midpoint_idx]
        print(f"  Midpoint position (max Y): {midpoint_position.round(3)}")
    
    # MODIFY: Calculate initial tangent at tip for shift direction
    if len(smoothed_centerline) > 1:
        if 0 < tip_idx < len(smoothed_centerline) - 1:
            # Use central difference for interior point
            tangent_vec = smoothed_centerline[tip_idx + 1] - smoothed_centerline[tip_idx - 1]
        elif tip_idx == 0:
            # Forward difference for first point
            tangent_vec = smoothed_centerline[1] - smoothed_centerline[0]
        else:
            # Backward difference for last point
            tangent_vec = smoothed_centerline[-1] - smoothed_centerline[-2]
            
        tangent_norm = np.linalg.norm(tangent_vec)
        if tangent_norm > 1e-9:
            tangent = tangent_vec / tangent_norm
        else:
            tangent = np.array([0.0, 1.0, 0.0])  # Default fallback
    else:
        tangent = np.array([0.0, 1.0, 0.0])  # Default for single point
    
    # MODIFY: Move a minimum distance from tip along tangent
    min_tip_distance = 0.05  # Adjust this value as needed
    if pore_center is not None:
        # Project pore_center onto the line defined by tip and tangent
        v = pore_center - tip_position
        along = float(np.dot(v, tangent))
        
        # Determine maximum slide based on minor radius
        max_slide = minor_radius if minor_radius and minor_radius > 0 else 0.5
        
        # Ensure we move at least min_tip_distance
        min_slide = min_tip_distance
        if max_slide < min_slide:
            max_slide = min_slide
            
        # Clamp the shift amount
        along_clamped = np.clip(along, min_slide, max_slide)
        
        # Calculate adjusted tip position
        adjusted_tip_position = tip_position + along_clamped * tangent
        
        # Find centerline point closest to this adjusted position
        distances = np.linalg.norm(smoothed_centerline - adjusted_tip_position, axis=1)
        adjusted_tip_idx = np.argmin(distances)
        print(f"  Adjusted tip position (offset from min Y): {smoothed_centerline[adjusted_tip_idx].round(3)}")
        
        # Use adjusted tip as new starting point
        tip_idx = adjusted_tip_idx
        tip_position = smoothed_centerline[tip_idx]
    else:
        print("  No pore center available for tip adjustment")
    
    # Ensure midpoint_idx > tip_idx for proper path extraction
    if midpoint_idx < tip_idx:
        # We need to adjust the indices to get the right path
        # This handles the case where the tip is later in the array than the midpoint
        midpoint_idx += len(smoothed_centerline)
    
    # Extract the path from tip to midpoint
    path_indices = []
    current_idx = tip_idx
    
    # Add indices wrapping around the end of the array if needed
    while current_idx % len(smoothed_centerline) != midpoint_idx % len(smoothed_centerline):
        path_indices.append(current_idx % len(smoothed_centerline))
        current_idx += 1
    
    path_indices.append(midpoint_idx % len(smoothed_centerline))  # Add midpoint
    
    # Extract the centerline segment
    centerline_segment = smoothed_centerline[path_indices]
    print(f"  Extracted path from tip to midpoint with {len(centerline_segment)} points")
    
    # Replace the full centerline with just this segment
    smoothed_centerline = centerline_segment
    
    # Step 4: Get total centerline segment length for normalization
    centerline_segments = np.diff(smoothed_centerline, axis=0)
    segment_lengths = np.linalg.norm(centerline_segments, axis=1)
    total_length = np.sum(segment_lengths)
    
    # Step 5: Sample positions along the centerline segment
    sampled_positions = []
    sampled_tangents = []
    normalized_positions = []
    
    # Better sampling approach: use equal arc-length sampling along the tip-to-midpoint segment
    total_points = len(smoothed_centerline)
    indices = np.linspace(0, total_points-1, num_sections).astype(int)
    
    # Precompute all normalized positions (tip=0.0, midpoint=1.0)
    cumulative_distances = np.zeros(total_points)
    for i in range(1, total_points):
        segment = smoothed_centerline[i] - smoothed_centerline[i-1]
        cumulative_distances[i] = cumulative_distances[i-1] + np.linalg.norm(segment)
    
    normalized_distances = cumulative_distances / cumulative_distances[-1]
    
    for i in indices:
        # Get position along centerline
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
        mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.15, edgecolor='gray', 
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
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
        
        # Adjust the view angle for better visualization
        ax.view_init(elev=25, azim=40)
        
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
    
    for idx, (position, tangent, norm_pos) in enumerate(zip(sampled_positions, sampled_tangents, normalized_positions)):
        print(f"  Analyzing section {idx+1}/{len(sampled_positions)} at position {norm_pos:.2f}")
        
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
            
        # Get plane origin in 2D for filtering
        plane_origin_2d, _ = _project_plane_origin_to_2d(position, transform_2d_to_3d, points_2D)
        
        # Get inner/outer points for this section (needed for minor_radius)
        _, _, _, minor_radius = get_radial_dimensions(mesh, center=position)
        
        if minor_radius is None:
            print(f"  Could not determine radius for section {idx+1}")
            aspect_ratios.append(None)
            widths.append(None)
            section_points_list.append(None)
            continue
            
        # Filter the section points to get clean outline
        filtered_points_2D, _ = filter_section_points(
            points_2D, 
            minor_radius,
            plane_origin_2d,
            eps_factor=0.15,
            min_samples=3
        )
        
        if len(filtered_points_2D) < 3:
            print(f"  Section {idx+1} has too few points after filtering")
            aspect_ratios.append(None)
            widths.append(None)
            section_points_list.append(None)
            continue
            
        # Calculate aspect ratio and width 
        aspect_ratio, pca_width = _calculate_pca_metrics(filtered_points_2D, "section")

        # Store the values
        aspect_ratios.append(aspect_ratio)
        widths.append(pca_width)
        section_points_list.append(filtered_points_2D)
        
        # After calculating aspect_ratio and width, add:
        if aspect_ratio <= 1.6:
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
                'points_3d': np.array(section_points_3d),
                'position': position,
                'tangent': tangent,
                'norm_pos': norm_pos,
                'aspect_ratio': aspect_ratio
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
        
        # NEW: Create 3D visualization with actual cross-sections
        if section_data_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh at low opacity
            vertices = mesh.vertices
            faces = mesh.faces
            
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            mesh_triangles = vertices[faces]
            mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.1, edgecolor='gray', 
                                              linewidth=0.05, facecolor='lightgray')
            ax.add_collection3d(mesh_collection)
            
            # Plot centerline
            ax.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], 
                    'k-', linewidth=2, label='Centerline')
            
            # Plot cross-sections
            cmap = plt.cm.plasma
            for idx, section in enumerate(section_data_3d):
                points_3d = section['points_3d']
                norm_pos = section['norm_pos']
                
                # Create color based on position
                color = cmap(norm_pos)
                
                # Order points for clean visualization
                # For 3D we'll just close the loop of the existing points
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
            
            ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
            ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
            ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
            
            # Set good viewing angle
            ax.view_init(elev=30, azim=45)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Cross-Sections Along Centerline\n{os.path.basename(mesh_file)}')
            
            # Save 3D cross-section visualization
            in_situ_path = os.path.join(output_dir, f"{base_name}_3d_sections.png")
            plt.savefig(in_situ_path, dpi=200)
            plt.close(fig)
            print(f"  Created 3D cross-section visualization: {in_situ_path}")
    
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
                     and aspect_ratios[i] <= 1.6]
    
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
            print("\nAnalysis complete. Results summary:")
            valid_ars = [ar for ar in results['aspect_ratios'] if ar is not None]
            print(f"  - Number of valid sections: {len(valid_ars)} / {len(results['positions'])}")
            print(f"  - Aspect ratio range: {min(valid_ars):.2f} - {max(valid_ars):.2f}")
            
            # Store results for combined plot
            all_results[mesh_name] = results
    
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
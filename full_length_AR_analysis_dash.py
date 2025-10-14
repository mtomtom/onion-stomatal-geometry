import numpy as np
import os
import trimesh
from sklearn.decomposition import PCA
from test_functions import get_radial_dimensions, filter_section_points
from helper_functions import (_smooth_centerline_savgol, _project_plane_origin_to_2d,
                             _calculate_pca_metrics, fit_ellipse_robust, generate_ellipse_points)
from plotting_helpers import (plot_aspect_ratio_curve, plot_width_curve, plot_inlier_ratio_curve,
                            plot_orientation_curve, create_section_montage,
                            plot_sections_3d_matplotlib, plot_sections_3d_plotly)
import edge_detection as ed

def _find_optimal_section(mesh, center_point, initial_normal, pivot_range_deg=20, num_pivots=11):
    """
    Pivots the section plane around the center_point to find the orientation 
    with the minimum cross-sectional area. This performs a 2D search.

    Args:
        mesh (trimesh.Trimesh): The mesh to section.
        center_point (ndarray): The point on the centerline to pivot around.
        initial_normal (ndarray): The initial plane normal (centerline tangent).
        pivot_range_deg (float): The range of angles (+/-) to search in degrees.
        num_pivots (int): The number of steps for the angle search.

    Returns:
        tuple: (best_section, best_normal) where best_section is a trimesh.Path3D object.
    """
    if num_pivots < 2:
        section = mesh.section(plane_origin=center_point, plane_normal=initial_normal)
        return section, initial_normal

    # Create a basis for pivoting (two orthogonal axes in the plane)
    u = np.array([1., 0., 0.])
    if np.abs(np.dot(u, initial_normal)) > 0.99:
        u = np.array([0., 1., 0.])
    
    pivot_axis1 = np.cross(initial_normal, u)
    pivot_axis1 /= np.linalg.norm(pivot_axis1)
    pivot_axis2 = np.cross(initial_normal, pivot_axis1)
    pivot_axis2 /= np.linalg.norm(pivot_axis2)

    best_section = None
    min_area = float('inf')
    best_normal = initial_normal
    angles = np.linspace(-np.deg2rad(pivot_range_deg), np.deg2rad(pivot_range_deg), num_pivots)

    # Perform a 2D search by iterating through combinations of pivots
    for angle1 in angles:
        for angle2 in angles:
            # Rotate around axis 1
            r1_normal = (initial_normal * np.cos(angle1) + 
                         np.cross(pivot_axis1, initial_normal) * np.sin(angle1))
            # Rotate the result around axis 2
            rotated_normal = (r1_normal * np.cos(angle2) + 
                              np.cross(pivot_axis2, r1_normal) * np.sin(angle2))
            
            rotated_normal /= np.linalg.norm(rotated_normal)
            
            try:
                section = mesh.section(plane_origin=center_point, plane_normal=rotated_normal)
                if section is not None and section.area > 1e-9:
                    if section.area < min_area:
                        min_area = section.area
                        best_section = section
                        best_normal = rotated_normal
            except Exception:
                continue # Ignore failures for any single pivot

    if best_section is None:
        print("  Warning: Optimal section search failed. Falling back to initial tangent.")
        best_section = mesh.section(plane_origin=center_point, plane_normal=initial_normal)
        best_normal = initial_normal

    return best_section, best_normal

def _find_seam_for_closed_stomata(mesh, output_dir=None, base_name=None):
    """
    Finds the internal seam wall by casting rays radially and selecting only the rays
    that have exactly 3 intersections, as per the geometric definition of the tip seam.
    """
    print("  Attempting to find internal seam by selecting rays with exactly 3 intersections...")
    try:
        # Step 1: Cast rays radially from the centroid.
        centroid = mesh.centroid
        ray_count = 360  # Use a high number of rays for good coverage
        angles = np.linspace(0, 2 * np.pi, ray_count, endpoint=False)
        ray_directions = np.zeros((ray_count, 3))
        ray_directions[:, 0] = np.cos(angles)
        ray_directions[:, 1] = np.sin(angles)
        ray_origins = np.tile(centroid, (ray_count, 1))
        
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        if len(locations) == 0:
            print("  Radial ray-casting found no intersections.")
            return None

        # Step 2: Collect points ONLY from rays that have exactly 3 hits.
        seam_points_collected = []
        for i in range(ray_count):
            hits_for_this_ray = locations[index_ray == i]
            
            # This is the key logic based on your insight:
            if len(hits_for_this_ray) == 3:
                # For a 3-hit ray, the middle point is the seam.
                distances = np.linalg.norm(hits_for_this_ray - centroid, axis=1)
                sorted_hits = hits_for_this_ray[np.argsort(distances)]
                seam_points_collected.append(sorted_hits[1]) # Add the middle point

        if not seam_points_collected:
            print("  Warning: No rays with exactly 3 intersections were found. Cannot identify tip seam.")
            return None
        
        junction_points_3d = np.array(seam_points_collected)
        print(f"  Found {len(junction_points_3d)} points on the tip seam using 3-intersection logic.")

        

        return junction_points_3d

    except Exception as e:
        print(f"  Error during internal seam detection: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def _generate_centerline_for_closed_stomata(mesh, seam_points = None, ray_count=90, output_dir=None, base_name=None):
    """
    Generates a centerline segment from tip to midpoint for a closed stoma.
    This version uses the mesh centroid for a stable centerline loop and the seam's
    principal axis to robustly define the tip and midpoint.
    """
    print("  Running closed stoma workflow: finding seam to define centerline segment.")

    # --- FIX: Use provided seam_points if available, otherwise find them ---
    if seam_points is None:
        print("  No manual seam provided, detecting automatically.")
        seam_points = _find_seam_for_closed_stomata(mesh)

    if seam_points is None or len(seam_points) < 3:
        print("  [FAIL] Could not define seam for centerline generation.")
        return None, None, None, None
    
    # Step 2: Generate the centerline loop from the true mesh centroid for stability.
    # The center of ray-casting is NOT adjusted to the seam, as this distorts the loop.
    estimated_center = mesh.centroid
    print(f"  Generating centerline loop via ray-casting from mesh centroid {estimated_center.round(3)}.")
    
    raw_centerline_points = []
    outer_wall_points = []
    for i in range(ray_count):
        angle = 2 * np.pi * i / ray_count
        ray_direction = np.array([np.cos(angle), np.sin(angle), 0])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[estimated_center], ray_directions=[ray_direction]
        )
        if len(locations) > 0:
            outer_wall_point = locations[np.argmax(np.linalg.norm(locations - estimated_center, axis=1))]
            outer_wall_points.append(outer_wall_point)
            midpoint = estimated_center + (outer_wall_point - estimated_center) / 2.0
            raw_centerline_points.append(midpoint)

    if len(raw_centerline_points) < 20:
        print("  Error: Not enough points generated from ray-casting to form a reliable loop.")
        return None, estimated_center, None, seam_points
    
    closed_loop_centerline = np.array(raw_centerline_points)

    # Step 3: Smooth the full loop.
    pad_len = len(closed_loop_centerline) // 2
    padded_loop = np.vstack([closed_loop_centerline[-pad_len:], closed_loop_centerline, closed_loop_centerline[:pad_len]])
    smoothed_padded_loop = _smooth_centerline_savgol(padded_loop)
    if smoothed_padded_loop is None:
        print("  Error: Smoothing the padded loop failed.")
        return None, estimated_center, None, seam_points
    smoothed_loop = smoothed_padded_loop[pad_len:-pad_len]
    print(f"  Successfully generated and smoothed a closed centerline loop with {len(smoothed_loop)} points.")

    # Step 4: Identify tip and midpoint using the seam's principal axis.
    N_cl = len(smoothed_loop)
    if seam_points is None or len(seam_points) < 3:
        print("  Warning: Could not find seam or too few seam points. Falling back to geometric tip/midpoint definition.")
        # Fallback: tip is point on loop furthest from center, midpoint is 90 degrees away.
        distances_from_center = np.linalg.norm(smoothed_loop - estimated_center, axis=1)
        tip_idx = np.argmax(distances_from_center)
        midpoint_idx = (tip_idx + N_cl // 4) % N_cl
    else:
        # Use PCA to find the orientation of the seam points.
        pca = PCA(n_components=2).fit(seam_points)
        seam_axis = pca.components_[0] # The primary direction of the seam
        print(f"  Determined seam orientation axis via PCA: {seam_axis.round(3)}")

        # Project the centerline loop points onto the seam axis to find the tip.
        # The tip is the point on the loop that extends furthest along the seam's direction.
        projections = (smoothed_loop - pca.mean_) @ seam_axis
        tip_idx = np.argmax(projections)
        
        # The midpoint is 90 degrees (a quarter of the loop) away from the tip.
        midpoint_idx = (tip_idx + N_cl // 4) % N_cl

    print(f"  Identified Tip (idx {tip_idx}) and Midpoint (idx {midpoint_idx}) on the loop based on seam orientation.")

    # Step 5: Extract the path from the new tip to the new midpoint.
    path_indices = []
    curr = tip_idx
    for _ in range(N_cl): # Safety break
        path_indices.append(curr)
        if curr == midpoint_idx: break
        curr = (curr + 1) % N_cl
    
    centerline_segment = smoothed_loop[path_indices]
    print(f"  Extracted centerline segment from tip to midpoint with {len(centerline_segment)} points.")

    # Step 6: Estimate minor radius for filtering.
    if len(outer_wall_points) > 0:
        # Use a more stable radius estimation based on the full loop
        radii = np.linalg.norm(np.array(outer_wall_points) - closed_loop_centerline, axis=1)
        estimated_minor_radius = np.mean(radii)
    else:
        estimated_minor_radius = np.min(mesh.bounding_box.extents) / 4.0
    print(f"  Estimated minor radius for filtering: {estimated_minor_radius:.3f}")

    return centerline_segment, estimated_center, estimated_minor_radius, seam_points

def calculate_centerline(mesh, is_closed, seam_points_manual=None):
    """
    Calculates the primary centerline for a mesh.
    Can use manually provided seam points for closed stomata.
    """
    if is_closed:
        # If manual seam points are provided, use them. Otherwise, detect automatically.
        seam_points_for_analysis = seam_points_manual if seam_points_manual is not None and len(seam_points_manual) > 2 else _find_seam_for_closed_stomata(mesh)
        
        centerline, pore_center, minor_radius, seam_points_for_plot = _generate_centerline_for_closed_stomata(mesh, seam_points=seam_points_for_analysis)
        if centerline is None: return None
        smoothed_centerline = _smooth_centerline_savgol(centerline)
        if smoothed_centerline is None: smoothed_centerline = centerline
        return {
            "centerline": smoothed_centerline,
            "pore_center": pore_center,
            "minor_radius": minor_radius,
            "seam_points_for_plot": seam_points_for_plot
        }
    else:
        # ... (The existing logic for open stomata centerline calculation would go here) ...
        # This part is complex and can be integrated later if needed.
        # For now, we focus on the closed case which is more defined.
        print("Open stomata centerline logic not yet integrated into this refactored function.")
        return None


def analyze_sections(mesh, centerline_data, section_positions_norm):
    """
    Analyzes cross-sections and returns all data needed for plotting and montages.
    This version does NOT perform ellipse fitting.
    """
    # --- SETUP and RESAMPLE CENTERLINE (No changes here) ---
    smoothed_centerline = np.array(centerline_data['centerline'])
    pore_center = centerline_data.get('pore_center')
    minor_radius = centerline_data.get('minor_radius')
    seam_points_for_plot = centerline_data.get('seam_points_for_plot')
    if pore_center is None or minor_radius is None: return None
    centerline_segments_diff = np.diff(smoothed_centerline, axis=0)
    segment_lengths = np.linalg.norm(centerline_segments_diff, axis=1)
    cumulative_distances = np.zeros(len(smoothed_centerline))
    cumulative_distances[1:] = np.cumsum(segment_lengths)
    total_length = cumulative_distances[-1]
    normalized_distances = cumulative_distances / total_length if total_length > 0 else np.zeros(len(smoothed_centerline))
    sampled_indices = [np.argmin(np.abs(normalized_distances - target)) for target in section_positions_norm]
    sampled_positions = [smoothed_centerline[i] for i in sampled_indices]
    sampled_tangents = []
    for i in sampled_indices:
        if len(smoothed_centerline) < 2: tangent_vec = np.array([0., 1., 0.])
        elif i == 0: tangent_vec = smoothed_centerline[1] - smoothed_centerline[0]
        elif i == len(smoothed_centerline) - 1: tangent_vec = smoothed_centerline[-1] - smoothed_centerline[-2]
        else: tangent_vec = smoothed_centerline[i + 1] - smoothed_centerline[i - 1]
        tangent_norm_val = np.linalg.norm(tangent_vec)
        tangent = tangent_vec / tangent_norm_val if tangent_norm_val > 1e-6 else np.array([0., 1., 0.])
        sampled_tangents.append(tangent)
    
    # --- FIT 3D SEAM PLANE (No changes here) ---
    seam_plane_origin, seam_plane_normal = None, None
    if seam_points_for_plot is not None and len(seam_points_for_plot) >= 3:
        from sklearn.decomposition import PCA
        seam_points_np = np.array(seam_points_for_plot)
        pca = PCA(n_components=3).fit(seam_points_np)
        seam_plane_origin = pca.mean_
        seam_plane_normal = pca.components_[2]

    # --- START OF THE SECTIONING LOGIC ---
    analysis_results = {'section_data_3d': [], 'section_points_list': [], 'positions': []}
    for idx, (s_pos, s_tan, s_norm_pos) in enumerate(zip(sampled_positions, sampled_tangents, section_positions_norm)):
        # --- Get Section and Split Cell (No changes here) ---
        section, optimal_normal = _find_optimal_section(mesh, center_point=s_pos, initial_normal=s_tan)
        if section is None or not hasattr(section, 'entities') or len(section.entities) == 0: continue
        try:
            path_2D, transform_2d_to_3d = section.to_2D()
            points_2D_geom = path_2D.vertices
        except Exception: continue
        plane_origin_2d, _ = _project_plane_origin_to_2d(s_pos, transform_2d_to_3d, points_2D_geom)
        filtered_points_2D, _ = filter_section_points(points_2D_geom, minor_radius, plane_origin_2d)
        if filtered_points_2D is None or len(filtered_points_2D) < 5: continue
        single_guard_cell_points_2d = filtered_points_2D
        if seam_plane_origin is not None and seam_plane_normal is not None:
            points_3d_full_section = trimesh.transform_points(np.hstack((filtered_points_2D, np.zeros((len(filtered_points_2D), 1)))), transform_2d_to_3d)
            vectors_from_plane_origin = points_3d_full_section - seam_plane_origin
            distances = np.dot(vectors_from_plane_origin, seam_plane_normal)
            centerline_dist = np.dot(s_pos - seam_plane_origin, seam_plane_normal)
            correct_mask = distances >= 0 if centerline_dist >= 0 else distances < 0
            if np.any(correct_mask): single_guard_cell_points_2d = filtered_points_2D[correct_mask]
        
        # --- Store results for plotting ---
        points_3d = trimesh.transform_points(np.hstack((single_guard_cell_points_2d, np.zeros((len(single_guard_cell_points_2d), 1)))), transform_2d_to_3d)
        
        analysis_results['section_data_3d'].append({'norm_pos': s_norm_pos, 'points_3d': points_3d, 'points_2d': single_guard_cell_points_2d})
        analysis_results['section_points_list'].append(single_guard_cell_points_2d)
        analysis_results['positions'].append(s_norm_pos)

    return analysis_results

# The old monolithic function is now deprecated in favor of the two new functions.
# You can keep it for command-line use or remove it.
def analyze_centerline_sections(*args, **kwargs):
    print("This function is deprecated for app use. Please use calculate_centerline and analyze_sections.")
    return None
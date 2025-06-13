from scipy.signal import savgol_filter
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.path import Path

def order_points(points, method="nearest", center=None):
    """
    Order points using specified method.
    
    Args:
        points: numpy array of 2D points
        method: "nearest" for nearest neighbor, "angular" for sorted by angle
        center: optional center point for angular method
        
    Returns:
        ordered points array
    """
    if len(points) <= 1:
        return points
        
    if method == "angular":
        # Center points if using angular method
        if center is None:
            center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]
    
    else:  # Default to nearest-neighbor method
        ordered_indices = []
        remaining = set(range(len(points)))
        
        # Start with leftmost point
        current = np.argmin(points[:, 0])
        ordered_indices.append(current)
        remaining.remove(current)
        
        # Greedily select nearest neighbors
        while remaining:
            current_point = points[current]
            distances = [np.linalg.norm(current_point - points[i]) for i in remaining]
            next_idx = list(remaining)[np.argmin(distances)]
            ordered_indices.append(next_idx)
            current = next_idx
            remaining.remove(current)
            
        return points[ordered_indices]

def _smooth_centerline_savgol(raw_centerline_points, min_points_for_smoothing=5, default_window_len=21, min_window_len=5, poly_order_val=3):
    """Smooths a 3D centerline using Savitzky-Golay filter."""
    if raw_centerline_points is None:
        return None
    
    num_cl_points = len(raw_centerline_points)
    
    if num_cl_points < min_points_for_smoothing:
        if num_cl_points > 0:
            print(f"  Skipping centerline smoothing: only {num_cl_points} points (requires >={min_points_for_smoothing}). Using raw.")
        return raw_centerline_points

    try:
        # Ensure window_length is odd and less than num_cl_points
        window_length = num_cl_points - (1 if num_cl_points % 2 == 0 else 0)
        window_length = min(default_window_len, window_length) 
        if window_length < min_window_len: window_length = min_window_len

        poly_order = min(poly_order_val, window_length - 1)
        
        if num_cl_points >= window_length:
            smoothed_x = savgol_filter(raw_centerline_points[:, 0], window_length, poly_order)
            smoothed_y = savgol_filter(raw_centerline_points[:, 1], window_length, poly_order)
            smoothed_z = savgol_filter(raw_centerline_points[:, 2], window_length, poly_order)
            smoothed_centerline = np.vstack((smoothed_x, smoothed_y, smoothed_z)).T
            print(f"  Centerline smoothed (window={window_length}, order={poly_order}).")
            return smoothed_centerline
        else:
            print(f"  Skipping centerline smoothing: not enough points ({num_cl_points}) for window ({window_length}).")
            return raw_centerline_points
    except Exception as e_smooth:
        print(f"  Warning: Error smoothing centerline: {e_smooth}. Using raw.")
        return raw_centerline_points

def _determine_midpoint_plane(centerline_to_use, detected_pore_center_3d_ed):
    """Determines plane origin and tangent for a 'midpoint' section."""
    if centerline_to_use is None or len(centerline_to_use) < 2:
        print("  Error (_determine_midpoint_plane): Not enough centerline points.")
        return None, None

    num_cl_points_final = len(centerline_to_use)
    y_coordinates = centerline_to_use[:, 1]
    target_y_for_midpoint = 0.0
    if detected_pore_center_3d_ed is not None:
        target_y_for_midpoint = detected_pore_center_3d_ed[1]
        print(f"  Using Y-coordinate from ED pore center for midpoint target: {target_y_for_midpoint:.3f}")
    else:
        print(f"  Warning: ED pore center Y not available for midpoint target. Using Y=0.0 as target.")
    
    closest_to_y_target_idx = np.argmin(np.abs(y_coordinates - target_y_for_midpoint))
    plane_origin = centerline_to_use[closest_to_y_target_idx]
    
    tangent_vec = np.array([0.0, 1.0, 0.0]) # Default
    if num_cl_points_final == 2:
        tangent_vec = centerline_to_use[1] - centerline_to_use[0]
    elif closest_to_y_target_idx == 0:
        tangent_vec = centerline_to_use[1] - centerline_to_use[0]
    elif closest_to_y_target_idx == num_cl_points_final - 1:
        tangent_vec = centerline_to_use[num_cl_points_final - 1] - centerline_to_use[num_cl_points_final - 2]
    else: # Internal point
        tangent_vec = centerline_to_use[closest_to_y_target_idx + 1] - centerline_to_use[closest_to_y_target_idx - 1]
    
    tangent_norm = np.linalg.norm(tangent_vec)
    tangent = tangent_vec / tangent_norm if tangent_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    print(f"  Initial Midpoint Plane Origin (CL idx {closest_to_y_target_idx}): {plane_origin.round(3)}, Tangent: {tangent.round(3)}")
    return plane_origin, tangent

def _determine_tip_plane_v2(centerline_to_use,
                            detected_pore_center_3d_ed,
                            shared_wall_points_3d, # Still unused
                            minor_radius_val,
                            inner_points_for_refinement, # Still unused
                            min_tip_distance=0.0,
                            estimated_centerline_3d_from_ed=None):
    """
    Replacement for _determine_tip_plane_iterative_refined:
    - Selects an initial tip-most centerline point (p0) by minimum Y.
    - Calculates an initial tangent (T_initial_direction) at p0 to define a shift direction.
    - Projects the pore center onto this line and clamps movement along it by 
      [min_tip_distance, minor_radius_val] to get a final_plane_origin.
    - Finds the centerline point (final_cl_point) closest to final_plane_origin.
    - Computes the final tangent (T_final) at final_cl_point. This T_final is the plane normal.

    Args:
        centerline_to_use (np.ndarray[N,3]): Centerline points.
        detected_pore_center_3d_ed (np.ndarray[3] or None): Pore center.
        shared_wall_points_3d (list or np.ndarray): Shared wall points (currently unused).
        minor_radius_val (float): Maximum slide distance along tangent.
        inner_points_for_refinement: (unused placeholder for compatibility).
        min_tip_distance (float): Minimum slide distance from the initial tip point along T_initial_direction.

    Returns:
        final_plane_origin (np.ndarray[3]): The calculated origin for the plane.
        T_final (np.ndarray[3]): The tangent to the centerline at/near final_plane_origin, used as plane normal.
        final_cl_idx (int): The index of the centerline point closest to final_plane_origin.
    """
    if centerline_to_use is None or len(centerline_to_use) < 1:
        print("Error (_determine_tip_plane_v2): No centerline provided.")
        return None, None, -1

    num_pts = len(centerline_to_use)

    # 1. Determine initial tip index (p0_idx) by lowest Y-coordinate
    #x_distances = np.abs(centerline_to_use[:, 0])
    #p0_idx = np.argmin(x_distances)
    #p0 = centerline_to_use[p0_idx]

    # Find the ED centerline point with minimum Y (the tip of the seam)
    seam_tip_idx = np.argmin(estimated_centerline_3d_from_ed[:, 1])
    seam_tip = estimated_centerline_3d_from_ed[seam_tip_idx]
    
    # Find the closest point on centerline_to_use to this seam tip
    distances = np.linalg.norm(centerline_to_use - seam_tip, axis=1)
    p0_idx = np.argmin(distances)
    p0 = centerline_to_use[p0_idx]

    # 2. Compute initial tangent T_initial_direction at p0_idx (used for shift direction)
    if 0 < p0_idx < num_pts - 1:
        T_initial_vec = centerline_to_use[p0_idx + 1] - centerline_to_use[p0_idx - 1]
    elif p0_idx == 0 and num_pts > 1:
        T_initial_vec = centerline_to_use[1] - centerline_to_use[0]
    elif p0_idx == num_pts - 1 and num_pts > 1: # Should not happen if p0_idx is min Y and CL extends
        T_initial_vec = centerline_to_use[-1] - centerline_to_use[-2]
    else: # Single point centerline
        T_initial_vec = np.array([0.0, 1.0, 0.0])

    norm_T_initial = np.linalg.norm(T_initial_vec)
    T_initial_direction = (T_initial_vec / norm_T_initial) if norm_T_initial > 1e-9 else np.array([0.0, 1.0, 0.0])

    # 3. Determine pore center, fallback to mean if needed
    pore_center = detected_pore_center_3d_ed
    if pore_center is None or not isinstance(pore_center, np.ndarray):
        pore_center = np.mean(centerline_to_use, axis=0)

    # 4. Project pore_center onto the line defined by p0 and T_initial_direction
    v = pore_center - p0
    along = float(np.dot(v, T_initial_direction))

    # 5. Determine clamp bounds and clamp the shift amount
    max_slide = float(minor_radius_val) if minor_radius_val and minor_radius_val > 0 else 0.0
    min_slide = float(min_tip_distance) if min_tip_distance and min_tip_distance >= 0 else 0.0
    
    # Ensure max_slide is not less than min_slide if both are positive
    if max_slide > 0 and min_slide > 0 and max_slide < min_slide:
        print(f"  Warning (_determine_tip_plane_v2): max_slide ({max_slide:.3f}) < min_slide ({min_slide:.3f}). Adjusting max_slide to be min_slide.")
        max_slide = min_slide
    elif max_slide == 0 and min_slide > 0: # If max_slide is effectively disabled but min_slide is active
        max_slide = min_slide # Allow shift up to min_slide

    along_clamped = np.clip(along, min_slide, max_slide)

    # 6. Calculate the final plane origin
    final_plane_origin = p0 + along_clamped * T_initial_direction
    
    # 7. Find the centerline point closest to this final_plane_origin
    if num_pts == 1:
        final_cl_idx = 0
    else:
        distances_to_final_origin = np.linalg.norm(centerline_to_use - final_plane_origin, axis=1)
        final_cl_idx = int(np.argmin(distances_to_final_origin))
    
    # 8. Compute the final tangent T_final at final_cl_idx
    # This T_final will be the normal of the cross-section plane

    if 0 < final_cl_idx < num_pts - 1:
        T_final_vec = centerline_to_use[final_cl_idx + 1] - centerline_to_use[final_cl_idx - 1]
    elif final_cl_idx == 0 and num_pts > 1:
        T_final_vec = centerline_to_use[1] - centerline_to_use[0]
    elif final_cl_idx == num_pts - 1 and num_pts > 1:
        T_final_vec = centerline_to_use[-1] - centerline_to_use[-2]
    else: # Single point centerline
        T_final_vec = np.array([0.0, 1.0, 0.0]) # Fallback, T_initial_direction could also be used

    norm_T_final = np.linalg.norm(T_final_vec)
    T_final = (T_final_vec / norm_T_final) if norm_T_final > 1e-9 else np.array([0.0, 1.0, 0.0])

    # The plane origin is final_plane_origin, and its normal is T_final
    # The target_cl_idx returned is final_cl_idx
    return final_plane_origin, T_final, final_cl_idx

def _determine_tip_plane_iterative_refined(centerline_to_use, detected_pore_center_3d_ed, 
                                           shared_wall_points_3d, minor_radius_val, inner_points_for_refinement):
    """
    Determines and refines the plane origin and tangent for a 'tip' section.
    Includes iterative search, fallback, shared wall pullback, and inner pore intersection check.
    Returns: plane_origin, tangent, target_cl_idx (for logging)
    """
    if centerline_to_use is None or len(centerline_to_use) < 1:
        print("  Error (_determine_tip_plane): Not enough centerline points for tip section.")
        return None, None, -1

    num_cl_points_final = len(centerline_to_use)
    sorted_cl_indices_by_y = np.argsort(centerline_to_use[:, 1])
    tip_most_cl_original_idx = sorted_cl_indices_by_y[0]
    
    plane_origin = None
    tangent = None
    target_cl_idx = -1 

    # --- Iterative Search for Angled Plane ---
    mesh_center_for_angling_check = detected_pore_center_3d_ed
    if mesh_center_for_angling_check is None:
        print("  Warning: ED pore center not available for tip angling check. Using centerline mean.")
        if num_cl_points_final > 0:
            mesh_center_for_angling_check = np.mean(centerline_to_use, axis=0)
        else:
            print("  Warning: Cannot determine mesh center for angling check.")
            # Fall through to default "11:55" logic if mesh_center_for_angling_check remains None
    
    found_suitable_plane_iteratively = False
    if num_cl_points_final >= 2 and mesh_center_for_angling_check is not None:
        num_candidates_to_check = min(5, num_cl_points_final - 1)
        if num_candidates_to_check > 0:
            print(f"  Iteratively searching for tip plane (checking up to {num_candidates_to_check} candidates)...")
            for i in range(num_candidates_to_check):
                candidate_idx_in_sorted_list = i + 1
                current_iter_target_cl_idx = sorted_cl_indices_by_y[candidate_idx_in_sorted_list]
                current_iter_plane_origin = centerline_to_use[current_iter_target_cl_idx]

                current_iter_tangent_vec = np.array([0.0, 1.0, 0.0])
                if num_cl_points_final == 1: # Should not happen due to num_cl_points_final >= 2 check above
                     current_iter_tangent_vec = np.array([0.0,1.0,0.0])
                elif num_cl_points_final == 2: # Handles case where num_candidates_to_check might be 1
                    current_iter_tangent_vec = centerline_to_use[sorted_cl_indices_by_y[1]] - centerline_to_use[sorted_cl_indices_by_y[0]]
                else:
                    # Simplified tangent logic for iteration, assuming current_iter_target_cl_idx is not an extreme end for simplicity here
                    # A more robust tangent calculation might be needed if this simplification is problematic
                    if current_iter_target_cl_idx == sorted_cl_indices_by_y[0]:
                        current_iter_tangent_vec = centerline_to_use[sorted_cl_indices_by_y[1]] - centerline_to_use[current_iter_target_cl_idx]
                    elif current_iter_target_cl_idx == sorted_cl_indices_by_y[-1]:
                         current_iter_tangent_vec = centerline_to_use[current_iter_target_cl_idx] - centerline_to_use[sorted_cl_indices_by_y[-2]]
                    else: # Attempt central difference if possible
                        # Find original index in the unsorted centerline_to_use to get neighbors
                        original_cl_index_pos = -1
                        for k_idx, cl_pt_idx_val in enumerate(sorted_cl_indices_by_y):
                            if cl_pt_idx_val == current_iter_target_cl_idx:
                                if k_idx > 0 and k_idx < len(sorted_cl_indices_by_y) -1:
                                     original_cl_index_pos = k_idx
                                break
                        if original_cl_index_pos != -1 and original_cl_index_pos > 0 and original_cl_index_pos < num_cl_points_final -1 :
                             # This logic for prev/next needs to be careful with sorted_cl_indices_by_y vs direct indexing
                             # For simplicity, let's use a simpler forward/backward if at ends of sorted list
                            idx_in_sorted = list(sorted_cl_indices_by_y).index(current_iter_target_cl_idx)
                            if idx_in_sorted == 0: # Should not happen if candidate_idx_in_sorted_list = i+1
                                 current_iter_tangent_vec = centerline_to_use[sorted_cl_indices_by_y[1]] - centerline_to_use[current_iter_target_cl_idx]
                            elif idx_in_sorted == len(sorted_cl_indices_by_y) -1:
                                 current_iter_tangent_vec = centerline_to_use[current_iter_target_cl_idx] - centerline_to_use[sorted_cl_indices_by_y[-2]]
                            else:
                                 prev_cl_point_idx = sorted_cl_indices_by_y[idx_in_sorted-1]
                                 next_cl_point_idx = sorted_cl_indices_by_y[idx_in_sorted+1]
                                 current_iter_tangent_vec = centerline_to_use[next_cl_point_idx] - centerline_to_use[prev_cl_point_idx]
                        elif current_iter_target_cl_idx == sorted_cl_indices_by_y[0]: # Redundant but safe
                             current_iter_tangent_vec = centerline_to_use[sorted_cl_indices_by_y[1]] - centerline_to_use[current_iter_target_cl_idx]
                        else: # Fallback to last point
                             current_iter_tangent_vec = centerline_to_use[current_iter_target_cl_idx] - centerline_to_use[sorted_cl_indices_by_y[list(sorted_cl_indices_by_y).index(current_iter_target_cl_idx)-1]]


                current_iter_tangent_norm = np.linalg.norm(current_iter_tangent_vec)
                if current_iter_tangent_norm < 1e-9: current_iter_tangent_vec = np.array([0.0, 1.0, 0.0]); current_iter_tangent_norm = 1.0
                current_iter_tangent = current_iter_tangent_vec / current_iter_tangent_norm

                vec_to_mesh_center = mesh_center_for_angling_check - current_iter_plane_origin
                vec_to_mesh_center_norm = np.linalg.norm(vec_to_mesh_center)
                
                tangent_y_points_inward = current_iter_tangent[1] > -0.1
                dot_product_with_center_vec = 0.0
                if vec_to_mesh_center_norm > 1e-6:
                    unit_vec_to_center = vec_to_mesh_center / vec_to_mesh_center_norm
                    dot_product_with_center_vec = np.dot(current_iter_tangent, unit_vec_to_center)

                print(f"    Testing CL Idx {current_iter_target_cl_idx} (Y={current_iter_plane_origin[1]:.3f}): tan_Y_ok={tangent_y_points_inward}, dot_center={dot_product_with_center_vec:.3f}")
                if tangent_y_points_inward and dot_product_with_center_vec > 0.3:
                    plane_origin = current_iter_plane_origin
                    tangent = current_iter_tangent
                    target_cl_idx = current_iter_target_cl_idx
                    found_suitable_plane_iteratively = True
                    print(f"  SUCCESS: Found suitable tip plane via iteration at CL Idx {target_cl_idx} (Y={plane_origin[1]:.3f}).")
                    break
    
    if not found_suitable_plane_iteratively:
        print("  Iterative search failed or was not applicable. Using default '11:55' logic for tip plane.")
        if num_cl_points_final == 1:
            target_cl_idx = sorted_cl_indices_by_y[0]
        elif len(sorted_cl_indices_by_y) > 1:
            _target_cl_idx_1155 = sorted_cl_indices_by_y[1]
            if _target_cl_idx_1155 == tip_most_cl_original_idx or \
               centerline_to_use[_target_cl_idx_1155, 1] <= centerline_to_use[tip_most_cl_original_idx, 1] + 1e-6:
                if len(sorted_cl_indices_by_y) > 2 and \
                   centerline_to_use[sorted_cl_indices_by_y[2], 1] > centerline_to_use[sorted_cl_indices_by_y[1], 1] + 1e-6:
                    target_cl_idx = sorted_cl_indices_by_y[2]
                else:
                    target_cl_idx = sorted_cl_indices_by_y[1] if len(sorted_cl_indices_by_y) > 1 else tip_most_cl_original_idx
            else:
                target_cl_idx = _target_cl_idx_1155
        else: # Should not be reached if num_cl_points_final >= 1
            target_cl_idx = sorted_cl_indices_by_y[0]
        
        plane_origin = centerline_to_use[target_cl_idx]
        print(f"  Default '11:55' Plane Origin (CL idx {target_cl_idx}): {plane_origin.round(3)}")

        # Default Tangent Calculation for the chosen plane_origin/target_cl_idx
        tangent_vec_default = np.array([0.0, 1.0, 0.0])
        if num_cl_points_final < 2:
            pass # Keep default
        elif num_cl_points_final == 2:
            tangent_vec_default = centerline_to_use[sorted_cl_indices_by_y[1]] - centerline_to_use[sorted_cl_indices_by_y[0]]
        else:
            idx_in_sorted_list = list(sorted_cl_indices_by_y).index(target_cl_idx)
            if idx_in_sorted_list == 0: # Tip-most point in sorted list
                tangent_vec_default = centerline_to_use[sorted_cl_indices_by_y[1]] - centerline_to_use[target_cl_idx]
            elif idx_in_sorted_list == len(sorted_cl_indices_by_y) - 1: # Other end of sorted list
                tangent_vec_default = centerline_to_use[target_cl_idx] - centerline_to_use[sorted_cl_indices_by_y[-2]]
            else: # Internal point in sorted list
                prev_cl_point_idx = sorted_cl_indices_by_y[idx_in_sorted_list-1]
                next_cl_point_idx = sorted_cl_indices_by_y[idx_in_sorted_list+1]
                tangent_vec_default = centerline_to_use[next_cl_point_idx] - centerline_to_use[prev_cl_point_idx]

        tangent_norm_default = np.linalg.norm(tangent_vec_default)
        if tangent_norm_default < 1e-9: tangent_vec_default = np.array([0.0, 1.0, 0.0]); tangent_norm_default = 1.0
        tangent = tangent_vec_default / tangent_norm_default

    if plane_origin is None or tangent is None:
        print("  FATAL ERROR (_determine_tip_plane): plane_origin or tangent is still None. Skipping section.")
        return None, None, -1
    
    print(f"  Selected Initial Tip Plane Origin (CL idx {target_cl_idx}): {plane_origin.round(3)}")
    print(f"  Selected Initial Tip Tangent (Plane Normal): {tangent.round(3)}")

    # --- Refinement 1: Shared Wall Pullback ---
    current_minor_radius = minor_radius_val if minor_radius_val is not None and minor_radius_val > 1e-6 else 0.5
    if shared_wall_points_3d is not None and len(shared_wall_points_3d) > 0:
        y_slice_check_lower = plane_origin[1] - (1.0 * current_minor_radius)
        y_slice_check_upper = plane_origin[1] + (0.05 * current_minor_radius)
        relevant_sw_points_for_pullback = [
            sw_pt for sw_pt in shared_wall_points_3d 
            if y_slice_check_lower <= sw_pt[1] < y_slice_check_upper and \
               np.linalg.norm(sw_pt[[0,2]] - plane_origin[[0,2]]) < (2.0 * current_minor_radius)
        ]
        if relevant_sw_points_for_pullback:
            highest_relevant_sw_y = np.max(np.array(relevant_sw_points_for_pullback)[:, 1])
            safety_margin_above_sw = 0.05 * current_minor_radius
            if plane_origin[1] < highest_relevant_sw_y + safety_margin_above_sw:
                print(f"    Current plane_origin Y={plane_origin[1]:.3f} is near/below shared wall at Y={highest_relevant_sw_y:.3f}.")
                new_target_y_for_origin = highest_relevant_sw_y + safety_margin_above_sw
                candidate_indices_for_new_origin = np.where(centerline_to_use[:, 1] >= new_target_y_for_origin)[0]
                if len(candidate_indices_for_new_origin) > 0:
                    new_target_cl_idx = candidate_indices_for_new_origin[np.argmin(centerline_to_use[candidate_indices_for_new_origin, 1])]
                    original_target_y = centerline_to_use[target_cl_idx,1] # Original target_cl_idx before pullback
                    if centerline_to_use[new_target_cl_idx,1] < original_target_y + (2 * current_minor_radius) :
                        plane_origin = centerline_to_use[new_target_cl_idx]
                        target_cl_idx = new_target_cl_idx # Update target_cl_idx
                        print(f"    Pulled back plane_origin to Y={plane_origin[1]:.3f} (CL Idx: {target_cl_idx}) due to shared wall.")
                    else:
                        print(f"    Pullback adjustment for shared wall resulted in Y too far from original tip target. Retaining previous origin.")
                else:
                    print(f"    Warning: Cannot pull back plane_origin; no centerline points above shared wall Y={new_target_y_for_origin:.3f}. Retaining previous origin.")
        else:
            print(f"    No shared walls found in immediate Y-vicinity of plane_origin Y={plane_origin[1]:.3f} for pullback.")
    else:
        print("    No shared walls available to refine tip placement (pullback).")

    # --- Refinement 2: Inner Pore Intersection Check ---
    if inner_points_for_refinement is not None and len(inner_points_for_refinement) > 0 and \
       minor_radius_val is not None and minor_radius_val > 1e-6:
        print(f"  Attempting refinement for tip plane to ensure inner pore intersection.")
        candidate_test_indices_on_cl = []
        if target_cl_idx in sorted_cl_indices_by_y: # target_cl_idx is now the one after potential pullback
            current_pos_in_sorted_list = list(sorted_cl_indices_by_y).index(target_cl_idx)
            candidate_test_indices_on_cl.append(target_cl_idx)
            if current_pos_in_sorted_list > 0:
                candidate_test_indices_on_cl.append(sorted_cl_indices_by_y[current_pos_in_sorted_list - 1])
            if current_pos_in_sorted_list < len(sorted_cl_indices_by_y) - 1:
                candidate_test_indices_on_cl.append(sorted_cl_indices_by_y[current_pos_in_sorted_list + 1])
            unique_indices = sorted(list(set(candidate_test_indices_on_cl)), key=lambda x: list(sorted_cl_indices_by_y).index(x))
            candidate_test_indices_on_cl = [idx for idx in unique_indices if 0 <= idx < len(centerline_to_use)]
        else: # Fallback if target_cl_idx somehow not in sorted list (should not happen)
            candidate_test_indices_on_cl = [target_cl_idx] if 0 <= target_cl_idx < len(centerline_to_use) else []


        print(f"    Testing CL indices for refinement: {candidate_test_indices_on_cl}")
        best_refined_origin_found = None
        best_refined_target_cl_idx_found = -1
        # Tangent is kept from the "11:55" or iterative search (before pullback)
        
        intersection_threshold = 0.25 * minor_radius_val
        for test_cl_idx in candidate_test_indices_on_cl:
            test_plane_origin_candidate = centerline_to_use[test_cl_idx]
            distances_to_plane = np.abs(np.dot(inner_points_for_refinement - test_plane_origin_candidate, tangent)) # Use original tangent
            min_dist_to_an_inner_point = np.min(distances_to_plane)
            print(f"      Testing CL Idx {test_cl_idx} (Y={test_plane_origin_candidate[1]:.3f}): min dist of plane to an inner point = {min_dist_to_an_inner_point:.4f}")
            if min_dist_to_an_inner_point <= intersection_threshold:
                print(f"        SUCCESS: Plane at CL Idx {test_cl_idx} intersects/is very close to inner pore (dist {min_dist_to_an_inner_point:.4f} <= thresh {intersection_threshold:.4f}).")
                best_refined_origin_found = test_plane_origin_candidate
                best_refined_target_cl_idx_found = test_cl_idx
                break
        
        if best_refined_origin_found is not None:
            print(f"    Refined tip plane origin to CL Idx {best_refined_target_cl_idx_found}: {best_refined_origin_found.round(3)} to ensure inner pore intersection.")
            plane_origin = best_refined_origin_found
            target_cl_idx = best_refined_target_cl_idx_found # Update for logging
            # Tangent remains unchanged from before this refinement block
        else:
            print(f"    No refinement found that ensures inner pore intersection within threshold {intersection_threshold:.4f}. Using current tip placement.")
    else:
        print("  Skipping inner pore intersection refinement for tip: missing inner_points or minor_radius.")
        
    return plane_origin, tangent, target_cl_idx

def _project_plane_origin_to_2d(plane_origin_3d, transform_2d_to_3d_matrix, current_points_2d_for_fallback):
    """
    Projects the 3D plane origin to 2D and returns the 3D-to-2D transform.
    Uses centroid of current_points_2d_for_fallback if projection fails.
    """
    plane_origin_2d_target = None
    transform_3d_to_2d_matrix = None
    if transform_2d_to_3d_matrix is not None:
        try:
            transform_3d_to_2d_matrix = np.linalg.inv(transform_2d_to_3d_matrix)
            plane_origin_h = np.append(plane_origin_3d, 1) # Homogeneous coordinates
            plane_origin_transformed_h = transform_3d_to_2d_matrix @ plane_origin_h
            plane_origin_2d_target = plane_origin_transformed_h[:2]
            print(f"  Plane origin projected to 2D for filtering target: {plane_origin_2d_target.round(3)}")
        except np.linalg.LinAlgError:
            print("  Error: Cannot invert section transformation matrix. Using 2D points centroid as filter target.")
            transform_3d_to_2d_matrix = None # Invalidate
        except Exception as e_proj:
            print(f"  Error projecting plane origin to 2D: {e_proj}. Using 2D points centroid as filter target.")
            transform_3d_to_2d_matrix = None
    
    if plane_origin_2d_target is None:
        if current_points_2d_for_fallback is not None and len(current_points_2d_for_fallback) > 0:
            plane_origin_2d_target = np.mean(current_points_2d_for_fallback, axis=0)
        else:
            plane_origin_2d_target = np.array([0.0, 0.0]) # Absolute fallback
        print(f"  Using centroid of 2D section points as filter target: {plane_origin_2d_target.round(3)}")
        
    return plane_origin_2d_target, transform_3d_to_2d_matrix

def _trim_tip_inner_boundary(final_points_2D_input, # Points after DBSCAN
                             final_mask_from_dbscan_input, # Mask from DBSCAN (vs original section.to_2D points)
                             inner_points_3d_for_trim, 
                             transform_3d_to_2d_for_trim,
                             original_section_to_2d_points_count # Length of the original points_2D from section.to_2D()
                             ):
    """
    Refines tip section by removing points inside the projected inner boundary.
    Returns updated final_points_2D and final_mask (relative to original section.to_2D points).
    """
    if not (inner_points_3d_for_trim is not None and len(inner_points_3d_for_trim) >= 3 and \
            transform_3d_to_2d_for_trim is not None and \
            final_points_2D_input is not None and len(final_points_2D_input) > 0):
        print("  Skipping inner boundary trimming: insufficient data.")
        return final_points_2D_input, final_mask_from_dbscan_input

    print("  Applying inner boundary trimming for tip section...")
    final_points_2D_after_trim = final_points_2D_input # Start with current points
    updated_final_mask = final_mask_from_dbscan_input.copy() # Start with current mask

    try:
        inner_points_h = np.hstack((inner_points_3d_for_trim, np.ones((len(inner_points_3d_for_trim), 1))))
        projected_inner_points_h = (transform_3d_to_2d_for_trim @ inner_points_h.T).T
        projected_inner_points_2D_on_plane = projected_inner_points_h[:, :2]

        if len(projected_inner_points_2D_on_plane) >= 3:
            ordered_projected_inner_polygon_pts = order_points(projected_inner_points_2D_on_plane, method="angular")
            inner_boundary_path = Path(ordered_projected_inner_polygon_pts)

            mask_for_trimming_within_filtered = [] # Relative to final_points_2D_input
            for pt_2d_to_check in final_points_2D_input:
                mask_for_trimming_within_filtered.append(not inner_boundary_path.contains_point(pt_2d_to_check, radius=1e-9))
            
            mask_for_trimming_within_filtered = np.array(mask_for_trimming_within_filtered, dtype=bool)
            
            num_before_trim = len(final_points_2D_input)
            temp_trimmed_points = final_points_2D_input[mask_for_trimming_within_filtered]
            
            if len(temp_trimmed_points) < 3 and len(final_points_2D_input) >= 3:
                print(f"  Inner boundary trimming resulted in too few points ({len(temp_trimmed_points)}). Reverting trim.")
            else:
                final_points_2D_after_trim = temp_trimmed_points
                
                # Update the overall final_mask (relative to original points_2D from section.to_2D())
                indices_selected_by_dbscan = np.where(final_mask_from_dbscan_input)[0]
                indices_survived_trim_relative_to_dbscan_set = np.where(mask_for_trimming_within_filtered)[0]
                
                true_indices_after_trim_and_dbscan = indices_selected_by_dbscan[indices_survived_trim_relative_to_dbscan_set]
                
                new_overall_final_mask = np.zeros(original_section_to_2d_points_count, dtype=bool) # Use count of original points_2D
                if len(true_indices_after_trim_and_dbscan) > 0:
                    new_overall_final_mask[true_indices_after_trim_and_dbscan] = True
                updated_final_mask = new_overall_final_mask
                print(f"  Inner boundary trimming: {num_before_trim} -> {len(final_points_2D_after_trim)} points.")
        else:
            print("  Not enough projected inner points to form boundary for trimming.")
    except Exception as e_trim:
        print(f"  Error during inner boundary trimming: {e_trim}. Skipping trim.")
    
    return final_points_2D_after_trim, updated_final_mask

def _perform_oriented_truncation(final_points_2D_input, # Points after DBSCAN & inner trim
                                 current_final_mask_input, # Mask after DBSCAN & inner trim (vs original section.to_2D points)
                                 estimated_centerline_3d, # From ED
                                 plane_origin_3d, # Section plane origin
                                 plane_normal_3d, # Section plane normal
                                 transform_3d_to_2d_matrix,
                                 plane_origin_2d_target_for_cut, # For choosing cut origin
                                 original_section_to_2d_points_count # Length of the original points_2D
                                 ):
    """
    Performs oriented truncation of 2D section points based on the ED centerline.
    Returns updated final_points_2D, final_mask, and debug visualization data.
    """
    debug_projected_cl_2d_output = None
    debug_reference_y_val_output = None
    final_points_2D_after_trunc = final_points_2D_input
    updated_final_mask = current_final_mask_input.copy()

    if not (estimated_centerline_3d is not None and len(estimated_centerline_3d) > 1 and \
            transform_3d_to_2d_matrix is not None and plane_origin_2d_target_for_cut is not None and \
            len(final_points_2D_input) > 0 and plane_origin_3d is not None and plane_normal_3d is not None):
        print("    Skipping oriented truncation: missing necessary data.")
        return final_points_2D_input, current_final_mask_input, debug_projected_cl_2d_output, debug_reference_y_val_output

    try:
        intersection_pts_3d = []
        for i in range(len(estimated_centerline_3d) - 1):
            p1, p2 = np.array(estimated_centerline_3d[i]), np.array(estimated_centerline_3d[i+1])
            d1, d2 = np.dot(p1 - plane_origin_3d, plane_normal_3d), np.dot(p2 - plane_origin_3d, plane_normal_3d)
            if d1 * d2 < 0:
                t = d1 / (d1 - d2)
                intersection_pts_3d.append(p1 + t * (p2 - p1))

        if not intersection_pts_3d:
            print("    Oriented Truncation: No ED centerline segments cross the section plane.")
        else:
            inter_h = np.hstack((np.array(intersection_pts_3d), np.ones((len(intersection_pts_3d),1))))
            proj_inter_2d = (transform_3d_to_2d_matrix @ inter_h.T).T[:, :2]
            debug_projected_cl_2d_output = proj_inter_2d

            diffs = proj_inter_2d - plane_origin_2d_target_for_cut[np.newaxis, :]
            idx_cut = np.argmin(np.hypot(diffs[:,0], diffs[:,1]))
            cut_origin_2d = proj_inter_2d[idx_cut]
            reference_y_on_section_plane = cut_origin_2d[1]
            debug_reference_y_val_output = reference_y_on_section_plane

            seed_pt_3d_h = np.append(plane_origin_3d, 1.0)
            seed_pt_2d = (transform_3d_to_2d_matrix @ seed_pt_3d_h)[:2]
            seed_side_indicator = 1 if seed_pt_2d[1] > reference_y_on_section_plane else -1
            print(f"    Oriented Truncation: Seed point is {'ABOVE' if seed_side_indicator > 0 else 'BELOW'} cut line Y={reference_y_on_section_plane:.3f}")

            points_sides_indicator = np.where(final_points_2D_input[:,1] > reference_y_on_section_plane, 1, -1)
            keep_mask_for_truncation = (points_sides_indicator == seed_side_indicator) # Relative to final_points_2D_input

            n_before_trunc, n_after_trunc = len(final_points_2D_input), np.sum(keep_mask_for_truncation)
            if n_after_trunc >= 3:
                temp_truncated_points = final_points_2D_input[keep_mask_for_truncation]
                
                indices_survived_previous_filters = np.where(current_final_mask_input)[0]
                indices_survived_truncation_relative_to_previous = np.where(keep_mask_for_truncation)[0]
                true_indices_after_all_filters = indices_survived_previous_filters[indices_survived_truncation_relative_to_previous]

                new_overall_final_mask_after_trunc = np.zeros(original_section_to_2d_points_count, dtype=bool)
                if len(true_indices_after_all_filters) > 0:
                    new_overall_final_mask_after_trunc[true_indices_after_all_filters] = True
                
                final_points_2D_after_trunc = temp_truncated_points
                updated_final_mask = new_overall_final_mask_after_trunc
                print(f"    Oriented Truncation at Y={reference_y_on_section_plane:.3f}: {n_before_trunc} -> {n_after_trunc} pts.")
            else:
                print(f"    Oriented Truncation would leave too few points ({n_after_trunc}); skipping.")
    except Exception as e_trunc:
        print(f"    ERROR during oriented centerline truncation: {e_trunc}")
        # Keep debug outputs as None or their current state if error occurs mid-process
    
    return final_points_2D_after_trunc, updated_final_mask, debug_projected_cl_2d_output, debug_reference_y_val_output

def _calculate_pca_metrics(points_2d_for_pca, section_location_name="section", reference_orientation_vector=None):
    """Calculates PCA-based aspect ratio and width.
    If reference_orientation_vector is provided, AR is relative to that orientation.
    """
    aspect_ratio_pca = None
    pca_aligned_width = None # This will be the length of the axis perpendicular to the reference orientation

    if points_2d_for_pca is not None and len(points_2d_for_pca) >= 3:
        try:
            pca = PCA(n_components=2)
            pca.fit(points_2d_for_pca)
            
            # Lengths along the principal components of the current section
            len_pc1 = np.sqrt(pca.explained_variance_[0])
            len_pc2 = np.sqrt(pca.explained_variance_[1])

            if reference_orientation_vector is None:
                # Original behavior: AR is always >= 1
                if len_pc2 > 1e-9:
                    aspect_ratio_pca = len_pc1 / len_pc2
                    pca_aligned_width = len_pc2 
                else:
                    aspect_ratio_pca = np.inf
                    pca_aligned_width = len_pc2
            else:
                # New behavior: Align with reference_orientation_vector
                current_pc1_vector = pca.components_[0]
                current_pc2_vector = pca.components_[1]

                # Compare absolute dot products to find which current PC is more aligned with the reference
                # (Handles 180-degree ambiguity)
                dot_product_pc1_ref = np.abs(np.dot(current_pc1_vector, reference_orientation_vector))
                dot_product_pc2_ref = np.abs(np.dot(current_pc2_vector, reference_orientation_vector))

                length_along_ref_orientation = 0.0
                length_perpendicular_to_ref_orientation = 0.0

                if dot_product_pc1_ref >= dot_product_pc2_ref:
                    # Current PC1 is more aligned with the reference "long" axis
                    length_along_ref_orientation = len_pc1
                    length_perpendicular_to_ref_orientation = len_pc2
                else:
                    # Current PC2 is more aligned with the reference "long" axis
                    length_along_ref_orientation = len_pc2
                    length_perpendicular_to_ref_orientation = len_pc1
                
                if length_perpendicular_to_ref_orientation > 1e-9:
                    aspect_ratio_pca = length_along_ref_orientation / length_perpendicular_to_ref_orientation
                else:
                    aspect_ratio_pca = np.inf if length_along_ref_orientation > 1e-9 else 0 # Or handle as None/NaN
                
                pca_aligned_width = length_perpendicular_to_ref_orientation

            print(f"  PCA Results ({section_location_name}): AR={aspect_ratio_pca:.3f}, ConsistentWidth(b)={pca_aligned_width:.3f}")

        except Exception as pca_err:
            print(f"  Error calculating PCA results ({section_location_name}): {pca_err}")
    elif points_2d_for_pca is not None and len(points_2d_for_pca) > 0:
        print(f"  Not enough points ({len(points_2d_for_pca)}) for PCA in {section_location_name} section.")
    
    return aspect_ratio_pca, pca_aligned_width
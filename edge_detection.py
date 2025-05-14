import numpy as np
import trimesh
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

# Optional: enable high-performance ray intersector if available
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    INTERSECTOR = RayMeshIntersector
except ImportError:
    from trimesh.ray.ray_triangle import RayMeshIntersector
    INTERSECTOR = RayMeshIntersector

# Add these constants at the beginning of your script or inside the function
POINT_TYPE_UNKNOWN = 0
POINT_TYPE_SHARED_WALL = 1
POINT_TYPE_PORE_WALL = 2
POINT_TYPE_EXTERNAL = 3
# Constants for pore center calculation
PORE_Y_FILTER_MIN_PERCENT = 0.25  # Ignore bottom 25% of mesh Y-range for pore points
PORE_Y_FILTER_MAX_PERCENT = 0.75  # Ignore top 25% of mesh Y-range for pore points
PORE_XZ_CENTRALITY_THRESHOLD = 1.5 # Max X or Z deviation for filtered pore points from mesh center


def _pick_seam_from_intersections(locs, tri_idx, mesh_normals, x_center_mesh):
    n = len(locs)
    picked_points = []

    # Calculate mesh-relative threshold scaling
    # Use the distance between inner points as a reference
    if n >= 4:
        inner_distance = np.linalg.norm(locs[2] - locs[1])  # Distance between inner points
        mesh_scale = inner_distance * 2  # Use this as a scale factor
    else:
        mesh_scale = 1.0  # Default if we don't have enough points

    # --- TEST THRESHOLDS (Temporarily Looser) ---
    DIST_THRESHOLD_SHARED = mesh_scale / 3.0  # Adjusted for mesh scale
    X_CENTER_THRESHOLD_SHARED = mesh_scale / 3.0
    NORMAL_OPPOSITION_THRESHOLD = -0.3 # Was -0.5 (allow less strictly opposed X-normals)
    X_CENTER_THRESHOLD_PORE = 3.0 # Original, seems reasonable

    # --- SHARED WALL DETECTION ---
    # Case 1: n=2 (classic thin shared wall)
    if n == 2:
        p1, p2 = locs[0], locs[1]
        dist = np.linalg.norm(p2 - p1)
        mid_point = (p1 + p2) / 2
        
        # Add your code right here:
        # Only consider points very close to central X plane
        if abs(mid_point[0] - x_center_mesh) > X_CENTER_THRESHOLD_SHARED:
            return None  # Reject points too far from center
            
        if dist < DIST_THRESHOLD_SHARED and abs(mid_point[0] - x_center_mesh) < X_CENTER_THRESHOLD_SHARED:
            picked_points.append(np.array([mid_point[0], mid_point[1], mid_point[2], POINT_TYPE_SHARED_WALL]))
            return picked_points # High confidence

    # Case 2: n=3 (classic cell1_outer -> shared_wall -> cell2_outer)
    elif n == 3:
        p1_shared = locs[1] # Middle point
        if abs(p1_shared[0] - x_center_mesh) < X_CENTER_THRESHOLD_SHARED:
            picked_points.append(np.array([p1_shared[0], p1_shared[1], p1_shared[2], POINT_TYPE_SHARED_WALL]))
            return picked_points # High confidence

    # Case 3: n>=4 - look for shared wall segments
    elif n >= 4:
        # Check 3a: Is the segment between the two innermost points (locs[1], locs[2]) a thin shared wall?
        inner_p1, inner_p2 = locs[1], locs[2]
        dist_inner_segment = np.linalg.norm(inner_p2 - inner_p1)
        mid_point_inner_segment = (inner_p1 + inner_p2) / 2
        if dist_inner_segment < DIST_THRESHOLD_SHARED and abs(mid_point_inner_segment[0] - x_center_mesh) < X_CENTER_THRESHOLD_SHARED:
            picked_points.append(np.array([mid_point_inner_segment[0], mid_point_inner_segment[1], mid_point_inner_segment[2], POINT_TYPE_SHARED_WALL]))
            # Not returning here, to allow Check 3b to potentially add more or different shared wall points

        # Check 3b: General opposing normals check for any segment if tri_idx is valid
        if tri_idx.size > 0 and tri_idx.shape[0] == n:
            face_norms_for_ray = mesh_normals[tri_idx]
            for i in range(n - 1):
                p_curr, p_next = locs[i], locs[i+1]
                norm_curr, norm_next = face_norms_for_ray[i], face_norms_for_ray[i+1]

                if norm_curr[0] * norm_next[0] < NORMAL_OPPOSITION_THRESHOLD: 
                    dist_segment = np.linalg.norm(p_next - p_curr)
                    if dist_segment < DIST_THRESHOLD_SHARED: 
                        mid_point_segment = (p_curr + p_next) / 2
                        if abs(mid_point_segment[0] - x_center_mesh) < X_CENTER_THRESHOLD_SHARED:
                            is_duplicate = False
                            for pp_existing in picked_points:
                                if pp_existing[3] == POINT_TYPE_SHARED_WALL and \
                                   np.linalg.norm(pp_existing[:3] - mid_point_segment[:3]) < 0.01:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                picked_points.append(np.array([mid_point_segment[0], mid_point_segment[1], mid_point_segment[2], POINT_TYPE_SHARED_WALL]))
                                # Consider 'break' if you only want the first such segment from this loop

    # --- PORE WALL DETECTION (only if no shared wall was conclusively identified by above checks) ---
    if not any(p[3] == POINT_TYPE_SHARED_WALL for p in picked_points):
        if n >= 4: 
            p1_pore, p2_pore = locs[1], locs[2] # Innermost points
            if abs(p1_pore[0] - x_center_mesh) < X_CENTER_THRESHOLD_PORE:
                picked_points.append(np.array([p1_pore[0], p1_pore[1], p1_pore[2], POINT_TYPE_PORE_WALL]))
            if abs(p2_pore[0] - x_center_mesh) < X_CENTER_THRESHOLD_PORE:
                picked_points.append(np.array([p2_pore[0], p2_pore[1], p2_pore[2], POINT_TYPE_PORE_WALL]))

    return picked_points if picked_points else None


def find_seam_by_raycasting(mesh_path, visualize=True):
    """
    Find the seam between guard cells by casting rays through the mesh.
    """
    print(f"Processing {mesh_path}...")
    mesh = _load_and_prepare_mesh(mesh_path)
    bounds = mesh.bounds
    y_min_orig, y_max_orig = bounds[:, 1] # Original bounds
    z_min, z_max = bounds[:, 2]

    # Slightly expand Y range for ray casting to ensure full coverage at extremes
    y_padding = (y_max_orig - y_min_orig) * 0.03 
    y_min_eff = y_min_orig - y_padding
    y_max_eff = y_max_orig + y_padding

    x_start = bounds[0, 0] - 5 
    ray_dir = np.array([1.0, 0.0, 0.0])
    mesh_center_x = np.mean(mesh.bounds[:, 0])

    # Calculate adaptive ray density based on mesh dimensions
    y_range = y_max_eff - y_min_eff
    z_range = z_max - z_min
    
    # Base values plus scaling factor for larger meshes
    y_steps = max(60, int(y_range / 0.5))  # One ray per 0.5 units in Y
    z_steps = max(20, int(z_range / 1.0))  # One ray per 1.0 units in Z
    
    # Cap to reasonable limits
    y_steps = min(y_steps, 120)  # Don't go beyond 120 rays in Y
    z_steps = min(z_steps, 40)   # Don't go beyond 40 rays in Z
    
    print(f"  Using adaptive ray density: {y_steps} Y-rays, {z_steps} Z-rays") 
    
    # OPTIONAL TEST: If threshold loosening doesn't help, try uniform ray spacing next.
    # y_vals = np.linspace(y_min_eff, y_max_eff, y_steps) # Uniform spacing
    y_vals = _nonuniform_space(y_min_eff, y_max_eff, y_steps) # Current non-uniform
    
    z_vals = np.linspace(z_min, z_max, z_steps)

    ray_origins, ray_intersections = [], []
    raw_wall_points = []
    seam_points = []

    for y in y_vals:
        level_seams = []
        for z in z_vals:
            origin = np.array([x_start, y, z])
            ray_origins.append(origin)
            hits, _, tri_idx = mesh.ray.intersects_location(
                ray_origins=[origin],
                ray_directions=[ray_dir],
                multiple_hits=True)
            if len(hits) < 2:
                continue
            sorted_pts = sorted(hits, key=lambda p: p[0])
            ray_intersections.append(sorted_pts)
            picked_info = _pick_seam_from_intersections(sorted_pts, tri_idx, mesh.face_normals, mesh_center_x)

            if picked_info is not None:
                for point_data in picked_info: # point_data is [x,y,z,type]
                    raw_wall_points.append(point_data)
                    # For seam_points (Y-level medians), only consider shared wall types for now
                    if point_data[3] == POINT_TYPE_SHARED_WALL:
                        level_seams.append(point_data[:3]) # Append only XYZ for median calculation
        if level_seams:
            seam_points.append(np.median(level_seams, axis=0))

    seam_points = np.array(seam_points)
    raw_wall_points = np.array(raw_wall_points) if raw_wall_points else np.empty((0,4))

    final_seam_for_smoothing = np.array([])
    raw_points_for_viz = None
    pore_center_coords = None # Initialize pore_center_coords
    smooth_pts = None

    if raw_wall_points.size and raw_wall_points.shape[1] == 4:
        raw_points_for_viz = raw_wall_points

        direct_shared_walls_xyz = raw_wall_points[raw_wall_points[:,3] == POINT_TYPE_SHARED_WALL][:,:3]
        print(f"Found {len(direct_shared_walls_xyz)} direct shared wall candidates from _pick_seam_from_intersections.")

        if direct_shared_walls_xyz.size > 0:
            final_seam_for_smoothing = direct_shared_walls_xyz
            save_seam_line(direct_shared_walls_xyz, "ray_wall_points_DIRECT_SHARED.obj")
            print("Using directly picked shared wall points (no additional clustering).")
        elif seam_points.size: 
            print("Warning: Direct picking yielded no shared wall points, falling back to Y-level medians.")
            final_seam_for_smoothing = seam_points
        
        # --- Calculate Pore Center ---
        pore_candidate_points_all = raw_wall_points[raw_wall_points[:,3] == POINT_TYPE_PORE_WALL][:,:3]
        if pore_candidate_points_all.size > 0:
            # Filter pore candidates by Y-range relative to the mesh bounds
            mesh_y_min, mesh_y_max = mesh.bounds[:, 1].min(), mesh.bounds[:, 1].max()
            mesh_y_range = mesh_y_max - mesh_y_min
            
            pore_y_abs_min_for_filter = mesh_y_min + PORE_Y_FILTER_MIN_PERCENT * mesh_y_range
            pore_y_abs_max_for_filter = mesh_y_min + PORE_Y_FILTER_MAX_PERCENT * mesh_y_range

            y_filtered_pore_points = pore_candidate_points_all[
                (pore_candidate_points_all[:,1] >= pore_y_abs_min_for_filter) &
                (pore_candidate_points_all[:,1] <= pore_y_abs_max_for_filter)
            ]
            
            if y_filtered_pore_points.size > 0:
                # Further filter by XZ centrality (assuming mesh is centered at/near X=0, Z=0 after alignment)
                central_pore_points = y_filtered_pore_points[
                    (np.abs(y_filtered_pore_points[:,0]) < PORE_XZ_CENTRALITY_THRESHOLD) &
                    (np.abs(y_filtered_pore_points[:,2]) < PORE_XZ_CENTRALITY_THRESHOLD)
                ]
                
                if central_pore_points.size > 0:
                    pore_center_coords = np.mean(central_pore_points, axis=0)
                    print(f"Calculated pore center at: {pore_center_coords}")
                else:
                    print("No central pore points found after XZ filtering.")
            else:
                print(f"No pore points found in the central Y-range ({pore_y_abs_min_for_filter:.2f} to {pore_y_abs_max_for_filter:.2f}).")
        else:
            print("No initial pore wall candidates found.")
        # --- End Calculate Pore Center ---

    elif seam_points.size: 
        print("Warning: No typed raw_wall_points available, falling back to Y-level medians.")
        final_seam_for_smoothing = seam_points
        if raw_wall_points.size and raw_wall_points.shape[1] != 4:
            raw_points_for_viz = raw_wall_points

    if final_seam_for_smoothing.size:
        if final_seam_for_smoothing.ndim == 2 and final_seam_for_smoothing.shape[0] > 1:
            final_seam_for_smoothing = final_seam_for_smoothing[np.argsort(final_seam_for_smoothing[:,1])]

        # Filter extreme X values before smoothing
        if final_seam_for_smoothing.size > 0:
            # Keep points with X values less than 1 standard deviation from the median X
            median_x = np.median(final_seam_for_smoothing[:, 0])
            x_std = np.std(final_seam_for_smoothing[:, 0])
            x_filter_threshold = x_std * 0.75  # Even more strict - less than 1 std dev
            original_count = len(final_seam_for_smoothing)
            final_seam_for_smoothing = final_seam_for_smoothing[
                np.abs(final_seam_for_smoothing[:, 0] - median_x) < x_filter_threshold
            ]
            filtered_count = len(final_seam_for_smoothing)
            if filtered_count < original_count:
                print(f"  Filtered out {original_count - filtered_count} outlier X points ({100*(1-filtered_count/original_count):.1f}%)")

        
        # Estimate minor_radius_val based on seam properties
        # Use average distance of seam points from X=0 as an approximation
        minor_radius_val = np.mean(np.abs(final_seam_for_smoothing[:, 0])) * 2.0
        if minor_radius_val < 0.5:  # Set a minimum threshold
            minor_radius_val = 0.5
        print(f"  Estimated minor radius for polynomial fit validation: {minor_radius_val:.2f}")

        # Initial smoothing
        smooth_pts_initial = smooth_seam(final_seam_for_smoothing)

        # --- Polynomial Simplification ---
        simplified_smooth_pts = smooth_pts_initial  # Default to initial smoothing
        if smooth_pts_initial is not None and smooth_pts_initial.shape[0] > 5:
            poly_degree = min(4, smooth_pts_initial.shape[0] - 2)
            if poly_degree >= 1:
                try:
                    y_coords = smooth_pts_initial[:, 1]
                    x_coords = smooth_pts_initial[:, 0]
                    z_coords = smooth_pts_initial[:, 2]
                    
                    # Try different polynomial degrees and pick the best
                    best_degree = poly_degree
                    best_error = float('inf')
                    best_fit = None
                    
                    # Try polynomial fits of decreasing degree
                    for degree in range(poly_degree, 1, -1):
                        coeffs_x = np.polyfit(y_coords, x_coords, degree)
                        coeffs_z = np.polyfit(y_coords, z_coords, degree)
                        
                        x_fit = np.polyval(coeffs_x, y_coords)
                        z_fit = np.polyval(coeffs_z, y_coords)
                        
                        fit_points = np.vstack((x_fit, y_coords, z_fit)).T
                        error = np.mean(np.sum((smooth_pts_initial - fit_points)**2, axis=1))
                        
                        # If error is acceptable, use this simpler model
                        if error < best_error * 1.2:  # Allow 20% more error for simpler models
                            best_degree = degree
                            best_error = error
                            best_fit = fit_points
                    
                    # Ensure the fit is not too far from the original points
                    max_deviation = np.max(np.sqrt(np.sum((smooth_pts_initial - best_fit)**2, axis=1)))
                    if max_deviation < minor_radius_val * 0.5:  # Use a fraction of minor_radius as threshold
                        simplified_smooth_pts = best_fit
                        print(f"  Applied polynomial simplification (degree {best_degree}) with controlled error.")
                    else:
                        print(f"  Polynomial fit rejected: max deviation {max_deviation:.3f} exceeds threshold.")
                except np.linalg.LinAlgError as e:
                    print(f"  Warning: Polynomial fit failed ({e}). Using initial smoothed points.")
                except Exception as e_poly:
                    print(f"  Warning: Error during polynomial simplification ({e_poly}). Using initial smoothed points.")
            else:
                print("  Not enough points for polynomial simplification after initial check.")
        # --- End Polynomial Simplification ---
        smooth_pts = simplified_smooth_pts # Assign the (potentially) simplified points to smooth_pts

        # Add the validation code here, right after setting smooth_pts
        # Validate seam orientation and position
        if smooth_pts is not None and smooth_pts.shape[0] > 3:
            # 1. Check seam orientation (should run primarily along Y-axis)
            y_extent = np.ptp(smooth_pts[:, 1])
            xz_extent = max(np.ptp(smooth_pts[:, 0]), np.ptp(smooth_pts[:, 2]))
            if y_extent < xz_extent:
                print("  Warning: Seam does not align primarily with Y-axis. May need manual inspection.")
            
            # 2. Check seam centrality (should be near X=0 for aligned mesh)
            x_mean = np.mean(np.abs(smooth_pts[:, 0]))
            if x_mean > 2.0:  # More than 2.0 units from center on average
                print(f"  Warning: Seam appears off-center (mean X distance: {x_mean:.2f}). Check alignment.")
            
            # 3. Check seam extent (should cover a significant portion of the mesh Y-range)
            mesh_y_range = y_max_orig - y_min_orig
            seam_y_range = np.ptp(smooth_pts[:, 1])
            coverage = seam_y_range / mesh_y_range
            if coverage < 0.5:  # Less than 50% coverage
                print(f"  Warning: Seam covers only {coverage*100:.1f}% of mesh Y-range. May be incomplete.")

        save_seam_line(simplified_smooth_pts, "ray_seam_smoothed.obj") # Save the simplified line
        if visualize:
            visualize_ray_results(mesh, simplified_smooth_pts, ray_origins, ray_intersections, # Visualize the simplified line
                                  raw_wall_points=raw_points_for_viz,
                                  pore_center=pore_center_coords)
    else:
        print("No seam points found to process or visualize.")
        smooth_pts = None # Ensure smooth_pts is None if no seam was processed

    output_data = {
        'mesh_object': mesh, # The aligned trimesh object
        'shared_wall_points': final_seam_for_smoothing if final_seam_for_smoothing is not None and final_seam_for_smoothing.size > 0 else None,
        'pore_center_coords': pore_center_coords,
        'raw_typed_points': raw_points_for_viz if raw_points_for_viz is not None and raw_points_for_viz.size > 0 else None,
        'estimated_centerline_points': smooth_pts # Add the smoothed seam as the centerline
    }
    return output_data


# Utility functions

def _load_and_prepare_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # align
    mesh = align_mesh(mesh)
    # attach ray intersector
    mesh.ray = INTERSECTOR(mesh)
    return mesh


def _nonuniform_space(vmin, vmax, steps):
    r = vmax - vmin
    e1 = np.linspace(vmin, vmin + 0.25*r, steps//4)
    m = np.linspace(vmin + 0.25*r, vmin + 0.75*r, steps//2)
    e2 = np.linspace(vmin + 0.75*r, vmax, steps//4)
    return np.concatenate((e1, m, e2))


def _pick_seam_from_intersections(locs, tri_idx, mesh_normals, x_center_mesh): # Added x_center_mesh
    """
    Analyzes ray intersections to identify potential seam or pore wall points.
    Returns a list of [x, y, z, type] points, or None.
    x_center_mesh is the approximate X-coordinate of the mesh center after alignment.
    """
    n = len(locs)
    picked_points = [] # Can return multiple points of interest from one ray

    # Heuristic: shared walls are often thin or result in 3 hits.
    # Pore walls are typically the inner surfaces when a ray passes through the opening.

    if n == 2: # Potentially a thin shared wall or grazing an external surface
        p1, p2 = locs[0], locs[1]
        dist = np.linalg.norm(p2 - p1)
        mid_point = (p1 + p2) / 2
        # Check if this thin segment is near the mesh's central X-plane
        if dist < 1.0 and abs(mid_point[0] - x_center_mesh) < 2.0: # Thresholds may need tuning
            # Assume it's a shared wall if thin and central
            picked_points.append(np.array([mid_point[0], mid_point[1], mid_point[2], POINT_TYPE_SHARED_WALL]))
        # else: could be external, ignore for now or classify as POINT_TYPE_EXTERNAL

    elif n == 3: # Often indicates passing: cell1_outer -> shared_wall -> cell2_outer
        p0, p1, p2 = locs[0], locs[1], locs[2]
        # The middle point is a strong candidate for a shared wall
        # Check if it's reasonably central in X
        if abs(p1[0] - x_center_mesh) < 2.0:
             picked_points.append(np.array([p1[0], p1[1], p1[2], POINT_TYPE_SHARED_WALL]))
        # p0 and p2 are likely external surfaces

    elif n >= 4: # Likely passing through the pore
        # Intersections p1 and p2 (0-indexed) are often the pore walls.
        # p0 is outer wall of first cell, p3 is outer wall of second cell.
        # Ensure we have enough points for p0, p1, p2, p3
        if n >= 4: # This check is slightly redundant but safe
            p0, p1, p2, p3 = locs[0], locs[1], locs[2], locs[3]

            # Add p1 as a pore wall candidate
            if abs(p1[0] - x_center_mesh) < 3.0: # Pore walls should also be somewhat central
                picked_points.append(np.array([p1[0], p1[1], p1[2], POINT_TYPE_PORE_WALL]))
            # Add p2 as a pore wall candidate
            if abs(p2[0] - x_center_mesh) < 3.0:
                picked_points.append(np.array([p2[0], p2[1], p2[2], POINT_TYPE_PORE_WALL]))

        # Alternative for n>=4: Check for opposing normals for shared wall
        # This might still be useful if cells are just touching without a clear pore pass
        # Ensure tri_idx is not empty and corresponds to the number of hits
        if tri_idx.size > 0 and tri_idx.shape[0] == n :
            face_norms_for_ray = mesh_normals[tri_idx]
            for i in range(n - 1):
                # Check if X components of normals are opposed and points are close in X
                if face_norms_for_ray[i][0] * face_norms_for_ray[i+1][0] < -0.5: # Strongly opposed
                    # And if the points are physically close along X, suggesting a thin internal feature
                    if abs(locs[i][0] - locs[i+1][0]) < 0.5: # Very close
                        mid_point = (locs[i] + locs[i+1]) / 2
                        if abs(mid_point[0] - x_center_mesh) < 2.0:
                            picked_points.append(np.array([mid_point[0], mid_point[1], mid_point[2], POINT_TYPE_SHARED_WALL]))
                            break # Found a candidate
        elif n >=4 : # Fallback if tri_idx is problematic, use simpler logic for pore walls if not already added
            # This part is a bit of a guess if normals aren't available/reliable for this ray
            # Consider if p1,p2 logic for pore walls is sufficient or if another heuristic is needed
            pass


    return picked_points if picked_points else None


def smooth_seam(points, window=5):
    if len(points) < 3:
        return points
    
    # Sort points by Y-coordinate to ensure proper ordering
    sorted_indices = np.argsort(points[:, 1])
    points = points[sorted_indices]
    
    # Check for gaps in the seam
    y_coords = points[:, 1]
    y_diffs = np.diff(y_coords)
    median_diff = np.median(y_diffs)
    
    # Handle zero or near-zero median difference
    if median_diff <= 1e-10:
        print("  Warning: Y-coordinates have zero median difference. Using default gap handling.")
        # Calculate average distance between points in 3D space instead
        point_dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        median_dist = np.median(point_dists)
        # If that's still zero, use a reasonable default
        if median_dist <= 1e-10:
            median_dist = 0.1
        
        # Use average 3D distance for gap detection
        gap_indices = np.where(point_dists > median_dist * 5)[0]
        default_gap_pts = 5  # Default number of points to insert in gaps
    else:
        gap_threshold = median_diff * 3  # Threshold for identifying gaps
        gap_indices = np.where(y_diffs > gap_threshold)[0]
        default_gap_pts = 5  # Default fallback
    
    # If there are large gaps, fill them with interpolated points
    if len(gap_indices) > 0:
        print(f"  Filling {len(gap_indices)} gaps in seam")
        new_points = []
        last_idx = 0
        
        for gap_idx in gap_indices:
            # Add all points up to the gap
            new_points.extend(points[last_idx:gap_idx+1])
            
            # Interpolate across the gap
            p1, p2 = points[gap_idx], points[gap_idx+1]
            
            # Safely calculate gap_size to avoid division by zero
            if median_diff > 1e-10:
                # Normal case - gap size based on y-difference
                gap_size = max(2, min(20, int(y_diffs[gap_idx] / median_diff)))
            else:
                # Fallback case - use default gap size
                gap_size = default_gap_pts
            
            # Create interpolated points
            for i in range(1, gap_size):
                t = i / gap_size
                interp_point = p1 + t * (p2 - p1)
                new_points.append(interp_point)
                
            last_idx = gap_idx + 1
            
        # Add remaining points
        new_points.extend(points[last_idx:])
        points = np.array(new_points)
    
    # Apply smoothing
    out = np.zeros_like(points)
    for i in range(3):
        out[:, i] = gaussian_filter1d(points[:, i], sigma=window/6.0)
    
    return out


def cluster_wall_points(typed_points): # Input is now Nx4 array [x,y,z,type]
    from sklearn.cluster import DBSCAN
    if typed_points.shape[0] < 5 or typed_points.shape[1] < 4:
        return np.empty((0,3)) # Return empty XYZ if not enough data or wrong format

    # Filter for points close to X=0 and of type SHARED_WALL
    # This is the most critical filter for shared walls
    shared_wall_candidates = typed_points[
        (np.abs(typed_points[:,0]) < 2.0) & # Stricter X filter for shared walls
        (typed_points[:,3] == POINT_TYPE_SHARED_WALL)
    ]
    pts_xyz = shared_wall_candidates[:, :3] # Work with XYZ for clustering

    if len(pts_xyz) < 5:
        print(f"Not enough shared wall candidates for clustering ({len(pts_xyz)} found).")
        return pts_xyz # Return what we have

    y_med = np.median(pts_xyz[:,1])
    clusters_xyz = [] # Store XYZ points of best clusters

    for mask_condition in [(pts_xyz[:,1] > y_med), (pts_xyz[:,1] <= y_med)]: # Top and bottom halves
        sub_xyz = pts_xyz[mask_condition]
        if len(sub_xyz) < 3: # Min samples for DBSCAN
            if len(sub_xyz) > 0: # If there are any points, keep them without clustering
                clusters_xyz.append(sub_xyz)
            continue

        # Adjust DBSCAN params: shared walls might be small, dense clusters
        lab = DBSCAN(eps=1.0, min_samples=3).fit_predict(sub_xyz) # Smaller eps for tighter clusters
        
        # Pass the original subset of points (sub_xyz) to _select_best_cluster
        best_cluster_xyz = _select_best_cluster(sub_xyz, lab, is_top_half=(np.mean(sub_xyz[:,1]) > y_med))
        if best_cluster_xyz.size > 0:
            clusters_xyz.append(best_cluster_xyz)

    return np.vstack(clusters_xyz) if clusters_xyz else np.empty((0,3))


def _select_best_cluster(pts_xyz, labels, is_top_half): # pts_xyz is XYZ
    # Selects the cluster most likely to be a shared wall segment
    valid_clusters = {l: pts_xyz[labels == l] for l in set(labels) if l >= 0}
    if not valid_clusters:
        return np.empty((0,3))

    scores = {}
    for label, cluster_xyz in valid_clusters.items():
        if len(cluster_xyz) < 3: continue # Skip tiny clusters

        y_coords = cluster_xyz[:,1]
        y_range = np.ptp(y_coords)
        x_std = cluster_xyz[:,0].std()
        x_center_abs_mean = np.mean(np.abs(cluster_xyz[:,0]))
        
        # Score: size, y-compactness (shared walls are not very long in Y), x-tightness, x-centrality
        # Shared walls are often short in Y extent compared to the whole pore.
        # We want dense, centrally located clusters.
        score = len(cluster_xyz) / ((x_std + 0.1) * (x_center_abs_mean + 0.1) * (y_range + 0.5)) # Penalize large y_range

        # Add bonus for being at the Y-extreme of the original point set for this half
        # This requires passing the original y_min/y_max of the half, or checking relative position
        # For simplicity, we assume DBSCAN has already grouped relevant points.
        scores[label] = score
    
    if not scores:
        return np.empty((0,3))

    best_label = max(scores, key=scores.get)
    return valid_clusters[best_label]


def align_mesh(mesh):
    """Align mesh using PCA so longest axis is along Y"""
    # Center the mesh
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Compute principal components
    pca = PCA(n_components=3).fit(mesh.vertices)
    axes = pca.components_

    # Define target axes
    target_y = np.array([0.0, 1.0, 0.0])
    target_x = np.array([1.0, 0.0, 0.0])

    # 1) Align first principal axis (longest) to Y
    R1 = trimesh.geometry.align_vectors(axes[0], target_y)
    mesh.apply_transform(R1)

    # 2) Align second principal axis to X, projecting out Y component
    v2 = R1[:3, :3].dot(axes[1])
    # Remove any Y component
    proj = v2 - np.dot(v2, target_y) * target_y
    if np.linalg.norm(proj) > 1e-6:
        R2 = trimesh.geometry.align_vectors(proj, target_x)
        mesh.apply_transform(R2)

    print("Mesh aligned with PCA (longest axis to Y)")
    return mesh


def save_seam_line(points, filename):
    with open(filename,'w') as f:
        for p in points:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for i in range(len(points)-1):
            f.write(f"l {i+1} {i+2}\n")
    print(f"Saved {filename}")


def visualize_ray_results(mesh, seam_points_line, ray_origins, ray_intersections,
                           ray_sample=10, show_rays=True, raw_wall_points=None, pore_center=None): # raw_wall_points can be Nx3 or Nx4
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12,9)) # Slightly larger figure for mesh + details
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    if mesh is not None:
        ax.scatter(*mesh.vertices.T, s=0.1, alpha=0.25, c='gray', label="Mesh")

    # Plot the final smoothed seam line (estimated mid-seam)
    if seam_points_line is not None and seam_points_line.size > 0:
        ax.plot(*seam_points_line.T, 'k-', linewidth=2.5, label="Estimated Mid-Seam (Centerline)")
        # Optionally, to make the line more visible, you can add markers at each point:
        # ax.plot(*seam_points_line.T, 'ko-', linewidth=2.5, markersize=4, label="Estimated Mid-Seam (Centerline)")


    # Plot the calculated pore center
    if pore_center is not None:
        ax.scatter(pore_center[0], pore_center[1], pore_center[2], 
                   c='red', s=180, marker='X', label="Calculated Pore Center", 
                   edgecolor='black', depthshade=False, linewidth=1.5)

    # Set plot limits based on the plotted data to ensure visibility
    all_points_for_limits = []
    if mesh is not None: # Include mesh bounds for limit calculation
        all_points_for_limits.append(mesh.bounds[0]) # min coords of mesh
        all_points_for_limits.append(mesh.bounds[1]) # max coords of mesh
    if seam_points_line is not None and seam_points_line.size > 0:
        all_points_for_limits.append(seam_points_line)
    if pore_center is not None:
        all_points_for_limits.append(pore_center.reshape(1,3))
    
    if all_points_for_limits:
        # If mesh bounds were added, they are already min/max for those points.
        # Otherwise, vstack and calculate min/max.
        if mesh is not None and len(all_points_for_limits) > 2: # mesh bounds + other points
            other_points = np.vstack(all_points_for_limits[2:])
            min_coords_data = other_points.min(axis=0)
            max_coords_data = other_points.max(axis=0)
            min_coords = np.minimum(mesh.bounds[0], min_coords_data)
            max_coords = np.maximum(mesh.bounds[1], max_coords_data)
        elif mesh is not None: # Only mesh bounds
            min_coords = mesh.bounds[0]
            max_coords = mesh.bounds[1]
        else: # No mesh, just other data
             all_points_for_limits_np = np.vstack(all_points_for_limits)
             min_coords = all_points_for_limits_np.min(axis=0)
             max_coords = all_points_for_limits_np.max(axis=0)

        ranges = max_coords - min_coords
        padding = ranges * 0.1 # Add 10% padding
        padding[padding < 1] = 1 # Ensure minimum padding
        # For Z, sometimes padding can be too much if Z range is small, cap it
        padding[2] = min(padding[2], 5)


        ax.set_xlim([min_coords[0] - padding[0], max_coords[0] + padding[0]])
        ax.set_ylim([min_coords[1] - padding[1], max_coords[1] + padding[1]])
        ax.set_zlim([min_coords[2] - padding[2], max_coords[2] + padding[2]])


    ax.set_xlabel("X")
    ax.set_ylabel("Y (Aligned Longest Axis)")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Estimated Mid-Seam, Pore Center, and Mesh")
    plt.show()

if __name__ == "__main__":
    find_seam_by_raycasting("Meshes/Onion_OBJ/Ac_DA_1_3.obj")




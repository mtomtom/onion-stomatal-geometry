import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import trimesh
from matplotlib.path import Path
import edge_detection as ed
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import LightSource
from helper_functions import _smooth_centerline_savgol, _determine_midpoint_plane, _determine_tip_plane_iterative_refined, _project_plane_origin_to_2d, _trim_tip_inner_boundary, _perform_oriented_truncation, _calculate_pca_metrics, _determine_tip_plane_v2, order_points

def get_pore_center_vertical_surface_points(inner_points_3d, 
                                            x_rel_threshold=0.15, 
                                            y_rel_threshold=0.25, 
                                            z_rel_threshold=0.25):
    """
    Isolates 3D points from the inner pore boundary that are:
    1. Close to the median X-value of the pore (forming a "vertical sheet" in XZ view).
    2. Close to the median Y-value (mid-region of stoma length).
    3. Close to the median Z-value (mid-region of stoma pore length).

    Args:
        inner_points_3d (np.ndarray): (N,3) array of 3D points on the inner pore boundary.
        x_rel_threshold (float): Relative threshold for X, as a fraction of the X-extent of inner_points.
                                 If X-extent is negligible, this value is used as an absolute threshold.
        y_rel_threshold (float): Relative threshold for Y, as a fraction of the Y-extent of inner_points.
                                 If Y-extent is negligible, this value is used as an absolute threshold.
        z_rel_threshold (float): Relative threshold for Z, as a fraction of the Z-extent of inner_points.
                                 If Z-extent is negligible, this value is used as an absolute threshold.

    Returns:
        np.ndarray: Filtered (M,3) array of points, or None if input is invalid or no points meet criteria.
    """
    if inner_points_3d is None or len(inner_points_3d) < 3:
        print("  get_pore_center_vertical_surface_points: Not enough inner points provided.")
        return None

    # Calculate actual thresholds based on the extent of inner_points
    # Points "close to the centre" and "aligned on a vertical axis" (in XZ view)
    
    # For X, we want points whose X-coordinate is close to the median X of all inner_points.
    # This identifies points forming a "sheet" roughly in the YZ plane passing through the pore's X-center.
    median_x_inner = np.median(inner_points_3d[:, 0])
    x_extent_inner = np.ptp(inner_points_3d[:, 0]) # ptp is "peak to peak" (max - min)
    actual_x_threshold = x_extent_inner * x_rel_threshold
    if x_extent_inner < 1e-6 : 
        actual_x_threshold = x_rel_threshold # If extent is tiny, use rel_threshold as an absolute threshold amount
                                              # to ensure a minimum filtering window.

    # For Y, filter around the median Y of inner_points to get the "central" part along stoma length.
    median_y_inner = np.median(inner_points_3d[:, 1])
    y_extent_inner = np.ptp(inner_points_3d[:, 1])
    actual_y_threshold = y_extent_inner * y_rel_threshold
    if y_extent_inner < 1e-6 : 
        actual_y_threshold = y_rel_threshold # If extent is tiny, use rel_threshold as an absolute threshold amount.

    # For Z, filter around the median Z of inner_points.
    median_z_inner = np.median(inner_points_3d[:, 2])
    z_extent_inner = np.ptp(inner_points_3d[:, 2])
    actual_z_threshold = z_extent_inner * z_rel_threshold
    if z_extent_inner < 1e-6 : 
        actual_z_threshold = z_rel_threshold # If extent is tiny, use rel_threshold as an absolute threshold amount.
    
    print(f"  Pore surface point filtering thresholds: dX < {actual_x_threshold:.3f} (from median X), dY < {actual_y_threshold:.3f} (from median Y), dZ < {actual_z_threshold:.3f} (from median Z)")

    # Apply filters
    x_condition = np.abs(inner_points_3d[:, 0] - median_x_inner) < actual_x_threshold
    candidate_points = inner_points_3d[x_condition]

    if len(candidate_points) == 0: 
        print("  No points found after X-filter for pore surface.")
        return np.empty((0,3))

    y_condition = np.abs(candidate_points[:, 1] - median_y_inner) < actual_y_threshold
    candidate_points = candidate_points[y_condition]

    if len(candidate_points) == 0: 
        print("  No points found after Y-filter for pore surface.")
        return np.empty((0,3))

    z_condition = np.abs(candidate_points[:, 2] - median_z_inner) < actual_z_threshold
    filtered_points = candidate_points[z_condition]
    
    if len(filtered_points) == 0:
        print("  No points found after Z-filter for pore surface.")
        return np.empty((0,3))
        
    print(f"  Isolated {len(filtered_points)} pore center vertical surface points.")
    return filtered_points
    
def get_radial_dimensions(mesh, center=None, ray_count=36):
    """
    Performs radial ray casting in the XY plane from a given 3D center point
    to find inner and outer boundary points of a mesh.

    This function assumes the mesh is oriented such that its cross-sectional
    profile is well captured by rays cast in the XY plane. It expects rays
    to intersect the mesh at least twice to identify an inner and outer point.

    Args:
        mesh (trimesh.Trimesh): The input mesh object.
        center (np.ndarray, optional): The 3D point from which rays are cast.
                                       Defaults to mesh.centroid if None.
        ray_count (int, optional): The number of rays to cast radially.
                                   Defaults to 36.

    Returns:
        tuple: (inner_points, outer_points, raw_centerline_points, avg_minor_radius)
               Returns (None, None, None, None) if ray casting fails to find
               sufficient points or if an error occurs.
               - inner_points (np.ndarray): (M,3) array of points on the inner boundary.
               - outer_points (np.ndarray): (M,3) array of points on the outer boundary.
               - raw_centerline_points (np.ndarray): (M,3) array of points forming a raw centerline.
               - avg_minor_radius (float): Average estimated minor radius.
    """
    if center is None:
        if mesh is None or len(mesh.vertices) == 0:
            print("  Error: Mesh is invalid or has no vertices, cannot determine centroid for ray casting.")
            return None, None, None, None
        center = mesh.centroid
        print(f"  Ray casting center not provided, using mesh centroid: {center.round(3)}")

    ray_angles = np.linspace(0, 2*np.pi, ray_count, endpoint=False)
    inner_points_list = [] # Use a different name to avoid confusion before conversion to array
    outer_points_list = []

    for angle in ray_angles:
        # Rays are cast in the XY plane relative to the center point's Z-level
        direction = np.array([np.cos(angle), np.sin(angle), 0.0]) 
        origins = np.array([center]) # Origin for this ray is the provided/calculated center
        directions = np.array([direction])
        try:
            # locations: (N,3) array of hit locations
            # index_ray: (N,) array of ray index (0 for single ray)
            # index_tri: (N,) array of triangle index
            locations, _, _ = mesh.ray.intersects_location(origins, directions)
            
            if len(locations) >= 2:
                # Calculate distances from the 3D center to each 3D intersection point
                dists = np.linalg.norm(locations - center, axis=1)
                sorted_idx = np.argsort(dists)
                inner_points_list.append(locations[sorted_idx[0]])
                outer_points_list.append(locations[sorted_idx[-1]])
            # else:
                # print(f"  Debug: Ray at angle {np.degrees(angle):.1f} had {len(locations)} intersections (expected >=2).")
        except Exception as ray_err:
             print(f"  Warning: Ray casting error at angle {np.degrees(angle):.1f}: {ray_err}")

    if not inner_points_list or not outer_points_list: # Check the lists
        print("  Warning: Ray casting failed to find sufficient inner/outer point pairs.")
        return None, None, None, None

    inner_points = np.array(inner_points_list)
    outer_points = np.array(outer_points_list)
    
    # Ensure we have the same number of inner and outer points before proceeding
    if len(inner_points) != len(outer_points) or len(inner_points) == 0:
        print(f"  Warning: Mismatch or zero inner/outer points after collecting results ({len(inner_points)} inner, {len(outer_points)} outer). Cannot calculate dimensions.")
        return None, None, None, None

    raw_centerline_points = (inner_points + outer_points) / 2.0
    avg_minor_radius = np.mean(np.linalg.norm(outer_points - inner_points, axis=1)) / 2.0

    print(f"  Ray casting complete. Found {len(inner_points)} point pairs. Avg minor radius: {avg_minor_radius:.4f}")
    return inner_points, outer_points, raw_centerline_points, avg_minor_radius


def filter_section_points(points_2D, minor_radius, origin_2d_target, eps_factor=0.20, min_samples=3):
    """
    Filters 2D section points to isolate the most relevant cluster.

    The filtering process involves:
    1. Performing DBSCAN clustering on the `points_2D`. The `eps` for DBSCAN
       is scaled by `minor_radius * eps_factor`.
    2. Identifying valid clusters (excluding noise points).
    3. If clusters are found:
        a. Calculating the centroid of each cluster and its distance to `origin_2d_target`.
        b. Sorting clusters by this distance (closest first).
        c. Iterating through sorted clusters:
            i.  Attempting to find a cluster that forms a polygon containing `origin_2d_target`.
                The first such cluster found is selected.
            ii. If no cluster contains `origin_2d_target`, it falls back to selecting the
                cluster whose centroid is closest to `origin_2d_target`.
    4. If DBSCAN fails or no suitable clusters are found, appropriate fallbacks are used
       (e.g., returning all points or empty arrays).

    Args:
        points_2D (np.ndarray): (N,2) array of 2D points from the mesh section.
        minor_radius (float): Estimated minor radius of the pore, used to scale DBSCAN `eps`.
        origin_2d_target (np.ndarray): (2,) array representing the target center in 2D
                                       (typically the projection of the 3D plane origin).
        eps_factor (float, optional): Factor to multiply `minor_radius` by to get DBSCAN `eps`.
                                      Defaults to 0.20.
        min_samples (int, optional): `min_samples` parameter for DBSCAN. Defaults to 3.

    Returns:
        tuple: (filtered_points, mask)
               - filtered_points (np.ndarray): (M,2) array of 2D points from the selected cluster.
               - mask (np.ndarray): Boolean array of shape (N,) indicating which of the
                                    input `points_2D` were selected.
               Returns (empty_array, empty_mask) if filtering is not possible or fails.
    """
    if points_2D is None or len(points_2D) < min_samples:
        return np.empty((0, 2)), np.array([], dtype=bool) # Return empty array and mask

    eps_value = minor_radius * eps_factor
    try:
        clustering = DBSCAN(eps=eps_value, min_samples=min_samples).fit(points_2D)
        labels = clustering.labels_
    except Exception as db_err:
         print(f"  Warning: DBSCAN failed: {db_err}")
         return points_2D, np.ones(len(points_2D), dtype=bool) # Return all points if DBSCAN fails

    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]

    best_label = -1
    final_mask = np.zeros(len(points_2D), dtype=bool)

    if len(valid_labels) > 0:
        cluster_distances = {}
        for label in valid_labels:
            label_mask = (labels == label)
            cluster_points_2d = points_2D[label_mask]
            if len(cluster_points_2d) < min_samples: continue
            # Calculate distance from origin_2d_target to the centroid of the cluster
            cluster_centroid = np.mean(cluster_points_2d, axis=0)
            distance_to_centroid = np.linalg.norm(cluster_centroid - origin_2d_target)
            cluster_distances[label] = distance_to_centroid # Store distance to centroid

        if not cluster_distances: # No valid clusters after size check
             print("  No substantial clusters found after DBSCAN.")
             return np.empty((0, 2)), final_mask

        # Sort labels by distance from their centroid to the origin_2d_target
        sorted_labels = sorted(cluster_distances, key=cluster_distances.get)

        for label_idx, label in enumerate(sorted_labels): # Iterate through sorted labels
            label_mask = (labels == label)
            cluster_points_2d = points_2D[label_mask]

            origin_is_inside = False
            try:
                ordered_cluster_pts = order_points(cluster_points_2d, method="angular")
                path = Path(ordered_cluster_pts)
                if path.contains_point(origin_2d_target):
                    origin_is_inside = True
                    print(f"  Cluster {label}: Origin is INSIDE polygon. Selecting.")
                else:
                    print(f"  Cluster {label}: Origin is OUTSIDE polygon (Dist to centroid: {cluster_distances[label]:.3f}).")
            except Exception as path_err:
                print(f"  Warning: Point-in-polygon check failed for cluster {label}: {path_err}. Considering based on distance.")
                # If check fails, we might still consider it if it's the closest and no other contains the origin
                origin_is_inside = False # Treat as outside for now, fallback will handle

            if origin_is_inside:
                best_label = label
                final_mask = (labels == best_label)
                break # Found suitable cluster containing the origin

        if best_label == -1:
            print("  Origin not strictly inside any cluster.")
            if sorted_labels: # If there are any clusters at all
                fallback_label = sorted_labels[0] # Choose the one whose centroid is closest to origin_2d_target
                final_mask = (labels == fallback_label)
                best_label = fallback_label # Mark that we've selected a label
                print(f"  Fallback: Selecting closest cluster {fallback_label} (Dist to centroid: {cluster_distances[fallback_label]:.3f}).")
            else:
                print("  No suitable cluster found even for fallback.")
    else:
        print("  No valid clusters found via DBSCAN.")

    return points_2D[final_mask], final_mask


def analyze_cross_section(file_paths,
                          section_location='midpoint',
                          output_dir=None,
                          visualize=False):
    """
    Analyzes multiple OBJ files to extract and characterize the cross-section
    at a specified location ('midpoint' or 'tip').

    Workflow for each file:
    1.  Uses `edge_detection.find_seam_by_raycasting` to get an aligned mesh,
        pore center, shared wall points, and an estimated 3D centerline.
    2.  Determines an origin for radial ray casting (preferring the ED pore center).
    3.  Calls `get_radial_dimensions` to find inner/outer boundary points,
        a raw centerline from these points, and an average minor radius.
    4.  Isolates pore center vertical surface points using `get_pore_center_vertical_surface_points`.
    5.  Smooths the raw centerline obtained from radial dimensions.
    6.  Determines an initial 3D plane (origin and normal/tangent) based on
        `section_location` and the smoothed centerline. For 'tip' sections,
        this involves an iterative search for a well-angled plane.
    7.  Refines the tip plane placement based on shared walls and ensuring
        intersection with the inner pore.
    8.  Takes a 3D section of the aligned mesh using the determined plane.
    9.  Converts the 3D section to 2D points.
    10. Filters these 2D points using `filter_section_points` (DBSCAN and
        geometric criteria) to isolate the primary pore outline.
    11. For 'tip' sections, applies inner boundary trimming to the filtered 2D points.
    12. Applies an oriented truncation based on the centerline from edge detection.
    13. Calculates aspect ratio and PCA-based width of the final 2D section.
    14. Optionally generates and saves 2D and 3D visualizations.

    Args:
        file_paths (list): List of paths to OBJ mesh files.
        section_location (str, optional): 'midpoint' or 'tip'. Defaults to 'midpoint'.
        output_dir (str, optional): Directory to save visualizations. Defaults to None.
        visualize (bool, optional): Whether to generate visualizations. Defaults to False.

    Returns:
        dict: A dictionary where keys are file paths and values are either:
              - A tuple: (final_points_2D, final_original_points_3D,
                          transform_2d_to_3d, aspect_ratio, pca_minor_std_dev)
                if analysis is successful.
              - None or {'error': 'message'} if analysis fails for that file.
    """
    results = {}
    print(f"\n--- Starting Cross-Section Analysis (Location: {section_location.upper()}) ---")

    if visualize and output_dir:
        location_output_dir = os.path.join(output_dir, section_location)
        os.makedirs(location_output_dir, exist_ok=True)
    else:
        location_output_dir = output_dir # Will be None if output_dir is None

    for file_path in file_paths:
        print(f"\nProcessing file: {file_path} for {section_location} section")
        # Initialize per-file results/variables
        aspect_ratio = None
        pca_minor_std_dev = None
        plane_origin = None 
        tangent = None
        # Variables from edge_detection
        mesh = None
        shared_wall_points_3d = None
        detected_pore_center_3d_ed = None
        estimated_centerline_3d_from_ed = None # Renamed for clarity
        # Variables from radial dimensions
        inner_points = None
        outer_points = None
        raw_centerline_points_from_radial = None # Renamed for clarity
        minor_radius = None
        # Smoothed centerline (derived from raw_centerline_points_from_radial)
        centerline_to_use = None 
        # Final 2D/3D points
        final_points_2D = None
        final_original_points_3D = None
        transform_2d_to_3d = None


        # --- Step 1: Use edge_detection to get processed mesh and key features ---
        print(f"  Running edge detection for {file_path}...")
        ed_output = ed.find_seam_by_raycasting(file_path, visualize=False) # visualize=False for ed_output here

        if ed_output is None or 'mesh_object' not in ed_output or ed_output['mesh_object'] is None:
            print(f"  Skipping file {file_path} due to edge detection failure or no mesh returned.")
            results[file_path] = {'error': 'Edge detection failed'}
            continue
        
        mesh = ed_output['mesh_object']
        shared_wall_points_3d = ed_output.get('shared_wall_points')
        detected_pore_center_3d_ed = ed_output.get('pore_center_coords') 
        estimated_centerline_3d_from_ed = ed_output.get('estimated_centerline_points') # Centerline from ED

        print(f"  Edge detection complete. Mesh vertices: {len(mesh.vertices)}")
        if detected_pore_center_3d_ed is not None:
            print(f"  Pore center (from ED): {detected_pore_center_3d_ed.round(3)}")
        if shared_wall_points_3d is not None:
            print(f"  Shared wall points (from ED): {len(shared_wall_points_3d)}")
        if estimated_centerline_3d_from_ed is not None:
            print(f"  Estimated centerline (from ED): {len(estimated_centerline_3d_from_ed)} points")


        # --- Step 2: Determine Ray Casting Origin for get_radial_dimensions ---
        ray_origin_for_radial_cast = np.array([0.0, 0.0, 0.0]) # Default to aligned mesh origin

        if detected_pore_center_3d_ed is not None:
            ray_origin_for_radial_cast = detected_pore_center_3d_ed
            print(f"  Using pore center from ED as ray origin for radial cast: {ray_origin_for_radial_cast.round(3)}")
        else:
            print("  Warning: Pore center from ED not available. Defaulting radial cast origin to [0,0,0].")
            # Midpoint logic might refine this later if section_location is 'midpoint' and it uses its own center finding.

        # --- Radial Ray Casting (uses the mesh from edge_detection) ---
        print(f"  Performing radial ray casting from origin: {ray_origin_for_radial_cast.round(3)}")
        ray_count = 45 # Number of rays for radial casting
        inner_points, outer_points, raw_centerline_points_from_radial, minor_radius = get_radial_dimensions(
            mesh, center=ray_origin_for_radial_cast, ray_count=ray_count
        )

        if inner_points is None or outer_points is None or raw_centerline_points_from_radial is None or minor_radius is None:
            print("  Error: Could not determine dimensions via radial ray casting.")
            results[file_path] = {'error': 'Radial ray casting failed'}
            continue
        print(f"  Estimated Minor Radius (aligned): {minor_radius:.3f}")

        # --- Isolate Pore Center Vertical Surface Points (using inner_points from radial cast) ---
        pore_vertical_surface_points_3d = None
        if inner_points is not None and len(inner_points) > 0:
            # Thresholds define how "thin" the sheet of points around the median X, Y, Z should be.
            pore_vertical_surface_points_3d = get_pore_center_vertical_surface_points(
                inner_points, x_rel_threshold=0.20, y_rel_threshold=0.25, z_rel_threshold=0.25
            )
        
        # --- Smooth Centerline (derived from raw_centerline_points_from_radial) ---
        smoothed_centerline_points = _smooth_centerline_savgol(raw_centerline_points_from_radial)
        
        centerline_to_use = smoothed_centerline_points if smoothed_centerline_points is not None else raw_centerline_points_from_radial
        
        if centerline_to_use is None or len(centerline_to_use) < 1: # Allow 1 point for tip case initial check
            print("  Error: Not enough centerline points for sectioning after smoothing attempts.")
            results[file_path] = {'error': 'Not enough centerline points for sectioning'}
            continue
        # For midpoint, we need at least 2 points for tangent. Tip logic handles < 2 points internally.
        if section_location == 'midpoint' and len(centerline_to_use) < 2:
            print("  Error: Midpoint sectioning requires at least 2 centerline points.")
            results[file_path] = {'error': 'Not enough centerline points for midpoint sectioning'}
            continue

        # --- Step 3: Determine Initial Plane Origin and Normal (Tangent) ---
        plane_origin = None # Ensure initialized before conditional assignment
        tangent = None      # Ensure initialized
        
        if section_location == 'midpoint':
            plane_origin, tangent = _determine_midpoint_plane(centerline_to_use, detected_pore_center_3d_ed)
        
        elif section_location == 'tip':
            # Note: minor_radius is from get_radial_dimensions, inner_points is also from there.
            plane_origin, tangent, target_cl_idx_tip_logging = _determine_tip_plane_v2(
                centerline_to_use, 
                detected_pore_center_3d_ed,
                shared_wall_points_3d,
                minor_radius, # Pass the actual minor_radius value
                inner_points,  # Pass inner_points for refinement
                min_tip_distance=2.0,
                estimated_centerline_3d_from_ed=estimated_centerline_3d_from_ed

            )
            # target_cl_idx_tip_logging is returned for potential print statements if needed here,
            # but the main outputs are plane_origin and tangent.
        
        else:
            print(f"  Error: Unknown section_location '{section_location}'.")
            results[file_path] = {'error': f'Unknown section location: {section_location}'}
            continue
        
        # --- At this point, plane_origin and tangent should be definitively set or None if error ---
        if plane_origin is None or tangent is None:
            # Error messages are printed within the helper functions
            results[file_path] = {'error': 'Plane origin/normal could not be determined'} 
            continue
        
        print(f"  Final Plane for Sectioning ({section_location}): Origin={plane_origin.round(3)}, Tangent={tangent.round(3)}")

        # --- Step 5: Take Cross-Section (on ALIGNED mesh) ---
        section = mesh.section(plane_origin=plane_origin, plane_normal=tangent)
        # Store the original 3D points of the section *before* filtering
        original_section_points_3d = section.vertices.copy() if section is not None and hasattr(section, 'vertices') else np.empty((0,3))
        
        if section is None or not hasattr(section, 'entities') or len(section.entities) == 0 or len(section.vertices) < 3:
            print(f"  {section_location.capitalize()} section failed, is empty, or has too few vertices.")
            results[file_path] = {'error': f'{section_location.capitalize()} section failed or empty'}
            continue

        # --- Step 6: Process and Filter Section (DBSCAN, PCA on 2D section) ---
        try:
            path_2D, transform_2d_to_3d = section.to_2D() # Get 2D path and the transform matrix
        except Exception as e_to_2d:
            print(f"  Error converting section to 2D: {e_to_2d}")
            results[file_path] = {'error': 'Failed to convert section to 2D'}
            continue
            
        points_2D = path_2D.vertices

        # Check if the number of original 3D points matches the number of 2D points
        if len(original_section_points_3d) != len(points_2D):
            print(f"  Warning: Mismatch between original 3D section points ({len(original_section_points_3d)}) and 2D points ({len(points_2D)}). Original 3D points mapping might be unreliable.")
            # We can still proceed with 2D analysis, but mapping back might be compromised.
            # original_section_points_3d is already set, so we just note the warning.

        # Transform 3D plane origin to 2D for distance/containment checks
        plane_origin_2d_target, transform_3d_to_2d = _project_plane_origin_to_2d(
            plane_origin, # The 3D plane origin
            transform_2d_to_3d, # The matrix from section.to_2D()
            points_2D # Fallback points if projection fails
        )

        # --- DBSCAN Clustering to get initial filtered 2D points ---
        # Epsilon factor for DBSCAN can be tuned based on section type
        if section_location == 'midpoint':
            eps_factor = 0.15 # Midpoints might be larger, allow slightly larger eps
        elif section_location == 'tip':
            eps_factor = 0.12 # Tips might be more delicate or have denser points, use smaller eps
        else: # Should not happen due to earlier check
            eps_factor = 0.20 

        min_samples_dbscan = 3 # Min samples for a point to be a core point in DBSCAN
        
        # final_mask is relative to points_2D (the output of section.to_2D().vertices)
        filtered_points_2D_after_dbscan, final_mask_from_dbscan = filter_section_points(
            points_2D,
            minor_radius, # minor_radius from get_radial_dimensions
            plane_origin_2d_target, # from _project_plane_origin_to_2d
            eps_factor=eps_factor,
            min_samples=min_samples_dbscan
        )
        print(f"  DBSCAN filtering: {len(points_2D)} -> {len(filtered_points_2D_after_dbscan)} points.")

        # Initialize final_points_2D and final_mask. These might be further modified by subsequent steps.
        final_points_2D = filtered_points_2D_after_dbscan
        final_mask = final_mask_from_dbscan 
        # This final_mask is relative to the original points_2D from section.to_2D()

        # --- Inner Boundary Trimming (for Tip Sections) ---
        if section_location == 'tip':
            final_points_2D, final_mask = _trim_tip_inner_boundary(
                final_points_2D, # Points after DBSCAN
                final_mask,      # Mask after DBSCAN (vs original points_2D)
                inner_points,    # 3D inner points from radial cast
                transform_3d_to_2d, # Matrix to project inner_points to the section plane
                len(points_2D) # Count of original points from section.to_2D() for mask regeneration
            )
        
        # --- Centerline Crossing Check and Oriented Truncation at Seam ---
        # debug_projected_cl_2d and debug_reference_y_val will be returned by the helper
        final_points_2D, final_mask, debug_projected_cl_2d, debug_reference_y_val = _perform_oriented_truncation(
            final_points_2D,    # Points after DBSCAN & potential inner trim
            final_mask,         # Mask after DBSCAN & potential inner trim (vs original points_2D)
            estimated_centerline_3d_from_ed,
            plane_origin,       # 3D origin of the section plane
            tangent,            # 3D normal of the section plane
            transform_3d_to_2d, # Matrix to project ED centerline intersections to section plane
            plane_origin_2d_target, # Target for choosing cut origin
            len(points_2D)      # Count of original points from section.to_2D() for mask regeneration
        )

        # Map back to 3D points (ALIGNED space) using the final cumulative mask
        final_original_points_3D = np.empty((0, 3)) 
        if original_section_points_3d is not None and len(original_section_points_3d) > 0 and \
           final_mask is not None and len(final_mask) == len(original_section_points_3d):
            final_original_points_3D = original_section_points_3d[final_mask]
        elif len(final_points_2D) > 0: 
            print(f"  Warning: Could not map filtered 2D points back to their original 3D coordinates.")
            
        # --- Step 7: Calculate Aspect Ratio and PCA Minor Axis Std Dev ---
        aspect_ratio, pca_minor_std_dev = _calculate_pca_metrics(final_points_2D, section_location)

        # --- Step 8: Store results ---
        # Ensure we have both 2D and corresponding 3D points for a successful result.
        # transform_2d_to_3d is also crucial for interpreting the 2D points in 3D space.
        if len(final_points_2D) > 0 and len(final_original_points_3D) == len(final_points_2D) and transform_2d_to_3d is not None:
            print(f"  Successfully extracted {section_location} section with {len(final_points_2D)} points.")
            results[file_path] = (final_points_2D, final_original_points_3D, transform_2d_to_3d, aspect_ratio, pca_minor_std_dev)
        else:
            if len(points_2D) > 0 and len(final_points_2D) == 0 : # Had initial section points but filtered to none
                print(f"  {section_location.capitalize()} section resulted in 0 points after all filtering.")
            elif len(final_points_2D) > 0 and len(final_original_points_3D) != len(final_points_2D):
                 print(f"  {section_location.capitalize()} section processing failed: Mismatch between final 2D ({len(final_points_2D)}) and 3D ({len(final_original_points_3D)}) points.")
            elif transform_2d_to_3d is None and len(final_points_2D) > 0:
                 print(f"  {section_location.capitalize()} section processing failed: Missing 2D-to-3D transformation.")
            else: # General failure or very few initial points
                 print(f"  {section_location.capitalize()} section processing failed or yielded no usable data.")
            results[file_path] = None # Mark as failed

        if visualize and location_output_dir and results[file_path] is not None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # debug_projected_cl_2d and debug_reference_y_val are taken directly as they were set
            # in the oriented truncation block.

            # --- Create and Save Individual 2D Plot ---
            fig_2d, ax2d = plt.subplots(figsize=(7, 7))
            title_prefix = f'2D {section_location.capitalize()} Cross-Section'
            ax2d.set_aspect('equal'); ax2d.grid(True)
            
            if len(final_points_2D) > 0:
                ordered_plot_points = order_points(final_points_2D, method="angular")
                ax2d.plot(np.append(ordered_plot_points[:, 0], ordered_plot_points[0, 0]), np.append(ordered_plot_points[:, 1], ordered_plot_points[0, 1]), 'b-', linewidth=1.5, label='Final Section')
                ax2d.plot(ordered_plot_points[:, 0], ordered_plot_points[:, 1], 'b.', markersize=4)

            if plane_origin_2d_target is not None:
                ax2d.plot(plane_origin_2d_target[0], plane_origin_2d_target[1], 'ro', markersize=8, label=f'Target Center (2D)', mfc='red', mec='black')

            if debug_projected_cl_2d is not None and len(debug_projected_cl_2d) > 0:
                sorted_proj_cl_indices = np.argsort(debug_projected_cl_2d[:, 1])
                sorted_proj_cl = debug_projected_cl_2d[sorted_proj_cl_indices]
                ax2d.plot(sorted_proj_cl[:, 0], sorted_proj_cl[:, 1], 'g--', linewidth=1.5, label='Projected Seam (ED)')
                ax2d.scatter(debug_projected_cl_2d[:, 0], debug_projected_cl_2d[:, 1], c='green', s=10, alpha=0.5)
            
            if debug_reference_y_val is not None:
                xmin_plot, xmax_plot = ax2d.get_xlim() 
                plot_xs_data = final_points_2D[:,0] if len(final_points_2D) > 0 else (debug_projected_cl_2d[:,0] if debug_projected_cl_2d is not None and len(debug_projected_cl_2d) > 0 else [-1,1])
                if not (np.isfinite(xmin_plot) and np.isfinite(xmax_plot) and xmin_plot < xmax_plot): # Check if xlim is valid
                    xmin_plot, xmax_plot = np.min(plot_xs_data), np.max(plot_xs_data)
                    if xmin_plot == xmax_plot: xmin_plot -=1; xmax_plot +=1 
                ax2d.axhline(y=debug_reference_y_val, color='purple', linestyle=':', linewidth=2, label=f'Cut Line (Y={debug_reference_y_val:.3f})')
                ax2d.set_xlim(xmin_plot, xmax_plot)

            title_str = f'{title_prefix}\n{base_name}'
            if aspect_ratio is not None and np.isfinite(aspect_ratio): title_str += f'\nAR: {aspect_ratio:.3f}'
            if pca_minor_std_dev is not None: title_str += f', Width(b): {pca_minor_std_dev:.3f}'
            ax2d.set_title(title_str); 
            ax2d.legend(fontsize=8)
            plt.tight_layout()
            save_path_2d = os.path.join(location_output_dir, f'{base_name}_{section_location}_section_2D.png')
            plt.savefig(save_path_2d, dpi=150); print(f"  Saved 2D visualization to {save_path_2d}"); plt.close(fig_2d)

            # --- Create and Save Individual 3D Plotly Scene (Aligned) ---
            plotly_traces = []
            vertices_vis = mesh.vertices; faces_vis = mesh.faces
            plotly_traces.append(go.Mesh3d(x=vertices_vis[:,0], y=vertices_vis[:,1], z=vertices_vis[:,2], i=faces_vis[:,0], j=faces_vis[:,1], k=faces_vis[:,2], opacity=0.5, color='lightgrey', name='Aligned Mesh'))

            if estimated_centerline_3d_from_ed is not None and len(estimated_centerline_3d_from_ed) > 0: # Corrected variable name
                plotly_traces.append(go.Scatter3d(x=estimated_centerline_3d_from_ed[:,0], y=estimated_centerline_3d_from_ed[:,1], z=estimated_centerline_3d_from_ed[:,2], mode='lines+markers', line=dict(color='blue', width=5), marker=dict(size=3, color='blue'), name='Estimated Seam (ED)'))
            
            if raw_centerline_points_from_radial is not None and len(raw_centerline_points_from_radial) > 0: # Corrected variable name
                plotly_traces.append(go.Scatter3d(x=raw_centerline_points_from_radial[:,0], y=raw_centerline_points_from_radial[:,1], z=raw_centerline_points_from_radial[:,2], mode='markers+lines', marker=dict(size=4, color='orange'), line=dict(color='orange', width=2), name='Raw CL (Radial)'))

            if smoothed_centerline_points is not None and smoothed_centerline_points is not raw_centerline_points_from_radial and len(smoothed_centerline_points) > 0:
                plotly_traces.append(go.Scatter3d(x=smoothed_centerline_points[:,0], y=smoothed_centerline_points[:,1], z=smoothed_centerline_points[:,2], mode='lines', line=dict(color='cyan', width=4, dash='dash'), name='Smoothed CL (Radial)'))

            if inner_points is not None and len(inner_points) > 0:
                plotly_traces.append(go.Scatter3d(x=inner_points[:,0], y=inner_points[:,1], z=inner_points[:,2], mode='markers', marker=dict(size=3, color='blue'), name='Inner Points (Radial)'))

            if outer_points is not None and len(outer_points) > 0:
                plotly_traces.append(go.Scatter3d(x=outer_points[:,0], y=outer_points[:,1], z=outer_points[:,2], mode='markers', marker=dict(size=3, color='red'), name='Outer Points (Radial)'))
            
            if ray_origin_for_radial_cast is not None:
                plotly_traces.append(go.Scatter3d(x=[ray_origin_for_radial_cast[0]], y=[ray_origin_for_radial_cast[1]], z=[ray_origin_for_radial_cast[2]], mode='markers', marker=dict(size=6, color='green', symbol='cross'), name='Ray Casting Origin'))
            
            if plane_origin is not None: # Check if plane_origin is defined
                plotly_traces.append(go.Scatter3d(x=[plane_origin[0]], y=[plane_origin[1]], z=[plane_origin[2]], mode='markers', marker=dict(size=6, color='purple', symbol='diamond'), name='Plane Origin'))

            if pore_vertical_surface_points_3d is not None and len(pore_vertical_surface_points_3d) > 0:
                plotly_traces.append(go.Scatter3d(x=pore_vertical_surface_points_3d[:,0], y=pore_vertical_surface_points_3d[:,1], z=pore_vertical_surface_points_3d[:,2], mode='markers', marker=dict(size=10, color='black', symbol='diamond-open'), name='Pore Center Surface'))

            # section_data is results[file_path]
            # final_points_2D_for_order = section_data[0]
            # final_section_points_3D_for_plot = section_data[1] # Use the stored 3D points

            if final_original_points_3D is not None and len(final_original_points_3D) >= 3 and \
               final_points_2D is not None and len(final_points_2D) == len(final_original_points_3D):
                center_2d_for_sort = np.mean(final_points_2D, axis=0)
                angles = np.arctan2(final_points_2D[:, 1] - center_2d_for_sort[1], final_points_2D[:, 0] - center_2d_for_sort[0])
                sorted_indices = np.argsort(angles)
                ordered_section_points_3d = final_original_points_3D[sorted_indices]
                loop_points_3d = np.vstack([ordered_section_points_3d, ordered_section_points_3d[0]])
                plotly_traces.append(go.Scatter3d(x=loop_points_3d[:,0], y=loop_points_3d[:,1], z=loop_points_3d[:,2], mode='lines', line=dict(color='magenta', width=4), name=f'{section_location.capitalize()} Section Outline'))

            fig_3d = go.Figure(data=plotly_traces)
            fig_3d.update_layout(title=f'3D Vis (Aligned) - {section_location.capitalize()} Section<br>{os.path.basename(file_path)}', scene=dict(xaxis_title='X', yaxis_title='Y (Aligned Length)', zaxis_title='Z', aspectmode='data'), margin=dict(l=0,r=0,b=0,t=40))
            save_path_3d_html = os.path.join(location_output_dir, f'{base_name}_{section_location}_scene_aligned.html')
            try:
                fig_3d.write_html(save_path_3d_html); print(f"  Saved 3D HTML scene (aligned) to {save_path_3d_html}")
            except Exception as export_err: print(f"  Failed to export 3D HTML scene: {export_err}")
    return results

def create_combined_2d_plot(results_data, output_path, location_name='Unknown'):
    """
    Creates a combined 2D plot overlaying all valid cross-sections.
    """
    print(f"\nCreating combined 2D overlay plot ({location_name}) at: {output_path}")

    # Filter for valid results that have 2D points
    valid_files = [
        f for f, data in results_data.items()
        if data is not None and isinstance(data, tuple) and len(data) > 0 and data[0] is not None and len(data[0]) > 0
    ]

    if not valid_files:
        print(f"  No valid data with 2D points to plot for {location_name}.")
        return

    fig, ax = plt.subplots(figsize=(10, 10)) # Adjust figure size as needed
    ax.set_aspect('equal')
    ax.set_title(f'Combined {location_name} Cross-Sections (Overlay)\n Y vs. X')
    ax.set_xlabel("X-axis") # Corresponds to ordered_points[:, 0]
    ax.set_ylabel("Y-axis") # Corresponds to ordered_points[:, 1]

    # This is where your selected code block begins
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_files)))
    max_extent = 0
    for i, file_path in enumerate(valid_files):
        # results_data[file_path] is expected to be a tuple:
        # (final_points_2D, final_original_points_3D, transform_2d_to_3d, aspect_ratio, pca_minor_std_dev)
        points_2d = results_data[file_path][0] # Get final_points_2D
        original_points_3d = results_data[file_path][1]  # Get original 3D points

        if points_2d is None or len(points_2d) < 3:
            print(f"  Skipping {os.path.basename(file_path)} for combined plot: not enough 2D points.")
            continue
            
        # Ensure points_2d and original_points_3d have the same length
        if len(points_2d) != len(original_points_3d):
            print(f"  Warning: 2D and 3D points count mismatch for {os.path.basename(file_path)}.")
            continue
 
        center_pt = np.mean(points_2d, axis=0)
        centered_points = points_2d - center_pt

        # REPLACE the landmark selection with Z-coordinate based approach:
        # Find highest Z point in 3D space
        z_coords = original_points_3d[:, 2]
        highest_z_idx = np.argmax(z_coords)
        landmark = centered_points[highest_z_idx]  # Use this 2D point as landmark
        
        print(f"  Using highest Z point as landmark for {os.path.basename(file_path)}")
        
        # Calculate angle to rotate so landmark is at the top (90 degrees)
        current_angle = np.arctan2(landmark[1], landmark[0])
        target_angle = np.pi/2  # 90 degrees = top
        rotation_angle = target_angle - current_angle
        
        # Create rotation matrix
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Apply rotation to align landmark to top
        rotated_points = np.dot(centered_points, rotation_matrix.T)
        
        # Order the rotated points for plotting
        ordered_points = order_points(rotated_points, method="angular")
        #ordered_points = order_points(centered_points, method="angular") # Ensure points are ordered for a closed polygon

        # Plotting: X is ordered_points[:, 0], Y is ordered_points[:, 1]
        # The selection had X and Y swapped in the ax.plot call, which might be intentional
        # depending on the desired orientation.
        # Original from selection: ax.plot(Y, X)
        # Standard: ax.plot(X, Y)
        # Let's stick to the selection's X/Y swap for now, assuming it was deliberate.
        # If you want X on horizontal and Y on vertical, it should be:
        # ax.plot(np.append(ordered_points[:, 0], ordered_points[0, 0]),
        #         np.append(ordered_points[:, 1], ordered_points[0, 1]), ...)

        ax.plot(np.append(ordered_points[:, 0], ordered_points[0, 0]),
            np.append(ordered_points[:, 1], ordered_points[0, 1]),
            '-', color=colors[i], linewidth=1.5, alpha=0.7, label=os.path.basename(file_path))
        
        # Calculate max_extent based on the actual plotted values (swapped)
        current_max_x_plot = np.max(np.abs(ordered_points[:, 1])) # Y values are plotted on X
        current_max_y_plot = np.max(np.abs(ordered_points[:, 0])) # X values are plotted on Y
        max_extent = max(max_extent, current_max_x_plot, current_max_y_plot)


    if max_extent == 0: # Handle case where no valid points were plotted
        print(f"  No points were plotted for {location_name}, cannot set limits.")
    else:
        limit = max_extent * 1.1
        # Axis limits should correspond to the data plotted on them.
        # Since Y was plotted first (horizontal) and X second (vertical):
        ax.set_xlim(-limit, limit) # Limits for the horizontal axis (which displayed ordered_points[:, 1])
        ax.set_ylim(-limit, limit) # Limits for the vertical axis (which displayed ordered_points[:, 0])

    if len(valid_files) <= 10:
        ax.legend(loc='upper right', fontsize=8) # Adjusted font size
    else:
        print("  Legend omitted for combined plot due to large number of files.")

    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved combined 2D plot ({location_name}) to {output_path}.")
    except Exception as e:
        print(f"  Error saving combined 2D plot ({location_name}): {e}")
    finally:
        plt.close(fig)

# --- NEW Function to create aspect ratio box plot ---
def create_data_boxplot(data_values, output_path, data_name='Aspect Ratio', location_name='Midpoint'):
    """
    Creates a box plot for a list of data values (e.g., aspect ratios, widths).
    """
    if not data_values:
        print(f"  No valid {data_name} values ({location_name}) to create a box plot.")
        return

    print(f"\nCreating {data_name} box plot ({location_name}) at: {output_path}")
    fig, ax = plt.subplots(figsize=(6, 6))

    jitter_strength = 0.08
    x_jitter = np.random.normal(1, jitter_strength, size=len(data_values))

    bp = ax.boxplot(data_values, vert=True, patch_artist=True, showmeans=False,
                    positions=[1], widths=0.5, showfliers=False,
                    boxprops=dict(facecolor='lightblue', alpha=0.8, zorder=2),
                    medianprops=dict(color='red', linewidth=2, zorder=3),
                    whiskerprops=dict(zorder=2), capprops=dict(zorder=2),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='orange', markeredgecolor='grey', zorder=2)
                   )
    ax.scatter(x_jitter, data_values, alpha=1.0, s=20, color='red', zorder=4, label='Individual Sections')
    ax.set_ylabel(data_name)
    ax.set_title(f'Distribution of {location_name} {data_name} (N={len(data_values)})')
    ax.set_xticks([1])
    ax.set_xticklabels([f'{location_name} Sections'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # ax.legend() # Optional

    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved {data_name} box plot ({location_name}).")
    except Exception as e:
        print(f"  Error saving {data_name} box plot ({location_name}): {e}")
    finally:
        plt.close(fig)


# --- Function to create mesh grid plot (MATPLOTLIB Version) ---
def create_mesh_grid_plot(file_paths, output_path, rows=3, cols=6):
    """
    Creates a single image file (e.g., PNG) with all meshes in a grid layout
    using Matplotlib's 3D plotting.
    """
    num_meshes = len(file_paths)
    if num_meshes == 0:
        print("No mesh files provided...")
        return
    num_to_process = min(num_meshes, rows * cols)
    if num_meshes > rows * cols:
        print(f"Warning: Number of meshes ({num_meshes}) exceeds grid size ({rows}x{cols}). Displaying first {num_to_process}.")

    print(f"\nCreating Matplotlib mesh grid plot ({rows}x{cols}) at: {output_path}")

    # --- Data Preparation: Load, Align, Collect Vertices ---
    mesh_data_for_plotting = []
    all_vertices_aligned = []
    print("  Preprocessing meshes for Matplotlib plot...")
    for i in range(rows * cols): # Prepare data for all potential cells
        mesh_info = {'vertices': None, 'faces': None, 'error': 'Empty Cell', 'basename': None}
        if i < num_meshes:
            file_path = file_paths[i]
            base_name = os.path.basename(file_path)
            mesh_info['basename'] = base_name
            mesh_info['error'] = f'Error: {base_name}' # Default error

            try:
                # Load mesh
                mesh = trimesh.load_mesh(file_path, process=False)
                if isinstance(mesh, trimesh.Scene): mesh = mesh.dump(concatenate=True)

                if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces') or len(mesh.vertices) == 0:
                    print(f"    ERROR: Invalid mesh data for {base_name}")
                    mesh_info['error'] = f'Invalid Mesh: {base_name}'
                else:
                    vertices = mesh.vertices; faces = mesh.faces
                    # Center mesh
                    center = vertices.mean(axis=0); vertices_centered = vertices - center
                    # Align mesh using PCA
                    vertices_aligned = vertices_centered
                    if len(vertices_centered) >= 3:
                        try:
                            pca = PCA(n_components=3); pca.fit(vertices_centered)
                            principal_axes = pca.components_; longest_axis = principal_axes[0]
                            target_axis = np.array([0.0, 1.0, 0.0]) # Align longest axis with Y
                            rotation_matrix = trimesh.geometry.align_vectors(longest_axis, target_axis)
                            vertices_aligned = trimesh.transform_points(vertices_centered, rotation_matrix)
                        except Exception as e: print(f"    Warning: PCA alignment failed for {base_name}: {e}")

                    mesh_info['vertices'] = vertices_aligned
                    mesh_info['faces'] = faces
                    mesh_info['error'] = None # Success for this mesh
                    all_vertices_aligned.append(vertices_aligned) # Collect for bounds calculation
                    print(f"    Prepared {base_name} successfully")

            except Exception as e:
                print(f"    ERROR processing {base_name} for plot: {e}")
                mesh_info['error'] = f'Processing Error: {base_name}'

        mesh_data_for_plotting.append(mesh_info)
    # --- End Data Preparation ---

    if not all_vertices_aligned:
        print("  No valid mesh data collected to create Matplotlib plot.")
        return

    # --- Calculate Overall Bounds for Consistent Axis Limits ---
    all_verts_np = np.concatenate(all_vertices_aligned, axis=0)
    min_bounds = np.min(all_verts_np, axis=0)
    max_bounds = np.max(all_verts_np, axis=0)
    center_bounds = (min_bounds + max_bounds) / 2
    # Determine the largest range along any axis and add padding
    max_range = np.max(max_bounds - min_bounds) * 0.3 # Use 60% of max range for padding

    plot_limits = [
        (center_bounds[0] - max_range, center_bounds[0] + max_range),
        (center_bounds[1] - max_range, center_bounds[1] + max_range),
        (center_bounds[2] - max_range, center_bounds[2] + max_range),
    ]
    print(f"  Calculated plot limits: X={plot_limits[0]}, Y={plot_limits[1]}, Z={plot_limits[2]}")

    # --- Create Matplotlib Figure and Subplots ---
    fig = plt.figure(figsize=(cols * 5, rows * 5)) # Increased multiplier from 2.5 to 3
    fig.patch.set_facecolor('white') # Set background of the whole figure to white

    ls = LightSource(azdeg=315, altdeg=60)

    processed_count = 0
    print("  Generating Matplotlib subplots...")
    for i in range(rows * cols):
        plot_index = i + 1
        # Add subplot with 3D projection
        ax = fig.add_subplot(rows, cols, plot_index, projection='3d', computed_zorder=False)
        ax.set_facecolor('white') # Set background of each subplot

        mesh_info = mesh_data_for_plotting[i]

        if mesh_info and mesh_info['error'] is None:
            vertices = mesh_info['vertices']
            faces = mesh_info['faces']
            # --- Apply light source and change color ---
            # Define color
            rgb = plt.cm.Greys(0.85) # Use a slightly darker grey from colormap
            # Plot the mesh using plot_trisurf with lightsource

            # Plot the mesh using plot_trisurf
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces,
                color=rgb,         # Use the defined color
                lightsource=ls,    
                shade=True, 
                edgecolor=None,
                linewidth=0,       # Line width for edges if shown
                antialiased=True, # Smoother rendering 
                zorder=1
            )
            processed_count += 1
        elif mesh_info and mesh_info['error'] and mesh_info['error'] != 'Empty Cell':
             # Display error message ONLY if it's not the default 'Empty Cell'
             ax.text(0.5, 0.5, 0.5, mesh_info['error'], ha='center', va='center', transform=ax.transAxes, fontsize=8, color='red', wrap=True)

        # --- Configure Axes Appearance ---
        # Set consistent limits for all axes
        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])
        ax.set_zlim(plot_limits[2])

        # Set aspect ratio to be equal based on limits (important!)
        ax.set_box_aspect([1,1,1]) # Forces cubic aspect ratio for the axes box

        # Set view angle (elevation 90 = top-down, azim -90 aligns Y vertically)
        ax.view_init(elev=90, azim=-90)

        # Hide grid, ticks, and labels for a cleaner look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Optionally hide the axis panes/box completely
        ax.set_axis_off()

    # Add overall title and adjust layout
    #fig.suptitle("Onion confocal images", fontsize=16) # Slightly larger title
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95, wspace=0, hspace=0) # Minimal margins, keep space for title

    # --- Save the Figure ---
    try:
        plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor()) # Added bbox_inches and pad_inches
        print(f"\nProcessed {processed_count} meshes successfully for Matplotlib grid plot.")
        print(f"Saved Matplotlib mesh grid visualization to {output_path}")
    except Exception as e:
        print(f"\nError writing Matplotlib file {output_path}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the list of mesh files to process
    files_to_process = [
        "Meshes/Onion_OBJ/Ac_DA_1_3.obj", "Meshes/Onion_OBJ/Ac_DA_1_2.obj", "Meshes/Onion_OBJ/Ac_DA_1_5.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_7.obj", "Meshes/Onion_OBJ/Ac_DA_3_6.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_3.obj", "Meshes/Onion_OBJ/Ac_DA_3_2.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_1.obj", "Meshes/Onion_OBJ/Ac_DA_2_7.obj",
         "Meshes/Onion_OBJ/Ac_DA_2_4.obj", "Meshes/Onion_OBJ/Ac_DA_2_3.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_8_mesh.obj", "Meshes/Onion_OBJ/Ac_DA_1_6.obj"
    ]

    # files_to_process = [
    #     "Meshes/OBJ/Ac_DA_1_3.obj", "Meshes/OBJ/Ac_DA_1_2.obj", "Meshes/OBJ/Ac_DA_1_5.obj",
    #     "Meshes/OBJ/Ac_DA_1_4.obj",  "Meshes/OBJ/Ac_DA_3_6.obj",
    #     "Meshes/OBJ/Ac_DA_3_4.obj",  
    #      "Meshes/OBJ/Ac_DA_2_7.obj", "Meshes/OBJ/Ac_DA_2_6b.obj",
    #     "Meshes/OBJ/Ac_DA_2_6a.obj", "Meshes/OBJ/Ac_DA_2_4.obj", "Meshes/OBJ/Ac_DA_2_3.obj",
    #     "Meshes/OBJ/Ac_DA_1_8_mesh.obj", "Meshes/OBJ/Ac_DA_1_6.obj"
    # ]

    # Check if files exist
    if not all(os.path.exists(f) for f in files_to_process):
         print("Error: One or more specified files do not exist. Please check paths.")
         print("Files expected:", files_to_process)
    else:
        # Define base output directory
        base_output_directory = "test_results" # Renamed for clarity
        os.makedirs(base_output_directory, exist_ok=True)

        # --- Run Analysis for MIDPOINT ---
        midpoint_data = analyze_cross_section(
            files_to_process,
            section_location='midpoint',
            output_dir=base_output_directory, # Pass base directory
            visualize=True
        )

        # --- Run Analysis for TIP ---
        tip_data = analyze_cross_section(
            files_to_process,
            section_location='tip',
            output_dir=base_output_directory, # Pass base directory
            visualize=True
        )

        # --- Process and Plot MIDPOINT Results ---
        if midpoint_data:
            # Combined 2D Plot
            combined_plot_path_mid = os.path.join(base_output_directory, "combined_midpoint_overlay_2D.png")
            create_combined_2d_plot(midpoint_data, combined_plot_path_mid, location_name='Midpoint')

            # Extract ARs and Widths
            midpoint_ars = []
            midpoint_widths = []
            for data in midpoint_data.values():
                if data and len(data) > 4:
                    ar, width = data[3], data[4]
                    if ar is not None and np.isfinite(ar): midpoint_ars.append(ar)
                    if width is not None and np.isfinite(width) and width > 1e-9: midpoint_widths.append(width)

            # Box Plots
            if midpoint_ars:
                boxplot_path_ar_mid = os.path.join(base_output_directory, "midpoint_aspect_ratio_boxplot.png")
                create_data_boxplot(midpoint_ars, boxplot_path_ar_mid, data_name='Aspect Ratio', location_name='Midpoint')
            if midpoint_widths:
                boxplot_path_width_mid = os.path.join(base_output_directory, "midpoint_pca_width_boxplot.png")
                create_data_boxplot(midpoint_widths, boxplot_path_width_mid, data_name='PCA Width (b)', location_name='Midpoint')
        else:
            print("\nNo valid data generated for midpoint analysis.")


        # --- Process and Plot TIP Results ---
        if tip_data:
            # Combined 2D Plot
            combined_plot_path_tip = os.path.join(base_output_directory, "combined_tip_overlay_2D.png")
            create_combined_2d_plot(tip_data, combined_plot_path_tip, location_name='Tip')

            # Extract ARs and Widths
            tip_ars = []
            tip_widths = []
            for data in tip_data.values():
                if data and len(data) > 4:
                    ar, width = data[3], data[4]
                    if ar is not None and np.isfinite(ar): tip_ars.append(ar)
                    if width is not None and np.isfinite(width) and width > 1e-9: tip_widths.append(width)

            # Box Plots
            if tip_ars:
                boxplot_path_ar_tip = os.path.join(base_output_directory, "tip_aspect_ratio_boxplot.png")
                create_data_boxplot(tip_ars, boxplot_path_ar_tip, data_name='Aspect Ratio', location_name='Tip')
            if tip_widths:
                boxplot_path_width_tip = os.path.join(base_output_directory, "tip_pca_width_boxplot.png")
                create_data_boxplot(tip_widths, boxplot_path_width_tip, data_name='PCA Width (b)', location_name='Tip')
        else:
            print("\nNo valid data generated for tip analysis.")


        # --- Create Input Mesh Grid Plot (Only needs to be done once) ---
        mesh_grid_path = os.path.join(base_output_directory, "input_mesh_grid_matplotlib.png")
        create_mesh_grid_plot(files_to_process, mesh_grid_path, rows=3, cols=6)

        print("\n--- Analysis and Visualization Complete ---")

import argparse
import cross_section_functions as csf
import trimesh
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.path as mpath


def analyze_stomata_cross_sections(torus_path, num_sections=16, visualize=True, output_dir=None, closed_stomata=False):
    """
    Analyze stomata cross-sections by extracting sections at various positions.
    
    Parameters:
    -----------
    torus_path : str
        Path to the mesh file (.obj, .stl, etc.)
    num_sections : int
        Number of evenly spaced cross-sections to generate
    visualize : bool
        Whether to create visualization plots
    output_dir : str or None
        Directory to save output files
    closed_stomata : bool
        Whether to process as closed stomata (will attempt to split guard cells)
    
    Returns:
    --------
    tuple
        (cross_sections, positions, centerline_points)
    """
    return analyze_torus_ring_sections_fixed(torus_path, num_sections, visualize=visualize, 
                                           output_dir=output_dir, closed_stomata=closed_stomata)

def analyze_torus_ring_sections_fixed(torus_path, num_sections=16, custom_angles=None, visualize=True, 
                                     output_dir=None, closed_stomata=False):
    """
    Analyze a torus/ring structure by taking cross-sections at specified positions.
    
    Parameters:
    -----------
    torus_path : str
        Path to the mesh file (.obj, .stl, etc.)
    num_sections : int
        Number of evenly spaced cross-sections to generate
    custom_angles : list of float or None
        Custom angles where cross-sections should be taken (in radians)
    visualize : bool
        Whether to create visualization plots
    output_dir : str or None
        Directory to save output files
    
    Returns:
    --------
    tuple
        (cross_sections, positions, centerline_points)
    """
    # 1. Load the mesh
    print(f"Loading mesh from {torus_path}...")
    mesh = trimesh.load_mesh(torus_path)
    center = mesh.centroid
    print(f"Mesh center: {center}")
    
    # 2. Determine torus dimensions using ray casting
    print("Determining torus dimensions using ray casting...")
    
    # We'll cast rays from the center point in the XY plane to find where they
    # intersect the torus surface. This helps determine the torus's shape.
    ray_count = 36
    ray_angles = np.linspace(0, 2*np.pi, ray_count, endpoint=False)
    inner_points = []
    outer_points = []
    
    for angle in ray_angles:
        # Create ray direction in XY plane
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        
        # Cast ray from center through direction
        origins = np.array([center])
        directions = np.array([direction])
        
        # Get all intersections with the mesh
        locations, _, _ = mesh.ray.intersects_location(origins, directions)
        
        if len(locations) >= 2:
            # Sort points by distance from center
            dists = np.linalg.norm(locations - center, axis=1)
            sorted_idx = np.argsort(dists)
            sorted_locations = locations[sorted_idx]
            
            # Store innermost and outermost points
            inner_points.append(sorted_locations[0])
            outer_points.append(sorted_locations[-1])
    
    inner_points = np.array(inner_points)
    outer_points = np.array(outer_points)
    
    # Calculate midpoints between inner and outer - these form the centerline
    raw_centerline_points = (inner_points + outer_points) / 2
    
    # 3. Estimate major/minor radii and fit an ellipse
    # Project all points to the XY plane
    xy_centerline = raw_centerline_points[:, :2]
    center_xy = center[:2]
    
    # Compute radial distances in the XY plane
    r = np.linalg.norm(xy_centerline - center_xy, axis=1)
    
    # Angular positions
    theta = np.arctan2(xy_centerline[:, 1] - center_xy[1], xy_centerline[:, 0] - center_xy[0])
    
    # Estimate minor radius (typical torus thickness)
    minor_radius = np.mean(np.linalg.norm(outer_points - inner_points, axis=1)) / 2
    
    # Estimate major radius (radius of center circle)
    major_radius = np.mean(r)
    
    print(f"Estimated major radius (R): {major_radius:.4f}")
    print(f"Estimated minor radius (r): {minor_radius:.4f}")
    
    # Fit an ellipse to account for non-circular shapes
    # a, b, phi = semi-major axis, semi-minor axis, rotation angle
    initial_guess = [major_radius, major_radius, 0]  # Start with circular guess
    
    try:
        params, _ = curve_fit(csf.ellipse, theta, r, p0=initial_guess)
        a, b, phi = params
        phi = phi % np.pi  # Keep angle in [0, π)
        
        # Make sure a >= b (a is semi-major axis)
        if a < b:
            a, b = b, a
            phi = phi + np.pi/2
        
        print(f"Fitted ellipse: a={a:.4f}, b={b:.4f}, rotation={phi*180/np.pi:.1f}°")
        
        # Generate refined centerline using ellipse fit
        theta_fit = np.linspace(0, 2*np.pi, num_sections, endpoint=False)
        r_fit = csf.ellipse(theta_fit, a, b, phi)
        x_fit = center_xy[0] + r_fit * np.cos(theta_fit)
        y_fit = center_xy[1] + r_fit * np.sin(theta_fit)
        z_fit = np.ones_like(x_fit) * center[2]  # Keep z at center level
        
        fitted_centerline = np.column_stack((x_fit, y_fit, z_fit))
        
        # Check if the fitted centerline is a good match
        # by comparing to original centerline
        avg_distance = 0
        for pt in raw_centerline_points:
            # Find closest point on fitted centerline
            dists = np.linalg.norm(fitted_centerline - pt, axis=1)
            avg_distance += np.min(dists)
        avg_distance /= len(raw_centerline_points)
        
        # If average distance is reasonable, use fitted centerline
        if avg_distance < minor_radius * 0.5:
            print(f"Using ellipse-fitted centerline (avg distance: {avg_distance:.4f})")
            centerline_points = fitted_centerline
        else:
            print(f"Ellipse fit not accurate (avg distance: {avg_distance:.4f}), using original centerline")
            centerline_points = raw_centerline_points
    except:
        print("Could not fit ellipse, using original centerline")
        centerline_points = raw_centerline_points
    
    # 4. Calculate tangent vectors and section positions
    # Compute path length along the centerline
    path_length = 0
    segment_lengths = []
    
    for i in range(len(centerline_points)):
        next_i = (i + 1) % len(centerline_points)
        segment = np.linalg.norm(centerline_points[next_i] - centerline_points[i])
        path_length += segment
        segment_lengths.append(segment)
    
    print(f"Total centerline path length: {path_length:.2f}")
    
    # We want evenly spaced positions along the centerline
    section_positions = np.linspace(0, 1, num_sections, endpoint=False)
    tangent_vectors = []
    
    # For each position, calculate the tangent vector
    for i in range(len(centerline_points)):
        prev_i = (i - 1) % len(centerline_points)
        next_i = (i + 1) % len(centerline_points)
        
        # Get points before and after
        prev_pt = centerline_points[prev_i]
        next_pt = centerline_points[next_i]
        
        # Calculate tangent as average of prev->current and current->next
        tangent = next_pt - prev_pt
        tangent = tangent / np.linalg.norm(tangent)
        tangent_vectors.append(tangent)
    
    tangent_vectors = np.array(tangent_vectors)
    
    # 5. Generate cross-sections with proper filtering
    cross_sections = []
    section_objects = []  # Store the original section objects too
    valid_sections = []
    section_transforms = []

    print(f"Taking {num_sections} cross-sections around the torus ring centerline...")
    for i, (point, tangent, position) in enumerate(zip(centerline_points, tangent_vectors, section_positions)):
        try:
            # Calculate local thickness for this specific position instead of using global minor radius
            # Find nearest inner and outer points
            inner_dists = np.linalg.norm(inner_points - point, axis=1)
            outer_dists = np.linalg.norm(outer_points - point, axis=1)
            nearest_inner = inner_points[np.argmin(inner_dists)]
            nearest_outer = outer_points[np.argmin(outer_dists)]
            
            # Calculate local thickness
            local_thickness = np.linalg.norm(nearest_outer - nearest_inner)
            
            # Use local thickness but don't go below minimum threshold
            local_box_size = max(local_thickness * 1.2, minor_radius * 2.0)
            print(f"  Using local box size: {local_box_size:.4f} at position {position:.2f}")
            
            # Debug the mesh properties
            print(f"  Mesh bounds: {mesh.bounds}")
            
            # Create local mesh using simplified approach (avoid slice_box which may not exist)
            # Create box bounds
            box_min = point - np.array([local_box_size/2, local_box_size/2, local_box_size/2])
            box_max = point + np.array([local_box_size/2, local_box_size/2, local_box_size/2])
            
            # Use full mesh for sectioning - trimesh's section() is fairly efficient
            # and we'll avoid complex subsetting operations that might corrupt the mesh
            local_mesh = mesh
            
            section = local_mesh.section(plane_origin=point, plane_normal=tangent)

            # Add this line:
            original_section_points = section.vertices.copy() if section is not None else None

            # Add diagnostics
            if section is not None:
                print(f"  Section found with {len(section.entities)} entities")
            
            # Add diagnostics
            if section is not None:
                print(f"  Section found with {len(section.entities)} entities")
                
                if len(section.entities) > 0:
                    # Convert to 2D coordinates
                    path_2D, transform = section.to_2D()
                    section_transforms.append(transform) # Store the transformation matrix
                    
                    # Get all closed paths
                    points_2D = path_2D.vertices
                    
                    # The key insight: in the planar space after to_planar(),
                    # the centerline point becomes the origin (0,0)
                    # So the cross-section we want should be centered near the origin
                    
                    # --- Get the 2D representation of the original 3D centerline point ---
                    point_3d_h = np.append(point, 1)
                    point_transformed_h = transform @ point_3d_h
                    point_2d_target = point_transformed_h[:2]
                    # --- End Get 2D representation ---

                    # Use DBSCAN to identify distinct cross-sections
                    eps_value = minor_radius * 0.3  # Adaptive clustering distance
                    clustering = DBSCAN(eps=eps_value, min_samples=3).fit(points_2D)
                    labels = clustering.labels_
                    
                    # Find distinct clusters
                    unique_labels = np.unique(labels)
                    valid_labels = unique_labels[unique_labels != -1]  # Exclude noise

                    # --- Define variables to hold the FINAL filtered points ---
                    final_points_2D = np.empty((0, 2))
                    final_original_points_3D = np.empty((0, 3))
                    # --- End Define ---

                    if len(valid_labels) > 1:
                        # --- New Selection Logic: Minimum Average Distance to Target Point ---
                        cluster_avg_distances = []
                        cluster_info = []
                        original_points_mapping = {}

                        for label in valid_labels:
                            label_mask = (labels == label)
                            cluster_points_2d = points_2D[label_mask]
                            num_points = len(cluster_points_2d)
                            if num_points < 3: continue

                            original_points_mapping[label] = original_section_points[label_mask]

                            # Calculate distances from target to each point in the cluster
                            distances = np.linalg.norm(cluster_points_2d - point_2d_target, axis=1)
                            avg_distance = np.mean(distances)

                            cluster_avg_distances.append({'label': label, 'avg_distance': avg_distance})
                            # Store other info for logging/fallback
                            center = np.mean(cluster_points_2d, axis=0)
                            cluster_info.append({'label': label, 'num_points': num_points, 'center': center})

                        # --- Decision based on minimum average distance ---
                        if cluster_avg_distances: # Check if list is not empty
                            cluster_avg_distances.sort(key=lambda x: x['avg_distance'])
                            best_label = cluster_avg_distances[0]['label']
                            min_avg_dist = cluster_avg_distances[0]['avg_distance']
                            print(f"  Selected cluster (label {best_label}) with minimum average distance ({min_avg_dist:.4f}) to target {point_2d_target}.")
                        else:
                            # Fallback if no valid clusters found after filtering small ones
                            print(f"  Warning: No substantial clusters found. Falling back to largest cluster overall (if any).")
                            if not cluster_info:
                                 print("  No clusters found at all.")
                                 best_label = -1
                            else:
                                cluster_info.sort(key=lambda x: x['num_points'], reverse=True)
                                best_label = cluster_info[0]['label']
                                print(f"  Fallback selected largest cluster (label {best_label}).")


                        # --- Get the FINAL points based on the best label ---
                        if best_label != -1:
                            final_mask = (labels == best_label)
                            final_points_2D = points_2D[final_mask]
                            final_original_points_3D = original_section_points[final_mask]
                            selected_center = next((info['center'] for info in cluster_info if info['label'] == best_label), None)
                            print(f"  Final selection: Cluster {best_label} with {len(final_points_2D)} points (center: {selected_center})")
                        else:
                             final_points_2D = np.empty((0, 2))
                             final_original_points_3D = np.empty((0, 3))
                        # --- End Selection Logic ---

                    elif len(valid_labels) == 1:
                        # Just filter out noise points if only one valid cluster
                        final_mask = (labels != -1) # Mask for the single valid cluster
                        # --- Get the FINAL points ---
                        final_points_2D = points_2D[final_mask]
                        final_original_points_3D = original_section_points[final_mask]
                        # --- End Get FINAL ---
                        print(f"  Single cross-section with {len(final_points_2D)} points")
                    else: # No valid clusters found
                        print("  No valid clusters found after DBSCAN.")
                        # final_points_2D and final_original_points_3D remain empty

                    # --- Assertion to catch mismatch during generation ---
                    assert len(final_points_2D) == len(final_original_points_3D), \
                        f"Mismatch during generation! Section {i}: final_points_2D len {len(final_points_2D)}, final_original_points_3D len {len(final_original_points_3D)}"
                    # --- End Assertion ---

                    # Store the FINAL filtered points
                    cross_sections.append((final_points_2D, final_original_points_3D))
                    section_transforms.append(transform) # Keep transform associated
                    valid_sections.append(len(final_points_2D) > 0) # Section is valid if points remain

                else: # Section was None or empty BEFORE DBSCAN
                    print(f"  Section at position {position:.2f} is invalid or empty.")
                    cross_sections.append((np.empty((0, 2)), np.empty((0, 3))))
                    section_transforms.append(None)
                    valid_sections.append(False)

        except Exception as e:
            # ... (exception handling remains the same) ...
            print(f"Error processing section at position {position:.2f}: {e}")
            cross_sections.append((np.empty((0, 2)), np.empty((0, 3))))
            section_transforms.append(None)
            valid_sections.append(False)

    # 6. Create visualizations
    if visualize:
        csf.create_visualizations(mesh, centerline_points, tangent_vectors, section_positions, 
                             cross_sections, section_objects, section_transforms, raw_centerline_points, 
                             inner_points, outer_points, minor_radius, valid_sections, 
                             output_dir, closed_stomata=closed_stomata)
    
    print("\nAnalysis complete:")
    print(f"- Total cross-sections: {num_sections}")
    print(f"- Valid cross-sections: {sum(valid_sections)}")
    
    if output_dir is not None:
        print(f"- Results saved to: {output_dir}")
    
    return cross_sections, section_positions, centerline_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze stomata cross-sections from 3D mesh")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file (.obj, .stl, etc.)")
    parser.add_argument("--num-sections", type=int, default=16, help="Number of cross-sections to generate")
    parser.add_argument("--no-vis", action="store_false", dest="visualize", help="Disable visualization")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--closed-stomata", action="store_true", help="Process as closed stomata (will attempt to split guard cells)")
    
    args = parser.parse_args()
    
    # Run the analysis
    cross_sections, positions, centerline_points = analyze_stomata_cross_sections(
    args.mesh_path, 
    num_sections=args.num_sections,
    visualize=args.visualize,
    output_dir=args.output_dir,
    closed_stomata=args.closed_stomata 
    )
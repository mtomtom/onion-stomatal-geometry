import os
import argparse
import cross_section_functions as csf
import trimesh
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit


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
            
            # Add diagnostics
            if section is not None:
                print(f"  Section found with {len(section.entities)} entities")
                
                if len(section.entities) > 0:
                    # Convert to 2D coordinates
                    path_2D, transform = section.to_planar()
                    
                    # Get all closed paths
                    points_2D = path_2D.vertices
                    
                    # The key insight: in the planar space after to_planar(),
                    # the centerline point becomes the origin (0,0)
                    # So the cross-section we want should be centered near the origin
                    
                    # Use DBSCAN to identify distinct cross-sections
                    eps_value = minor_radius * 0.3  # Adaptive clustering distance
                    clustering = DBSCAN(eps=eps_value, min_samples=3).fit(points_2D)
                    labels = clustering.labels_
                    
                    # Find distinct clusters
                    unique_labels = np.unique(labels)
                    valid_labels = unique_labels[unique_labels != -1]  # Exclude noise
                
                if len(valid_labels) > 1:
                    print(f"  Section at position {position:.2f} has {len(valid_labels)} separate cross-sections")
                    
                    # Simple selection criterion: pick cluster closest to origin
                    cluster_centers = []
                    for label in valid_labels:
                        cluster_points = points_2D[labels == label]
                        center = np.mean(cluster_points, axis=0)
                        cluster_centers.append((label, center, np.linalg.norm(center)))
                    
                    # Sort by distance to origin and take the closest
                    closest_label = sorted(cluster_centers, key=lambda x: x[2])[0][0]
                    
                    # Filter to just the closest cluster
                    points_2D = points_2D[labels == closest_label]
                    print(f"  Selected cross-section with {len(points_2D)} points (distance to origin: {sorted(cluster_centers, key=lambda x: x[2])[0][2]:.4f})")
                else:
                    # Just filter out noise points if only one valid cluster
                    filtered_points = points_2D[labels != -1] if -1 in unique_labels else points_2D
                    print(f"  Single cross-section with {len(filtered_points)} points")
                    points_2D = filtered_points
  
                
                # Calculate area using ordered points
                center_point = np.mean(points_2D, axis=0)
                angles = np.arctan2(points_2D[:, 1] - center_point[1], points_2D[:, 0] - center_point[0])
                sorted_indices = np.argsort(angles)

                # Calculate convexity ratio
                convexity = csf.calculate_convexity(points_2D)                       

                # Calculate additional shape metrics
                # 1. Find the bounding box and check aspect ratio
                min_coords = np.min(points_2D, axis=0)
                max_coords = np.max(points_2D, axis=0)
                width = max_coords[0] - min_coords[0]
                height = max_coords[1] - min_coords[1]
                aspect_ratio = max(width/height, height/width) if min(width, height) > 0 else 1.0

                # 2. Measure the roundness using PCA
                pca = PCA(n_components=2)
                pca.fit(points_2D)
                variance_ratio = pca.explained_variance_ratio_[0] / (pca.explained_variance_ratio_[1] + 1e-10)

                print(f"  Shape metrics - Aspect ratio: {aspect_ratio:.2f}, Variance ratio: {variance_ratio:.2f}")

                # BRANCHING LOGIC: Different approaches based on closed_stomata flag

                if closed_stomata:
                    # Assume points_2D and labels have been computed already.
                    unique_labels = np.unique(labels[labels != -1])
                    if len(unique_labels) > 1:
                        # Compute centroids for each cluster
                        cluster_centers = []
                        for label in unique_labels:
                            cluster_points = points_2D[labels == label]
                            cluster_center = np.mean(cluster_points, axis=0)
                            distance = np.linalg.norm(cluster_center)  # distance to origin
                            cluster_centers.append((label, cluster_center, distance))
                        # Select cluster with smallest distance
                        best_label = sorted(cluster_centers, key=lambda x: x[2])[0][0]
                        points_2D = points_2D[labels == best_label]
                        print(f"  Closed stomata: selected cluster {best_label} based on proximity")
                    else:
                        print("  Closed stomata: only one cluster detected")

                else:
                    # Only consider splitting for open stomata
                    if (convexity < 0.45) and (variance_ratio > 3.5 or aspect_ratio > 2.5):
                        # Process merged cells only for open stomata
                        print(f"  Open stomata with merged guard cells detected")
                        
                        # Find the best splitting line using PCA
                        main_axis = pca.components_[0]  # Primary axis
                        
                        # Project points onto the main axis
                        projections = points_2D @ main_axis
                    
                        # First check if we need to perform any splitting
                        # Use PCA to check if there might be two guard cells present
                        pca = PCA(n_components=2).fit(points_2D)
                        variance_ratio = pca.explained_variance_ratio_[0] / (pca.explained_variance_ratio_[1] + 1e-10)
                        
                        # If the shape is elongated, try to split between cells
                        if variance_ratio > 2.5:  # Fairly elongated shape
                            main_axis = pca.components_[0]  # Primary axis
                            
                            # Project points onto the main axis
                            projections = points_2D @ main_axis
                            
                            # Find center point projection
                            center_proj = np.mean(projections)
                            
                            # Calculate distance from center of section to each point
                            center_point = np.mean(points_2D, axis=0)
                            distances = np.linalg.norm(points_2D - center_point, axis=1)
                            
                            # Find the median distance - useful for threshold
                            median_dist = np.median(distances)
                            
                            # Find the split point that maximizes separation between cells
                            # Try multiple approaches
                            
                            # Method 1: Use histogram to find valleys
                            hist, bin_edges = np.histogram(projections, bins=min(40, len(projections)//2))
                            hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')  # Smooth histogram
                            
                            # Look for valleys in the histogram
                            neg_hist = -hist_smooth
                            peaks, _ = find_peaks(neg_hist)
                            
                            split_point = None
                            
                            if len(peaks) > 0:
                                # Find a good valley near the center
                                hist_max = np.max(hist_smooth)
                                peak_depths = hist_max - hist_smooth[peaks]
                                
                                # Get the deepest valleys
                                sorted_peaks = sorted(zip(peaks, peak_depths), key=lambda x: -x[1])
                                
                                for peak_idx, depth in sorted_peaks:
                                    # Get the actual bin center for this valley
                                    valley_proj = bin_edges[peak_idx:peak_idx+2].mean()
                                    
                                    # Check if this valley is a good split point
                                    # We want it to be away from center, so it preserves the inner cell
                                    if abs(valley_proj - center_proj) > median_dist * 0.3:
                                        split_point = valley_proj
                                        print(f"  Found good split point at projection {split_point:.4f}")
                                        break
                            
                            # If we found a valid split point, apply it
                            if split_point is not None:
                                # Split the points
                                if split_point < center_proj:
                                    # Keep points to the right of the split
                                    points_2D = points_2D[projections >= split_point]
                                else:
                                    # Keep points to the left of the split
                                    points_2D = points_2D[projections <= split_point]
                                
                                print(f"  Split cross-section to keep inner guard cell with {len(points_2D)} points")
                            else:
                                print(f"  Could not find good split point, keeping entire cross-section")
                        else:
                            print(f"  Cross-section doesn't appear to have multiple cells (variance_ratio={variance_ratio:.2f})")


                
                # Store data
                cross_sections.append(points_2D)
                
                # Create a modified section object containing only the filtered points
                # Get proper ordering for the points around the circumference
                try:
                    # Find center of the points
                    center_point = np.mean(points_2D, axis=0)
                    
                    # Order points by their angle around this center
                    angles = np.arctan2(points_2D[:, 1] - center_point[1], 
                                    points_2D[:, 0] - center_point[0])
                    sorted_indices = np.argsort(angles)
                    sorted_points = points_2D[sorted_indices]
                    
                    # Create a new path entity for the processed section
                    entity = section.entities[0].copy()
                    entity.points = np.arange(len(sorted_points))
                    
                    # Store this modified section for visualization
                    modified_section = section.copy()
                    modified_section.entities = [entity]
                    modified_section.vertices = sorted_points
                except Exception as e:
                    modified_section = section  # Fall back to original section
                
                section_objects.append(modified_section)
                valid_sections.append(True)
                print(f"  Section at position {position:.2f} successful")
            else:
                print(f"  No intersection at position {position:.2f}")
                cross_sections.append(None)
                section_objects.append(None)
                valid_sections.append(False)
        except Exception as e:
            print(f"  Error at position {position:.2f}: {str(e)}")
            cross_sections.append(None)
            section_objects.append(None)
            valid_sections.append(False)
    
    # 6. Create visualizations
    if visualize:
        csf.create_visualizations(mesh, centerline_points, tangent_vectors, section_positions, 
                             cross_sections, section_objects, raw_centerline_points, 
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
    closed_stomata=args.closed_stomata  # Pass this parameter through
    )
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
# Ensure this import is at the top of the file
from scipy.spatial import ConvexHull

# Replace the cross-section processing code in both overlay and individual plots

def process_cross_section(section, section_points):
    """
    Process a cross-section to get properly ordered points that follow the shape contour.
    
    Parameters:
    -----------
    section : trimesh.path.Path3D or trimesh.path.Path2D
        The section object (can be either 3D or our custom 2D version)
    section_points : array
        The 2D points from path_2D.vertices
        
    Returns:
    --------
    list
        List of arrays, each containing ordered points for a continuous segment
    """
    # Special case handling for simple sections
    if len(section_points) <= 3:
        return [section_points]
    
    # First method: Try to extract entity connectivity directly
    try:
        # Add the intended logic here or remove the try block if unnecessary
        pass
    except Exception as e:
        print(f"  Error in KDE analysis: {e}")
    try:
        # Check if this is a Path2D object (our manually created one)
        if hasattr(section, 'entities') and not hasattr(section, 'to_planar'):
            # It's already a Path2D object, use its entities directly
            for entity in section.entities:
                if hasattr(entity, 'points') and len(entity.points) >= 2:
            path_2D, _ = section.to_planar()  # Replace unused variable 'transform' with '_'
                    segments.append(segment)
        # Check if it's a Path3D object (original section)
        elif hasattr(section, 'to_planar'):
            # Convert to 2D for processing
            path_2D, transform = section.to_planar()
            
            if hasattr(path_2D, 'entities'):
                for entity in path_2D.entities:
                    if hasattr(entity, 'points') and len(entity.points) >= 2:
                        segment = path_2D.vertices[entity.points]
    except Exception as e:
        print(f"  Error in KDE analysis: {e}")
        
        # If we found valid segments, return them
        if segments and len(segments) > 0:
            return segments
    except Exception as e:
        print(f"  Warning: Could not extract entity connectivity: {e}")
    
    # Fallback: Use proximity-based ordering if entity extraction failed
    print(f"  Using proximity-based ordering for {len(section_points)} points")
    from scipy.spatial import distance_matrix
    
    # Calculate distances between all points
    dists = distance_matrix(section_points, section_points)
    
    # Identify points that are connected (close to each other)
    # Use a threshold based on the average distance to nearest neighbors
    threshold = np.median(np.sort(dists, axis=1)[:, 1]) * 2
    
    # Start with the first point
    ordered_points = [0]
    unvisited = set(range(1, len(section_points)))
    
    # Keep adding the closest unvisited point
    while unvisited:
        current = ordered_points[-1]
        # Find the closest unvisited point to the current one
        distances = dists[current]
        closest = None
        min_dist = float('inf')
        
        for idx in unvisited:
            if distances[idx] < min_dist:
                min_dist = distances[idx]
                closest = idx
        
        # If the closest point is too far, this is a gap - start a new segment
        if min_dist > threshold:
            # Try to connect back to the start to close the loop
            if len(ordered_points) > 2 and dists[current, ordered_points[0]] <= threshold:
                # Close the loop by adding the first point again
                ordered_points.append(ordered_points[0])
            
            # If there are remaining points, start a new segment
            if unvisited:
                next_start = list(unvisited)[0]
                ordered_points.append(-1)  # Use -1 as a separator
                ordered_points.append(next_start)
                unvisited.remove(next_start)
        else:
            # Add the closest point to our ordered list
            ordered_points.append(closest)
            unvisited.remove(closest)
    
    # Split into continuous segments
    segments = []
    current_segment = []
    
    for idx in ordered_points:
        if idx == -1:
            # End of segment
            if current_segment:
                segments.append(section_points[current_segment])
                current_segment = []
        else:
            current_segment.append(idx)
            
    # Add the last segment if it exists
    if current_segment:
        segments.append(section_points[current_segment])
    
    # If we didn't find any segments, fall back to the original points
    if not segments:
        segments = [section_points]
        
    return segments

def analyze_stomata_cross_sections(torus_path, num_sections=16, visualize=True, output_dir=None):
    """
    Analyze cross-sections of a stomata mesh at evenly spaced positions.

    Parameters:
    -----------
    file_number : str or int
        The file number to analyze (e.g., "1_2" for Ac_DA_1_2.obj)
    num_sections : int, optional
        Number of evenly spaced cross-sections to generate (default: 16)
    visualize : bool, optional
        Whether to create visualization plots
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    tuple
        (cross_sections, positions, centerline_points)
    """
    
    # Call the modified implementation
    return analyze_torus_ring_sections_fixed(
        torus_path=torus_path,
        num_sections=num_sections,
        visualize=visualize,
        output_dir=output_dir
    )


def analyze_torus_ring_sections_fixed(torus_path, num_sections=16, custom_angles=None, visualize=True, output_dir=None):
    """
    Fixed version that correctly isolates cross-sections and visualizes them properly.
    
    Parameters:
    -----------
    torus_path : str
        Path to the mesh file
    num_sections : int, optional
        Number of evenly spaced sections to take (used only if custom_angles is None)
    custom_angles : array-like, optional
        Specific angles in radians where cross-sections should be taken
    visualize : bool, optional
        Whether to create visualization plots
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    tuple
        (cross_sections, positions, centerline_points)
    """
    import trimesh
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import ConvexHull
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the mesh
    print(f"Loading mesh from {torus_path}...")
    mesh = trimesh.load_mesh(torus_path)
    verts = mesh.vertices
    
    # 2. Find the center
    center = mesh.centroid
    print(f"Mesh center: {center}")
    
    # 3. Improved ray casting to find torus dimensions with correction for shape
    print("Determining torus dimensions using ray casting...")
    
    n_rays = 72  # More rays for better accuracy
    ray_angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    
    # Arrays to store inner and outer intersection points
    inner_points = []
    outer_points = []
    all_intersections = []  # Store all intersections for diagnostics
    
    for angle in ray_angles:
        # Ray direction in XY plane
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        
        # Cast the ray from center
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[center],
            ray_directions=[direction],
            multiple_hits=True
        )
        
        if len(locations) >= 2:
            # Calculate distances from center
            distances = np.linalg.norm(locations - center, axis=1)
            
            # Sort by distance
            sorted_indices = np.argsort(distances)
            sorted_locations = locations[sorted_indices]
            
            # Store inner and outer points
            inner_points.append(sorted_locations[0])
            outer_points.append(sorted_locations[-1])
            
            # Store all intersections for visualization
            all_intersections.extend([(loc, angle) for loc in sorted_locations])
    
    if len(inner_points) == 0 or len(outer_points) == 0:
        print("Error: Could not determine torus dimensions. Ray casting failed.")
        return None, None, None
    
    # Convert to numpy arrays
    inner_points = np.array(inner_points)
    outer_points = np.array(outer_points)
    
    # Calculate average inner and outer radii
    inner_radii = np.linalg.norm(inner_points[:, :2] - center[:2], axis=1)  # XY plane only
    outer_radii = np.linalg.norm(outer_points[:, :2] - center[:2], axis=1)
    
    avg_inner_radius = np.mean(inner_radii)
    avg_outer_radius = np.mean(outer_radii)
    
    # Calculate major and minor radii
    major_radius = (avg_inner_radius + avg_outer_radius) / 2
    minor_radius = (avg_outer_radius - avg_inner_radius) / 2
    
    print(f"Estimated major radius (R): {major_radius:.4f}")
    print(f"Estimated minor radius (r): {minor_radius:.4f}")
    
    # 4. Define centerline by ray casting
    # Create the raw centerline first by collecting all intersection midpoints
    angles_to_positions = {}
    for loc, angle in all_intersections:
        if angle not in angles_to_positions:
            angles_to_positions[angle] = []
        angles_to_positions[angle].append(loc)
    
    # Collect all centerline points from ray casting
    raw_centerline = []
    for angle in sorted(angles_to_positions.keys()):
        positions = angles_to_positions[angle]
        if len(positions) >= 2:
            # Calculate the midpoint between inner and outer surface
            distances = np.linalg.norm(positions - center, axis=1)
            sorted_indices = np.argsort(distances)
            inner_pt = positions[sorted_indices[0]]
            outer_pt = positions[sorted_indices[-1]]
            
            # Find midpoint of the torus ring
            midpoint = (inner_pt + outer_pt) / 2
            raw_centerline.append(midpoint)
    
    raw_centerline = np.array(raw_centerline)

    # Fit an ellipse to improve centerline for elliptical stomata
    try:
        from scipy.optimize import curve_fit
        
        # Prepare data for ellipse fitting - project to XY plane
        points_2D = raw_centerline[:, :2]
        center_2D = center[:2]
        
        # Convert to polar coordinates relative to estimated center
        r = np.sqrt(np.sum((points_2D - center_2D)**2, axis=1))
        theta = np.arctan2(points_2D[:, 1] - center_2D[1], points_2D[:, 0] - center_2D[0])
        
        # Sort by angle
        sort_idx = np.argsort(theta)
        theta = theta[sort_idx]
        r = r[sort_idx]
        
        # Define ellipse function
        def ellipse(t, a, b, phi):
            """Parametric ellipse with semi-major axis a, semi-minor axis b, and rotation phi"""
            return a * b / np.sqrt((b * np.cos(t - phi))**2 + (a * np.sin(t - phi))**2)
        
        # Fit ellipse parameters
        initial_guess = [major_radius, major_radius*0.8, 0]
        try:
            params, _ = curve_fit(ellipse, theta, r, p0=initial_guess)
            a, b, phi = params
            print(f"Fitted ellipse: a={a:.4f}, b={b:.4f}, rotation={np.degrees(phi):.1f}°")
            
            # Create improved centerline based on the ellipse
            refined_centerline = []
            refined_angles = np.linspace(0, 2*np.pi, len(raw_centerline), endpoint=False)
            for angle in refined_angles:
                # Calculate radius at this angle
                r_ellipse = ellipse(angle, a, b, phi)
                # Convert back to Cartesian
                x = center[0] + r_ellipse * np.cos(angle)
                y = center[1] + r_ellipse * np.sin(angle)
                z = np.mean(raw_centerline[:, 2])  # Maintain average Z height
                refined_centerline.append([x, y, z])
            
            # Use refined centerline if it looks good
            # (Check if it's reasonably close to original points)
            refined_centerline = np.array(refined_centerline)
            avg_distance = np.mean(
                [np.min(np.linalg.norm(refined_centerline - p, axis=1)) for p in raw_centerline]
            )
            if avg_distance < minor_radius:
                print(f"Using ellipse-fitted centerline (avg distance: {avg_distance:.4f})")
                raw_centerline = refined_centerline
            else:
                print(f"Keeping original centerline (fitted ellipse too far: {avg_distance:.4f})")
        except:
            print("Ellipse fitting failed, using original centerline")
    except:
        print("Could not perform ellipse fitting, continuing with raw centerline")
    
    # Calculate distances along the centerline
    centerline_distances = [0.0]
    for i in range(1, len(raw_centerline)):
        dist = np.linalg.norm(raw_centerline[i] - raw_centerline[i-1])
        centerline_distances.append(centerline_distances[-1] + dist)
    
    # Total path length
    total_path_length = centerline_distances[-1]
    print(f"Total centerline path length: {total_path_length:.2f}")
    
    # Generate evenly spaced positions along the centerline
    centerline_points = []
    tangent_vectors = []
    section_positions = []
    
    for i in range(num_sections):
        # Calculate target distance along the path (evenly spaced)
        target_distance = (i / num_sections) * total_path_length
        
        # Find closest point on the centerline
        closest_idx = np.abs(np.array(centerline_distances) - target_distance).argmin()
        point = raw_centerline[closest_idx]
        
        # Get neighboring points for tangent calculation
        prev_idx = (closest_idx - 1) % len(raw_centerline)
        next_idx = (closest_idx + 1) % len(raw_centerline)
        
        # Calculate tangent from neighbors
        tangent = raw_centerline[next_idx] - raw_centerline[prev_idx]
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
        else:
            # Fallback to geometric approach
            # Calculate tangent based on center and current point
            radial_dir = point[:2] - center[:2]
            if np.linalg.norm(radial_dir) > 0:
                radial_dir = radial_dir / np.linalg.norm(radial_dir)
                tangent = np.array([-radial_dir[1], radial_dir[0], 0.0])
            else:
                # Complete fallback
                tangent = np.array([0, 1, 0])
        
        centerline_points.append(point)
        tangent_vectors.append(tangent)
        section_positions.append(target_distance / total_path_length)  # Normalize to [0,1]
    
    centerline_points = np.array(centerline_points)
    tangent_vectors = np.array(tangent_vectors)
   
    
    # 5. Generate cross-sections with proper filtering
    cross_sections = []
    section_objects = []  # Store the original section objects too
    valid_sections = []

    print(f"Taking {num_sections} cross-sections around the torus ring centerline...")
    for i, (point, tangent, position) in enumerate(zip(centerline_points, tangent_vectors, section_positions)):
        try:
            # Create a local bounding box around the current point to isolate this part of the mesh
            # Calculate local thickness for this specific position instead of using global minor radius
            # Find nearest inner and outer points
            inner_dists = np.linalg.norm(inner_points - point, axis=1)
            outer_dists = np.linalg.norm(outer_points - point, axis=1)
            nearest_inner = inner_points[np.argmin(inner_dists)]
            nearest_outer = outer_points[np.argmin(outer_dists)]
            
            # Calculate local thickness
            local_thickness = np.linalg.norm(nearest_outer - nearest_inner)
            
            # Use local thickness but don't go below minimum threshold
            box_size = max(local_thickness * 1.25, minor_radius * 2.0)
            print(f"  Using local box size: {box_size:.4f} at position {position:.2f}")
        except:
            # Fallback to global minor radius
            box_size = minor_radius * 2.5  # Just large enough for the local cross-section
            
        # Find faces within the box_size distance from the point
        from scipy.spatial import KDTree
        tree = KDTree(mesh.vertices)
        indices = tree.query_ball_point(point, box_size)
        if indices:
            # Find all faces that use any of these vertices
            local_face_indices = np.unique(np.concatenate([
                np.where(np.any(np.isin(mesh.faces, idx), axis=1))[0] 
                for idx in indices
            ])) if indices else []
        else:
            local_face_indices = []
        
        if len(local_face_indices) == 0:
            print(f"  No faces found near position {position:.2f}")
            cross_sections.append(None)
            section_objects.append(None)
            valid_sections.append(False)
            continue
        
        # Create submesh with only these faces
        local_mesh = mesh.submesh([local_face_indices], append=True)
        
        # Now section only this local mesh
        try:
            section = local_mesh.section(plane_origin=point, plane_normal=tangent)
            
            if section is not None and len(section.entities) > 0:
                # Convert to 2D coordinates
                path_2D, transform = section.to_planar()
                
                # Get all closed paths
                points_2D = path_2D.vertices
                
                # The key insight: in the planar space after to_planar(),
                # the centerline point becomes the origin (0,0)
                # So the cross-section we want should be centered near the origin
                
                # Use DBSCAN to identify distinct cross-sections
                from sklearn.cluster import DBSCAN
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
                
                # Check if this might be a fully closed stomata (both guard cells in one cluster)
                if len(points_2D) > 0:
                    # Calculate convexity and area to detect potentially merged guard cells
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(points_2D)
                        hull_area = hull.volume  # In 2D, volume is actually area

                        # First ensure points are properly ordered for area calculation
                        center_point = np.mean(points_2D, axis=0)
                        angles = np.arctan2(points_2D[:, 1] - center_point[1], points_2D[:, 0] - center_point[0])
                        sorted_indices = np.argsort(angles)
                        ordered_pts = points_2D[sorted_indices]

                        # Calculate area using ordered points
                        actual_area = 0
                        for j in range(len(ordered_pts)):
                            x1, y1 = ordered_pts[j]
                            x2, y2 = ordered_pts[(j+1) % len(ordered_pts)]
                            actual_area += 0.5 * abs(x1*y2 - x2*y1)
                        
                        # Calculate convexity ratio
                        convexity = actual_area / hull_area if hull_area > 0 else 1.0                        
                        print(f"  Cross-section convexity: {convexity:.2f}")
                        print(f"  DEBUG: Hull area: {hull_area:.2f}, Actual area: {actual_area:.2f}")
                        
                        # Calculate additional shape metrics
                        # 1. Find the bounding box and check aspect ratio
                        from sklearn.decomposition import PCA
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
                        
                        # Detect merged cells if specific criteria are met
                        if ((convexity < 0.45) and (variance_ratio > 3.5 or aspect_ratio > 2.5)):
                            print(f"  Possible merged guard cells detected! (convexity: {convexity:.2f}, "
                                f"variance ratio: {variance_ratio:.2f}, aspect ratio: {aspect_ratio:.2f})")
                            
                            # Find the best splitting line using PCA
                            main_axis = pca.components_[0]  # Primary axis
                            
                            # Project points onto the main axis
                            projections = points_2D @ main_axis
                            
                            # Find bimodality using multiple methods
                            from scipy.signal import find_peaks
                            from scipy.stats import gaussian_kde
                            
                            # Method 1: Histogram peaks
                            hist, bin_edges = np.histogram(projections, bins=30)  # More bins
                            neg_hist = -hist
                            peaks, _ = find_peaks(neg_hist)
                            
                            # Check for significant valleys in the histogram (evidence of bimodality)
                            has_significant_valley = False
                            if len(peaks) > 0:
                                hist_max = np.max(hist)
                                peak_depths = hist_max - hist[peaks]
                                if np.max(peak_depths) > 0.4 * hist_max:  # Valley at least 40% deep
                                    has_significant_valley = True
                            
                            if has_significant_valley:
                                # Method 2: KDE for smoother distribution
                                try:
                                    kde = gaussian_kde(projections)
                                    x = np.linspace(min(projections), max(projections), 100)
                                    y = kde(x)
                                    # Find valleys in KDE
                                    neg_kde = -y
                                    kde_peaks, _ = find_peaks(neg_kde)
                                    if len(kde_peaks) > 0:
                                        # Use KDE valley as split point
                                        split_point = x[kde_peaks[np.argmax(neg_kde[kde_peaks])]]
                                        print(f"  Using KDE-based split point: {split_point:.4f}")
                                    elif len(peaks) > 0:
                                        # Fall back to histogram-based split
                                        deepest = peaks[np.argmax(neg_hist[peaks])]
                                        split_point = bin_edges[deepest:deepest+2].mean()
                                        print(f"  Using histogram-based split point: {split_point:.4f}")
                                    else:
                                        # Just split in the middle if no peaks found
                                        split_point = (max(projections) + min(projections)) / 2
                                        print(f"  Using midpoint split: {split_point:.4f}")
                                    
                                    # Split the points
                                    left_half = points_2D[projections < split_point]
                                    right_half = points_2D[projections >= split_point]
                                    
                                    # Use the half with more points or based on position
                                    if len(left_half) > 0 and len(right_half) > 0:
                                        # Choose based on position in the stomata ring
                                        if position < 0.5:  # First half of the stomata
                                            points_2D = left_half
                                        else:  # Second half of the stomata
                                            points_2D = right_half
                                        print(f"  Split merged guard cells, using half with {len(points_2D)} points")
                                        
                                        # ADDITIONAL CLEANUP: Remove "rabbit ears" using convexity
                                        try:
                                            # Step 1: Find centroid of the half we selected
                                            centroid = np.mean(points_2D, axis=0)
                                            
                                            # Step 2: Calculate distances from each point to centroid
                                            distances = np.linalg.norm(points_2D - centroid, axis=1)
                                            
                                            # Step 3: Find the median distance
                                            median_dist = np.median(distances)
                                            
                                            # Step 4: Filter out points that are far from centroid
                                            close_points = points_2D[distances <= 1.5 * median_dist]
                                            
                                            # Only use this filtering if we don't lose too many points
                                            if len(close_points) >= len(points_2D) * 0.7:
                                                print(f"  Cleaned up {len(points_2D) - len(close_points)} potential 'rabbit ear' points")
                                                points_2D = close_points
                                            
                                            # This requires scipy's ConvexHull as a simpler alternative to alpha shapes
                                            if len(points_2D) > 5:  # Need at least 3 points for ConvexHull
                                                from matplotlib.path import Path
                                                hull = ConvexHull(points_2D)
                                                # Get a slightly smaller convex hull by scaling around centroid
                                                scale_factor = 0.7  # More aggressive scaling
                                                hull_points = points_2D[hull.vertices]
                                                scaled_hull_points = centroid + (hull_points - centroid) * scale_factor
                                                
                                                # Use only points inside this smaller hull
                                                hull_path = Path(scaled_hull_points)
                                                inside_mask = hull_path.contains_points(points_2D)
                                                inside_points = points_2D[inside_mask]
                                                
                                                # Use this filtered set if we keep enough points
                                                if len(inside_points) >= len(points_2D) * 0.6:
                                                    print(f"  Further filtered to {len(inside_points)} points using convex hull")
                                                    points_2D = inside_points
                                        except Exception as e:
                                            print(f"  Error during rabbit ear cleanup: {e}")
                                        
                                        # Reorder the points for proper connectivity
                                        center_point = np.mean(points_2D, axis=0)
                                        angles = np.arctan2(points_2D[:, 1] - center_point[1], 
                                                        points_2D[:, 0] - center_point[0])
                                        sorted_indices = np.argsort(angles)
                                        points_2D = points_2D[sorted_indices]
                                except Exception as e:
                                    print(f"  Error in KDE analysis: {e}")
                    except Exception as e:
                        print(f"  Error in convex hull analysis: {e}")
                                
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
                    
                    # Sort points by angle to get proper circumferential ordering
                    sorted_indices = np.argsort(angles)
                    ordered_points = points_2D[sorted_indices]
                    
                    # Create a simplified Path2D with properly ordered points
                    from trimesh.path.entities import Line
                    from trimesh.path.path import Path2D
                    
                    # Create line segments connecting points in the correct order
                    entities = []
                    for j in range(len(ordered_points)-1):  # Using j instead of i to avoid conflict
                        entities.append(Line([j, j+1]))
                    
                    # Close the loop
                    entities.append(Line([len(ordered_points)-1, 0]))
                    
                    # Create a new path with the ordered points
                    filtered_path2d = Path2D(entities=entities, vertices=ordered_points)
                    section_objects.append(filtered_path2d)
                    
                    # Replace the original array of points with the ordered ones
                    cross_sections[-1] = ordered_points
                
                except Exception as e:
                    print(f"  Warning: Could not create ordered section object: {e}")
                    section_objects.append(None)  # Don't use a bad object
                
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
        
    # 6. Advanced visualization with diagnostic information
    if visualize and sum(valid_sections) > 0:
        fig = plt.figure(figsize=(18, 12))
        
        # 6a. 3D visualization with more detail
        ax3d = fig.add_subplot(231, projection='3d')
        
        # Plot a subset of the mesh faces
        face_count = min(len(mesh.faces), 3000)
        face_indices = np.random.choice(len(mesh.faces), face_count, replace=False)
        ax3d.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=mesh.faces[face_indices], alpha=0.2, color='lightblue')
        
        # Plot inner and outer points
        ax3d.scatter(inner_points[:, 0], inner_points[:, 1], inner_points[:, 2], 
                color='blue', s=10, alpha=0.5)
        ax3d.scatter(outer_points[:, 0], outer_points[:, 1], outer_points[:, 2], 
                color='purple', s=10, alpha=0.5)
        
        # Plot the centerline
        ax3d.plot(centerline_points[:, 0], centerline_points[:, 1], centerline_points[:, 2], 
                'r-', linewidth=3, label='Torus Ring Centerline')
        for point, tangent in zip(centerline_points, tangent_vectors):
            # Scale the tangent vector for visibility
            scale = minor_radius * 0.8
            ax3d.quiver(point[0], point[1], point[2],
                    tangent[0], tangent[1], tangent[2],
                    color='orange', length=scale, normalize=True)
        
        # Plot cross-section planes
        for i, (point, tangent, valid) in enumerate(zip(centerline_points, tangent_vectors, valid_sections)):
            if not valid:
                continue
                
            # Draw section point and normal
            ax3d.scatter(point[0], point[1], point[2], color='green', s=50)
            
            # For key angles, show the section plane
            if i % (num_sections // 4) == 0:
                # Find perpendicular vectors to create a visualization of the plane
                up = np.array([0, 0, 1])
                up = up - np.dot(up, tangent) * tangent
                up = up / np.linalg.norm(up)
                
                # Third orthogonal vector 
                third = np.cross(tangent, up)
                third = third / np.linalg.norm(third)
                
                # Create a grid to represent the plane
                plane_size = minor_radius * 2
                grid_points = []
                
                for u in np.linspace(-plane_size, plane_size, 5):
                    for v in np.linspace(-plane_size, plane_size, 5):
                        grid_point = point + u * up + v * third
                        grid_points.append(grid_point)
                
                grid_points = np.array(grid_points)
                ax3d.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], 
                        color='green', s=10, alpha=0.3)
                
                # Add a label
                ax3d.text(point[0], point[1], point[2], f"Pos {section_positions[i]:.2f}", 
                    color='black', fontsize=9)
        
        ax3d.set_title('Torus with Corrected Cross-Section Planes')
        ax3d.legend(loc='upper right', fontsize=8)
        
        # 6b. Top view with enhanced details
        ax_top = fig.add_subplot(232)
        ax_top.set_title('Top View with Section Lines')
        
        # Plot points in XY projection
        ax_top.scatter(verts[:, 0], verts[:, 1], s=1, alpha=0.1, color='lightgray')
        
        # Plot inner and outer detected edges
        ax_top.scatter(inner_points[:, 0], inner_points[:, 1], color='blue', s=20, alpha=0.5, label='Inner Edge')
        ax_top.scatter(outer_points[:, 0], outer_points[:, 1], color='purple', s=20, alpha=0.5, label='Outer Edge')
        
        # Plot the centerline
        ax_top.plot(centerline_points[:, 0], centerline_points[:, 1], 'r-', linewidth=2, label='Ring Centerline')
        
        # Process and center all sections
        processed_sections = []
        centers = []
        max_extent = 0

        # FIRST populate the processed_sections list BEFORE using it
        for i, (section_points, valid) in enumerate(zip(cross_sections, valid_sections)):
            if not valid or section_points is None:
                processed_sections.append(None)
                centers.append(None)
                continue
            
            # Calculate the center of the section points
            center_pt = np.mean(section_points, axis=0)
            centers.append(center_pt)
            
            # Center the points
            centered = section_points - center_pt
            processed_sections.append(centered)
            
            # Update max extent
            max_extent = max(max_extent, np.max(np.abs(centered)))
        # Draw the cross-section planes
        for i, (point, tangent, valid) in enumerate(zip(centerline_points, tangent_vectors, valid_sections)):
            if not valid or cross_sections[i] is None:
                continue
                
            # Calculate a perpendicular direction in XY plane
            perp = np.array([-tangent[1], tangent[0], 0])
            
            # Calculate the actual width of this specific cross-section
            # Get the 2D points from the cross-section at this position
            points_2D = processed_sections[i] if processed_sections[i] is not None else cross_sections[i]
            
            if points_2D is not None and len(points_2D) > 0:
                # Project points onto the perpendicular direction
                proj = points_2D @ np.array([perp[0], perp[1]])
                min_proj = np.min(proj)
                max_proj = np.max(proj)
                
                # Add a small margin (10%)
                span = max_proj - min_proj
                margin = span * 0.1
                
                # Draw line that spans just the width of the actual cross-section
                scale_min = min_proj - margin
                scale_max = max_proj + margin
                
                p1 = point[:2] + perp[:2] * scale_min
                p2 = point[:2] + perp[:2] * scale_max
                ax_top.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)
            else:
                # Fallback to a shorter default if no cross-section data
                line_length = minor_radius * 1.2
                p1 = point[:2] + perp[:2] * line_length
                p2 = point[:2] - perp[:2] * line_length
                ax_top.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', linewidth=1, alpha=0.5)
            
            # Add labels for key points
            if i % (num_sections // 4) == 0:
                ax_top.scatter(point[0], point[1], color='green', s=50)
                ax_top.text(point[0], point[1], f"Pos {section_positions[i]:.2f}", 
                fontsize=9, ha='center', va='center')
        
        ax_top.set_aspect('equal')
        ax_top.grid(True)
        ax_top.legend(fontsize=8)
        
        # 6c. RAW CROSS-SECTION DATA - Show the unprocessed section for debugging
        ax_raw = fig.add_subplot(233)
        ax_raw.set_title('Raw Section Data at Starting Position')
        
        # Show the raw data for the 0° section
        zero_idx = np.argmin(np.abs(section_positions))  # Section closest to starting position
        if valid_sections[zero_idx] and cross_sections[zero_idx] is not None:
            raw_points = cross_sections[zero_idx]
            ax_raw.scatter(raw_points[:, 0], raw_points[:, 1], s=15, c='blue', alpha=0.7)
            
            # Add indices to points to see their ordering
            for i, (x, y) in enumerate(raw_points):
                if i % 10 == 0:  # Label every 10th point for clarity
                    ax_raw.text(x, y, str(i), fontsize=8)
            
            # Try to connect the points to show the actual path
            # Be careful with the ordering - it might not be consecutive
            # Try to fit a convex hull to see the rough shape
            try:
                hull = ConvexHull(raw_points)
                for simplex in hull.simplices:
                    ax_raw.plot(raw_points[simplex, 0], raw_points[simplex, 1], 'r-', alpha=0.5)
            except:
                # If convex hull fails, just connect points based on their order
                ax_raw.plot(raw_points[:, 0], raw_points[:, 1], 'b--', alpha=0.3)
            
            ax_raw.set_aspect('equal')
            ax_raw.grid(True)
            
            # Show a clean circle for reference
            circle_theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.mean(raw_points[:, 0]) + minor_radius * np.cos(circle_theta)
            circle_y = np.mean(raw_points[:, 1]) + minor_radius * np.sin(circle_theta)
            ax_raw.plot(circle_x, circle_y, 'g--', linewidth=1, alpha=0.7)
            
            # Set consistent limits
            margin = np.max(np.ptp(raw_points, axis=0)) * 0.2
            ax_raw.set_xlim(np.min(raw_points[:, 0]) - margin, np.max(raw_points[:, 0]) + margin)
            ax_raw.set_ylim(np.min(raw_points[:, 1]) - margin, np.max(raw_points[:, 1]) + margin)
            
        # 6d. Cross-sections with improved connectivity
        ax_all = fig.add_subplot(234)
        ax_all.set_title('Processed Cross-Sections (Properly Connected)')

        # Plot a reference circle
        circle_theta = np.linspace(0, 2*np.pi, 100)
        circle_x = minor_radius * np.cos(circle_theta)
        circle_y = minor_radius * np.sin(circle_theta)
        ax_all.plot(circle_x, circle_y, 'k--', linewidth=2, alpha=0.5, label='Reference Circle')

        # Plot each processed section with proper connectivity
        for i, (section_obj, centered, valid) in enumerate(zip(section_objects, processed_sections, valid_sections)):
            if not valid or centered is None or section_obj is None:
                continue
                
            # Use angle-based color
            color = plt.cm.hsv(i / num_sections)
            
            try:
                # Process the cross section to get properly ordered segments
                segments = process_cross_section(section_obj, centered)
                
                # Plot each segment with proper connectivity
                for segment in segments:
                    ax_all.plot(segment[:, 0], segment[:, 1], '-', color=color, linewidth=1, alpha=0.7)
                
                # Label key sections
                if i % (num_sections // 4) == 0 and segments and len(segments[0]) > 0:
                    ax_all.scatter(segments[0][0, 0], segments[0][0, 1], 
                    color=color, s=30, 
                    label=f"Pos {section_positions[i]:.2f}")
            except Exception as e:
                print(f"  Warning: Could not process section at {section_positions[i]:.2f}: {e}")
                # Fallback to scatter plot if processing fails
                ax_all.scatter(centered[:, 0], centered[:, 1], s=5, color=color, alpha=0.5)

        ax_all.set_aspect('equal')
        ax_all.grid(True)
        #ax_all.set_xlim(-max_extent*1.2, max_extent*1.2)
        #ax_all.set_ylim(-max_extent*1.2, max_extent*1.2)
        ax_all.legend(loc='upper right', fontsize=8)
        
        # 6e. Two most important individual cross-sections (0° and 90°)
        positions_to_show = [0.0, 0.25]  # At start and 1/4 around the torus
        subplot_positions = [5, 6]

        for idx, target_pos in enumerate(positions_to_show):
            # Find closest section to this position
            i = np.argmin(np.abs(np.array(section_positions) - target_pos))
            
            if valid_sections[i] and processed_sections[i] is not None and section_objects[i] is not None:
                # Create subplot
                ax = fig.add_subplot(2, 3, subplot_positions[idx])
                
                # Get processed section
                processed = processed_sections[i]
                section_obj = section_objects[i]
                
                try:
                    # Get properly ordered segments using the process_cross_section function
                    segments = process_cross_section(section_obj, processed)
                    
                    # Compute aspect ratio from the overall bounds
                    all_points = np.vstack(segments) if segments else processed
                    min_vals = np.min(all_points, axis=0)
                    max_vals = np.max(all_points, axis=0)
                    width = max_vals[0] - min_vals[0]
                    height = max_vals[1] - min_vals[1]
                    aspect = width / height if height > 0 else float('inf')
                    
                    # Plot each segment with proper connectivity
                    color = plt.cm.hsv(i / num_sections)
                    for segment in segments:
                        ax.plot(segment[:, 0], segment[:, 1], '-', color=color, linewidth=2)
                    
                    # Add reference circle
                    ax.plot(circle_x, circle_y, 'k--', linewidth=1, alpha=0.5)
                    
                    # Add diagnostic info
                    total_points = sum(len(s) for s in segments)
                    ax.set_title(f"Position {section_positions[i]:.2f} - {total_points} points, AR={aspect:.2f}")
                except Exception as e:
                    # Fallback to scatter plot
                    ax.scatter(processed[:, 0], processed[:, 1], s=5, alpha=0.7)
                    ax.set_title(f"Position {section_positions[i]:.2f} - Error processing")
                    
                ax.set_aspect('equal')
                ax.grid(True)
                #ax.set_xlim(-max_extent*1.2, max_extent*1.2)
                #ax.set_ylim(-max_extent*1.2, max_extent*1.2)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "torus_ring_sections_fixed.png"), dpi=200)
        
        plt.show()

        # 6. Additional figure showing all cross-sections individually
        if visualize and sum(valid_sections) > 0:
            # Calculate grid dimensions for subplots
            n_valid = sum(valid_sections)
            cols = min(4, n_valid)  # Maximum 4 columns
            rows = (n_valid + cols - 1) // cols  # Ceiling division
            
            # Create a new figure for all cross-sections
            fig_all = plt.figure(figsize=(16, 3 * rows))
            fig_all.suptitle("All Cross-Sections", fontsize=16)
            
            # Reference circle for comparison
            circle_theta = np.linspace(0, 2*np.pi, 100)
            circle_x = minor_radius * np.cos(circle_theta)
            circle_y = minor_radius * np.sin(circle_theta)
            
            # Plot each cross-section in its own subplot
            plot_idx = 1
            for i, (section_obj, processed, valid) in enumerate(zip(section_objects, processed_sections, valid_sections)):
                if not valid or processed is None or section_obj is None:
                    continue
                    
                ax = fig_all.add_subplot(rows, cols, plot_idx)
                plot_idx += 1
                
                try:
                    # Get properly ordered segments
                    segments = process_cross_section(section_obj, processed)
                    
                    # Compute aspect ratio
                    all_points = np.vstack(segments) if segments else processed
                    min_vals = np.min(all_points, axis=0)
                    max_vals = np.max(all_points, axis=0)
                    width = max_vals[0] - min_vals[0]
                    height = max_vals[1] - min_vals[1]
                    aspect = width / height if height > 0 else float('inf')
                    
                    # Plot each segment with proper connectivity
                    color = plt.cm.hsv(i / num_sections)
                    for segment in segments:
                        ax.plot(segment[:, 0], segment[:, 1], '-', color=color, linewidth=2)
                    
                    # Add reference circle
                    ax.plot(circle_x, circle_y, 'k--', linewidth=1, alpha=0.5)
                    
                    # Show position info and aspect ratio
                    position = section_positions[i]
                    ax.set_title(f"Position: {position:.2f}\nAR: {aspect:.2f}")
                    
                except Exception as e:
                    # Fallback to scatter plot
                    ax.scatter(processed[:, 0], processed[:, 1], s=5, alpha=0.7)
                    ax.set_title(f"Position: {section_positions[i]:.2f}\nError processing")
                    
                ax.set_aspect('equal')
                ax.grid(True)
                #ax.set_xlim(-max_extent*1.2, max_extent*1.2)
                #ax.set_ylim(-max_extent*1.2, max_extent*1.2)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
            
            if output_dir:
                all_sections_filename = os.path.join(output_dir, "all_cross_sections.png")
                fig_all.savefig(all_sections_filename, dpi=200)
                print(f"All cross-sections visualization saved to: {all_sections_filename}")
                
            plt.show()
    
    # Return the cross-sections, angles used, and centerline points
    return cross_sections, section_positions, centerline_points

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze stomata cross-sections')
    parser.add_argument('torus_path', help='Mesh path (e.g., "Ac_DA_1_2.obj")')
    parser.add_argument('--num-sections', type=int, default=16, 
                        help='Number of evenly spaced cross-sections (default: 16)')
    parser.add_argument('--no-vis', action='store_false', dest='visualize', 
                        help='Disable visualization')
    parser.add_argument('--output-dir', help='Directory to save output files')
    args = parser.parse_args()
    
    # Call the function with the provided arguments
    cross_sections, positions, centerline_points = analyze_stomata_cross_sections(
        torus_path=args.torus_path,
        num_sections=args.num_sections,
        visualize=args.visualize,
        output_dir=args.output_dir
    )
    
    # Print summary
    valid_sections = [cs is not None for cs in cross_sections]
    print(f"\nAnalysis complete:")
    print(f"- Total cross-sections: {len(cross_sections)}")
    print(f"- Valid cross-sections: {sum(valid_sections)}")
    
    if args.output_dir:
        print(f"- Results saved to: {args.output_dir}")
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# Replace the cross-section processing code in both overlay and individual plots

def process_cross_section(section, section_points):
    """
    Process a cross-section to get properly ordered points that follow the shape contour.
    
    Parameters:
    -----------
    section : trimesh.path.Path3D
        The original section object from mesh.section()
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
        
    # Get the original entities and connectivity
    try:
        # If this fails, we'll fall back to the convex hull approach
        path_2D, transform = section.to_planar()
        
        # Extract the entities (line segments, arcs, etc.)
        segments = []
        
        # Check if we have direct access to path entities
        if hasattr(path_2D, 'entities') and len(path_2D.entities) > 0:
            # Process each entity to extract connected segments
            for entity in path_2D.entities:
                if hasattr(entity, 'points') and len(entity.points) >= 2:
                    # Extract the points that form this entity
                    segment = path_2D.vertices[entity.points]
                    segments.append(segment)
            
            # If we didn't get any segments, try another approach
            if not segments:
                # Try to use discrete entities if available
                if hasattr(path_2D, 'discrete'):
                    for discrete in path_2D.discrete:
                        segments.append(discrete)
            
            # If we found segments, return them
            if segments:
                return segments
    except Exception as e:
        print(f"  Warning: Could not extract entity connectivity: {e}")
    
    # Fallback: If we couldn't extract proper entity information,
    # try to find continuous segments based on proximity
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

def analyze_stomata_cross_sections(file_number, angles_deg=None, visualize=True, output_dir=None):
    """
    Analyze cross-sections of a stomata mesh at specified angles.

    Parameters:
    -----------
    file_number : str or int
        The file number to analyze (e.g., "1_2" for Ac_DA_1_2.obj)
    angles_deg : list of float, optional
        Specific angles in degrees where cross-sections should be taken
        If None, 16 evenly spaced angles will be used
    visualize : bool, optional
        Whet her to create visualization plots
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    tuple
        (cross_sections, angles_rad, centerline_points)
    """
    # Construct the file path
    torus_path = f'/home/tomkinsm/stomata-air-mattress/Ac_DA_{file_number}.obj'
    
    # If angles aren't specified, use evenly spaced angles
    if angles_deg is None:
        num_sections = 16
        angles_rad = np.linspace(0, 2*np.pi, num_sections, endpoint=False)
    else:
        # Convert specified angles from degrees to radians
        angles_rad = np.array([angle * np.pi / 180 for angle in angles_deg])
        num_sections = len(angles_rad)
    
    print(f"Analyzing file: {torus_path}")
    print(f"Taking cross-sections at {num_sections} angles: {[f'{angle:.1f}°' for angle in (angles_rad * 180/np.pi)]}")
    
    # Call the original implementation
    return analyze_torus_ring_sections_fixed(
        torus_path=torus_path,
        custom_angles=angles_rad,  # Pass the angles to the implementation
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
        (cross_sections, angles, centerline_points)
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
        locations, index_ray, index_tri = mesh.ray.intersects_location(
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
            sorted_distances = distances[sorted_indices]
            
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
    
    # 4. Define centerline with correction for the top/bottom positioning
    # Use more accurate shape-based centerline calculation
    if custom_angles is not None:
        theta = custom_angles
    else:
        theta = np.linspace(0, 2*np.pi, num_sections, endpoint=False)
        
    centerline_points = []
    tangent_vectors = []
    
    # Calculate centerline based on actual mesh shape, not just a perfect circle
    angles_to_positions = {}
    for loc, angle in all_intersections:
        if angle not in angles_to_positions:
            angles_to_positions[angle] = []
        angles_to_positions[angle].append(loc)
    
    for t in theta:
        # Find the closest angle we have data for
        closest_angle = min(angles_to_positions.keys(), key=lambda a: abs(a - t))
        positions = angles_to_positions[closest_angle]
        
        if len(positions) >= 2:
            # Calculate the midpoint between inner and outer surface at this angle
            # This gives us the true centerline of the torus ring
            distances = np.linalg.norm(positions - center, axis=1)
            sorted_indices = np.argsort(distances)
            inner_pt = positions[sorted_indices[0]]
            outer_pt = positions[sorted_indices[-1]]
            
            # Find midpoint of the torus ring (not just the global center)
            midpoint = (inner_pt + outer_pt) / 2
            centerline_points.append(midpoint)
            
            # Tangent vector is perpendicular to the radial direction
            radial_dir = midpoint[:2] - center[:2]
            radial_dir = radial_dir / np.linalg.norm(radial_dir)
            tangent = np.array([-radial_dir[1], radial_dir[0], 0.0])
            tangent_vectors.append(tangent)
        else:
            # Fallback to geometric calculation
            x = center[0] + major_radius * np.cos(t)
            y = center[1] + major_radius * np.sin(t)
            z = center[2]
            centerline_points.append([x, y, z])
            
            tangent = np.array([-np.sin(t), np.cos(t), 0.0])
            tangent_vectors.append(tangent)
    
    centerline_points = np.array(centerline_points)
    tangent_vectors = np.array(tangent_vectors)
    
    # 5. Generate cross-sections with proper filtering
    cross_sections = []
    section_objects = []  # Store the original section objects too
    valid_sections = []

    print(f"Taking {len(theta)} cross-sections perpendicular to the torus ring centerline...")
    for i, (point, tangent) in enumerate(zip(centerline_points, tangent_vectors)):
        try:
            # Get section
            section = mesh.section(plane_origin=point, plane_normal=tangent)
            
            if section is not None and len(section.entities) > 0:
                # Convert to 2D coordinates
                path_2D, transform = section.to_planar()
                
                # Get all closed paths
                points_2D = path_2D.vertices
                
                # If we detect a problem with having too many points (potential double intersection)
                if len(points_2D) > 100:  # Arbitrary threshold for "too many points"
                    print(f"  Section at {theta[i]*180/np.pi:.1f}° has {len(points_2D)} points - filtering...")
                    
                    # Try to segment the points into clusters
                    from sklearn.cluster import DBSCAN
                    clustering = DBSCAN(eps=minor_radius*0.5, min_samples=5).fit(points_2D)
                    labels = clustering.labels_
                    
                    # Find the largest cluster
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    if len(unique_labels) > 1 and -1 in unique_labels:
                        # Remove noise points (label -1)
                        valid_labels = unique_labels[unique_labels != -1]
                        valid_counts = counts[unique_labels != -1]
                        largest_cluster = valid_labels[np.argmax(valid_counts)]
                        
                        # Filter to keep only points in the largest cluster
                        filtered_points = points_2D[labels == largest_cluster]
                        print(f"    Filtered from {len(points_2D)} to {len(filtered_points)} points")
                        points_2D = filtered_points
                
                # Store data
                cross_sections.append(points_2D)
                section_objects.append(section)  # Store the original section object
                valid_sections.append(True)
                print(f"  Section at {theta[i]*180/np.pi:.1f}° successful with {len(points_2D)} points")
            else:
                print(f"  No intersection at angle {theta[i]*180/np.pi:.1f}°")
                cross_sections.append(None)
                section_objects.append(None)  # Store None for section object
                valid_sections.append(False)
        except Exception as e:
            print(f"  Error at angle {theta[i]*180/np.pi:.1f}°: {str(e)}")
            cross_sections.append(None)
            section_objects.append(None)  # Store None for section object
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
                ax3d.text(point[0], point[1], point[2], f"{theta[i]*180/np.pi:.0f}°", 
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
        
        # Draw the cross-section planes
        for i, (point, tangent, valid) in enumerate(zip(centerline_points, tangent_vectors, valid_sections)):
            if not valid:
                continue
                
            # Calculate a perpendicular direction in XY plane
            perp = np.array([-tangent[1], tangent[0]])  # 90° rotation in XY
            
            # Draw the section line
            line_length = minor_radius * 3
            p1 = point[:2] + perp * line_length
            p2 = point[:2] - perp * line_length
            ax_top.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)
            
            # Add labels for key points
            if i % (num_sections // 4) == 0:
                ax_top.scatter(point[0], point[1], color='green', s=50)
                ax_top.text(point[0], point[1], f"{theta[i]*180/np.pi:.0f}°", 
                          fontsize=9, ha='center', va='center')
        
        ax_top.set_aspect('equal')
        ax_top.grid(True)
        ax_top.legend(fontsize=8)
        
        # 6c. RAW CROSS-SECTION DATA - Show the unprocessed section for debugging
        ax_raw = fig.add_subplot(233)
        ax_raw.set_title('Raw Section Data at 0°')
        
        # Show the raw data for the 0° section
        zero_idx = np.argmin(np.abs(theta))
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

        # Process and center all sections
        processed_sections = []
        centers = []
        max_extent = 0

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
                                label=f"{theta[i]*180/np.pi:.0f}°")
            except Exception as e:
                print(f"  Warning: Could not process section at {theta[i]*180/np.pi:.1f}°: {e}")
                # Fallback to scatter plot if processing fails
                ax_all.scatter(centered[:, 0], centered[:, 1], s=5, color=color, alpha=0.5)

        ax_all.set_aspect('equal')
        ax_all.grid(True)
        ax_all.set_xlim(-max_extent*1.2, max_extent*1.2)
        ax_all.set_ylim(-max_extent*1.2, max_extent*1.2)
        ax_all.legend(loc='upper right', fontsize=8)
        
        # 6e. Two most important individual cross-sections (0° and 90°)
        angles_to_show = [0, 90]  # degrees
        subplot_positions = [5, 6]

        for idx, angle_deg in enumerate(angles_to_show):
            angle_rad = angle_deg * np.pi / 180
            
            # Find closest section to this angle
            i = np.argmin(np.abs(theta - angle_rad))
            
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
                    ax.set_title(f"{theta[i]*180/np.pi:.1f}° - {total_points} points, AR={aspect:.2f}")
                except Exception as e:
                    # Fallback to scatter plot
                    ax.scatter(processed[:, 0], processed[:, 1], s=5, alpha=0.7)
                    ax.set_title(f"{theta[i]*180/np.pi:.1f}° - Error processing")
                    
                ax.set_aspect('equal')
                ax.grid(True)
                ax.set_xlim(-max_extent*1.2, max_extent*1.2)
                ax.set_ylim(-max_extent*1.2, max_extent*1.2)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "torus_ring_sections_fixed.png"), dpi=200)
        
        plt.show()
    
    # Return the cross-sections, angles used, and centerline points
    return cross_sections, theta, centerline_points

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze stomata cross-sections')
    parser.add_argument('file_number', help='File number (e.g., "1_2" for Ac_DA_1_2.obj)')
    parser.add_argument('--angles', type=float, nargs='+', help='Specific angles (in degrees) for cross-sections')
    parser.add_argument('--no-vis', action='store_false', dest='visualize', help='Disable visualization')
    parser.add_argument('--output-dir', help='Directory to save output files')
    args = parser.parse_args()
    
    # Call the function with the provided arguments
    cross_sections, angles, centerline_points = analyze_stomata_cross_sections(
        file_number=args.file_number,
        angles_deg=args.angles,
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
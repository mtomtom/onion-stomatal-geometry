import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import argparse
from scipy.spatial import KDTree

def ellipse(theta, a, b, phi):
    """Parameterized equation of an ellipse."""
    r = a*b / np.sqrt((b*np.cos(theta-phi))**2 + (a*np.sin(theta-phi))**2)
    return r


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

def calculate_polygon_area(points):
    """
    Calculate the area of a polygon using the shoelace formula.
    
    Args:
        points: ordered array of points forming a polygon
        
    Returns:
        area value
    """
    area = 0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += 0.5 * abs(x1*y2 - x2*y1)
    return area

def calculate_convexity(points):
    """
    Calculate the convexity of a shape (actual area / convex hull area).
    
    Args:
        points: array of 2D points
        
    Returns:
        convexity value (0-1)
    """
    if len(points) < 3:
        return 1.0
        
    try:
        # Calculate convex hull
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is actually area
        
        # Order points for area calculation
        ordered_pts = order_points(points, method="angular")
        
        # Calculate actual area
        actual_area = calculate_polygon_area(ordered_pts)
        
        # Calculate convexity
        return actual_area / hull_area if hull_area > 0 else 1.0
    except:
        return 1.0
    
# Replace/update the detect_gaps function:

def detect_gaps(points, threshold=2.5):
    """
    Detect if there are abnormally large gaps in an ordered sequence of points.
    
    Args:
        points: ordered array of points
        threshold: multiplier of average gap to consider as large
        
    Returns:
        tuple: (has_big_gap, gap_indices, gap_sizes)
    """
    if len(points) < 3:
        return False, [], []
        
    # Calculate distances between consecutive points
    gaps = []
    distances = []
    for i in range(len(points)):
        pt1 = points[i]
        pt2 = points[(i + 1) % len(points)]
        dist = np.linalg.norm(pt2 - pt1)
        gaps.append((i, dist))
        distances.append(dist)
    
    # Use robust statistics (median absolute deviation)
    median_dist = np.median(distances)
    mad = np.median(np.abs(np.array(distances) - median_dist))
    threshold_dist = median_dist + threshold * mad
    
    # Collect all significant gaps
    gap_indices = []
    gap_sizes = []
    for i, dist in gaps:
        if dist > threshold_dist:
            gap_indices.append(i)
            gap_sizes.append(dist)
    
    has_big_gap = len(gap_indices) > 0
    
    return has_big_gap, gap_indices, gap_sizes

def process_cross_section(cross_section):
    """
    Process a cross-section to ensure points are properly ordered.
    
    Args:
        cross_section: Array of 2D points
        
    Returns:
        tuple or None: (segments, ordered_points) if successful, None otherwise
    """
    try:
        ordered_points = order_points(cross_section, method="nearest")
        
        # Create segments from consecutive points
        segments = []
        for i in range(len(ordered_points)):
            next_i = (i + 1) % len(ordered_points)  # Wrap around for last point
            segments.append((ordered_points[i], ordered_points[next_i]))
        
        return segments, ordered_points
    
    except Exception as e:
        print(f"Error processing cross-section: {e}")
        return None
    
    ### Plotting functions

# Replace the existing create_combined_cross_section_figure function with this:
def create_combined_cross_section_figure(cross_sections, valid_sections, minor_radius, output_dir, closed_stomata=False):
    """Create a single figure with all cross-sections in a grid layout, using boundary detection for closed stomata."""
    if output_dir is None or sum(valid_sections) == 0:
        return

    valid_count = sum(valid_sections)
    cols = min(5, valid_count)
    rows = (valid_count + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    fig.suptitle('All Cross-Sections Comparison (Boundary Split)', fontsize=16)

    plot_idx = 1

    for i, (cross_section, valid) in enumerate(zip(cross_sections, valid_sections)):
        if not valid or cross_section is None or len(cross_section) < 5: # Need more points for boundary detection
            continue

        ax = fig.add_subplot(rows, cols, plot_idx)
        ax.set_title(f'Section {i}')
        plot_idx += 1

        original_center = np.mean(cross_section, axis=0)
        centered = cross_section - original_center

        selected_points = centered # Default
        has_split = False # Flag if boundary split was successful

        # --- Boundary Detection Splitting Logic (for closed_stomata) ---
        if closed_stomata:
            print(f"Section {i}: Applying boundary detection split.")
            points_to_process = centered
            initial_point_count = len(points_to_process)

            

        # --- End of Boundary Detection Logic ---
        
            used_fallback = False
            try:
                # --- Projected Space Clustering Method ---
                print(f"  Attempting Projected Space Clustering method...")

                if len(points_to_process) < 10: # Need enough points for PCA and clustering
                    raise ValueError("Not enough points for Projected Space Clustering.")

                # 1. PCA to find Major Axis (Axis of elongation)
                pca = PCA(n_components=2)
                pca.fit(points_to_process)
                v_major = pca.components_[0] # Axis of elongation
                # v_minor = pca.components_[1] # Axis perpendicular to elongation
                print(f"  Major axis vector (projection axis): [{v_major[0]:.3f}, {v_major[1]:.3f}]") # Changed print

                # 2. Project points onto the MAJOR Axis
                projections_1d = (points_to_process @ v_major).reshape(-1, 1) # Project onto MAJOR axis

                # 3. 1D K-Means Clustering (k=2)
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(projections_1d)

                # 4. Select Cluster based on centroid proximity to origin
                group1_indices = np.where(labels == 0)[0]
                group2_indices = np.where(labels == 1)[0]

                if len(group1_indices) >= 5 and len(group2_indices) >= 5:
                    group1 = points_to_process[group1_indices]
                    group2 = points_to_process[group2_indices]

                    centroid1 = np.mean(group1, axis=0)
                    centroid2 = np.mean(group2, axis=0)

                    if np.linalg.norm(centroid1) <= np.linalg.norm(centroid2):
                        selected_points = group1
                        print(f"  Projected Clustering split successful. Kept group 1 ({len(group1)} pts).")
                    else:
                        selected_points = group2
                        print(f"  Projected Clustering split successful. Kept group 2 ({len(group2)} pts).")
                    has_split = True
                    used_fallback = False # Ensure fallback flag is false
                else:
                    print("  Projected Clustering split resulted in invalid groups. Reverting.")
                    selected_points, has_split_fallback = _apply_proximity_fallback(centered, minor_radius, initial_point_count)
                    if has_split_fallback: has_split = True; used_fallback = True

            except Exception as e:
                print(f"  Error during Projected Space Clustering method: {e}. Reverting.")
                selected_points, has_split_fallback = _apply_proximity_fallback(centered, minor_radius, initial_point_count)
                if has_split_fallback: has_split = True; used_fallback = True

        # Order points using angular sorting - simpler and more predictable
        ordered_points_plot = order_points(selected_points, method="angular")
        print(f"  Ordering final {len(ordered_points_plot)} points for plotting using 'angular' method.")
        
        # Check for significant gaps
        has_gaps, gap_indices, gap_sizes = detect_gaps(ordered_points_plot, threshold=3.0)
        
        # Define plot color
        plot_color = 'g' if closed_stomata else 'b'

        # Plot with special handling for gaps
        if has_split or has_gaps:
            print(f"  Plotting section {i} as open line.")
            if has_gaps:
                print(f"    Found {len(gap_indices)} significant gaps.")
                
                # If we have multiple gaps, split into segments
                if len(gap_indices) > 0:
                    segments = []
                    start_idx = 0
                    
                    # Sort gap indices in descending order so we can remove segments properly
                    sorted_gaps = sorted(gap_indices, reverse=True)
                    
                    # Create a copy of the points that we can modify
                    points_copy = ordered_points_plot.copy()
                    
                    # Cut at each gap (starting from the end)
                    for gap_idx in sorted_gaps:
                        # If this is the gap between last and first point, handle specially
                        if gap_idx == len(points_copy) - 1:
                            # We'll handle this by plotting from the next starting point
                            start_idx = 0
                        else:
                            # Split at this gap - the points before the gap form one segment
                            segment = points_copy[start_idx:gap_idx+1]
                            if len(segment) >= 2:  # Only add if segment has at least 2 points
                                segments.append(segment)
                            start_idx = gap_idx + 1
                    
                    # Add the final segment if needed
                    if start_idx < len(points_copy):
                        segment = points_copy[start_idx:]
                        if len(segment) >= 2:
                            segments.append(segment)
                    
                    # Plot each segment separately
                    for j, segment in enumerate(segments):
                        if len(segment) >= 2:
                            ax.plot(segment[:, 0], segment[:, 1], f"{plot_color}-", linewidth=2)
                            
                            # Mark endpoints of segments
                            if j == 0:
                                ax.plot(segment[0, 0], segment[0, 1], f"{plot_color}o", markersize=5)
                            if j == len(segments) - 1:
                                ax.plot(segment[-1, 0], segment[-1, 1], f"{plot_color}x", markersize=6)
                else:
                    # No gaps found, just plot as a regular line
                    ax.plot(ordered_points_plot[:, 0], ordered_points_plot[:, 1], 
                           f"{plot_color}-", linewidth=2)
            else:
                # No gaps but has_split is True - plot as is
                ax.plot(ordered_points_plot[:, 0], ordered_points_plot[:, 1], 
                       f"{plot_color}-", linewidth=2)
        else:
            # Plot as closed loop
            ax.plot(np.append(ordered_points_plot[:, 0], ordered_points_plot[0, 0]),
                   np.append(ordered_points_plot[:, 1], ordered_points_plot[0, 1]),
                   f"{plot_color}-", linewidth=2)
        
        # Plot individual points
        ax.plot(ordered_points_plot[:, 0], ordered_points_plot[:, 1], 
               f"{plot_color}.", markersize=3)

             
        # Plot reference circle
        #ax.plot(circle_x, circle_y, 'k--', linewidth=1, alpha=0.7)

        # Compute metrics based on the final selected points
        points_for_metrics = selected_points
        if len(points_for_metrics) > 0:
            width = np.ptp(points_for_metrics[:, 0])
            height = np.ptp(points_for_metrics[:, 1])
            aspect = width / height if height > 0 else 0
            try:
                convexity_val2 = calculate_convexity(order_points(points_for_metrics, method="angular"))
                props_text = f"Pts: {len(points_for_metrics)}\nAR: {aspect:.2f}\nC: {convexity_val2:.2f}"
            except Exception as e:
                props_text = f"Pts: {len(points_for_metrics)}\nAR: {aspect:.2f}"
        else:
             props_text = "Pts: 0"

        ax.text(0.05, 0.95, props_text, transform=ax.transAxes, verticalalignment='top',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.set_aspect('equal')
        #ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Final figure adjustments and saving
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'all_individual_cross_sections.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved grid figure with {plot_idx-1} cross-sections to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def create_visualizations(mesh, centerline_points, tangent_vectors, section_positions, 
                         cross_sections, section_objects, raw_centerline_points, 
                         inner_points, outer_points, minor_radius, valid_sections, 
                         output_dir=None, closed_stomata=False):
    """Create visualization figures for the analysis results."""
    # Import plotly here to make it optional
    try:
        import plotly.graph_objects as go
        use_plotly = True
    except ImportError:
        use_plotly = False
        
    # 3D Visualization using plotly if available
    if use_plotly:
        # Convert mesh to plotly format
        vertices = mesh.vertices
        faces = mesh.faces
        
        i = faces[:, 0]
        j = faces[:, 1]
        k = faces[:, 2]
        
        # Create mesh3d object
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i, j=j, k=k,
            opacity=0.5,
            color='lightblue'
        )
        
        # Add raw centerline points
        raw_centerline_trace = go.Scatter3d(
            x=raw_centerline_points[:, 0],
            y=raw_centerline_points[:, 1],
            z=raw_centerline_points[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='orange',
            ),
            name='Raw Centerline'
        )
        
        # Add fitted centerline points
        centerline_trace = go.Scatter3d(
            x=centerline_points[:, 0],
            y=centerline_points[:, 1],
            z=centerline_points[:, 2],
            mode='markers+lines',
            marker=dict(
                size=5,
                color='red',
            ),
            name='Fitted Centerline'
        )
        
        # Add section planes
        section_traces = []
        for i, (point, tangent, valid) in enumerate(zip(centerline_points, tangent_vectors, valid_sections)):
            if not valid:
                continue
                
            # Create a disc to represent the section plane
            r = np.linspace(0, minor_radius * 2, 20)
            theta = np.linspace(0, 2*np.pi, 36)
            r_grid, theta_grid = np.meshgrid(r, theta)
            
            # Coordinates in the plane
            x_disc = r_grid * np.cos(theta_grid)
            y_disc = r_grid * np.sin(theta_grid)
            
            # Ensure we have a valid normal vector
            if np.linalg.norm(tangent) < 1e-10:
                continue
                
            # Find two orthogonal vectors in the plane
            if np.abs(tangent[0]) > 0.9:
                v1 = np.array([0, 1, 0])
            else:
                v1 = np.array([1, 0, 0])
            
            v1 = v1 - np.dot(v1, tangent) * tangent
            v1 = v1 / np.linalg.norm(v1)
            
            v2 = np.cross(tangent, v1)
            
            # Transform disc to 3D
            x_plane = point[0] + x_disc * v1[0] + y_disc * v2[0]
            y_plane = point[1] + x_disc * v1[1] + y_disc * v2[1]
            z_plane = point[2] + x_disc * v1[2] + y_disc * v2[2]
            
            # Create surface
            section_trace = go.Surface(
                x=x_plane,
                y=y_plane,
                z=z_plane,
                colorscale=[[0, f'rgba(255,0,0,0.3)'], [1, f'rgba(255,0,0,0.3)']],
                showscale=False,
                opacity=0.3,
                name=f'Section {i}'
            )
            section_traces.append(section_trace)
        
        # Create figure with the traces
        fig = go.Figure(data=[mesh_trace, raw_centerline_trace, centerline_trace] + section_traces)
        
        # Update layout
        fig.update_layout(title='3D Visualization of Stomata with Cross-Sections',
                         scene=dict(
                             xaxis_title='X',
                             yaxis_title='Y',
                             zaxis_title='Z'
                         ))
        
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig.write_html(os.path.join(output_dir, '3d_visualization.html'))
    
    # 2D Visualizations using matplotlib
    # Create a 2x3 grid of plots
    fig = plt.figure(figsize=(15, 10))
    fig.tight_layout(pad=3.0)
    
    # 6a. 3D view with section planes
    ax_3d = fig.add_subplot(231, projection='3d')
    ax_3d.set_title('3D View with Cross-Sections')
    
    # Draw centerline
    ax_3d.plot(centerline_points[:, 0], centerline_points[:, 1], centerline_points[:, 2], 
               'ro-', linewidth=2, markersize=4, label='Centerline')
    
    # Draw section planes as discs
    for i, (point, tangent, valid) in enumerate(zip(centerline_points, tangent_vectors, valid_sections)):
        if not valid:
            continue
            
        # Create a disc to represent the section plane
        r = np.linspace(0, minor_radius * 1.5, 2)  # Just draw the edge
        theta = np.linspace(0, 2*np.pi, 36)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        # Coordinates in the plane
        x_disc = r_grid * np.cos(theta_grid)
        y_disc = r_grid * np.sin(theta_grid)
        
        # Ensure we have a valid normal vector
        if np.linalg.norm(tangent) < 1e-10:
            continue
            
        # Find two orthogonal vectors in the plane
        if np.abs(tangent[0]) > 0.9:
            v1 = np.array([0, 1, 0])
        else:
            v1 = np.array([1, 0, 0])
        
        v1 = v1 - np.dot(v1, tangent) * tangent
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(tangent, v1)
        
        # Transform disc to 3D
        for j in range(r_grid.shape[0]):
            x_plane = point[0] + x_disc[j] * v1[0] + y_disc[j] * v2[0]
            y_plane = point[1] + x_disc[j] * v1[1] + y_disc[j] * v2[1]
            z_plane = point[2] + x_disc[j] * v1[2] + y_disc[j] * v2[2]
            
            if r_grid[j, 0] == r[-1]:  # Only draw the outer edge
                ax_3d.plot(x_plane, y_plane, z_plane, 'g-', alpha=0.5)
    
    # Set reasonable aspect ratio
    ax_3d.set_box_aspect([1, 1, 1])
    
    # 6b. Top view with section lines
    ax_top = fig.add_subplot(232)
    ax_top.set_title('Top View (XY) with Section Lines')
    
    # Draw central points
    ax_top.plot(centerline_points[:, 0], centerline_points[:, 1], 'ro-', linewidth=2, markersize=4, label='Centerline')
    
    # Draw section lines from the center outward
    for i, (point, tangent, valid) in enumerate(zip(centerline_points, tangent_vectors, valid_sections)):
        if not valid:
            continue
            
        # Project tangent vector to XY
        tangent_xy = np.array([tangent[0], tangent[1], 0])
        if np.linalg.norm(tangent_xy) > 0.01:
            tangent_xy = tangent_xy / np.linalg.norm(tangent_xy)
            
            # Orthogonal vector in XY plane
            ortho = np.array([-tangent_xy[1], tangent_xy[0], 0])
            
            # Draw section line
            line_length = minor_radius * 3
            line_start = point + ortho * line_length/2
            line_end = point - ortho * line_length/2
            
            ax_top.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'g-', alpha=0.7)
            
            # Draw section number
            if i % 2 == 0:  # Label every other section for clarity
                text_pos = point + ortho * line_length/2 * 0.7
                ax_top.text(text_pos[0], text_pos[1], str(i), fontsize=8, 
                           horizontalalignment='center', verticalalignment='center')
    
    # Set equal aspect ratio
    ax_top.set_aspect('equal')
    ax_top.grid(True)
    
    # 6c. Raw cross-section shapes (without processing)
    ax_raw = fig.add_subplot(233)
    ax_raw.set_title('Original Cross-Sections')
    
    # Reference circle
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = minor_radius * np.cos(circle_theta)
    circle_y = minor_radius * np.sin(circle_theta)
    
    # Find a valid zero-angle section to use as reference
    zero_idx = 0
    while zero_idx < len(valid_sections) and not valid_sections[zero_idx]:
        zero_idx += 1
    
    if sum(valid_sections) > 0:
        # Plot each raw section
        raw_points = None
        for i, (section, valid) in enumerate(zip(section_objects, valid_sections)):
            if not valid or section is None:
                continue
                
            # Get 2D points
            try:
                # Convert section to 2D
                path_2D, _ = section.to_planar()
                points = path_2D.vertices
                
                # Store points for setting limits
                if raw_points is None:
                    raw_points = points
                else:
                    raw_points = np.vstack((raw_points, points))
                
                # Use angle-based coloring
                color = plt.cm.hsv(i / len(section_objects))
                
                # Plot outline
                ax_raw.plot(points[:, 0], points[:, 1], '-', color=color, alpha=0.8, 
                           linewidth=1.5, label=f'Section {i}')
                
                # Mark center
                center = np.mean(points, axis=0)
                ax_raw.plot(center[0], center[1], 'o', color=color, markersize=4)
            except Exception as e:
                print(f"  Could not plot raw section {i}: {e}")
        
        # Add reference circle for section at 0 degrees
        if valid_sections[zero_idx] and cross_sections[zero_idx] is not None:
            ref_center = np.mean(cross_sections[zero_idx], axis=0)
            ax_raw.plot(circle_x + ref_center[0], circle_y + ref_center[1], 'g--', linewidth=1, alpha=0.7)
            
        # Set consistent limits
        if raw_points is not None:
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
    for i, (section_obj, cross_section, valid) in enumerate(zip(section_objects, cross_sections, valid_sections)):
        if not valid or cross_section is None or section_obj is None:
            continue
            
        # Use angle-based color
        color = plt.cm.hsv(i / len(valid_sections))
        
        try:
            # Process the cross section to get properly ordered segments and points
            result = process_cross_section(cross_section)
            
            if result is not None:
                segments, ordered_points = result
                
                # Plot as a closed loop
                x = np.append(ordered_points[:, 0], ordered_points[0, 0])
                y = np.append(ordered_points[:, 1], ordered_points[0, 1])
                ax_all.plot(x, y, '-', color=color, alpha=0.8, linewidth=1.5)
                
                # Mark center
                center = np.mean(cross_section, axis=0)
                ax_all.plot(center[0], center[1], 'o', color=color, markersize=4)
        except Exception as e:
            # Fall back to simple plotting if processing fails
            print(f"Error plotting section {i}: {e}")
            ax_all.plot(cross_section[:, 0], cross_section[:, 1], '-', color=color, 
                        alpha=0.8, linewidth=1.5)
    
    # Set equal aspect ratio
    ax_all.set_aspect('equal')
    ax_all.grid(True)
    
    # Replace plot code for individual cross-sections (around line 900-910):

# 6e. Plot each cross-section individually
    ax_ind = fig.add_subplot(235)
    ax_ind.set_title('Individual Cross-Sections (Centered)')

    # Update the individual cross-sections plot:

    # Plot all processed sections with zero centerpoint for comparison
    for i, (cross_section, valid) in enumerate(zip(cross_sections, valid_sections)):
        if not valid or cross_section is None:
            continue
            
        # Center each section at the origin
        center = np.mean(cross_section, axis=0)
        centered = cross_section - center
        
        # Use angle-based color
        color = plt.cm.hsv(i / len(valid_sections))
        
        try:
            # Use nearest-neighbor ordering instead of angular sorting
            ordered_points = order_points(centered, method="nearest")
            
            # Connect points in the correct order
            ax_ind.plot(np.append(ordered_points[:, 0], ordered_points[0, 0]), 
                    np.append(ordered_points[:, 1], ordered_points[0, 1]), 
                    '-', color=color, alpha=0.8, linewidth=1.5, label=f'Section {i}')
        except Exception as e:
            # Fall back to simple plotting
            print(f"Error plotting individual section {i}: {e}")

    # Plot reference circle
    ax_ind.plot(circle_x, circle_y, 'k--', linewidth=2, alpha=0.5, label='Reference Circle')
    
    # Set equal aspect ratio and reasonable limits
    ax_ind.set_aspect('equal')
    ax_ind.grid(True)
    
    # 6f. Show example cross-section with details
    ax_example = fig.add_subplot(236)
    
    # Find a good example (middle of the valid sections)
    valid_indices = [i for i, v in enumerate(valid_sections) if v]
    example_idx = valid_indices[len(valid_indices)//2] if valid_indices else 0
    
    # Replace plot code for example section (around line 935-940):

    if valid_sections[example_idx] and cross_sections[example_idx] is not None:
        example = cross_sections[example_idx]
        ax_example.set_title(f'Detailed View of Section {example_idx}')
        
        # Center the example
        center = np.mean(example, axis=0)
        centered = example - center
        
        # Sort points by angle for proper plotting
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        sorted_indices = np.argsort(angles)
        ordered_points = centered[sorted_indices]
        
        # Plot with points and vectors (ensure closed loop)
        ax_example.plot(np.append(ordered_points[:, 0], ordered_points[0, 0]), 
                    np.append(ordered_points[:, 1], ordered_points[0, 1]), 
                    'b-', linewidth=2)
        
        # Mark points
        ax_example.plot(ordered_points[:, 0], ordered_points[:, 1], 'ro', markersize=3)
        
        # Add indices for key points (every Nth point)
        n = max(1, len(centered) // 10)
        for i in range(0, len(centered), n):
            ax_example.text(centered[i, 0], centered[i, 1], str(i), fontsize=8,
                         ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Calculate properties
        width = np.max(centered[:, 0]) - np.min(centered[:, 0])
        height = np.max(centered[:, 1]) - np.min(centered[:, 1])
        aspect = width / height if height > 0 else 0
        
        # Add property box
        props_text = (f"Points: {len(centered)}\n" 
                      f"Width: {width:.2f}\n"
                      f"Height: {height:.2f}\n"
                      f"Aspect: {aspect:.2f}")
        ax_example.text(0.05, 0.95, props_text, transform=ax_example.transAxes, 
                      verticalalignment='top', bbox=dict(boxstyle='round', 
                                                        facecolor='wheat', alpha=0.7))
        
        # Plot reference circle
        ax_example.plot(circle_x, circle_y, 'g--', linewidth=1, alpha=0.7)
        
        # Set equal aspect ratio and reasonable limits
        ax_example.set_aspect('equal')

    else:
        ax_example.set_title('No Valid Example Section')
    
    ax_example.grid(True)
    
    # Save or show all visualizations
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'all_cross_sections.png'), dpi=150)
    else:
        plt.show()
    plt.close(fig)

    # Create individual figures for each cross-section
    if output_dir is not None:
        create_combined_cross_section_figure(cross_sections, valid_sections, minor_radius, output_dir, closed_stomata)

def _apply_proximity_fallback(centered_points, minor_radius, initial_point_count):
    """Applies the proximity filter/region growing as a fallback."""
    print("  Applying proximity filter fallback...")
    selected_points = centered_points
    has_filtered = False
    points_to_process = centered_points

    # Simplified DBSCAN cleanup
    try:
        dbscan_eps = minor_radius * 0.40
        clustering = DBSCAN(eps=dbscan_eps, min_samples=5).fit(points_to_process)
        labels = clustering.labels_
        unique_labels = np.unique(labels[labels != -1])
        if len(unique_labels) > 1:
            min_dist = float('inf')
            best_label = -1
            for label in unique_labels:
                cluster_points = points_to_process[labels == label]
                if len(cluster_points) < 5: continue
                dist = np.linalg.norm(np.mean(cluster_points, axis=0))
                if dist < min_dist:
                    min_dist = dist
                    best_label = label
            if best_label != -1: points_to_process = points_to_process[labels == best_label]
            else: points_to_process = points_to_process[labels != -1]
        elif len(unique_labels) == 1: points_to_process = points_to_process[labels == unique_labels[0]]
        elif len(labels[labels != -1]) > 5: points_to_process = points_to_process[labels != -1]
        else: pass # Keep original if cleanup failed badly
        points_after_dbscan = points_to_process.copy()
    except Exception:
        points_after_dbscan = points_to_process.copy()

    if len(points_to_process) < 5:
        return points_to_process, (len(points_to_process) < initial_point_count)

    # Core identification and region growing (simplified from previous version)
    core_radius = minor_radius * 0.4
    distances_from_origin = np.linalg.norm(points_to_process, axis=1)
    core_indices = np.where(distances_from_origin <= core_radius)[0]
    start_indices = []
    if len(core_indices) > 0: start_indices = list(core_indices)
    else:
        k_closest = min(3, len(points_to_process))
        start_indices = list(np.argsort(distances_from_origin)[:k_closest])

    if start_indices:
        selection_mask = np.zeros(len(points_to_process), dtype=bool)
        selection_mask[start_indices] = True
        try:
            temp_tree = KDTree(points_to_process)
            distances, _ = temp_tree.query(points_to_process, k=min(6, len(points_to_process)))
            if distances.ndim > 1 and distances.shape[1] > 1:
                 avg_neighbor_dist = np.median(distances[:, 1])
                 growth_threshold = avg_neighbor_dist * 3.5
            else: growth_threshold = minor_radius * 0.25
        except Exception: growth_threshold = minor_radius * 0.25

        processed_indices = set(start_indices)
        queue = start_indices[:]
        grow_tree = KDTree(points_to_process)
        while queue:
            current_idx = queue.pop(0)
            neighbor_indices = grow_tree.query_ball_point(points_to_process[current_idx], r=growth_threshold)
            for neighbor_idx in neighbor_indices:
                if neighbor_idx not in processed_indices:
                    selection_mask[neighbor_idx] = True
                    processed_indices.add(neighbor_idx)
                    queue.append(neighbor_idx)
        grown_points = points_to_process[selection_mask]

        if len(grown_points) < 10: # Sanity check
            selected_points = points_after_dbscan
            has_filtered = (len(selected_points) < initial_point_count)
        else:
            selected_points = grown_points
            has_filtered = (len(selected_points) < initial_point_count)
    else:
        selected_points = points_after_dbscan
        has_filtered = (len(selected_points) < initial_point_count)

    print(f"  Proximity fallback finished. Selected {len(selected_points)} points. Filtered: {has_filtered}")
    return selected_points, has_filtered

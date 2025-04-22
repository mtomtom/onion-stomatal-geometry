import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import trimesh

def ellipse(theta, a, b, phi):
    """Parameterized equation of an ellipse."""
    r = a*b / np.sqrt((b*np.cos(theta-phi))**2 + (a*np.sin(theta-phi))**2)
    return r

def get_2d_points(cross_section):
    """Extract 2D points from a cross-section, handling both tuple and direct formats."""
    if cross_section is None:
        return None
    return cross_section[0] if isinstance(cross_section, tuple) else cross_section


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
        # Ensure points are in 2D
        points_2d = get_2d_points(cross_section)
        ordered_points = order_points(points_2d, method="nearest")
        
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
    valid_indices = [i for i, valid in enumerate(valid_sections) if valid]

    # Add section index mapping as text in the figure
    mapping_text = "Section Mapping: " + ", ".join([f"Plot {j+1}→Sec {i}" for j, i in enumerate(valid_indices)])
    fig.text(0.5, 0.01, mapping_text, ha='center', fontsize=9)

    for i, (cross_section, valid) in enumerate(zip(cross_sections, valid_sections)):
        points_2d = get_2d_points(cross_section)
        if not valid or cross_section is None or len(points_2d) < 5: # Need more points for boundary detection
            continue

        ax = fig.add_subplot(rows, cols, plot_idx)
        ax.set_title(f'Plot {plot_idx}: Section {i}')
        plot_idx += 1

        original_center = np.mean(points_2d, axis=0)
        centered = points_2d - original_center

        selected_points = centered # Default
        has_split = False # Flag if boundary split was successful

        # --- Boundary Detection Splitting Logic (for closed_stomata) ---
        if closed_stomata:
            print(f"Section {i}: Applying boundary detection split.")
            points_to_process = centered
            initial_point_count = len(points_to_process)

        
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
        #points_for_metrics = selected_points
        #if len(points_for_metrics) > 0:
        #    width = np.ptp(points_for_metrics[:, 0])
        #    height = np.ptp(points_for_metrics[:, 1])
        #    aspect = width / height if height > 0 else 0
        #    try:
        #        convexity_val2 = calculate_convexity(order_points(points_for_metrics, method="angular"))
        #        props_text = f"Pts: {len(points_for_metrics)}\nAR: {aspect:.2f}\nC: {convexity_val2:.2f}"
        #    except Exception as e:
        #        props_text = f"Pts: {len(points_for_metrics)}\nAR: {aspect:.2f}"
        #else:
        #     props_text = "Pts: 0"

        #ax.text(0.05, 0.95, props_text, transform=ax.transAxes, verticalalignment='top',
        #        fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
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
                         cross_sections, section_objects, section_transforms, raw_centerline_points, 
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

        section_point_traces = []
        for i, (cross_section, transform, valid, point, tangent) in enumerate(zip(
                cross_sections, section_transforms, valid_sections, centerline_points, tangent_vectors)):
            if not valid or cross_section is None:
                continue
                
            # Unpack the cross-section data - these are the already processed 2D points
            processed_points_2d, original_points_3d = cross_section if isinstance(cross_section, tuple) else (cross_section, None)
            
            try:
                # --- Start Modification ---
                # Always calculate the correct order based on the processed 2D points first
                center_2d = np.mean(processed_points_2d, axis=0)
                centered_points_2d = processed_points_2d - center_2d
                angles = np.arctan2(centered_points_2d[:, 1], centered_points_2d[:, 0])
                sorted_indices = np.argsort(angles)

                if original_points_3d is not None and len(original_points_3d) > 0:
                    # Use original 3D points, but apply the calculated order for connectivity
                    # Check if the number of points matches before applying indices
                    if len(original_points_3d) == len(sorted_indices):
                         points_3d = original_points_3d[sorted_indices]
                         print(f"  Section {i}: Using ORDERED original 3D points ({len(points_3d)} points)")
                    else:
                         # Fallback if point counts mismatch (should not happen often with filtering)
                         print(f"  Section {i}: Warning - Mismatch between original ({len(original_points_3d)}) and processed ({len(processed_points_2d)}) points. Using fallback.")
                         # Use the fallback transformation (copied from below)
                         ordered_centered_points_2d = centered_points_2d[sorted_indices]
                         if np.abs(tangent[0]) > 0.9: v1 = np.array([0, 1, 0])
                         else: v1 = np.array([1, 0, 0])
                         v1 = v1 - np.dot(v1, tangent) * tangent; v1 /= np.linalg.norm(v1)
                         v2 = np.cross(tangent, v1)
                         points_3d = np.array([point + ordered_centered_points_2d[j, 0] * v1 + ordered_centered_points_2d[j, 1] * v2 for j in range(len(ordered_centered_points_2d))])

                else:
                    # FALLBACK: Reconstruct 3D points using the manual plane basis.
                    print(f"  Section {i}: Warning - Original 3D points not available. Using fallback transformation.")
                    # Use the CENTERED and ORDERED points for transformation
                    ordered_centered_points_2d = centered_points_2d[sorted_indices]

                    # Calculate the plane basis vectors (v1, v2)
                    if np.abs(tangent[0]) > 0.9:
                        v1 = np.array([0, 1, 0])
                    else:
                        v1 = np.array([1, 0, 0])
                    v1 = v1 - np.dot(v1, tangent) * tangent
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = np.cross(tangent, v1)

                    # Transform the CENTERED points relative to the centerline point 'point'
                    points_3d = np.array([
                        point + ordered_centered_points_2d[j, 0] * v1 + ordered_centered_points_2d[j, 1] * v2
                        for j in range(len(ordered_centered_points_2d))
                    ])
                    print(f"  Section {i}: Using transformed centered 2D points ({len(points_3d)} points)")
                
                # Create color based on section index for consistent coloring
                color = plt.cm.hsv(i / len(valid_sections))
                rgb_color = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
                
                # Create visualization for the selected cross-section points
                # Add the first point again at the end to close the loop
                if len(points_3d) > 0: # Check if points_3d is not empty
                    closed_points_3d = np.vstack([points_3d, points_3d[0]])
                else:
                    closed_points_3d = np.empty((0, 3)) # Handle empty case
                
                section_point_trace = go.Scatter3d(
                    x=closed_points_3d[:, 0],
                    y=closed_points_3d[:, 1],
                    z=closed_points_3d[:, 2],
                    mode='lines',  # Just use lines for a clean visualization
                    line=dict(color=rgb_color, width=5),  # Thicker lines for visibility
                    name=f'Section {i}'  # Consistent naming with section planes
                )
                section_point_traces.append(section_point_trace)
                
                # Also add points for better visibility
                points_trace = go.Scatter3d(
                    x=points_3d[:, 0],
                    y=points_3d[:, 1],
                    z=points_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=rgb_color,
                        symbol='circle',
                    ),
                    name=f'Section {i} Points',
                    showlegend=False
                )
                section_point_traces.append(points_trace)
                
            except Exception as e:
                print(f"Could not visualize section {i} points: {str(e)}")
                print(f"  Debug info: {e.__class__.__name__}, processed_points_2d shape: {processed_points_2d.shape if processed_points_2d is not None else 'None'}")
        # Update the figure creation to include these new traces
        fig = go.Figure(data=[mesh_trace, raw_centerline_trace, centerline_trace] + section_traces + section_point_traces)
        valid_indices = [i for i, valid in enumerate(valid_sections) if valid]
        buttons = [
            {
                'label': 'Show All',
                'method': 'update',
                'args': [{'visible': [True] * len(fig.data)}]
            }
        ]

        # Add buttons for individual sections
        base_traces = 3  # mesh, raw_centerline, centerline
        for i, valid in enumerate(valid_sections):
            if not valid:
                continue
            
            # Calculate indices within the visible traces
            plane_idx = base_traces + valid_indices.index(i)
            points_idx = base_traces + len(section_traces) + valid_indices.index(i)
            
            # Create a visibility list where only this section's plane and points are visible
            visible = [True] * base_traces  # Always show mesh and centerlines
            visible.extend([False] * (len(section_traces) + len(section_point_traces)))
            
            # Make this section's plane and points visible
            if plane_idx < len(visible):
                visible[plane_idx] = True
            if points_idx < len(visible):
                visible[points_idx] = True
            
            buttons.append({
                'label': f'Section {i}',
                'method': 'update',
                'args': [{'visible': visible}]
            })

        # --- Update Button Generation Logic ---
        num_base_traces = 3 # mesh_trace, raw_centerline_trace, centerline_trace
        num_plane_traces = len(section_traces)
        # Each section point visualization adds TWO traces (lines + markers)
        num_point_traces_per_section = 2
        total_traces = num_base_traces + num_plane_traces + len(section_point_traces)

        buttons = [
            {
                'label': 'Show All',
                'method': 'update',
                'args': [{'visible': [True] * total_traces}]
            }
        ]

        # Keep track of which index in the original loop corresponds to which trace index
        valid_section_indices = [i for i, valid in enumerate(valid_sections) if valid]
        plane_trace_map = {original_idx: trace_idx for trace_idx, original_idx in enumerate(valid_section_indices)}
        # Point traces start after base traces and plane traces
        point_trace_start_idx = num_base_traces + num_plane_traces
        point_trace_map = {original_idx: point_trace_start_idx + trace_idx * num_point_traces_per_section
                           for trace_idx, original_idx in enumerate(valid_section_indices)}

        # Add buttons for individual sections
        for original_idx in valid_section_indices:
            visibility = [False] * total_traces
            # Show base traces
            visibility[0:num_base_traces] = [True] * num_base_traces

            # Find the correct trace indices for this section
            plane_idx = plane_trace_map.get(original_idx)
            point_start_idx = point_trace_map.get(original_idx)

            if plane_idx is not None:
                 # Plane traces start right after base traces
                visibility[num_base_traces + plane_idx] = True
            if point_start_idx is not None:
                # Show both line and marker traces for this section
                visibility[point_start_idx] = True # Line trace
                visibility[point_start_idx + 1] = True # Marker trace

            buttons.append({
                'label': f'Section {original_idx}',
                'method': 'update',
                'args': [{'visible': visibility}]
            })
        # --- End Update Button Generation Logic ---

        # Update layout with buttons
        fig.update_layout(
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            }],
            # ... rest of layout settings ...
        )

        # Save or show the figure
        # ... (rest of the function) ...
            
        
        # Create figure with the traces
        #fig = go.Figure(data=[mesh_trace, raw_centerline_trace, centerline_trace] + section_traces)
        
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
            ref_points = get_2d_points(cross_sections[zero_idx])
            ref_center = np.mean(ref_points, axis=0)
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
            points_2d = get_2d_points(cross_section)
            result = process_cross_section(points_2d)
            
            if result is not None:
                segments, ordered_points = result
                
                # Plot as a closed loop
                x = np.append(ordered_points[:, 0], ordered_points[0, 0])
                y = np.append(ordered_points[:, 1], ordered_points[0, 1])
                ax_all.plot(x, y, '-', color=color, alpha=0.8, linewidth=1.5)
                
                # Mark center
                center = np.mean(points_2d, axis=0)
                ax_all.plot(center[0], center[1], 'o', color=color, markersize=4)
        except Exception as e:
            # Fall back to simple plotting if processing fails
            print(f"Error plotting section {i}: {e}")
            ax_all.plot(points_2d[:, 0], points_2d[:, 1], '-', color=color, 
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
            
        # Extract 2D points
        points_2d = get_2d_points(cross_section)
        # Center each section at the origin
        center = np.mean(points_2d, axis=0)
        centered = points_2d - center
        
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
        points_2d = get_2d_points(cross_sections[example_idx])
        ax_example.set_title(f'Detailed View of Section {example_idx}')
        
        # Center the example
        center = np.mean(points_2d, axis=0)
        centered = points_2d - center
        
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
    """
    Applies the proximity filter/region growing as a fallback method for separating
    inner boundary points from outer points.
    
    Args:
        centered_points: Array of points centered around origin
        minor_radius: Estimated minor radius of the structure
        initial_point_count: Original number of points for comparison
        
    Returns:
        (selected_points, has_filtered): Selected subset of points and whether filtering occurred
    """
    print("  Applying proximity filter fallback...")
    points_to_process = centered_points
    
    # Step 1: DBSCAN cleanup to remove outliers and find main clusters
    try:
        # Use 40% of minor radius as a reasonable neighborhood size
        dbscan_eps = minor_radius * 0.40
        clustering = DBSCAN(eps=dbscan_eps, min_samples=5).fit(points_to_process)
        labels = clustering.labels_
        unique_labels = np.unique(labels[labels != -1])
        
        if len(unique_labels) > 1:
            # Multiple clusters found - select the one closest to origin
            min_dist = float('inf')
            best_label = -1
            
            for label in unique_labels:
                cluster_points = points_to_process[labels == label]
                if len(cluster_points) < 5:
                    continue  # Skip tiny clusters
                    
                # Calculate distance from origin to cluster centroid
                dist = np.linalg.norm(np.mean(cluster_points, axis=0))
                if dist < min_dist:
                    min_dist = dist
                    best_label = label
                    
            if best_label != -1:
                points_to_process = points_to_process[labels == best_label]
            else:
                # No good cluster found, just remove noise points
                points_to_process = points_to_process[labels != -1]
                
        elif len(unique_labels) == 1:
            # Only one cluster - keep it
            points_to_process = points_to_process[labels == unique_labels[0]]
            
        elif len(labels[labels != -1]) > 5:
            # No distinct clusters but some non-noise points exist
            points_to_process = points_to_process[labels != -1]
            
        # If all points are noise or DBSCAN failed completely, keep original points
        
        # Store DBSCAN results
        points_after_dbscan = points_to_process.copy()
        print(f"  DBSCAN filtering: {len(centered_points)} → {len(points_after_dbscan)} points")
        
    except Exception as e:
        print(f"  DBSCAN clustering failed: {e}")
        points_after_dbscan = points_to_process.copy()

    # Early exit if too few points remain
    if len(points_to_process) < 5:
        print(f"  Too few points after DBSCAN ({len(points_to_process)}), returning early")
        return points_to_process, (len(points_to_process) < initial_point_count)

    # Step 2: Region growing from core points
    try:
        # Core identification - points close to origin
        core_radius = minor_radius * 0.4  # 40% of minor radius defines the core region
        distances_from_origin = np.linalg.norm(points_to_process, axis=1)
        core_indices = np.where(distances_from_origin <= core_radius)[0]
        
        # Select starting points - either core points or closest to origin
        start_indices = []
        if len(core_indices) > 0:
            start_indices = list(core_indices)
        else:
            # If no core points, use the 3 closest points (or fewer if < 3 points total)
            k_closest = min(3, len(points_to_process))
            start_indices = list(np.argsort(distances_from_origin)[:k_closest])

        if start_indices:
            # Create selection mask and KDTree just once
            selection_mask = np.zeros(len(points_to_process), dtype=bool)
            selection_mask[start_indices] = True
            
            # Build KD-Tree for efficient neighbor searches
            grow_tree = KDTree(points_to_process)
            
            # Determine growth threshold based on median neighbor distance
            try:
                distances, _ = grow_tree.query(points_to_process, k=min(6, len(points_to_process)))
                if distances.ndim > 1 and distances.shape[1] > 1:
                    # Use median distance to nearest neighbor times 3.5 as growth threshold
                    avg_neighbor_dist = np.median(distances[:, 1])
                    growth_threshold = avg_neighbor_dist * 3.5
                else:
                    growth_threshold = minor_radius * 0.25
            except Exception as e:
                print(f"  Error calculating growth threshold: {e}")
                growth_threshold = minor_radius * 0.25
                
            print(f"  Using growth threshold: {growth_threshold:.4f}")

            # Perform region growing using BFS
            from collections import deque  # More efficient than list for queue operations
            processed_indices = set(start_indices)
            queue = deque(start_indices)
            
            while queue:
                current_idx = queue.popleft()  # O(1) vs O(n) for list.pop(0)
                neighbor_indices = grow_tree.query_ball_point(
                    points_to_process[current_idx], r=growth_threshold)
                
                for neighbor_idx in neighbor_indices:
                    if neighbor_idx not in processed_indices:
                        selection_mask[neighbor_idx] = True
                        processed_indices.add(neighbor_idx)
                        queue.append(neighbor_idx)
            
            grown_points = points_to_process[selection_mask]
            
            # Verify growth produced reasonable results
            if len(grown_points) < 10:
                print(f"  Region growing produced too few points ({len(grown_points)}), reverting to DBSCAN result")
                selected_points = points_after_dbscan
            else:
                print(f"  Region growing: {len(points_to_process)} → {len(grown_points)} points")
                selected_points = grown_points
        else:
            # No starting points found (unlikely but possible)
            print("  No starting points for region growing, using DBSCAN result")
            selected_points = points_after_dbscan
            
    except Exception as e:
        print(f"  Region growing failed: {e}")
        selected_points = points_after_dbscan

    # Determine if filtering occurred
    has_filtered = (len(selected_points) < initial_point_count)
    
    print(f"  Proximity fallback finished. Selected {len(selected_points)} points. Filtered: {has_filtered}")
    return selected_points, has_filtered

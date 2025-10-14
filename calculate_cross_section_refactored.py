import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay, ConvexHull, KDTree
import argparse

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

PARAMS = {
    'gap_threshold_multiplier': 3.0,   # Multiplier for detecting significant gaps
    'dbscan_eps_factor': 0.4,          # Epsilon factor for DBSCAN relative to minor_radius
    'dbscan_min_samples': 5,           # Minimum samples for DBSCAN
    'kmeans_n_init': 10,               # Number of KMeans initializations
    'projection_min_points': 10,       # Minimum points needed for projection clustering
    'plot_linewidth': 2,               # Line width for plotted cross sections
    'plot_markersize': {               # Marker sizes for different point types
        'start': 5,
        'end': 6,
        'single': 4,
        'points': 3
    },
    'plot_colors': {                   # Colors for different plot elements
        'open': 'b',                   # Default color for open stomata
        'closed': 'g',                 # Color for closed stomata
    }
}

# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def order_points(points, method="angular", center=None):
    """Order points in a clockwise fashion around a center using different methods.
    
    Args:
        points (np.ndarray): Points to order (shape: Nx2)
        method (str): Method to use ("angular", "nearest", "convex_hull")
        center (np.ndarray, optional): Center point for angular sorting. If None, use centroid.
        
    Returns:
        np.ndarray: Ordered points
    """
    if len(points) <= 1:
        return points
        
    if method == "nearest":
        # Nearest-neighbor ordering
        ordered_indices = np.zeros(len(points), dtype=int)
        remaining_indices = set(range(len(points)))
        current_idx = 0  # Start with the first point
        ordered_indices[0] = current_idx
        remaining_indices.remove(current_idx)
        
        for i in range(1, len(points)):
            current_point = points[current_idx]
            distances = np.linalg.norm(points - current_point, axis=1)
            # Find closest unvisited point
            distances[ordered_indices[:i]] = np.inf
            next_idx = np.argmin(distances)
            ordered_indices[i] = next_idx
            current_idx = next_idx
            
        return points[ordered_indices]
        
    elif method == "angular":
        # Angular sorting around centroid
        if center is None:
            center = np.mean(points, axis=0)
        
        # Calculate angles from center to each point
        centered_points = points - center
        angles = np.arctan2(centered_points[:, 1], centered_points[:, 0])
        
        # Sort points by angle
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]
    
    else:
        # Default to nearest if method not recognized
        return order_points(points, "nearest")


def detect_gaps(ordered_points, threshold=3.0):
    """Detect significant gaps between adjacent points in the ordered sequence.
    
    Args:
        ordered_points (np.ndarray): Ordered points to check for gaps
        threshold (float): Multiplier of average distance to consider a gap significant
        
    Returns:
        tuple: (has_gap, max_gap_index, max_gap_size)
    """
    if len(ordered_points) < 3:
        return False, 0, 0
    
    # Calculate distances between consecutive points
    distances = np.zeros(len(ordered_points))
    for i in range(len(ordered_points) - 1):
        distances[i] = np.linalg.norm(ordered_points[i+1] - ordered_points[i])
    # Add distance from last to first point to close the loop
    distances[-1] = np.linalg.norm(ordered_points[0] - ordered_points[-1])
    
    # Calculate average distance
    avg_distance = np.mean(distances)
    max_gap = np.max(distances)
    max_gap_idx = np.argmax(distances)
    
    # Check if max gap exceeds the threshold
    if max_gap > (threshold * avg_distance):
        return True, max_gap_idx, max_gap
    else:
        return False, 0, 0


def calculate_aspect_ratio(points):
    """Calculate the aspect ratio of a set of points using PCA.
    
    Args:
        points (np.ndarray): Points to analyze (shape: Nx2)
        
    Returns:
        float: Aspect ratio (ratio of major to minor axis lengths)
    """
    if len(points) < 2:
        return 1.0
        
    try:
        pca = PCA(n_components=2)
        pca.fit(points)
        
        # Variance ratio gives the aspect ratio
        variances = pca.explained_variance_
        if variances[1] < 1e-10:  # Avoid division by zero
            return 100.0
            
        aspect_ratio = variances[0] / variances[1]
        return np.sqrt(aspect_ratio)
    except Exception as e:
        print(f"Error calculating aspect ratio: {e}")
        return 1.0


def calculate_convexity(points):
    """Calculate convexity as the ratio of the area to the convex hull area.
    
    Args:
        points (np.ndarray): Points to analyze (shape: Nx2)
        
    Returns:
        float: Convexity value (1.0 = perfectly convex)
    """
    if len(points) < 3:
        return 1.0
        
    try:
        # Get convex hull
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        # Approximate original shape area using the ordered points
        # This is an approximation using the shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # Convexity is the ratio of the area to the convex hull area
        if hull_area > 0:
            return area / hull_area
        else:
            return 1.0
    except Exception as e:
        print(f"Error calculating convexity: {e}")
        return 1.0


def apply_proximity_fallback(centered_points, minor_radius, initial_point_count):
    """Applies proximity filtering as a fallback method for boundary detection.
    
    Args:
        centered_points (np.ndarray): Points centered around the origin
        minor_radius (float): Estimate of cell minor radius
        initial_point_count (int): Number of points before filtering
        
    Returns:
        tuple: (selected_points, has_filtered)
    """
    print("  Applying proximity filter fallback...")
    selected_points = centered_points
    has_filtered = False

    try:
        # Apply DBSCAN to find the main cluster
        dbscan_eps = minor_radius * PARAMS['dbscan_eps_factor']
        clustering = DBSCAN(
            eps=dbscan_eps, 
            min_samples=PARAMS['dbscan_min_samples']
        ).fit(centered_points)
        
        labels = clustering.labels_
        unique_labels = np.unique(labels[labels != -1])
        
        if len(unique_labels) > 1:
            # Multiple clusters - select the one closest to origin
            min_dist = float('inf')
            best_label = -1
            
            for label in unique_labels:
                cluster_points = centered_points[labels == label]
                if len(cluster_points) < PARAMS['dbscan_min_samples']:
                    continue
                    
                dist = np.linalg.norm(np.mean(cluster_points, axis=0))
                if dist < min_dist:
                    min_dist = dist
                    best_label = label
                    
            if best_label != -1:
                selected_points = centered_points[labels == best_label]
                has_filtered = True
            else:
                # No suitable cluster found - keep all non-noise points
                selected_points = centered_points[labels != -1]
                has_filtered = True
                
        elif len(unique_labels) == 1:
            # Only one cluster - use it
            selected_points = centered_points[labels == unique_labels[0]]
            has_filtered = True
            
        elif len(labels[labels != -1]) > PARAMS['dbscan_min_samples']:
            # No clusters but enough non-noise points
            selected_points = centered_points[labels != -1]
            has_filtered = True
            
        print(f"  Proximity fallback finished. Selected {len(selected_points)} points. Filtered: {has_filtered}")
        
    except Exception as e:
        print(f"  Error during proximity fallback: {e}")
        # Just return the original points if there's an error
        has_filtered = False
        
    return selected_points, has_filtered


# =============================================================================
# CROSS-SECTION SPLITTING FUNCTIONS
# =============================================================================

def split_cross_section_by_projection(points, minor_radius):
    """Split a cross-section using projection onto the major PCA axis.
    
    Args:
        points (np.ndarray): Points to split
        minor_radius (float): Estimated minor radius of the cell
        
    Returns:
        tuple: (selected_points, has_split, used_fallback)
    """
    if len(points) < PARAMS['projection_min_points']:
        print("  Not enough points for projection clustering.")
        return points, False, False
        
    try:
        # 1. PCA to find Major Axis (axis of elongation)
        pca = PCA(n_components=2)
        pca.fit(points)
        v_major = pca.components_[0]  # Axis of elongation
        print(f"  Major axis vector: [{v_major[0]:.3f}, {v_major[1]:.3f}]")
        
        # 2. Project points onto the major axis
        projections_1d = (points @ v_major).reshape(-1, 1)
        
        # 3. 1D K-Means Clustering (k=2)
        kmeans = KMeans(
            n_clusters=2, 
            random_state=42, 
            n_init=PARAMS['kmeans_n_init']
        )
        labels = kmeans.fit_predict(projections_1d)
        
        # 4. Select Cluster based on centroid proximity to origin
        group1_indices = np.where(labels == 0)[0]
        group2_indices = np.where(labels == 1)[0]
        
        if len(group1_indices) >= 5 and len(group2_indices) >= 5:
            group1 = points[group1_indices]
            group2 = points[group2_indices]
            
            centroid1 = np.mean(group1, axis=0)
            centroid2 = np.mean(group2, axis=0)
            
            # Select the group closer to the origin
            if np.linalg.norm(centroid1) <= np.linalg.norm(centroid2):
                selected_points = group1
                print(f"  Projection split successful. Kept group 1 ({len(group1)} pts).")
            else:
                selected_points = group2
                print(f"  Projection split successful. Kept group 2 ({len(group2)} pts).")
                
            return selected_points, True, False
        else:
            print("  Projection split resulted in invalid groups. Reverting.")
            return points, False, False
            
    except Exception as e:
        print(f"  Error during projection splitting: {e}")
        return points, False, False


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cross_section(ax, points, has_split, minor_radius, section_index, is_closed_stomata=False):
    """Plot a single cross-section with appropriate styling.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        points (np.ndarray): Points to plot
        has_split (bool): Whether the section was split
        minor_radius (float): Estimated minor radius
        section_index (int): Index of this section
        is_closed_stomata (bool): Whether this is a closed stomata
    """
    # Order points for plotting
    ordered_points = order_points(points, method="angular")
    print(f"  Ordered {len(ordered_points)} points for plotting using 'angular' method.")
    
    # Check for significant gaps
    has_big_gap, max_gap_idx, max_gap_size = detect_gaps(
        ordered_points, 
        threshold=PARAMS['gap_threshold_multiplier']
    )
    
    # Determine plot color
    plot_color = PARAMS['plot_colors']['closed'] if is_closed_stomata else PARAMS['plot_colors']['open']
    
    # Plot as open line if split or has a large gap
    if has_split or has_big_gap:
        print(f"  Plotting section {section_index} as open line.")
        
        if has_big_gap:
            print(f"    Break detected after index {max_gap_idx} (gap size: {max_gap_size:.2f})")
        elif has_split:
            print(f"    Plotting open due to boundary split.")
            
        if len(ordered_points) > 1:
            # Roll the points to start after the gap if there is one
            start_plot_idx = (max_gap_idx + 1) if has_big_gap else 0
            points_to_plot = np.roll(ordered_points, -start_plot_idx, axis=0)
            
            # Plot the line
            ax.plot(
                points_to_plot[:, 0], 
                points_to_plot[:, 1], 
                f"{plot_color}-", 
                linewidth=PARAMS['plot_linewidth']
            )
            
        if len(ordered_points) > 0:
            # Plot start and end markers
            start_marker_idx = (max_gap_idx + 1) % len(ordered_points) if has_big_gap else 0
            end_marker_idx = max_gap_idx if has_big_gap else -1
            
            # Handle potential index errors
            if end_marker_idx == -1:
                end_marker_idx = len(ordered_points) - 1
                
            # Plot markers
            ax.plot(
                ordered_points[start_marker_idx, 0], 
                ordered_points[start_marker_idx, 1], 
                f"{plot_color}o", 
                markersize=PARAMS['plot_markersize']['start'], 
                label='Start'
            )
            ax.plot(
                ordered_points[end_marker_idx, 0], 
                ordered_points[end_marker_idx, 1], 
                f"{plot_color}x", 
                markersize=PARAMS['plot_markersize']['end'], 
                label='End'
            )
    else:
        # Plot as closed loop
        print(f"  Plotting section {section_index} as closed loop.")
        
        if len(ordered_points) > 1:
            # Plot a closed loop
            ax.plot(
                np.append(ordered_points[:, 0], ordered_points[0, 0]),
                np.append(ordered_points[:, 1], ordered_points[0, 1]),
                f"{plot_color}-", 
                linewidth=PARAMS['plot_linewidth']
            )
        elif len(ordered_points) == 1:
            # Plot a single point
            ax.plot(
                ordered_points[0, 0], 
                ordered_points[0, 1], 
                f"{plot_color}o", 
                markersize=PARAMS['plot_markersize']['single']
            )
            
    # Plot individual points
    ax.plot(
        ordered_points[:, 0], 
        ordered_points[:, 1], 
        f"{plot_color}.", 
        markersize=PARAMS['plot_markersize']['points']
    )
    
    # Plot reference circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = minor_radius * np.cos(theta)
    circle_y = minor_radius * np.sin(theta)
    ax.plot(circle_x, circle_y, 'k:', linewidth=1, alpha=0.3)


def add_metrics_text(ax, points):
    """Add metrics text box to the plot.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to add text to
        points (np.ndarray): Points to calculate metrics for
    """
    if len(points) > 0:
        try:
            aspect = calculate_aspect_ratio(points)
            convexity = calculate_convexity(order_points(points, method="angular"))
            props_text = f"Pts: {len(points)}\nAR: {aspect:.2f}\nC: {convexity:.2f}"
        except Exception:
            props_text = f"Pts: {len(points)}\nAR: -\nC: -"
    else:
        props_text = "Pts: 0"
        
    ax.text(
        0.05, 0.95, 
        props_text, 
        transform=ax.transAxes, 
        verticalalignment='top',
        fontsize=8, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    )


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_cross_section(cross_section, minor_radius, is_closed_stomata=False):
    """Process a single cross-section to extract points for plotting.
    
    Args:
        cross_section (np.ndarray): Cross-section points
        minor_radius (float): Estimated minor radius
        is_closed_stomata (bool): Whether this is a closed stomata
        
    Returns:
        tuple: (selected_points, has_split, used_fallback)
    """
    # Center the points around the origin
    if len(cross_section) == 0:
        return np.empty((0, 2)), False, False
        
    centroid = np.mean(cross_section, axis=0)
    centered = cross_section - centroid
    initial_point_count = len(centered)
    
    # For closed stomata, attempt to split the cross-section
    if is_closed_stomata:
        print(f"Section: Applying boundary detection split.")
        
        # Try projection-based splitting first
        selected_points, has_split, used_fallback = split_cross_section_by_projection(
            centered, minor_radius
        )
        
        # If projection splitting failed, use proximity fallback
        if not has_split:
            selected_points, has_fallback = apply_proximity_fallback(
                centered, minor_radius, initial_point_count
            )
            if has_fallback:
                has_split = True
                used_fallback = True
            else:
                # If fallback also failed, use all points
                selected_points = centered
                has_split = False
                used_fallback = False
    else:
        # For open stomata, use all points
        selected_points = centered
        has_split = False
        used_fallback = False
        
    return selected_points, has_split, used_fallback


def create_combined_cross_section_figure(cross_sections, valid_sections, minor_radius, output_dir=None):
    """Create a figure showing all cross-sections.
    
    Args:
        cross_sections (list): List of cross-section point arrays
        valid_sections (list): List of booleans indicating valid sections
        minor_radius (float): Estimated minor radius
        output_dir (str, optional): Directory to save output image
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Determine grid size based on number of cross-sections
    valid_count = sum(valid)
    grid_size = int(np.ceil(np.sqrt(valid_count)))
    
    # Create figure with grid
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(grid_size, grid_size, figure=fig)
    fig.suptitle(f"Cross Sections - Minor Radius: {minor_radius:.2f}", fontsize=16)
    
    # Keep track of next plot position
    plot_idx = 0
    
    # Process each valid cross-section
    for i, (cross_section, valid) in enumerate(zip(cross_sections, valid_sections)):
        if not valid:
            continue
            
        # Calculate subplot row and column
        row = plot_idx // grid_size
        col = plot_idx % grid_size
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f"Position {i}")
        
        # Check if this is for closed stomata
        is_closed_stomata = ('args' in globals() and 
                           hasattr(args, 'closed_stomata') and 
                           args.closed_stomata)
        
        # Process the cross-section
        selected_points, has_split, used_fallback = process_cross_section(
            cross_section, minor_radius, is_closed_stomata
        )
        
        # Plot the cross-section
        plot_cross_section(
            ax, selected_points, has_split, minor_radius, i, is_closed_stomata
        )
        
        # Add metrics text
        add_metrics_text(ax, selected_points)
        
        # Set axes properties
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Increment plot index
        plot_idx += 1
        
    # Finalize and save/show figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'all_individual_cross_sections.png')
        plt.savefig(save_path, dpi=150)
        print(f"Saved grid figure with {plot_idx} cross-sections to {save_path}")
    else:
        plt.show()
        
    plt.close(fig)
    return fig

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
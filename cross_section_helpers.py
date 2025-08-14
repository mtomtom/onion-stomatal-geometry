## Helper functions for the cross section analysis

import trimesh
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


## Define function to visualize the mesh

def visualize_mesh(mesh, extra_details=None, title="Mesh Visualization"):
    traces = [
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color='lightgray',
            opacity=0.75,
            name='Mesh'
        )
    ]
    if extra_details is not None:
        if isinstance(extra_details, list):
            traces.extend(extra_details)
        else:
            traces.append(extra_details)
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        title=title
    )
    fig.show()


def find_outer_edge(mesh, smoothing=50, n_slices=100, visualize=True):
    """
    Find the smooth outer edge of a mesh, resembling a C-shape.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The mesh to analyze
    smoothing : float
        Smoothing factor for the spline fit (higher = smoother)
    n_slices : int
        Number of slices to take along the y-axis
    visualize : bool
        Whether to show a 3D visualization of the result
        
    Returns:
    --------
    outer_points : ndarray
        The smoothed outer edge points (n_points, 3)
    """
    import numpy as np
    from scipy.interpolate import splprep, splev
    from scipy.stats import zscore
    
    # Get mesh vertices
    vertices = mesh.vertices
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Slice the mesh along y-axis
    y_min, y_max = y.min(), y.max()
    y_slices = np.linspace(y_min, y_max, n_slices)
    
    # Storage for edge points (x-min and x-max at each slice)
    left_points = []
    right_points = []
    
    for y_val in y_slices:
        # Define window size proportional to y-range and number of slices
        window = 0.2 * (y_max - y_min) / n_slices
        mask = np.abs(y - y_val) < window
        
        if np.sum(mask) < 3:  # Skip slices with too few points
            continue
            
        x_slice = x[mask]
        z_slice = z[mask]
        y_slice = y[mask]
        
        # Detect and remove outliers in x using z-score
        x_scores = np.abs(zscore(x_slice)) if len(x_slice) > 5 else np.zeros_like(x_slice)
        valid_mask = x_scores < 3.0  # Keep points within 3 standard deviations
        
        if np.sum(valid_mask) < 3:  # Skip if too few valid points
            continue
            
        x_slice = x_slice[valid_mask]
        y_slice = y_slice[valid_mask]
        z_slice = z_slice[valid_mask]
        
        # Get min/max x points
        idx_min = np.argmin(x_slice)
        idx_max = np.argmax(x_slice)
        
        left_points.append([x_slice[idx_min], y_slice[idx_min], z_slice[idx_min]])
        right_points.append([x_slice[idx_max], y_slice[idx_max], z_slice[idx_max]])
    
    # Convert to arrays
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    
    # Check if we have enough points
    if len(left_points) < 5 or len(right_points) < 5:
        raise ValueError("Not enough points to create smooth edges. Try increasing n_slices.")
    
    # Fit splines with increased smoothing
    left_tck, left_u = splprep(left_points.T, s=smoothing)
    right_tck, right_u = splprep(right_points.T, s=smoothing)
    
    # Generate smooth curves with more points for better interpolation
    u_dense = np.linspace(0, 1, 200)
    left_curve = np.array(splev(left_u, left_tck)).T
    right_curve = np.array(splev(right_u, right_tck)).T
    
    # Determine which curve is the outer edge (typically the right/max x curve)
    # For a C-shape, the outer edge is usually the one with larger x-coordinate span
    left_x_range = np.max(left_curve[:, 0]) - np.min(left_curve[:, 0])
    right_x_range = np.max(right_curve[:, 0]) - np.min(right_curve[:, 0])
    
    outer_points = right_curve if right_x_range >= left_x_range else left_curve
    
    # Apply final smoothing to ensure a very smooth curve
    tck, u = splprep(outer_points.T, s=smoothing * 1.5)
    outer_points = np.array(splev(u, tck)).T
    
    return outer_points

## Find the midpoint of the smooth outer points, and take a cross section at that point

def get_midpoint_trace(outer_points, mesh):

    # Find the midpoint of the smooth outer points
    mid_idx = len(outer_points) // 2
    midpoint = outer_points[mid_idx]

    # Estimate tangent at the midpoint (using neighbors)
    tangent = outer_points[mid_idx + 1] - outer_points[mid_idx - 1]
    tangent /= np.linalg.norm(tangent)

    # Take a cross section at the midpoint
    section = mesh.section(plane_origin=midpoint, plane_normal=tangent)

    # Get section points
    if section is not None:
        if hasattr(section, 'discrete'):
            section_points = np.vstack([seg for seg in section.discrete])
        else:
            section_points = section.vertices

       
        midpoint_trace = go.Scatter3d(
            x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]],
            mode='markers', marker=dict(size=8, color='black'), name='Midpoint'
        )

        section_trace = go.Scatter3d(
            x=section_points[:, 0], y=section_points[:, 1], z=section_points[:, 2],
            mode='markers', marker=dict(size=5, color='orange'), name='Midpoint Cross Section'
        )
        return midpoint_trace, section_trace, section_points
    else:
        print("No cross section found at midpoint.")


def create_projected_curve(mesh, outer_points, smooth=50, n_points=120, end_segment_size=10):
    """
    Create a curve that extends to the minimum x boundaries of the cell with 
    parallel end segments perpendicular to circle axes.
    """
    from scipy.interpolate import splprep, splev, CubicSpline
    import numpy as np
    
    # 1. Fit spline to original curve
    tck, u = splprep(outer_points.T, s=smooth)
    
    # 2. Make a copy of the input points for the central part
    central_curve = outer_points.copy()
    
    # 3. Find the min x value in the mesh (with a small buffer)
    min_x = mesh.vertices[:, 0].min() + 0.5  # Add a small buffer to avoid edge cases
    max_x = mesh.vertices[:, 0].max() - 0.5  # Add a small buffer to avoid edge cases
    
    # 4. Find mesh vertices near minimum x
    x_tol = 2.0  # Tolerance for considering points at minimum x
    near_min_x = mesh.vertices[np.abs(mesh.vertices[:, 0] - min_x) < x_tol]

    if len(near_min_x) > 0:
        # 5. For the start point: find the minimum x point with closest y-coordinate
        start_pt = outer_points[0]
        y_diffs_start = np.abs(near_min_x[:, 1] - start_pt[1])
        start_candidate_idx = np.argmin(y_diffs_start)
        start_boundary = near_min_x[start_candidate_idx]
        
        # 6. For the end point: find the minimum x point with closest y-coordinate
        end_pt = outer_points[-1]
        y_diffs_end = np.abs(near_min_x[:, 1] - end_pt[1])
        end_candidate_idx = np.argmin(y_diffs_end)
        end_boundary = near_min_x[end_candidate_idx]
        
        # 7. Create new curve with these endpoints
        extended_curve = np.vstack([start_boundary, central_curve, end_boundary])
    else:
        # Fallback to original curve if no min x points found
        extended_curve = central_curve
    
    # 8. Project onto mesh surface to ensure all points are on the mesh
    extended_curve = trimesh.proximity.closest_point(mesh, extended_curve)[0]
    
    # 9. Resample for smoothness with higher density at ends
    if len(extended_curve) >= 3:
        t = np.linspace(0, 1, len(extended_curve))
        t_new = np.linspace(0, 1, n_points)
        
        # Use cubic interpolation to maintain shape
        cs_x = CubicSpline(t, extended_curve[:,0])
        cs_y = CubicSpline(t, extended_curve[:,1])
        cs_z = CubicSpline(t, extended_curve[:,2])
        
        final_curve = np.column_stack([
            cs_x(t_new), cs_y(t_new), cs_z(t_new)
        ])
        
        # Ensure endpoints are exactly preserved
        final_curve[0] = extended_curve[0]
        final_curve[-1] = extended_curve[-1]
    else:
        # Fall back to original method if insufficient points
        tck2, u2 = splprep(extended_curve.T, s=smooth)
        final_curve = np.array(splev(np.linspace(0, 1, n_points), tck2)).T
    
    # 10. Apply additional smoothing
    tck3, u3 = splprep(final_curve.T, s=smooth*0.5)  # Use lighter smoothing to preserve shape
    final_curve_smooth = np.array(splev(np.linspace(0, 1, n_points), tck3)).T
    
    # 11. Make end segments parallel and perpendicular to circle axes
    # We assume the Y axis is the main axis of the circles based on code
    
    # Number of points to adjust at each end
    if end_segment_size > n_points // 4:
        end_segment_size = n_points // 4  # Limit to 1/4 of total points
    
    # Get tangent directions at the transition points
    start_transition_idx = end_segment_size
    end_transition_idx = n_points - end_segment_size - 1
    
    start_tangent = final_curve_smooth[start_transition_idx+1] - final_curve_smooth[start_transition_idx-1]
    end_tangent = final_curve_smooth[end_transition_idx+1] - final_curve_smooth[end_transition_idx-1]
    
    # Normalize tangents
    start_tangent = start_tangent / np.linalg.norm(start_tangent)
    end_tangent = end_tangent / np.linalg.norm(end_tangent)
    
    # Create vectors perpendicular to Y axis (in X-Z plane)
    # These will form the direction of our end segments
    y_axis = np.array([0, 1, 0])
    
    # For start segment: Project start_tangent to be perpendicular to Y
    start_dir = start_tangent - np.dot(start_tangent, y_axis) * y_axis
    start_dir = start_dir / np.linalg.norm(start_dir)
    
    # For end segment: Project end_tangent to be perpendicular to Y
    end_dir = end_tangent - np.dot(end_tangent, y_axis) * y_axis
    end_dir = end_dir / np.linalg.norm(end_dir)
    
    # Adjust the start segment points
    for i in range(end_segment_size):
        # Transition factor (0 at transition point, 1 at endpoint)
        factor = (end_segment_size - i) / end_segment_size
        # Original direction from transition point to current point
        orig_dir = final_curve_smooth[i] - final_curve_smooth[start_transition_idx]
        dist = np.linalg.norm(orig_dir)
        # New position using start_dir, preserving distance
        final_curve_smooth[i] = final_curve_smooth[start_transition_idx] - dist * start_dir * factor
    
    # Adjust the end segment points
    for i in range(end_segment_size):
        idx = n_points - i - 1  # Count from the end
        # Transition factor (0 at transition point, 1 at endpoint)
        factor = (end_segment_size - i) / end_segment_size
        # Original direction from transition point to current point
        orig_dir = final_curve_smooth[idx] - final_curve_smooth[end_transition_idx]
        dist = np.linalg.norm(orig_dir)
        # New position using end_dir, preserving distance
        final_curve_smooth[idx] = final_curve_smooth[end_transition_idx] + dist * end_dir * factor
    
    # 12. Project back onto mesh surface
    final_curve_smooth = trimesh.proximity.closest_point(mesh, final_curve_smooth)[0]
    
    # 13. Apply one final light smoothing to blend the adjusted ends
    tck4, u4 = splprep(final_curve_smooth.T, s=smooth*0.3)
    final_curve_smooth = np.array(splev(np.linspace(0, 1, n_points), tck4)).T
    
    # Ensure endpoints remain fixed
    final_curve_smooth[0] = final_curve[0]
    final_curve_smooth[-1] = final_curve[-1]
    
    return final_curve_smooth

def define_end_point_circles(outer_curve_projected, section_points):

    def cross_section_area_2d(points):
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points)
        hull = ConvexHull(points_2d)
        return hull.volume

    # Calculate area and radius
    area_mid = cross_section_area_2d(section_points)
    radius = np.sqrt(area_mid / np.pi)

    # Get endpoints of the projected outer curve
    start_pt = outer_curve_projected[0]
    end_pt = outer_curve_projected[-1]

    def make_circle_y(center, radius, n_points=50, y_offset=0):
        # Circle normal is Y axis, so circle lies in XZ plane
        angles = np.linspace(0, 2*np.pi, n_points)
        circle = np.zeros((n_points, 3))
        circle[:, 0] = center[0]   # X
        circle[:, 1] = center[1] + y_offset + radius * np.cos(angles)                         # Y (fixed)
        circle[:, 2] = center[2] + radius * np.sin(angles)  # Z
        return circle

    # Create circles at start and end, parallel to Y axis
    circle_start = make_circle_y(start_pt, radius, y_offset=radius)
    circle_end = make_circle_y(end_pt, radius, y_offset=-radius)

    # Plot mesh, projected outer curve, and circles
    circle_start_trace = go.Scatter3d(
        x=circle_start[:,0], y=circle_start[:,1], z=circle_start[:,2],
        mode='lines', line=dict(color='red', width=6), name='Start Circle'
    )
    circle_end_trace = go.Scatter3d(
        x=circle_end[:,0], y=circle_end[:,1], z=circle_end[:,2],
        mode='lines', line=dict(color='blue', width=6), name='End Circle'
    )
    outer_projected_trace = go.Scatter3d(
        x=outer_curve_projected[:,0], y=outer_curve_projected[:,1], z=outer_curve_projected[:,2],
        mode='lines+markers', marker=dict(size=4, color='green'), name='Outer Curve Projected'
    )

    ## Find and return the circle centres
    def find_circle_center(circle):
        # Find the average of all coordinates for the true center
        return np.mean(circle, axis=0)
    center_start = find_circle_center(circle_start)
    center_end = find_circle_center(circle_end)
    # Plot the centers
    center_start_trace = go.Scatter3d(
        x=[center_start[0]], y=[center_start[1]], z=[center_start[2]],
        mode='markers', marker=dict(size=8, color='red'), name='Start Circle Center'
    )
    center_end_trace = go.Scatter3d(
        x=[center_end[0]], y=[center_end[1]], z=[center_end[2]],
        mode='markers', marker=dict(size=8, color='blue'), name='End Circle Center'
    )

    return circle_start_trace, circle_end_trace, outer_projected_trace, center_start_trace, center_end_trace, center_start, center_end


def create_bezier_centreline(center_start, section_midpoint, center_end, curve_points=50):
    """
    Create a centreline using a cubic Bezier curve that exactly passes through
    the start, the specified midpoint (at t=0.5), and the end point,
    while keeping the control handles aligned toward the midpoint direction.

    Parameters:
    -----------
    center_start : ndarray
        Center of the start circle (3,)
    section_midpoint : ndarray
        Midpoint of the cross section to interpolate (3,)
    center_end : ndarray
        Center of the end circle (3,)
    curve_points : int
        Number of sample points along the Bezier curve

    Returns:
    --------
    centreline : ndarray
        The fitted centreline points (curve_points, 3)
    centreline_trace : go.Scatter3d
        Plotly trace of the centreline
    """
    import numpy as np
    import plotly.graph_objects as go

    # Ensure floats for arithmetic
    P0 = np.array(center_start, dtype=float)
    M  = np.array(section_midpoint, dtype=float)
    P3 = np.array(center_end, dtype=float)

    # Compute exact handle_sum so that B(0.5) == M
    handle_sum = (8 * M - P0 - P3) / 3.0

    # Compute primary direction from P0 toward midpoint (project off Y)
    y_axis = np.array([0, 1, 0], dtype=float)
    dir_to_mid = M - P0
    dir_to_mid -= np.dot(dir_to_mid, y_axis) * y_axis
    if np.linalg.norm(dir_to_mid) > 1e-6:
        dir_to_mid /= np.linalg.norm(dir_to_mid)
    # Place P1 along this direction, with length = 2/3 of chord length
    length_P1 = 2.0 / 3.0 * np.linalg.norm(M - P0)
    P1 = P0 + dir_to_mid * length_P1

    # Derive P2 so the handles still sum correctly
    P2 = handle_sum - P1

    # Parameterize t and build the curve
    t = np.linspace(0.0, 1.0, curve_points)[:, None]
    centreline = ((1 - t)**3) * P0 \
               + 3 * ((1 - t)**2) * t * P1 \
               + 3 * (1 - t) * (t**2) * P2 \
               + (t**3) * P3

    # Build Plotly 3D trace
    centreline_trace = go.Scatter3d(
        x=centreline[:, 0],
        y=centreline[:, 1],
        z=centreline[:, 2],
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(width=3),
        name='Bezier Centreline'
    )

    return centreline, centreline_trace

def align_mesh_to_y_axis(mesh):
    """
    Aligns the mesh so its longest principal axis is parallel to the Y-axis,
    and then centers the mesh at the origin.

    This standardizes the mesh's orientation for consistent analysis.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh.

    Returns:
    --------
    aligned_mesh : trimesh.Trimesh
        A new mesh, rotated and centered.
    """
    # 1. Use PCA on the vertices to find the principal axes
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)
    
    # The first component corresponds to the longest axis
    longest_axis = pca.components_[0]
    
    # The target axis is the world Y-axis
    y_axis = np.array([0, 1, 0])

    # 2. Find the 4x4 transformation matrix that aligns the longest axis with the Y-axis
    transform_matrix = trimesh.geometry.align_vectors(longest_axis, y_axis)

    # 3. Apply the transformation to a copy of the mesh to avoid modifying the original
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(transform_matrix)
    
    # 4. Center the aligned mesh at the origin for consistency
    aligned_mesh.vertices -= aligned_mesh.centroid

    return aligned_mesh

def extract_cross_section_data(file, path, side="single_a", n_sections=20):
    """
    Extract cross-section data from a mesh without creating visualizations.
    
    Parameters:
    -----------
    file_index : int
        Index of the file to analyze from the files list
    side : str
        Side to analyze ("single_a" or "single_b")
    n_sections : int
        Number of cross-sections to generate along the centreline
        
    Returns:
    --------
    data : dict
        Dictionary containing all computed data
    """
    from sklearn.decomposition import PCA
    from skimage.measure import EllipseModel
    
    # Initialize results dictionary
    results = {
        'file_info': {
            'file_name': file,
            'side': side,
            'mesh_path': f"{path}{file}/{file}_{side}.obj"
        },
        'mesh_data': {},
        'cross_sections': {
            'indices': [],
            '3d_points': [],
            '2d_points': [],
            'pca_components': [],
            'std_devs': [],
            'aspect_ratios': [],
            'ellipse_fits': []
        }
    }
    
    # Load and process mesh
    mesh_path = results['file_info']['mesh_path']
    mesh = trimesh.load(mesh_path, process=True)
    
    # Align the mesh to the Y-axis and center it
    mesh = align_mesh_to_y_axis(mesh)
    
    # Ensure consistent orientation
    outer_points_check = find_outer_edge(mesh, smoothing=50)
    if np.mean(outer_points_check[:, 0]) < 0:
        mesh.vertices[:, 0] *= -1
    
    # Process mesh to find centreline
    outer_points = find_outer_edge(mesh, smoothing=50)
    outer_curve_projected = create_projected_curve(
        mesh, outer_points, smooth=50, n_points=120, end_segment_size=10
    )

    # Get midpoint cross-section
    _, _, section_points = get_midpoint_trace(outer_points, mesh)

    # Get the circle centers
    _, _, _, _, _, center_start, center_end = define_end_point_circles(outer_curve_projected, section_points)

    # Create the centreline using the actual circle centers
    section_midpoint = np.mean(section_points, axis=0)
    centreline, _ = create_bezier_centreline(
        center_start, section_midpoint, center_end, curve_points=50
    )

    # Store centreline in results
    results['mesh_data']['centreline'] = centreline
    results['mesh_data']['outer_points'] = outer_points
    results['mesh_data']['outer_curve_projected'] = outer_curve_projected
    
    # Define cross-section indices
    end_margin_points = 3  # Avoid unstable sections at the very ends
    start_index = end_margin_points
    end_index = len(centreline) - 1 - end_margin_points
    indices = np.linspace(start_index, end_index, n_sections, dtype=int)
    results['cross_sections']['indices'] = indices.tolist()
    
    # Get all cross-sections
    all_section_points = get_cross_section_points(mesh, centreline, indices)
    
    # Calculate tangents along the centreline
    tangents = np.gradient(centreline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    results['mesh_data']['tangents'] = tangents
    
    # Define a consistent reference for orientation
    pattern_reference = np.array([1.0, 0.0, 1.0])  # Diagonal pattern in X-Z plane
    pattern_reference = pattern_reference / np.linalg.norm(pattern_reference)  # Normalize
    x_axis = np.array([1, 0, 0])
    
    # Process each cross-section
    for i, section_points in enumerate(all_section_points):
        if section_points is None or len(section_points) < 3:
            # Store empty data for missing sections
            results['cross_sections']['3d_points'].append(None)
            results['cross_sections']['2d_points'].append(None)
            results['cross_sections']['pca_components'].append(None)
            results['cross_sections']['std_devs'].append(None)
            results['cross_sections']['aspect_ratios'].append(np.nan)
            results['cross_sections']['ellipse_fits'].append(None)
            continue
            
        idx = indices[i]
        tangent = tangents[idx]
        
        # Store 3D points
        results['cross_sections']['3d_points'].append(section_points.tolist())
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(section_points)
        std_devs = np.sqrt(pca.explained_variance_)
        
        # Store PCA components and std_devs
        results['cross_sections']['pca_components'].append(pca.components_.tolist())
        results['cross_sections']['std_devs'].append(std_devs.tolist())

        # Get PCA components
        component_0 = pca.components_[0]
        component_1 = pca.components_[1]

        # CONSISTENT HORIZONTAL ALIGNMENT:
        # Check which component is more aligned with X-axis
        dot_0_x = np.abs(np.dot(component_0, x_axis))
        dot_1_x = np.abs(np.dot(component_1, x_axis))

        # The component more aligned with X is the major axis
        if dot_0_x >= dot_1_x:
            main_axis_vec = component_0 if np.dot(component_0, x_axis) >= 0 else -component_0
            minor_axis_vec = component_1 if np.cross(main_axis_vec, component_1)[1] >= 0 else -component_1
            main_axis_std, minor_axis_std = std_devs[0], std_devs[1]
        else:
            main_axis_vec = component_1 if np.dot(component_1, x_axis) >= 0 else -component_1
            minor_axis_vec = component_0 if np.cross(main_axis_vec, component_0)[1] >= 0 else -component_0
            main_axis_std, minor_axis_std = std_devs[1], std_devs[0]
        
        # Calculate aspect ratio
        if minor_axis_std > 1e-9:
            aspect_ratio = main_axis_std / minor_axis_std
        else:
            aspect_ratio = np.nan
            
        # Store aspect ratio
        results['cross_sections']['aspect_ratios'].append(float(aspect_ratio))
        
        # Project points to 2D using the principal axes
        centroid = np.mean(section_points, axis=0)
        centered = section_points - centroid
        x_coords = centered @ main_axis_vec
        y_coords = centered @ minor_axis_vec
        points_2d = np.column_stack([x_coords, y_coords])
        
        # Store 2D points
        results['cross_sections']['2d_points'].append(points_2d.tolist())
        
        # Fit an ellipse to the 2D points
        ellipse_data = None
        try:
            ellipse_model = EllipseModel()
            if len(points_2d) >= 5 and ellipse_model.estimate(points_2d):
                xc, yc, a, b, theta = ellipse_model.params
                
                # Generate points along the ellipse
                t = np.linspace(0, 2*np.pi, 100)
                ellipse_x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
                ellipse_y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
                ellipse_points = np.column_stack([ellipse_x, ellipse_y])
                
                # Calculate ellipse aspect ratio
                ellipse_ar = max(a, b) / min(a, b)
                
                # Store ellipse data
                ellipse_data = {
                    'center': [float(xc), float(yc)],
                    'axes_lengths': [float(a), float(b)],
                    'angle': float(theta),
                    'aspect_ratio': float(ellipse_ar),
                    'points': ellipse_points.tolist()
                }
        except Exception as e:
            pass
            
        # Store ellipse data
        results['cross_sections']['ellipse_fits'].append(ellipse_data)
    
    return results

## Function to iterate over a number of meshes, and create plots showing the aspect ratio of the cross sections along the guard cells
def get_cross_section_points(mesh, centreline, indices):
    """
    For a given set of indices along a centreline, compute the cross-section points.
    Ensures only the segment nearest the centreline point is returned.
    """
    tangents = np.gradient(centreline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    
    sections_points_list = []
    for idx in indices:
        midpoint = centreline[idx]
        tangent = tangents[idx]
        
        section = mesh.section(plane_origin=midpoint, plane_normal=tangent)
        if section is None:
            continue

        if hasattr(section, 'discrete') and section.discrete:
            segments = section.discrete
            centroids = [seg.mean(axis=0) for seg in segments]
            dists = [np.linalg.norm(c - midpoint) for c in centroids]
            nearest_seg = segments[int(np.argmin(dists))]
            sections_points_list.append(nearest_seg)
        else:
            if section.vertices.shape[0] > 0:
                sections_points_list.append(section.vertices)
                
    return sections_points_list

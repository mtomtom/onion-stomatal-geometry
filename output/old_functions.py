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

def move_circle_out_of_mesh(circle, center, plane_normal, mesh, step=0.05, max_dist=10.0, margin=0.1):
    """
    Move the circle along plane_normal until it is outside the mesh, then move back by 'margin'.
    """
    import trimesh
    moved_center = center.copy()
    moved_circle = circle.copy()
    dist = 0.0
    while dist < max_dist:
        # Check if any point is inside the mesh
        inside = np.any(mesh.contains(moved_circle))
        if not inside:
            # Move back by 'margin'
            moved_center += plane_normal * margin
            moved_circle += plane_normal * margin
            break
        # Move further out
        moved_center -= plane_normal * step
        moved_circle -= plane_normal * step
        dist += step
    return moved_circle, moved_center



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

def load_and_analyze(args):
    pressure, obj_path, mesh_id = args
    filename = f"{obj_path}{mesh_id}_{pressure:.1f}.obj"
    mesh = trimesh.load_mesh(filename)
    section_points_left, section_points_right, [spline_x, spline_y, spline_z] = analyze_stomata_mesh(
        filename, num_sections=20, n_points=40, visualize=False
    )
    return mesh, section_points_left, section_points_right, [spline_x, spline_y, spline_z]

def define_end_point_circles(outer_curve_projected, section_points, n_points=64, eps=1e-9, y_rotation_deg=0.0, mesh=None, flip_circle_directions=False):
    """
    Place two circles at ends of outer_curve_projected that lie FLAT on the ground (YZ plane).
    - Ground plane normal = X axis.
    - Each circle passes through its endpoint; its diameter (e1) is parallel to the midpoint tangent
      projected into the ground plane (so it matches the midsection orientation).
    - Radius is computed from the midpoint cross-section (section_points).
    Returns the same Plotly traces and centers as before.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    def unit(v):
        n = np.linalg.norm(v)
        return v if n < eps else v / n

    def cross_section_area_2d(points):
        pca = PCA(n_components=2)
        pts2 = pca.fit_transform(points)
        hull = ConvexHull(pts2)
        return hull.volume  # 2D 'volume' == area

    # radius from midpoint cross-section
    area_mid = cross_section_area_2d(section_points)
    radius = np.sqrt(max(area_mid, 0.0) / np.pi)

    # curve and endpoints
    P = np.asarray(outer_curve_projected, dtype=float)
    start_pt = P[0]
    end_pt = P[-1]

    # midpoint tangent (central difference)
    nP = len(P)
    if nP < 3:
        T_mid = unit(P[-1] - P[0])
    else:
        mid = nP // 2
        i0 = max(0, mid - 1)
        i2 = min(nP - 1, mid + 1)
        T_mid = unit(P[i2] - P[i0])
    if np.linalg.norm(T_mid) < eps:
        T_mid = np.array([0.0, 1.0, 0.0])

    # Plane normal = X axis so circle lies flat on ground (YZ plane)
    plane_normal = np.array([1.0, 0.0, 0.0])

    # Reference direction: midpoint tangent projected into ground plane (remove X component)
    ref = np.array([0.0, T_mid[1], T_mid[2]])
    if np.linalg.norm(ref) < eps:
        ref = np.array([0.0, 1.0, 0.0])
    # Ensure ref is in-plane by removing any component along plane_normal (should already be)
    ref = ref - np.dot(ref, plane_normal) * plane_normal
    e1_candidate = unit(ref)
    if np.linalg.norm(e1_candidate) < eps:
        # fallback
        e1_candidate = np.array([0.0, 1.0, 0.0])

    # midpoint centroid used to choose sign so center moves inward
    mid_centroid = np.mean(np.asarray(section_points, dtype=float), axis=0)

    def build_flat_circle(endpoint):
        """
        Choose e1 = ±e1_candidate such that center = endpoint - radius*e1 (endpoint on rim)
        yields a center closer to mid_centroid (so circle interior faces inward).
        """
        e1_pos = e1_candidate
        center_pos = endpoint - radius * e1_pos
        dist_pos = np.linalg.norm(mid_centroid - center_pos)

        e1_neg = -e1_candidate
        center_neg = endpoint - radius * e1_neg
        dist_neg = np.linalg.norm(mid_centroid - center_neg)

        if dist_pos < dist_neg:
            e1 = e1_pos
            center = center_pos
        else:
            e1 = e1_neg
            center = center_neg

        # e2 is the orthogonal in-plane axis (plane_normal x e1 gives e2 or -e2)
        e2 = unit(np.cross(plane_normal, e1))

        # Build circle so endpoint corresponds to angle 0: center + radius*e1 == endpoint
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)
        circle = np.array([center + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles])
        return circle, center, e1, e2

    circle_start, center_start, e1s, e2s = build_flat_circle(start_pt)
    circle_end,   center_end,   e1e, e2e   = build_flat_circle(end_pt)

    circle_start, center_start = move_circle_out_of_mesh(circle_start, center_start, plane_normal, mesh)
    circle_end, center_end = move_circle_out_of_mesh(circle_end, center_end, plane_normal, mesh)

    # optional final yaw around global Y (kept for compatibility)
    if abs(y_rotation_deg) > 1e-6:
        angle_rad = np.deg2rad(y_rotation_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[ c, 0.0,  s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0,  c]])
        circle_start = (circle_start - center_start) @ R.T + center_start
        circle_end   = (circle_end   - center_end)   @ R.T + center_end

    # ---- plotly traces ----
    circle_start_trace = go.Scatter3d(
        x=circle_start[:, 0], y=circle_start[:, 1], z=circle_start[:, 2],
        mode='lines', line=dict(width=6), name='Start Circle'
    )
    circle_end_trace = go.Scatter3d(
        x=circle_end[:, 0], y=circle_end[:, 1], z=circle_end[:, 2],
        mode='lines', line=dict(width=6), name='End Circle'
    )
    outer_projected_trace = go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2],
        mode='lines+markers', marker=dict(size=4), name='Outer Curve Projected'
    )
    center_start_trace = go.Scatter3d(
        x=[center_start[0]], y=[center_start[1]], z=[center_start[2]],
        mode='markers', marker=dict(size=8), name='Start Circle Center'
    )
    center_end_trace = go.Scatter3d(
        x=[center_end[0]], y=[center_end[1]], z=[center_end[2]],
        mode='markers', marker=dict(size=8), name='End Circle Center'
    )

    return (
        circle_start_trace, circle_end_trace, outer_projected_trace,
        center_start_trace, center_end_trace, center_start, center_end, circle_start, circle_end
    )

def visualize_mesh_with_cross_sections(mesh=None, outer_points = None, centreline = None, end_margin_points = 3, n_sections=20, colormap='viridis'):
    """
    Create a 3D visualization of a single mesh with all cross-sections.
    
    Parameters:
    -----------
    file_index : int
        Index of the file to visualize from the files list
    side : str
        Side to analyze ("single_a" or "single_b")
    n_sections : int
        Number of cross-sections to generate along the centreline
    colormap : str
        Matplotlib colormap name for coloring the cross-sections
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The 3D figure object that can be further customized if needed
    """
    import plotly.graph_objects as go
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    # Load and process mesh
    #mesh_path = f"{path}/{file}_{side}.obj"
    #mesh = trimesh.load(mesh_path, process=True)
    
    # Align the mesh to the Y-axis and center it
    #mesh = align_mesh_to_y_axis(mesh)
    
    # Ensure consistent orientation
    #outer_points_check = find_outer_edge(mesh, smoothing=50)
    #if np.mean(outer_points_check[:, 0]) < 0:
    #    mesh.vertices[:, 0] *= -1
    
    # Process mesh to find centreline
    #outer_points = find_outer_edge(mesh, smoothing=50)
    #outer_curve_projected = create_projected_curve(
    #    mesh, outer_points, smooth=50, n_points=120, #end_segment_size=10
    #)

    # Get midpoint cross-section
    #midpoint_trace, section_trace, section_points = #get_midpoint_trace(outer_points, mesh)

    # Get the circle centers
    #_, _, _, _, _, center_start, center_end, circle_start, #circle_end = define_end_point_circles(outer_points, section_points, mesh=mesh)

    # Create the centreline using the actual circle centers
    #section_midpoint = np.mean(section_points, axis=0)
    #centreline, centreline_trace = create_bezier_centreline(
    #   center_start, section_midpoint, center_end, curve_points=50
    #)

    # Define cross-section indices
    end_margin_points = end_margin_points  # Avoid unstable sections at the very ends
    start_index = end_margin_points
    end_index = len(centreline) - 1 - end_margin_points
    indices = np.linspace(start_index, end_index, n_sections, dtype=int)
    
    # Get all cross-sections
    all_section_points = get_cross_section_points(mesh, centreline, indices)
    
    # Calculate tangents along the centreline
    tangents = np.gradient(centreline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    
    # Create a colormap for cross-sections based on position
    color_scale = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(all_section_points)))
    
    # Create visualization traces
    traces = []
    
    # Add the mesh with transparency
    traces.append(go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        color='lightgray', opacity=0.5, name='Mesh'
    ))
    
    # Add the centreline
    traces.append(go.Scatter3d(
        x=centreline[:, 0], y=centreline[:, 1], z=centreline[:, 2],
        mode='lines', line=dict(color='black', width=3), name='Centreline'
    ))
    
    # Process each cross-section
    for i, section_points in enumerate(all_section_points):
        if section_points is None or len(section_points) < 3:
            continue
            
        idx = indices[i]
        section_centroid = np.mean(section_points, axis=0)
        
        # Get RGB color for this cross-section
        color_rgb = color_scale[i][:3]  # Extract RGB (ignore alpha)
        color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
        
        # Sort points for a clean polygon (circular order around centroid)
        tangent = tangents[idx]
        points_2d = section_points - section_centroid
        
        # Project onto plane perpendicular to tangent
        ref = np.array([0, 0, 1]) if not np.allclose(tangent, [0, 0, 1]) else np.array([1, 0, 0])
        v1 = np.cross(tangent, ref)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(tangent, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        x_2d = points_2d @ v1
        y_2d = points_2d @ v2
        
        # Sort by polar angle
        angles = np.arctan2(y_2d, x_2d)
        sorted_indices = np.argsort(angles)
        
        # Get sorted 3D points
        sorted_points = section_points[sorted_indices]
        
        # Close the loop by adding the first point at the end
        closed_points = np.vstack([sorted_points, sorted_points[0]])
        
        # Add the cross-section as a line trace
        traces.append(go.Scatter3d(
            x=closed_points[:, 0], y=closed_points[:, 1], z=closed_points[:, 2],
            mode='lines', line=dict(color=color, width=5),
            name=f'Section {i+1}'
        ))
    
    # Create the figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        #title=f"Cross-Sections for {file}_{side}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=0, z=0.5)  # Adjust for better viewing angle
            )
        )
    )
    
    # Save the visualization as an HTML file for interactive viewing
    output_filename = f"cross_sections_3d.html"
    fig.write_html(output_filename, include_plotlyjs='cdn')
    print(f"Saved 3D visualization to {output_filename}")
    
    return fig, all_section_points

def find_wall_vertices(mesh: trimesh.Trimesh, dihedral_deg=175.0):
    """
    Return indices of vertices that lie on 'wall' seams where adjacent faces
    have almost opposite normals (dihedral angle near 180°).
    """
    if mesh.face_adjacency.shape[0] == 0:
        return np.array([], dtype=int)

    angles = mesh.face_adjacency_angles  # angle between face normals (0 ~ same dir, pi ~ opposite)
    wall_mask = angles > np.deg2rad(dihedral_deg)
    if not np.any(wall_mask):
        return np.array([], dtype=int)

    wall_face_pairs = mesh.face_adjacency[wall_mask]              # (k,2)
    wall_faces = np.unique(wall_face_pairs.ravel())               # face indices involved
    wall_vertices = np.unique(mesh.faces[wall_faces].ravel())     # vertex indices
    return wall_vertices

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
    rotation_matrix = trimesh.geometry.align_vectors(longest_axis, y_axis)

    # 3. Apply the transformation to a copy of the mesh to avoid modifying the original
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(rotation_matrix)

    # 4. Compute translation to center at origin
    centroid = aligned_mesh.centroid
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -centroid

    # 5. Apply translation
    aligned_mesh.apply_transform(translation_matrix)
    # 6. Combine rotation and translation into one matrix
    full_transform = translation_matrix @ rotation_matrix

    return aligned_mesh, full_transform



## Function to iterate over a number of meshes, and create plots showing the aspect ratio of the cross sections along the guard cells: OBSOLETE


import numpy as np

def get_cross_section_points_with_normals(
    mesh,
    centreline,
    indices,
    normal_source='centreline',   # 'centreline' or 'svd'
    enforce_consistent_normal=True
):
    """
    Compute cross-section point clouds and return slice normals.

    Returns (sections_list, normals_list, midpoints_list) where each list has the
    same length as `indices`. Failed slices produce None entries.

    normal_source:
      - 'centreline' : return the tangent at the centreline index (fast)
      - 'svd'        : compute normal from the section points via SVD (robust)

    enforce_consistent_normal: if True, flip a computed normal so it faces the
    same hemisphere as the previous non-None normal (reduces sign flip issues).
    """
    centreline = np.asarray(centreline, dtype=float)
    # compute stable centreline tangents (same approach as before)
    tangents = np.gradient(centreline, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    tangents = tangents / norms
    for i in range(1, len(tangents)):
        if np.dot(tangents[i], tangents[i-1]) < 0:
            tangents[i] *= -1.0

    # containers aligned to indices
    n_idx = len(indices)
    sections = [None] * n_idx
    normals = [None] * n_idx
    midpoints = [None] * n_idx

    prev_normal = None

    # fallback ordering if user has no order_points_consistently function
    def _order_points_consistently_fallback(pts3, tangent, midpoint):
        # order by angle in-plane using SVD to get a stable plane basis
        X = pts3 - pts3.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(X, full_matrices=False)
            normal_est = vh[-1]
        except Exception:
            normal_est = tangent
        # build in-plane basis
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, normal_est)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0])
        v1 = np.cross(normal_est, ref)
        nv1 = np.linalg.norm(v1)
        if nv1 < 1e-12:
            return pts3  # give up
        v1 /= nv1
        v2 = np.cross(normal_est, v1)
        v2 /= max(np.linalg.norm(v2), 1e-12)
        # project to 2D and sort by angle around centroid
        pts2 = np.column_stack([ (pts3 - midpoint) @ v1, (pts3 - midpoint) @ v2 ])
        centroid2 = pts2.mean(axis=0)
        angles = np.arctan2(pts2[:,1] - centroid2[1], pts2[:,0] - centroid2[0])
        order = np.argsort(angles)
        return pts3[order]

    for out_i, idx in enumerate(indices):
        # bounds guard
        if idx < 0 or idx >= len(centreline):
            continue
        midpoint = centreline[idx]
        tangent = tangents[idx]

        section = mesh.section(plane_origin=midpoint, plane_normal=tangent)
        if section is None:
            # leave None entries
            continue

        # obtain an ordered polyline for the segment nearest midpoint
        chosen_seg = None
        if hasattr(section, 'discrete') and section.discrete:
            # pick the segment whose centroid is nearest to the midpoint
            segments = [seg for seg in section.discrete if seg.shape[0] >= 3]
            if len(segments) == 0:
                # nothing useful
                continue
            centroids = [seg.mean(axis=0) for seg in segments]
            dists = [np.linalg.norm(c - midpoint) for c in centroids]
            chosen_seg = segments[int(np.argmin(dists))]
        else:
            # fallback to merged vertices - try to order them
            if not (hasattr(section, 'vertices') and section.vertices is not None and section.vertices.shape[0] >= 3):
                continue
            verts = section.vertices
            # try to use user's ordering function if present
            try:
                ordered = order_points_consistently(verts, tangent, midpoint)
            except NameError:
                ordered = _order_points_consistently_fallback(verts, tangent, midpoint)
            chosen_seg = ordered

        # store chosen segment and midpoint
        sections[out_i] = chosen_seg
        midpoints[out_i] = midpoint

        # compute the returned normal according to requested source
        if normal_source == 'centreline':
            n = tangent.copy()
        elif normal_source == 'svd':
            # compute from chosen segment points
            pts = chosen_seg - chosen_seg.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(pts, full_matrices=False)
                n = vh[-1].copy()
            except Exception:
                n = tangent.copy()
        else:
            raise ValueError("normal_source must be 'centreline' or 'svd'")

        # ensure unit and non-zero
        nnorm = np.linalg.norm(n)
        if nnorm < 1e-12:
            n = tangent.copy()
            nnorm = np.linalg.norm(n)
        n /= nnorm

        # enforce hemisphere consistency to reduce basis flipping
        if enforce_consistent_normal and prev_normal is not None and np.dot(n, prev_normal) < 0:
            n = -n

        normals[out_i] = n
        prev_normal = n

    return sections, normals, midpoints

def get_guard_cell_centreline(top_anchor, bottom_anchor, section_centres):
    """
    Estimate a smooth centreline for a single guard cell.
    
    Parameters
    ----------
    top_anchor : (3,) array
        Point at the top of the guard cell (must be included in the spline).
    bottom_anchor : (3,) array
        Point at the bottom of the guard cell (must be included in the spline).
    section_centres : (M,3) array
        Optional intermediate centres along the guard cell cross-sections.

    Returns
    -------
    spline_trace : go.Scatter3d
        Plotly line trace of the spline.
    spline_points : (N,3) array
        Interpolated centreline points.
    """
    # Stack points in order: top → intermediates → bottom
    control_points = np.vstack([top_anchor, section_centres, bottom_anchor])
    
    # Parametric coordinate
    t = np.linspace(0, 1, len(control_points))
    t_fine = np.linspace(0, 1, 200)
    
    # Natural cubic spline (not periodic)
    spline_x = CubicSpline(t, control_points[:, 0], bc_type='natural')(t_fine)
    spline_y = CubicSpline(t, control_points[:, 1], bc_type='natural')(t_fine)
    spline_z = CubicSpline(t, control_points[:, 2], bc_type='natural')(t_fine)
    
    spline_points = np.vstack([spline_x, spline_y, spline_z]).T
    
    spline_trace = go.Scatter3d(
        x=spline_x, y=spline_y, z=spline_z,
        mode='lines',
        line=dict(color='black', width=6),
        name='Guard Cell Centreline'
    )
    
    return spline_trace, spline_points


def improve_centre_line_with_anchors( mesh, spline_x, spline_y, spline_z, centre_top, centre_bottom, left_section_centre, right_section_centre, num_sections=40):

    indices = np.linspace(0, len(spline_x) - 1, num_sections + 6, dtype=int)  # +6 to allow for removal

    # Find the closest index in the spline to each wall centre
    spline_points = np.column_stack([spline_x, spline_y, spline_z])
    top_idx = np.argmin(np.linalg.norm(spline_points - centre_top, axis=1))
    bottom_idx = np.argmin(np.linalg.norm(spline_points - centre_bottom, axis=1))

    def remove_near_wall(idx, indices, window=1):
        closest = np.argmin(np.abs(indices - idx))
        to_remove = [(closest + offset) % len(indices) for offset in range(-window, window+1)]
        return set(to_remove)

    remove_set = remove_near_wall(top_idx, indices, window=1) | remove_near_wall(bottom_idx, indices, window=1)
    keep_mask = np.ones(len(indices), dtype=bool)
    for i in remove_set:
        keep_mask[i] = False
    final_indices = indices[keep_mask]
    final_indices = final_indices[:num_sections]

    section_points_list = []
    section_centroids = []

    for idx in final_indices:
        section_points = get_cross_section_points(mesh, spline_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            centroid = section_points.mean(axis=0)
            section_centroids.append(centroid)

    centroids = np.array(section_centroids)

    # Insert anchor points at the closest centroid locations
    anchors = [
        centre_top,
        left_section_centre,
        centre_bottom,
        right_section_centre,
        centre_top  # repeat to close the loop
    ]
    anchor_indices = [np.argmin(np.linalg.norm(centroids - anchor, axis=1)) for anchor in anchors[:-1]]
    centroids_with_anchors = centroids.copy()
    offset = 0
    for idx, anchor in sorted(zip(anchor_indices, anchors[:-1])):
        centroids_with_anchors = np.insert(centroids_with_anchors, idx + offset, anchor, axis=0)
        offset += 1
    centroids_with_anchors = np.vstack([centroids_with_anchors, centroids_with_anchors[0]])

    # Polyline with anchors
    polyline_trace = go.Scatter3d(
        x=centroids_with_anchors[:,0], y=centroids_with_anchors[:,1], z=centroids_with_anchors[:,2],
        mode='lines', line=dict(color='orange', width=4), name='Centroid Polyline + Anchors'
    )

    # Smoothed polyline
    window_length = 7 if len(centroids_with_anchors) > 7 else len(centroids_with_anchors) // 2 * 2 + 1
    smoothed = np.column_stack([
        savgol_filter(centroids_with_anchors[:,i], window_length=window_length, polyorder=2, mode='wrap')
        for i in range(3)
    ])
    smoothed_trace = go.Scatter3d(
        x=smoothed[:,0], y=smoothed[:,1], z=smoothed[:,2],
        mode='lines', line=dict(color='red', width=6), name='Smoothed Centroid Line'
    )

    # Optionally, show anchor points as markers
    anchor_trace = go.Scatter3d(
        x=[a[0] for a in anchors], y=[a[1] for a in anchors], z=[a[2] for a in anchors],
        mode='markers+text', marker=dict(size=8, color='black'), name='Anchors',
        text=['Top','Left','Bottom','Right','Top'], textposition='top center'
    )

    return polyline_trace, smoothed_trace, anchor_trace, smoothed

from scipy.signal import savgol_filter
import numpy as np
import plotly.graph_objects as go

def improve_guard_cell_centreline(
    mesh,
    spline_x, spline_y, spline_z,
    centre_top, centre_bottom,
    num_sections=40
):
    """
    Refine a guard cell centreline by sampling cross-sections
    and re-fitting through centroids, anchored at top and bottom.
    """
    spline_points = np.column_stack([spline_x, spline_y, spline_z])

    # Choose evenly spaced sample indices along the spline
    indices = np.linspace(0, len(spline_points) - 1, num_sections, dtype=int)

    section_points_list = []
    section_centroids = []

    for idx in indices:
        section_points = get_cross_section_points(mesh, spline_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            centroid = section_points.mean(axis=0)
            section_centroids.append(centroid)

    centroids = np.array(section_centroids)

    # Insert top and bottom anchors at the ends
    centroids_with_anchors = np.vstack([centre_top, centroids, centre_bottom])

    # Polyline trace before smoothing
    polyline_trace = go.Scatter3d(
        x=centroids_with_anchors[:,0], y=centroids_with_anchors[:,1], z=centroids_with_anchors[:,2],
        mode='lines+markers', line=dict(color='orange', width=4), name='Centroid Polyline + Anchors'
    )

    # Smooth using Savitzky-Golay (open, no wrap)
    window_length = 7 if len(centroids_with_anchors) > 7 else max(3, len(centroids_with_anchors)//2*2+1)
    smoothed = np.column_stack([
        savgol_filter(centroids_with_anchors[:,i], window_length=window_length, polyorder=2, mode='interp')
        for i in range(3)
    ])

    smoothed_trace = go.Scatter3d(
        x=smoothed[:,0], y=smoothed[:,1], z=smoothed[:,2],
        mode='lines', line=dict(color='red', width=6), name='Smoothed Guard Cell Line'
    )

    # Anchor markers
    anchor_trace = go.Scatter3d(
        x=[centre_top[0], centre_bottom[0]],
        y=[centre_top[1], centre_bottom[1]],
        z=[centre_top[2], centre_bottom[2]],
        mode='markers+text',
        marker=dict(size=8, color='black'),
        text=['Top','Bottom'], textposition='top center',
        name='Anchors'
    )

    return polyline_trace, smoothed_trace, anchor_trace, smoothed

def get_regularly_spaced_cross_sections(mesh, smoothed, centre_top, centre_bottom, num_sections=30):
    """
    Given a mesh and a smoothed centreline (Nx3 array), return
    cross section points at regularly spaced intervals along the centreline,
    avoiding points too close to the top and bottom wall centroids.
    """
    smoothed_x = smoothed[:,0]
    smoothed_y = smoothed[:,1]
    smoothed_z = smoothed[:,2]
    smoothed_points = np.column_stack([smoothed_x, smoothed_y, smoothed_z])

    dists = np.linalg.norm(np.diff(smoothed_points, axis=0), axis=1)
    arc_length = np.concatenate([[0], np.cumsum(dists)])
    total_length = arc_length[-1]
    target_lengths = np.linspace(0, total_length, num_sections + 4)

    # Interpolate to get regularly spaced points
    interp_points = np.empty((len(target_lengths), 3))
    for i in range(3):
        interp_points[:, i] = np.interp(target_lengths, arc_length, smoothed_points[:, i])

    smoothed_points = interp_points

    # Find indices of closest points to top and bottom wall centroids
    top_idx = np.argmin(np.linalg.norm(smoothed_points - centre_top, axis=1))
    bottom_idx = np.argmin(np.linalg.norm(smoothed_points - centre_bottom, axis=1))

    # Sample num_sections + 4 indices evenly along the centreline (to allow for ±1 removal at each wall)
    indices = np.linspace(0, len(smoothed_points) - 1, num_sections + 4, dtype=int)

    def remove_near_wall(idx, indices, window=1):
        # Find the closest index in indices to idx, then remove ±window around it
        closest = np.argmin(np.abs(indices - idx))
        to_remove = [(closest + offset) % len(indices) for offset in range(-window, window+1)]
        return set(to_remove)

    # Indices to remove: ±1 around top and bottom wall
    remove_set = remove_near_wall(top_idx, indices, window=1) | remove_near_wall(bottom_idx, indices, window=1)

    keep_mask = np.ones(len(indices), dtype=bool)
    for i in remove_set:
        keep_mask[i] = False

    final_indices = indices[keep_mask]

    section_points_list = []
    section_traces = []
    section_bary_data = []

    for idx in final_indices:
        section_points = get_cross_section_points(mesh, smoothed_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            section_trace = go.Scatter3d(
                x=section_points[:, 0],
                y=section_points[:, 1],
                z=section_points[:, 2],
                mode='markers',
                marker=dict(size=5, color='green'),
                name=f'Section {idx}'
            )
            section_traces.append(section_trace)

            # Compute barycentric data for each point in this section
            section_bary = []
            for pt in section_points:
                _, _, face_idx = mesh.nearest.on_surface([pt])
                face_idx = face_idx[0]
                face_vertices = mesh.vertices[mesh.faces[face_idx]]
                bary = get_barycentric_coords(pt, face_vertices)
                section_bary.append((face_idx, bary))
            section_bary_data.append(section_bary)

    return section_points_list, section_traces, section_bary_data

def get_regularly_spaced_cross_sections_with_normals(mesh, smoothed, centre_top, centre_bottom, num_sections=30):
    """
    Given a mesh and a smoothed centreline (Nx3 array), return
    cross section points at regularly spaced intervals along the centreline,
    avoiding points too close to the top and bottom wall centroids.
    """
    smoothed_x = smoothed[:,0]
    smoothed_y = smoothed[:,1]
    smoothed_z = smoothed[:,2]
    smoothed_points = np.column_stack([smoothed_x, smoothed_y, smoothed_z])

    dists = np.linalg.norm(np.diff(smoothed_points, axis=0), axis=1)
    arc_length = np.concatenate([[0], np.cumsum(dists)])
    total_length = arc_length[-1]
    target_lengths = np.linspace(0, total_length, num_sections + 4)

    # Interpolate to get regularly spaced points
    interp_points = np.empty((len(target_lengths), 3))
    for i in range(3):
        interp_points[:, i] = np.interp(target_lengths, arc_length, smoothed_points[:, i])

    smoothed_points = interp_points

    # Find indices of closest points to top and bottom wall centroids
    top_idx = np.argmin(np.linalg.norm(smoothed_points - centre_top, axis=1))
    bottom_idx = np.argmin(np.linalg.norm(smoothed_points - centre_bottom, axis=1))

    # Sample num_sections + 4 indices evenly along the centreline (to allow for ±1 removal at each wall)
    indices = np.linspace(0, len(smoothed_points) - 1, num_sections + 4, dtype=int)

    def remove_near_wall(idx, indices, window=1):
        # Find the closest index in indices to idx, then remove ±window around it
        closest = np.argmin(np.abs(indices - idx))
        to_remove = [(closest + offset) % len(indices) for offset in range(-window, window+1)]
        return set(to_remove)

    # Indices to remove: ±1 around top and bottom wall
    remove_set = remove_near_wall(top_idx, indices, window=1) | remove_near_wall(bottom_idx, indices, window=1)

    keep_mask = np.ones(len(indices), dtype=bool)
    for i in remove_set:
        keep_mask[i] = False

    final_indices = indices[keep_mask]

    section_points_list = []
    section_traces = []
    section_bary_data = []

    for idx in final_indices:
        section_points, normals, midpoints = get_cross_section_points_with_normals(mesh, smoothed_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            section_trace = go.Scatter3d(
                x=section_points[:, 0],
                y=section_points[:, 1],
                z=section_points[:, 2],
                mode='markers',
                marker=dict(size=5, color='green'),
                name=f'Section {idx}'
            )
            section_traces.append(section_trace)

            # Compute barycentric data for each point in this section
            section_bary = []
            for pt in section_points:
                _, _, face_idx = mesh.nearest.on_surface([pt])
                face_idx = face_idx[0]
                face_vertices = mesh.vertices[mesh.faces[face_idx]]
                bary = get_barycentric_coords(pt, face_vertices)
                section_bary.append((face_idx, bary))
            section_bary_data.append(section_bary)

    return section_points_list, section_traces, section_bary_data, normals

def refine_centre_line_with_three_anchors(
    mesh,
    spline_x,
    spline_y,
    spline_z,
    centre_top,
    centre_bottom,
    side_section_centre,
    num_sections=40
):
    """
    Refine the centreline using only three anchor points (top, bottom, one side),
    passing through those anchors and refined by cross-section centroids.
    The curve will pass through the three anchors and be smoothed by cross-section centroids.
    The result is an open curve (not a loop).
    """
    import numpy as np
    from scipy.interpolate import splprep, splev
    import plotly.graph_objects as go

    # Build cross-section centroids along the initial spline
    spline_points = np.column_stack([spline_x, spline_y, spline_z])
    indices = np.linspace(0, len(spline_points) - 1, num_sections, dtype=int)
    section_points_list = []
    section_centroids = []
    for idx in indices:
        section_points = get_cross_section_points(mesh, spline_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            centroid = section_points.mean(axis=0)
            section_centroids.append(centroid)
    centroids = np.array(section_centroids)

    # Insert anchors at start, side, end (top, side, bottom)
    anchors = [centre_top, side_section_centre, centre_bottom]
    # Find closest centroid to side anchor
    side_idx = np.argmin(np.linalg.norm(centroids - side_section_centre, axis=1))
    # Build ordered points: top anchor, centroids up to side, side anchor, centroids after side, bottom anchor
    ordered_points = [centre_top]
    if side_idx > 0:
        ordered_points.extend(centroids[1:side_idx])
    ordered_points.append(side_section_centre)
    if side_idx < len(centroids) - 1:
        ordered_points.extend(centroids[side_idx+1:-1])
    ordered_points.append(centre_bottom)
    ordered_points = np.array(ordered_points)

    # Interpolate a smooth, non-periodic spline through these points
    tck, u = splprep(ordered_points.T, s=0, per=0)
    u_fine = np.linspace(0, 1, num_sections)
    smoothed = np.array(splev(u_fine, tck)).T

    # Polyline trace (unsmoothed, just anchor+centroid points)
    polyline_trace = go.Scatter3d(
        x=ordered_points[:,0], y=ordered_points[:,1], z=ordered_points[:,2],
        mode='lines+markers', line=dict(color='orange', width=4), name='Centroid Polyline + Anchors (3)'
    )
    # Smoothed spline trace
    smoothed_trace = go.Scatter3d(
        x=smoothed[:,0], y=smoothed[:,1], z=smoothed[:,2],
        mode='lines+markers', line=dict(color='red', width=6), name='Smoothed Centreline (3 anchors)'
    )
    # Anchor trace
    anchor_trace = go.Scatter3d(
        x=[a[0] for a in anchors], y=[a[1] for a in anchors], z=[a[2] for a in anchors],
        mode='markers+text', marker=dict(size=8, color='black'), name='Anchors (3)',
        text=['Top','Side','Bottom'], textposition='top center'
    )
    return polyline_trace, smoothed_trace, anchor_trace, smoothed
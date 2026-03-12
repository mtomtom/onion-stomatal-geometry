# Old, unused functions extracted from the codebase

# === calculate_cross_section_areas (from src/cross_section_helpers.py) ===
def calculate_cross_section_areas(sections_points_list):
    """Calculate area of each cross section using convex hull.
    
    Projects each 3D cross section to 2D using PCA and computes the convex hull area.
    
    Parameters
    ----------
    sections_points_list : list of array_like
        Each element is an (N, 3) array of cross-section points.
    
    Returns
    -------
    list of float
        Area for each cross section. Returns 0.0 for sections with <3 points.
    """
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull
    areas = []
    for points in sections_points_list:
        if points is None or len(points) < 3:
            areas.append(0.0)
            continue
        pca = PCA(n_components=2)
        pts2 = pca.fit_transform(np.asarray(points))
        try:
            hull = ConvexHull(pts2)
            area = hull.volume  # 2D 'volume' is area
        except Exception:
            area = 0.0
        areas.append(area)
    return areas


# === visualize_wall_vertex_methods (from src/cross_section_helpers.py) ===
def visualize_wall_vertex_methods(mesh_entry, dot_thresh=0.2, axis=1, quantile=0.08):
    """Visualize wall-vertex selections from normals and axis-extrema methods.

    Parameters
    ----------
    mesh_entry : str, Path, or trimesh.Trimesh
        Mesh path or loaded mesh.
    dot_thresh : float, optional
        Threshold for normal-based wall detection.
    axis : int, optional
        Axis used by axis-extrema fallback (0=x, 1=y, 2=z).
    quantile : float, optional
        Tail quantile used by axis-extrema method.

    Returns
    -------
    dict
        Summary with vertex index arrays and counts for each method.
    """
    mesh, label = _coerce_mesh_entry(mesh_entry)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("visualize_wall_vertex_methods requires a trimesh.Trimesh input.")
    mesh_name = label or "in_memory_mesh"

    wall_normals = find_wall_vertices_vertex_normals(mesh, dot_thresh=dot_thresh)
    wall_extrema = find_wall_vertices_axis_extrema(mesh, axis=axis, quantile=quantile)
    overlap = np.intersect1d(wall_normals, wall_extrema)

    traces = []
    verts = mesh.vertices

    if wall_normals.size:
        traces.append(
            go.Scatter3d(
                x=verts[wall_normals, 0],
                y=verts[wall_normals, 1],
                z=verts[wall_normals, 2],
                mode='markers',
                marker=dict(size=2.5, color='#D55E00'),
                name='Wall (normals)'
            )
        )
    if wall_extrema.size:
        traces.append(
            go.Scatter3d(
                x=verts[wall_extrema, 0],
                y=verts[wall_extrema, 1],
                z=verts[wall_extrema, 2],
                mode='markers',
                marker=dict(size=2.0, color='#009E73'),
                name='Wall (axis extrema)'
            )
        )
    if overlap.size:
        traces.append(
            go.Scatter3d(
                x=verts[overlap, 0],
                y=verts[overlap, 1],
                z=verts[overlap, 2],
                mode='markers',
                marker=dict(size=3.0, color='#CC79A7'),
                name='Overlap'
            )
        )

    if wall_normals.size >= 2:
        n_top, n_bottom, *_ = get_top_bottom_wall_centres(mesh, wall_normals)
        traces.extend([
            go.Scatter3d(
                x=[n_top[0]], y=[n_top[1]], z=[n_top[2]],
                mode='markers', marker=dict(size=7, color='#A73F00'),
                name='Normals top centre'
            ),
            go.Scatter3d(
                x=[n_bottom[0]], y=[n_bottom[1]], z=[n_bottom[2]],
                mode='markers', marker=dict(size=7, color='#A73F00', symbol='diamond'),
                name='Normals bottom centre'
            ),
        ])
    if wall_extrema.size >= 2:
        e_top, e_bottom, *_ = get_top_bottom_wall_centres(mesh, wall_extrema)
        traces.extend([
            go.Scatter3d(
                x=[e_top[0]], y=[e_top[1]], z=[e_top[2]],
                mode='markers', marker=dict(size=7, color='#007A58'),
                name='Extrema top centre'
            ),
            go.Scatter3d(
                x=[e_bottom[0]], y=[e_bottom[1]], z=[e_bottom[2]],
                mode='markers', marker=dict(size=7, color='#007A58', symbol='diamond'),
                name='Extrema bottom centre'
            ),
        ])

    title = (
        f"Wall method comparison: {Path(mesh_name).name} "
        f"(normals={len(wall_normals)}, extrema={len(wall_extrema)}, overlap={len(overlap)})"
    )
    visualize_mesh(mesh, extra_details=traces, title=title, opacity=0.25)

    return {
        "mesh": mesh_name,
        "wall_vertices_normals": wall_normals,
        "wall_vertices_axis_extrema": wall_extrema,
        "overlap_vertices": overlap,
        "n_normals": int(len(wall_normals)),
        "n_axis_extrema": int(len(wall_extrema)),
        "n_overlap": int(len(overlap)),
    }


# === barycentric_to_point (from src/cross_section_helpers.py) ===
def barycentric_to_point(face_vertices, bary):
    """Convert barycentric coordinates to 3D point.
    
    Parameters
    ----------
    face_vertices : ndarray
        Triangle vertices as (3, 3) array.
    bary : ndarray
        Barycentric coordinates (u, v, w).
    
    Returns
    -------
    ndarray
        3D point coordinates.
    """
    return bary[0]*face_vertices[0] + bary[1]*face_vertices[1] + bary[2]*face_vertices[2]


# === get_midsection_and_tip_data (from src/idealised_mesh_functions.py) ===
def get_midsection_and_tip_data(cross_section_ratios, major_lengths, minor_lengths):
    """Extract midsection and tip measurements from cross-section data.
    
    Processes cross-section aspect ratios and lengths to extract measurements
    at the midsection and tip locations for both left and right guard cells.
    
    Parameters
    ----------
    cross_section_ratios : list
        Aspect ratios along cross-sections for left and right guard cells.
    major_lengths : list
        Major axis lengths along cross-sections.
    minor_lengths : list
        Minor axis lengths along cross-sections.
    
    Returns
    -------
    tuple of (list, list, list, list, list, list, list, list)
        Eight lists containing:
        - left_midsection_ar, right_midsection_ar: Aspect ratios at midsection
        - left_tip_ar, right_tip_ar: Aspect ratios at tip
        - left_midsection_major, right_midsection_major: Major lengths at midsection
        - left_midsection_minor, right_midsection_minor: Minor lengths at midsection
    """
    left_midsection_ar = []
    right_midsection_ar = []
    left_tip_ar = []
    right_tip_ar = []
    left_midsection_major = []
    right_midsection_major = []
    left_midsection_minor = []
    right_midsection_minor = []

    for r, major, minor in zip(cross_section_ratios, major_lengths, minor_lengths):
        ## Get the midsection cross section for each guard cell
        mid_left = r[0][len(r[0]) // 2]
        mid_right = r[1][len(r[1]) // 2]
        
        left_midsection_ar.append(mid_left)
        right_midsection_ar.append(mid_right)
        ## Get the tip cross section for each guard cell
        tip_left = r[0][-1]
        tip_right = r[1][-1]
        
        left_tip_ar.append(tip_left)
        right_tip_ar.append(tip_right)
        ## Get the major lengths for each guard cell - get the midsection values
        major_left = major[0][len(r[0]) // 2]
        major_right = major[1][len(r[1]) // 2]
        
        left_midsection_major.append(major_left)
        right_midsection_major.append(major_right)
        ## Get the minor lengths for each guard cell
        minor_left = minor[0][len(r[0]) // 2]
        minor_right = minor[1][len(r[0]) // 2]
        
        left_midsection_minor.append(minor_left)
        right_midsection_minor.append(minor_right)
    return left_midsection_ar, right_midsection_ar, left_tip_ar, right_tip_ar, left_midsection_major, right_midsection_major, left_midsection_minor, right_midsection_minor


# === get_aspect_ratios (from src/idealised_mesh_functions.py) ===
def get_aspect_ratios(section_right, section_left):
    """Calculate aspect ratios and lengths from cross-section data.
    
    Parameters
    ----------
    section_right : list
        Cross-section points for right guard cells.
    section_left : list
        Cross-section points for left guard cells.
    
    Returns
    -------
    tuple of (list, list, list)
        Cross-section aspect ratios, major lengths, and minor lengths for each section.
    """
    cross_section_ratios = []
    major_lengths = []
    minor_lengths = []
    for right, left in zip(section_right, section_left):
        lr, major_length_l, minor_length_l = csh.calculate_cross_section_aspect_ratios_and_lengths(left)
        rr, major_length_r, minor_length_r = csh.calculate_cross_section_aspect_ratios_and_lengths(right)
        cross_section_ratios.append((lr, rr))
        major_lengths.append((major_length_l, major_length_r))
        minor_lengths.append((minor_length_l, minor_length_r))
    return cross_section_ratios, major_lengths, minor_lengths


# === get_cross_sections (from src/idealised_mesh_functions.py) ===
def get_cross_sections(mesh_list, meshdir_path, mid_area_left_0, mid_area_right_0):
    """Extract cross-section points from a list of stomata meshes.
    
    Parameters
    ----------
    mesh_list : list
        List of mesh identifiers to process.
    meshdir_path : str
        Path to directory containing mesh files.
    mid_area_left_0 : float
        Reference mid-area for left guard cell.
    mid_area_right_0 : float
        Reference mid-area for right guard cell.
    
    Returns
    -------
    tuple of (list, list)
        Cross-section points for right and left guard cells.
    """
    section_right = []
    section_left = []
    for sm in mesh_list:
        mesh_path = meshdir_path + "Ac_DA_" + sm + ".obj"
        section_points_right, section_points_left, _, _ = csh.analyze_stomata_mesh(mesh_path, num_sections=20, n_points=40, visualize=False, mid_area_left_0=mid_area_left_0, mid_area_right_0=mid_area_right_0)
        section_right.append(section_points_right)
        section_left.append(section_points_left)
    return section_right, section_left


# === cross_section_points_and_aspect (from src/mesh_functions.py) ===
def cross_section_points_and_aspect(mesh_path, tol=0.5, side="left", visualize=False):
    """Extract cross-section points and aspect ratio for one guard cell.
    
    Slices a torus-like stomata mesh near the Y-midplane and extracts points
    for one guard cell (left or right half). Assumes mesh is centered at origin
    and symmetric along Y and X axes.
    
    Parameters
    ----------
    mesh_path : Path
        Path to OBJ mesh file.
    tol : float, optional
        Half-thickness of slice around Y=0 (default: 0.5).
    side : {'left', 'right'}, optional
        Which guard cell to extract: 'left' for x < 0, 'right' for x > 0 (default: 'left').
    visualize : bool, optional
        If True, displays 2D scatter plot of the cross-section (default: False).
    
    Returns
    -------
    dict
        Dictionary containing mesh ID, cross-section type, pressure,
        cross-section points (N, 2) array in XZ plane, and aspect ratio.
    
    Raises
    ------
    ValueError
        If filename format is unexpected, no vertices found near Y=0,
        no vertices on specified side, or invalid side parameter.
    """

    parts = mesh_path.stem.split("_")

    try:
        mesh_id = "_".join(parts[3:5])          # e.g. "1_2"
        cross_section_type = parts[5]           # e.g. "circular"
        pressure = round(float(parts[-1]), 2)   # e.g. 0.0
    except (IndexError, ValueError):
        raise ValueError(f"Unexpected filename format: {mesh_path.name}")

    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    # --- Slice points near the Y midplane ---
    y_mid = 0.0
    low, high = y_mid - tol, y_mid + tol
    mask = (mesh.vertices[:, 1] > low) & (mesh.vertices[:, 1] < high)
    midsection_points = mesh.vertices[mask]

    if midsection_points.shape[0] == 0:
        raise ValueError("No vertices found near Y=0 midplane.")

    # --- Take only one half (left or right) ---
    if side == "left":
        half_points = midsection_points[midsection_points[:, 0] < 0]
    elif side == "right":
        half_points = midsection_points[midsection_points[:, 0] > 0]
    else:
        raise ValueError("side must be 'left' or 'right'.")

    if half_points.shape[0] == 0:
        raise ValueError(f"No vertices found on {side} half.")

    # --- Project to XZ plane ---
    points_2D = half_points[:, [0, 2]]

    # --- Compute aspect ratio ---
    min_pt = points_2D.min(axis=0)
    max_pt = points_2D.max(axis=0)
    width, height = max_pt - min_pt
    aspect_ratio = width / height if height > 0 else np.nan

    # --- Optional visualization ---
    if visualize:
        plt.figure(figsize=(5, 5))
        plt.scatter(points_2D[:, 0], points_2D[:, 1], s=2, alpha=0.6)
        plt.gca().set_aspect("equal")
        plt.title(f"{side.capitalize()} cross-section (aspect = {aspect_ratio:.3f})")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()

    return {
            "Mesh ID": mesh_id,
            "Cross-section type": cross_section_type,
            "Pressure": pressure,
            "Cross section": points_2D,
            "Aspect Ratio": aspect_ratio,
        }


# === process_idealised_mesh (from src/mesh_functions.py) ===
def process_idealised_mesh(file, debug=False):
    """Process an idealised stomata mesh to extract geometric properties.
    
    Analyzes an idealised mesh by slicing through the Y-midplane, clustering
    vertices into guard cells, and calculating aspect ratio and pore area.
    
    Parameters
    ----------
    file : Path
        Path to idealised mesh file. Filename must follow convention:
        'idealised_final_{mesh_id}_{cross_section_type}_{pressure}.obj'
    debug : bool, optional
        If True, displays debug visualization of selected guard cell (default: False).
    
    Returns
    -------
    dict
        Dictionary containing mesh ID, cross-section type, pressure,
        aspect ratio, and pore area.
    
    Raises
    ------
    ValueError
        If no vertices found within midsection tolerance or if vertex array
        has unexpected shape.
    """
    parts = file.stem.split("_")
    mesh_id = "_".join(parts[3:5])          # '2_6a'
    cross_section_type = parts[5]           # 'circular'
    pressure = round(float(parts[-1]), 2)   # 0.8


    # --- Load mesh ---
    mesh = trimesh.load(file, process=False)

    # --- Slice through midplane perpendicular to Y ---
    y_mid = mesh.bounds[:, 1].mean()
    tol = 0.25  # thickness of slice

    # Efficient slicing using searchsorted
    y_sorted_idx = np.argsort(mesh.vertices[:,1])
    y_sorted = mesh.vertices[y_sorted_idx,1]
    low, high = y_mid - tol, y_mid + tol
    start = np.searchsorted(y_sorted, low, side='left')
    end   = np.searchsorted(y_sorted, high, side='right')
    midsection_points = mesh.vertices[y_sorted_idx[start:end]]

    if len(midsection_points) == 0:
        raise ValueError("No vertices found within midsection tolerance.")

    # --- Cluster into two halves ---
    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=100, n_init=1, random_state=0)
    labels = kmeans.fit_predict(midsection_points)

    group1 = midsection_points[labels == 0]
    group2 = midsection_points[labels == 1]

    # Pick upper cell (higher z mean)
    one_guard_cell_points = group1 if group1[:,2].mean() > group2[:,2].mean() else group2

    # Ensure 3D shape
    if one_guard_cell_points.ndim != 2 or one_guard_cell_points.shape[1] != 3:
        raise ValueError(f"Expected shape (N,3), got {one_guard_cell_points.shape}")

    # --- Calculate aspect ratio ---
    aspect_ratio = csh.calculate_cross_section_aspect_ratios(one_guard_cell_points)

    # --- Optional plot for debugging (random subset to speed up) ---
    if debug:
        import matplotlib.pyplot as plt
        subset = one_guard_cell_points
        if len(one_guard_cell_points) > 2000:
            idx = np.random.choice(len(one_guard_cell_points), 2000, replace=False)
            subset = one_guard_cell_points[idx]
        plt.figure(figsize=(6,6))
        plt.scatter(subset[:,0], subset[:,2], s=10)
        plt.gca().set_aspect('equal')
        plt.title("Selected guard cell cross-section (X vs Z)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()

    # --- Calculate pore area ---
    pore_area = fast_pore_area(mesh, step=0.05)

    return {
        "Mesh ID": mesh_id,
        "Cross-section type": cross_section_type,
        "Pressure (MPa)": pressure,
        "Aspect Ratio": aspect_ratio,
        "Pore Area (um^2)": pore_area,
    }


# === plot_radius_distribution (from src/generate_idealised_mesh_new.py) ===
def plot_radius_distribution(side_radius_a, tip_radius_a, side_radius_b=None, tip_radius_b=None, transition_point=0.8):
    """
    Plots the distribution of cross-section radii around the elliptical path.
    
    Args:
        side_radius_a: The a-axis radius at the sides (horizontal axis points)
        tip_radius_a: The a-axis radius at the tips (vertical axis points)
        side_radius_b: The b-axis radius at the sides (optional, for plotting both dimensions)
        tip_radius_b: The b-axis radius at the tips (optional, for plotting both dimensions)
        transition_point: Fraction of distance where transition completes
    """
    theta_values = np.linspace(0, 2*np.pi, 100)
    
    # Calculate radius_a values
    radius_a_values = [calculate_cross_section_radius(theta, side_radius_a, tip_radius_a, transition_point) 
                      for theta in theta_values]
    
    # Convert theta to degrees for easier reading
    theta_degrees = np.degrees(theta_values)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot radius_a
    plt.plot(theta_degrees, radius_a_values, 'b-', label='radius_a')
    plt.axhline(y=side_radius_a, color='r', linestyle='--', 
               label=f'Side radius_a: {side_radius_a}')
    plt.axhline(y=tip_radius_a, color='g', linestyle='--', 
               label=f'Tip radius_a: {tip_radius_a}')
    
    # If b-axis values are provided, plot those too
    if side_radius_b is not None and tip_radius_b is not None:
        radius_b_values = [calculate_cross_section_radius(theta, side_radius_b, tip_radius_b, transition_point) 
                          for theta in theta_values]
        plt.plot(theta_degrees, radius_b_values, 'c-', label='radius_b')
        plt.axhline(y=side_radius_b, color='m', linestyle='--', 
                   label=f'Side radius_b: {side_radius_b}')
        plt.axhline(y=tip_radius_b, color='y', linestyle='--', 
                   label=f'Tip radius_b: {tip_radius_b}')
    
    # Mark the sides and tips
    plt.axvline(x=0, color='blue', linestyle=':', label='Side (θ=0°)')
    plt.axvline(x=180, color='blue', linestyle=':')
    plt.axvline(x=90, color='orange', linestyle=':', label='Tip (θ=90°)')
    plt.axvline(x=270, color='orange', linestyle=':')
    
    # Mark the transition points
    transition_angle = transition_point * 90
    plt.axvline(x=transition_angle, color='purple', linestyle='-.', 
               label=f'Transition point: {transition_point*100}%')
    plt.axvline(x=180-transition_angle, color='purple', linestyle='-.')
    plt.axvline(x=180+transition_angle, color='purple', linestyle='-.')
    plt.axvline(x=360-transition_angle, color='purple', linestyle='-.')
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cross-section radius')
    plt.title('Distribution of cross-section radii around the elliptical path')
    plt.grid(True)
    plt.legend()
    plt.savefig('radius_distribution.png')
    plt.show()

    # After plotting radius_a_values
    shrink_start = 0.95
    theta_values = np.linspace(0, 2*np.pi, 100)
    norm_pos_from_tip = np.abs(np.sin(theta_values))  # For a circle; for ellipse use path_y/major_radius_b

    plt.plot(np.degrees(theta_values), norm_pos_from_tip, 'k--', alpha=0.3, label='norm_pos_from_tip')

    # Mark where shrink starts
    shrink_angles = np.degrees(theta_values[norm_pos_from_tip >= shrink_start])
    if len(shrink_angles) > 0:
        plt.axvline(x=shrink_angles[0], color='purple', linestyle='-.', label='Shrink start')
        plt.axvline(x=shrink_angles[-1], color='purple', linestyle='-.')

    plt.legend()
    plt.show()
    
    return radius_a_values


# === calculate_cross_section_radius (from src/generate_idealised_mesh_new.py) ===
def calculate_cross_section_radius(theta, side_radius, tip_radius, transition_point=0.8):
    """
    Calculate the cross-section radius at a given angle theta around the elliptical path.
    
    Args:
        theta: Angle in radians (0 = right side, pi/2 = top, pi = left side, 3pi/2 = bottom)
        side_radius: Radius at the sides of the elliptical path (theta=0 or pi)
        tip_radius: Radius at the tips of the elliptical path (theta=pi/2 or 3pi/2)
        transition_point: Fraction of distance between side and tip where transition completes
    
    Returns:
        The radius at the specified angle
    """
    # Map theta to a value between 0 and pi/2 (a quadrant)
    quadrant_theta = theta % np.pi
    if quadrant_theta > np.pi/2:
        quadrant_theta = np.pi - quadrant_theta
    
    # Normalize position from side (0) to tip (1)
    # At theta=0, this equals 0 (side)
    # At theta=pi/2, this equals 1 (tip)
    norm_pos = quadrant_theta / (np.pi/2)
    
    # Apply piecewise linear interpolation
    if norm_pos <= transition_point:
        # Linear change from side to transition point
        interp_factor = norm_pos / transition_point
        radius = side_radius * (1 - interp_factor) + tip_radius * interp_factor
    else:
        # Constant value (tip_radius) for the remainder
        radius = tip_radius
        
    return radius


# === create_elliptical_torus_bulged (from src/generate_idealised_mesh_new.py) ===
def create_elliptical_torus_bulged(
    major_radius_a, major_radius_b, 
    mid_radius_a, mid_radius_b,
    tip_radius_a, tip_radius_b, 
    major_segments, minor_segments
):
    """
    Creates a mesh where an elliptical cross-section travels along an elliptical path.

    Args:
        major_radius_a: Semi-axis of the major (path) ellipse along the x-axis.
        major_radius_b: Semi-axis of the major (path) ellipse along the y-axis.
        mid_radius_a: Semi-axis of the minor (cross-section) ellipse at midpoint (sides).
        mid_radius_b: Semi-axis of the minor (cross-section) ellipse at midpoint (sides).
        tip_radius_a: Semi-axis of the minor (cross-section) ellipse at tip (top/bottom).
        tip_radius_b: Semi-axis of the minor (cross-section) ellipse at tip (top/bottom).
        major_segments: Number of segments for the major elliptical path.
        minor_segments: Number of segments for the minor elliptical cross-section.
    """
    vertices = []
    faces = []
    for i in range(major_segments):
        theta = i * 2 * np.pi / major_segments

        # Position on the major (path) ellipse
        path_x = major_radius_a * np.cos(theta)
        path_y = major_radius_b * np.sin(theta)

        # Use vertical position (y) to determine proximity to tip (top/bottom)
        norm_pos_from_tip = abs(path_y) / major_radius_b  # 0 at sides, 1 at tips

        # --- REVERSED INTERPOLATION ---
        # Now: 0 at midpoint (sides), 1 at tips
        # Instead of: 1 at midpoint, 0 at tips
        transition_point = 0.8
        if norm_pos_from_tip < transition_point:
            interp_factor = norm_pos_from_tip / transition_point  # Changed from (1.0 - norm_pos...)
        else:
            interp_factor = 1.0  # Changed from 0.0

        interp_factor = 1.0
        # Corrected interpolation formula - applies mid_radius at midpoints and tip_radius at tips
        current_minor_radius_a = mid_radius_a * (1 - interp_factor) + tip_radius_a * interp_factor
        current_minor_radius_b = mid_radius_b * (1 - interp_factor) + tip_radius_b * interp_factor

        shrink_start = 0.9  # Start shrinking at 95% of the way to the tip
        # if norm_pos_from_tip >= shrink_start:
        #     # Quadratic shrink: increases faster near the tip
        #     t = (norm_pos_from_tip - shrink_start) / (1.0 - shrink_start)
        #     shrink_factor = 1.0 - 0.2 * (t ** 5)
        #     current_minor_radius_a *= shrink_factor
        #     current_minor_radius_b *= shrink_factor
            
        # Tangent vector to the major ellipse at this point (for orientation)
        tx = -major_radius_a * np.sin(theta)
        ty =  major_radius_b * np.cos(theta)
        tangent = np.array([tx, ty, 0])
        tangent /= np.linalg.norm(tangent)

        # Normal and binormal vectors to define the plane of the cross-section
        normal = np.array([-ty, tx, 0])  # Perpendicular to tangent in XY plane
        normal /= np.linalg.norm(normal)
        binormal = np.array([0, 0, 1])   # Perpendicular to XY plane

        for j in range(minor_segments):
            phi = j * 2 * np.pi / minor_segments

            local_x = current_minor_radius_a * np.cos(phi)
            local_z = current_minor_radius_b * np.sin(phi)

            # Only apply shrink near the tip
            if norm_pos_from_tip >= shrink_start:
                t = (norm_pos_from_tip - shrink_start) / (1.0 - shrink_start)
                full_shrink = 1.0 - 0.2 * (t ** 5)

                # Normalize local_x: -max_a (outside) to +max_a (inside)
                # So: 0 (outside) ... 1 (inside)
                x_norm = (local_x + current_minor_radius_a) / (2 * current_minor_radius_a)
                # Blend shrink: full_shrink at outside, 1 at inside
                blend_shrink = full_shrink * (1 - x_norm) + 1.0 * x_norm

                local_x *= blend_shrink
                local_z *= blend_shrink  # If you want to keep aspect ratio

            vertex = np.array([path_x, path_y, 0]) + local_x * normal + local_z * binormal
            vertices.append(vertex)

    # Create faces connecting vertices (quads as two triangles)
    for i in range(major_segments):
        for j in range(minor_segments):
            next_i = (i + 1) % major_segments
            next_j = (j + 1) % minor_segments

            v1 = i * minor_segments + j
            v2 = next_i * minor_segments + j
            v3 = next_i * minor_segments + next_j
            v4 = i * minor_segments + next_j

            faces.append([v1, v4, v2])
            faces.append([v2, v4, v3])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# === create_elliptical_torus (from src/generate_idealised_mesh_new.py) ===
def create_elliptical_torus(
    major_radius_a, major_radius_b, 
    minor_radius_a, minor_radius_b, 
    major_segments, minor_segments
):
    """
    Creates a mesh where an elliptical cross-section travels along an elliptical path.

    Args:
        major_radius_a: Semi-axis of the major (path) ellipse along the x-axis.
        major_radius_b: Semi-axis of the major (path) ellipse along the y-axis.
        minor_radius_a: Semi-axis of the minor (cross-section) ellipse (width).
        minor_radius_b: Semi-axis of the minor (cross-section) ellipse (height).
        major_segments: Number of segments for the major elliptical path.
        minor_segments: Number of segments for the minor elliptical cross-section.
    """
    vertices = []
    faces = []

    # Create vertices
    for i in range(major_segments):
        theta = i * 2 * np.pi / major_segments

        # Position on the major (path) ellipse
        path_x = major_radius_a * np.cos(theta)
        path_y = major_radius_b * np.sin(theta)
        
        # Tangent vector to the major ellipse at this point (for orientation)
        # The derivative of the path gives the tangent direction
        tx = -major_radius_a * np.sin(theta)
        ty =  major_radius_b * np.cos(theta)
        tangent = np.array([tx, ty, 0])
        tangent /= np.linalg.norm(tangent)

        # Normal and binormal vectors to define the plane of the cross-section
        normal = np.array([-ty, tx, 0]) # Perpendicular to tangent in XY plane
        normal /= np.linalg.norm(normal)
        binormal = np.array([0, 0, 1]) # Perpendicular to the XY plane

        for j in range(minor_segments):
            phi = j * 2 * np.pi / minor_segments

            # Position on the minor (cross-section) ellipse
            # These are local coordinates on the plane of the cross-section
            local_x = minor_radius_a * np.cos(phi)
            local_z = minor_radius_b * np.sin(phi)

            # Combine the local coordinates with the orientation vectors
            # to position the vertex in 3D space
            vertex = np.array([path_x, path_y, 0]) + local_x * normal + local_z * binormal
            vertices.append(vertex)

    # Create faces (this logic remains the same)
    for i in range(major_segments):
        for j in range(minor_segments):
            next_i = (i + 1) % major_segments
            next_j = (j + 1) % minor_segments

            v1 = i * minor_segments + j
            v2 = next_i * minor_segments + j
            v3 = next_i * minor_segments + next_j
            v4 = i * minor_segments + next_j

            faces.append([v1, v4, v2])
            faces.append([v2, v4, v3])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


# === create_elliptical_torus_with_wall (from src/generate_idealised_mesh_new.py) ===
def create_elliptical_torus_with_wall(
    major_radius_a=2.0,
    major_radius_b=1.0,
    minor_radius_a=0.3,
    minor_radius_b=0.4,
    major_segments=120,
    minor_segments=40,
    wall_thickness=0.0
):
    """
    Create two half-oval guard cell meshes forming a continuous ring along the y-axis.
    
    Returns:
        left_mesh, right_mesh: two separate Trimesh objects
    """
    
    def generate_half(theta_start, theta_end, offset_sign):
        vertices = []
        faces = []
        major_seg = major_segments // 2
        for i in range(major_seg):
            theta = theta_start + i * (theta_end - theta_start) / (major_seg - 1)
            path_x = major_radius_a * np.sin(theta)   # X now vertical in plane
            path_y = major_radius_b * np.cos(theta)   # Y now horizontal in plane

            
            # Tangent and normal
            tx =  major_radius_a * np.cos(theta)      # derivative of path_x
            ty = -major_radius_b * np.sin(theta)      # derivative of path_y

            tangent = np.array([tx, ty, 0])
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-ty, tx, 0])
            normal /= np.linalg.norm(normal)
            binormal = np.array([0, 0, 1])
            
            # Optional offset for wall separation
            offset = normal * (wall_thickness / 2.0) * offset_sign
            
            for j in range(minor_segments):
                phi = j * 2 * np.pi / minor_segments
                local_x = minor_radius_a * np.cos(phi)
                local_z = minor_radius_b * np.sin(phi)
                vertex = np.array([path_x, path_y, 0]) - local_x * normal + local_z * binormal + offset
                vertices.append(vertex)
        
        # Create faces
        for i in range(major_seg - 1):
            for j in range(minor_segments):
                nj = (j + 1) % minor_segments
                v1 = i * minor_segments + j
                v2 = (i + 1) * minor_segments + j
                v3 = (i + 1) * minor_segments + nj
                v4 = i * minor_segments + nj
                # Counter-clockwise winding
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
        mesh.fix_normals()
        return mesh

    # Left half: theta 0 -> π, offset to left side
    left_mesh = generate_half(0, np.pi, offset_sign=-1)
    # Right half: theta π -> 2π, offset to right side
    right_mesh = generate_half(np.pi, 2*np.pi, offset_sign=1)
    
    return left_mesh, right_mesh


# === create_elliptical_torus_with_shared_wall_taper (from src/generate_idealised_mesh_new.py) ===
def create_elliptical_torus_with_shared_wall_taper(
    major_radius_a=2.0,
    major_radius_b=1.0,
    minor_radius_a=0.3,
    minor_radius_b=0.4,
    major_segments=120,
    minor_segments=40,
    wall_thickness=0.0,
    cap_inset_frac=0.05           # fraction of minor_radius to inset cap centre along normal
):
    """
    Create two half-oval guard cell meshes forming a continuous ring along the y-axis.
    Each half has tip caps. This version smoothly tapers the cross-section near tips
    and insets the tip centre to avoid a 90-degree corner.
    Returns left_mesh, right_mesh (Trimesh objects).
    """
    import numpy as np
    import trimesh

    # top-level parameters
    ref_major_radius = 2.0
    base_taper_frac = 0.05
    base_taper_length_fraction = 0.05

    # inside generate_half
    


    def generate_half(theta_start, theta_end, offset_sign):
        vertices = []
        faces = []
        major_seg = major_segments // 2
        taper_frac = base_taper_frac * (ref_major_radius / major_radius_a)
        taper_length_fraction = base_taper_length_fraction * (ref_major_radius / major_radius_a)

        # precompute taper region size in indices
        taper_len = max(1, int(round(taper_length_fraction * major_seg)))
        # Construct a taper weight function along i=0..major_seg-1 that is 1.0 in the middle
        # and down to (1 - taper_frac) at the ends (over taper_len indices).
        taper_weights = np.ones(major_seg, dtype=float)
        if taper_len > 0:
            # linear taper; you can change to cosine for smoother slope
            for i in range(taper_len):
                w = (i + 1) / (taper_len + 1)  # 0..1
                # use cosine ease for smoother derivative
                ease = 0.5 * (1 - np.cos(np.pi * w))
                taper_weights[i] = 1.0 - taper_frac * (1 - ease)   # near start -> slightly reduced
                taper_weights[major_seg - 1 - i] = 1.0 - taper_frac * (1 - ease)

        # Generate vertices with taper applied based on i index
        normals_at_section = []  # store normal for each section for cap inset
        for i in range(major_seg):
            theta = theta_start + i * (theta_end - theta_start) / (major_seg - 1)
            path_x = major_radius_a * np.sin(theta)   # X vertical in plane
            path_y = major_radius_b * np.cos(theta)   # Y horizontal in plane

            # Tangent and normal
            tx = major_radius_a * np.cos(theta)
            ty = -major_radius_b * np.sin(theta)
            tangent = np.array([tx, ty, 0.0])
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-ty, tx, 0.0])
            normal /= np.linalg.norm(normal)
            binormal = np.array([0.0, 0.0, 1.0])

            normals_at_section.append(normal.copy())

            # taper factor for this section
            taper_scale = taper_weights[i]

            for j in range(minor_segments):
                phi = j * 2.0 * np.pi / minor_segments
                local_x = minor_radius_a * np.cos(phi) * taper_scale
                local_z = minor_radius_b * np.sin(phi)  # don't taper z much; keep thickness
                vertex = np.array([path_x, path_y, 0.0]) - local_x * normal + local_z * binormal \
                         + (normal * (wall_thickness / 2.0) * offset_sign)
                vertices.append(vertex)

        # Generate side faces
        for i in range(major_seg - 1):
            for j in range(minor_segments):
                nj = (j + 1) % minor_segments
                v1 = i * minor_segments + j
                v2 = (i + 1) * minor_segments + j
                v3 = (i + 1) * minor_segments + nj
                v4 = i * minor_segments + nj
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        # Tip wall at first cross-section (i=0)
        tip_indices_start = np.arange(0, minor_segments)
        # compute center as mean of tip indices then inset along local normal
        center_start_pos = np.mean(np.array(vertices)[tip_indices_start], axis=0)
        normal_start = normals_at_section[0]
        cap_inset = cap_inset_frac * minor_radius_a
        cap_inset_frac_scaled = cap_inset_frac * (ref_major_radius / major_radius_a)
        cap_inset = cap_inset_frac_scaled * minor_radius_a

        center_start = len(vertices)
        vertices.append(center_start_pos + normal_start * (-cap_inset))  # inset towards interior
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            # ordering chosen to keep face normals consistent (may need flip depending on orientation)
            faces.append([int(tip_indices_start[j]), int(tip_indices_start[nj]), center_start])

        # Tip wall at last cross-section (i = major_seg - 1)
        tip_indices_end = np.arange((major_seg - 1) * minor_segments, major_seg * minor_segments)
        center_end_pos = np.mean(np.array(vertices)[tip_indices_end], axis=0)
        normal_end = normals_at_section[-1]
        center_end = len(vertices)
        vertices.append(center_end_pos + normal_end * (-cap_inset))
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            faces.append([int(tip_indices_end[nj]), int(tip_indices_end[j]), center_end])

        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
        mesh.fix_normals()
        return mesh

    left_mesh = generate_half(0.0, np.pi, offset_sign=-1)
    right_mesh = generate_half(np.pi, 2.0 * np.pi, offset_sign=1)

    return left_mesh, right_mesh


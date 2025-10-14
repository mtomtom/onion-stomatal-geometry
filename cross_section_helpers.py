def get_regularly_spaced_cross_sections_batch(mesh, smoothed, centre_top, centre_bottom, num_sections=30):
    """
    Optimized version: Given a mesh and a smoothed centreline (Nx3 array), return
    cross section points at regularly spaced intervals along the centreline,
    avoiding points too close to the top and bottom wall centroids.
    Uses batched nearest face search for barycentric data.
    """
    import numpy as np
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
        closest = np.argmin(np.abs(indices - idx))
        to_remove = [(closest + offset) % len(indices) for offset in range(-window, window+1)]
        return set(to_remove)

    remove_set = remove_near_wall(top_idx, indices, window=1) | remove_near_wall(bottom_idx, indices, window=1)

    keep_mask = np.ones(len(indices), dtype=bool)
    for i in remove_set:
        keep_mask[i] = False

    final_indices = indices[keep_mask]

    section_points_list = []
    section_traces = []
    section_bary_data = []

    # Collect all section points for batch nearest search
    all_pts = []
    section_lengths = []
    for idx in final_indices:
        section_points = get_cross_section_points(mesh, smoothed_points, [idx])
        if section_points and section_points[0] is not None:
            section_points = section_points[0]
            section_points_list.append(section_points)
            section_lengths.append(len(section_points))
            section_trace = go.Scatter3d(
                x=section_points[:, 0],
                y=section_points[:, 1],
                z=section_points[:, 2],
                mode='markers',
                marker=dict(size=5, color='green'),
                name=f'Section {idx}'
            )
            section_traces.append(section_trace)
            all_pts.append(section_points)

    if len(all_pts) > 0:
        all_pts_flat = np.vstack(all_pts)
        _, _, face_indices = mesh.nearest.on_surface(all_pts_flat)
        # Split face_indices for each section
        split_indices = np.split(face_indices, np.cumsum(section_lengths)[:-1])
        for section, face_idx_list in zip(section_points_list, split_indices):
            section_bary = []
            for pt, face_idx in zip(section, face_idx_list):
                face_vertices = mesh.vertices[mesh.faces[face_idx]]
                bary = get_barycentric_coords(pt, face_vertices)
                section_bary.append((face_idx, bary))
            section_bary_data.append(section_bary)
    else:
        section_bary_data = [[] for _ in section_points_list]

    return section_points_list, section_traces, section_bary_data
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import time

def curve_length(x, y, z):
    # Stack coordinates into (N, 3) array
    points = np.column_stack((x, y, z))
    # Compute distances between consecutive points
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    # Sum to get total length
    return segment_lengths.sum()

def analyze_stomata_mesh(mesh_path, num_sections=20, n_points=40, visualize=False):
    ## Load in the mesh
    mesh = trimesh.load(mesh_path, process=False)
    ## Get the wall vertices
    wall_vertices = find_wall_vertices_vertex_normals(mesh, dot_thresh=0.2)


    centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace = get_top_bottom_wall_centres(mesh, wall_vertices)

    midpoint, traces, section_points, local_axes = get_midpoint_cross_section_from_centres(mesh, centre_top, centre_bottom)

    ## Label the left and right cross sections

    left_section, right_section, left_section_centre, right_section_centre, left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace = get_left_right_midsections(section_points, midpoint, local_axes)

    ## The radius of the circles to place at the top and bottom walls is taken from the radius of the midsections
    area1 = cross_section_area_2d(left_section)
    area2 = cross_section_area_2d(right_section)

    avg_area = 0.5 * (area1 + area2)

    ## Calculate the area of the circles at either end, based on the midsection areas
    radius = np.sqrt(max(avg_area, 0.0) / np.pi)

    ## Create the circles
    circle_top = make_circle(top_wall_coords, radius=radius)
    circle_bottom = make_circle(bottom_wall_coords, radius=radius)

    circle_top_trace = get_circle_trace(circle_top, colour = 'red', name = 'Top Circle')
    circle_bottom_trace = get_circle_trace(circle_bottom, colour = 'blue', name = 'Bottom Circle')

    spline_trace, spline_x, spline_y, spline_z = get_centreline_estimate(centre_top, centre_bottom, left_section_centre, right_section_centre)

    ## Visualise the first step
    if visualize:
        output = visualize_mesh(mesh,[top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace, *traces, left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace, circle_top_trace, circle_bottom_trace, spline_trace])

    ## To simplify analysis, we will split the mesh into left and right guard cells
    left, right = split_mesh_at_wall_vertices(mesh, wall_vertices, left_section_centre, right_section_centre)
    full_trace, left_trace, right_trace, left_half, right_half = get_centreline_estimate_and_split(centre_top, centre_bottom, left_section_centre, right_section_centre, n_points=40)
    section_points_left, section_traces_left, section_bary_data_left = get_regularly_spaced_cross_sections_batch(left, left_half, centre_top, centre_bottom, num_sections=20)
    section_points_right, section_traces_right, section_bary_data_right = get_regularly_spaced_cross_sections_batch(right, right_half, centre_top, centre_bottom, num_sections=20)

    return section_points_right, section_points_left, section_traces_left, section_traces_right, [spline_x, spline_y, spline_z]

def load_and_analyze(args):
    pressure, obj_path, mesh_id = args
    filename = f"{obj_path}{mesh_id}_{pressure:.1f}.obj"
    mesh = trimesh.load_mesh(filename)
    section_points_left, section_points_right, [spline_x, spline_y, spline_z] = analyze_stomata_mesh(
        filename, num_sections=20, n_points=40, visualize=False
    )
    return mesh, section_points_left, section_points_right, [spline_x, spline_y, spline_z]

def calculate_cross_section_aspect_ratios_and_lengths(sections_points_list):
    """
    Compute aspect ratio (major/minor), major axis length, and minor axis length for each cross section.
    Returns three lists: aspect_ratios, major_lengths, minor_lengths.
    """
    from sklearn.decomposition import PCA
    import numpy as np

    # Normalize input: allow a single (N,3) array
    if isinstance(sections_points_list, np.ndarray):
        if sections_points_list.ndim == 2 and sections_points_list.shape[1] == 3:
            sections_points_list = [sections_points_list]
        else:
            raise ValueError("If passing a numpy array it must be shape (N,3).")

    if not sections_points_list:
        return [], [], []

    valid_indices = [i for i, s in enumerate(sections_points_list) if s is not None and len(s) >= 3]
    if not valid_indices:
        n = len(sections_points_list)
        return [0.0]*n, [0.0]*n, [0.0]*n

    mid_idx = len(sections_points_list) // 2
    if sections_points_list[mid_idx] is None or len(sections_points_list[mid_idx]) < 3:
        mid_idx = valid_indices[len(valid_indices)//2]

    mid_points = np.asarray(sections_points_list[mid_idx])
    pca_mid = PCA(n_components=2)
    pca_mid.fit(mid_points)
    major_axis_ref = pca_mid.components_[0]

    aspect_ratios = []
    major_lengths = []
    minor_lengths = []

    for section in sections_points_list:
        if section is None or len(section) < 3:
            aspect_ratios.append(0.0)
            major_lengths.append(0.0)
            minor_lengths.append(0.0)
            continue
        pts = np.asarray(section)
        if pts.ndim != 2 or pts.shape[1] != 3:
            aspect_ratios.append(0.0)
            major_lengths.append(0.0)
            minor_lengths.append(0.0)
            continue
        pca = PCA(n_components=2)
        section_2d = pca.fit_transform(pts)
        comps = pca.components_
        dot0 = abs(np.dot(comps[0], major_axis_ref))
        dot1 = abs(np.dot(comps[1], major_axis_ref))
        if dot0 >= dot1:
            major_vals = section_2d[:, 0]
            minor_vals = section_2d[:, 1]
        else:
            major_vals = section_2d[:, 1]
            minor_vals = section_2d[:, 0]
        major_length = major_vals.max() - major_vals.min()
        minor_length = minor_vals.max() - minor_vals.min()
        if major_length <= 1e-12 or minor_length <= 1e-12:
            aspect_ratios.append(0.0)
            major_lengths.append(0.0)
            minor_lengths.append(0.0)
        else:
            aspect_ratios.append(major_length / minor_length)
            major_lengths.append(major_length)
            minor_lengths.append(minor_length)
    return aspect_ratios, major_lengths, minor_lengths

def calculate_cross_section_aspect_ratios(sections_points_list):
    """Compute aspect ratio (major/minor) for each cross section.

    Parameters
    ----------
    sections_points_list : list[array_like] or ndarray
        Either a list where each element is an (N_i, 3) array of points for a
        cross section, OR a single (N, 3) array (treated as one section).

    Returns
    -------
    list[float]
        Aspect ratio per cross section. 0.0 for invalid/degenerate sections.

    Notes
    -----
    1. A PCA is fit per section to obtain a local 2D plane.
    2. The *major* axis for each section is chosen as the PCA axis most
       aligned (by absolute dot) to the midpoint section's first PCA axis.
    3. The *minor* axis is enforced as the perpendicular in the 2D plane.
    4. Width/height are taken as peak-to-peak extents along major/minor.
    """
    from sklearn.decomposition import PCA
    import numpy as np

    # Normalize input: allow a single (N,3) array
    if isinstance(sections_points_list, np.ndarray):
        if sections_points_list.ndim == 2 and sections_points_list.shape[1] == 3:
            sections_points_list = [sections_points_list]
        else:
            raise ValueError("If passing a numpy array it must be shape (N,3).")

    if not sections_points_list:
        return []

    # Ensure all valid sections have >=3 pts; collect indices of valid sections
    valid_indices = [i for i, s in enumerate(sections_points_list) if s is not None and len(s) >= 3]
    if not valid_indices:
        return [0.0] * len(sections_points_list)

    # Midpoint index chosen among the *list* (not only valid subset) for stability
    mid_idx = len(sections_points_list) // 2
    if sections_points_list[mid_idx] is None or len(sections_points_list[mid_idx]) < 3:
        # Fallback: pick the central valid index
        mid_idx = valid_indices[len(valid_indices)//2]

    mid_points = np.asarray(sections_points_list[mid_idx])
    pca_mid = PCA(n_components=2)
    pca_mid.fit(mid_points)
    major_axis_ref = pca_mid.components_[0]

    aspect_ratios = []
    for section in sections_points_list:
        if section is None or len(section) < 3:
            aspect_ratios.append(0.0)
            continue
        pts = np.asarray(section)
        if pts.ndim != 2 or pts.shape[1] != 3:
            aspect_ratios.append(0.0)
            continue
        # Fit PCA (2D) on the 3D points
        pca = PCA(n_components=2)
        section_2d = pca.fit_transform(pts)  # shape (N,2) in PCA basis
        comps = pca.components_              # shape (2,3) in 3D
        # Decide which PCA component is the major axis (aligned with reference)
        dot0 = abs(np.dot(comps[0], major_axis_ref))
        dot1 = abs(np.dot(comps[1], major_axis_ref))
        if dot0 >= dot1:
            width_vals = section_2d[:, 0]
            height_vals = section_2d[:, 1]
        else:
            # swap if second component is closer to reference
            width_vals = section_2d[:, 1]
            height_vals = section_2d[:, 0]
        width = width_vals.max() - width_vals.min()
        height = height_vals.max() - height_vals.min()
        if width <= 1e-12 or height <= 1e-12:
            aspect_ratios.append(0.0)
        else:
            aspect_ratios.append(width / height)
    return aspect_ratios
def calculate_cross_section_areas(sections_points_list):
    """
    Calculate the area of each cross section in a list.
    Each cross section should be an (N, 3) array-like of points.
    Returns a list of areas (float), one per cross section. If a section has <3 points, area is 0.
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
## Helper functions for the cross section analysis

import trimesh
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_cross_sections_grid_overlay(sections_points_list1, sections_points_list2, n_cols=5, figsize=(15, 10), filename=None, colors=('k-', 'r-'), align_to_x=True):
    """
    Plot each pair of cross sections (Nx3 arrays) in a grid of 2D subplots, overlaid.
    Projects both sections to the best-fit 2D plane of the first section using PCA.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n_sections = min(len(sections_points_list1), len(sections_points_list2))
    n_rows = int(np.ceil(n_sections / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Ensure axes is always an array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()

    # Build a reference PCA basis from a stable section so that the first axis
    # maps to horizontal (x-axis) consistently across all subplots.
    pca_ref = None
    # Prefer a valid mid section from list1; fallback to any valid in list1 then list2
    def _pick_valid_section(lst):
        idxs = [i for i, s in enumerate(lst) if s is not None and len(s) >= 3]
        if not idxs:
            return None
        mid = idxs[len(idxs)//2]
        return np.asarray(lst[mid])

    ref_points = _pick_valid_section(sections_points_list1)
    if ref_points is None:
        ref_points = _pick_valid_section(sections_points_list2)
    if ref_points is not None:
        pca_ref = PCA(n_components=2).fit(np.asarray(ref_points))

    # Compute a single global rotation so the reference section is level with the x-axis.
    # This eliminates any consistent tilt relative to the horizontal axis across all subplots.
    R_global = np.eye(2)
    if align_to_x and ref_points is not None:
        try:
            if pca_ref is not None:
                ref2d = pca_ref.transform(np.asarray(ref_points))
            else:
                p = PCA(n_components=2)
                ref2d = p.fit_transform(np.asarray(ref_points))
            ref2d = ref2d - ref2d.mean(axis=0)
            if ref2d.shape[0] >= 1:
                idx_far = int(np.argmax(np.linalg.norm(ref2d, axis=1)))
                vec = ref2d[idx_far]
                theta0 = np.arctan2(vec[1], vec[0])
                cth, sth = np.cos(-theta0), np.sin(-theta0)
                R_global = np.array([[cth, -sth], [sth, cth]])
        except Exception:
            R_global = np.eye(2)

    for i in range(n_sections):
        section1 = sections_points_list1[i]
        section2 = sections_points_list2[i]
        ax = axes[i]

        if (section1 is None or len(section1) < 3) and (section2 is None or len(section2) < 3):
            ax.set_axis_off()
            continue

        # Project both sections using PCA from section1 (if available), else section2
        if section1 is not None and len(section1) >= 3:
            section1 = np.asarray(section1)
            # Use global reference PCA basis if available; else fit per-section
            if pca_ref is not None:
                section1_2d = pca_ref.transform(section1)
            else:
                pca = PCA(n_components=2)
                section1_2d = pca.fit_transform(section1)
            # Center at centroid for alignment
            centroid1 = section1_2d.mean(axis=0)
            rel1 = section1_2d - centroid1
            # Sort by angle around origin (centroid-aligned)
            angles1 = np.arctan2(rel1[:, 1], rel1[:, 0])
            sort_idx1 = np.argsort(angles1)
            s1_sorted = rel1[sort_idx1]
            s1_sorted = np.vstack([s1_sorted, s1_sorted[0]])
            # Apply a single global rotation so sections are level with the x-axis
            s1_plot = s1_sorted @ R_global.T
            ax.plot(s1_plot[:, 0], s1_plot[:, 1], colors[0], label='Mesh 1')
            # Track radius for symmetric limits
            rmax = float(np.linalg.norm(s1_plot, axis=1).max())
            if section2 is not None and len(section2) >= 3:
                section2 = np.asarray(section2)
                if pca_ref is not None:
                    section2_2d = pca_ref.transform(section2)
                else:
                    section2_2d = pca.transform(section2)
                centroid2 = section2_2d.mean(axis=0)
                rel2 = section2_2d - centroid2
                angles2 = np.arctan2(rel2[:, 1], rel2[:, 0])
                sort_idx2 = np.argsort(angles2)
                s2_sorted = rel2[sort_idx2]
                s2_sorted = np.vstack([s2_sorted, s2_sorted[0]])
                s2_plot = s2_sorted @ R_global.T
                ax.plot(s2_plot[:, 0], s2_plot[:, 1], colors[1], label='Mesh 2')
                rmax = max(rmax, float(np.linalg.norm(s2_plot, axis=1).max()))
            # Set symmetric limits about origin for better overlay alignment
            if np.isfinite(rmax) and rmax > 0:
                ax.set_xlim(-rmax, rmax)
                ax.set_ylim(-rmax, rmax)
        elif section2 is not None and len(section2) >= 3:
            section2 = np.asarray(section2)
            if pca_ref is not None:
                section2_2d = pca_ref.transform(section2)
            else:
                pca = PCA(n_components=2)
                section2_2d = pca.fit_transform(section2)
            centroid2 = section2_2d.mean(axis=0)
            rel2 = section2_2d - centroid2
            angles2 = np.arctan2(rel2[:, 1], rel2[:, 0])
            sort_idx2 = np.argsort(angles2)
            s2_sorted = rel2[sort_idx2]
            s2_sorted = np.vstack([s2_sorted, s2_sorted[0]])
            s2_plot = s2_sorted @ R_global.T
            ax.plot(s2_plot[:, 0], s2_plot[:, 1], colors[1], label='Mesh 2')
            rmax = float(np.linalg.norm(s2_plot, axis=1).max())
            if np.isfinite(rmax) and rmax > 0:
                ax.set_xlim(-rmax, rmax)
                ax.set_ylim(-rmax, rmax)

        ax.set_title(f'Section {i+1}')
        ax.axis('equal')
        ax.set_xticks([])
        plt.legend()
        plt.savefig(filename, dpi=300) if filename else None
    #mesh_trace = go.Mesh3d(
    #    x=mesh.vertices[:, 0],
    #    y=mesh.vertices[:, 1],
    #    z=mesh.vertices[:, 2],
    #    i=mesh.faces[:, 0],
    #    j=mesh.faces[:, 1],
    #    k=mesh.faces[:, 2],
    #    color='lightgray',
    #    opacity=0.75,
    #    name='Mesh'
    #)
    #return mesh_trace

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
            color="#0072B2",
            opacity=0.65,
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
    return fig


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

def get_circle_trace(circle, name="Circle", colour="red"):
    ## Create the circle traces
    circle_trace = go.Scatter3d(
        x=circle[:, 0],
        y=circle[:, 1],
        z=circle[:, 2],
        mode='lines',
        line=dict(color=colour, width=2),
        name=name
    )
    return circle_trace

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

def order_points_consistently(points, normal, midpoint):
    # build a stable 2D basis in the slicing plane
    ref = np.array([0,0,1]) if abs(np.dot(normal,[0,0,1])) < 0.9 else np.array([1,0,0])
    v1 = np.cross(normal, ref); v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    R = points - midpoint
    coords2d = np.column_stack([R @ v1, R @ v2])
    angles = np.arctan2(coords2d[:,1], coords2d[:,0])
    order = np.argsort(angles)
    return points[order]

## Function to iterate over a number of meshes, and create plots showing the aspect ratio of the cross sections along the guard cells: OBSOLETE
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
                ordered_vertices = order_points_consistently(section.vertices, tangent, midpoint)
                sections_points_list.append(ordered_vertices)

    return sections_points_list

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

def find_wall_vertices_vertex_normals(mesh: trimesh.Trimesh, dot_thresh=0.2):
    """
    Fallback: mark a vertex as wall if its incident face normals contain
    at least one pair with dot < - (1 - dot_thresh) (i.e., strong opposition).
    """
    face_normals = mesh.face_normals
    # Build incident face list per vertex
    incident = [[] for _ in range(len(mesh.vertices))]
    for f_idx, face in enumerate(mesh.faces):
        for v in face:
            incident[v].append(f_idx)

    wall_vertices = []
    opposite_limit = -(1.0 - dot_thresh)
    for vidx, f_list in enumerate(incident):
        if len(f_list) < 2:
            continue
        fn = face_normals[f_list]
        # Fast prune: compare against first normal
        dots = fn @ fn[0]
        if np.min(dots) > opposite_limit:  # no strong opposite to first
            # full pairwise if ambiguous
            opp = False
            for i in range(len(fn)):
                if opp: break
                d = fn[i+1:] @ fn[i]
                if d.size and np.min(d) < opposite_limit:
                    opp = True
            if not opp:
                continue
        wall_vertices.append(vidx)
    return np.array(wall_vertices, dtype=int)

def get_top_bottom_wall_centres(mesh, wall_vertices):
    ## Separate these into top and bottom walls
    verts = mesh.vertices
    wall_coords = verts[wall_vertices]
    # Use KMeans clustering to separate into two groups (top and bottom walls)
    if len(wall_coords) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(wall_coords)
        labels = kmeans.labels_
        group1 = wall_coords[labels == 0]
        group2 = wall_coords[labels == 1]
        # Assign top/bottom by comparing mean y values
        if group1[:, 1].mean() > group2[:, 1].mean():
            top_wall_coords = group1
            bottom_wall_coords = group2
        else:
            top_wall_coords = group2
            bottom_wall_coords = group1
    else:
        # Fallback: use all as top, none as bottom
        top_wall_coords = wall_coords
        bottom_wall_coords = np.empty((0, 3))

    # Create the wall traces
    top_wall_trace = go.Scatter3d(
        x=top_wall_coords[:, 0],
        y=top_wall_coords[:, 1],
        z=top_wall_coords[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Top Wall Vertices'
    )
    bottom_wall_trace = go.Scatter3d(
        x=bottom_wall_coords[:, 0],
        y=bottom_wall_coords[:, 1],
        z=bottom_wall_coords[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Bottom Wall Vertices'
    )

    centre_top = top_wall_coords.mean(axis=0)
    centre_bottom = bottom_wall_coords.mean(axis=0)

    centre_top_trace = go.Scatter3d(
        x=[centre_top[0]],
        y=[centre_top[1]],
        z=[centre_top[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Top Wall Centre'
    )
    centre_bottom_trace = go.Scatter3d(
        x=[centre_bottom[0]],
        y=[centre_bottom[1]],
        z=[centre_bottom[2]],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Bottom Wall Centre'
    )

    return centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace

def get_midpoint_cross_section_from_centres(mesh, centre_top, centre_bottom):
    """
    Take a cross section at the midpoint between two precomputed wall centres.
    The cross section plane is perpendicular to the line joining the centres.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to section.
    centre_top : np.ndarray
        3D coordinates of the top wall centre.
    centre_bottom : np.ndarray
        3D coordinates of the bottom wall centre.

    Returns
    -------
    midpoint : np.ndarray
        The midpoint between wall centres.
    traces : list
        Plotly traces for visualization.
    section_points : np.ndarray
        Points of the cross section at the midpoint.
    local_axes : np.ndarray
        3x3 array: [wall_vec, left_right_vec, normal_vec]
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    # Define wall axis (from bottom to top)
    wall_vec = centre_top - centre_bottom
    wall_vec /= np.linalg.norm(wall_vec)

    # Midpoint
    midpoint = (centre_top + centre_bottom) / 2

    # Find all mesh vertices near the midpoint plane (for local PCA)
    verts = mesh.vertices
    dists = np.dot(verts - midpoint, wall_vec)
    close_mask = np.abs(dists) < np.percentile(np.abs(dists), 10)  # 10% closest to plane
    midplane_points = verts[close_mask]

    # Use PCA to find the two main axes in the midplane
    pca = PCA(n_components=2)
    pca.fit(midplane_points)
    left_right_vec = pca.components_[0]
    normal_vec = np.cross(wall_vec, left_right_vec)
    normal_vec /= np.linalg.norm(normal_vec)

    # Take cross section at midpoint, normal to wall_vec
    section = mesh.section(plane_origin=midpoint, plane_normal=wall_vec)
    if section is not None:
        if hasattr(section, 'discrete'):
            section_points = np.vstack([seg for seg in section.discrete])
        else:
            section_points = section.vertices
    else:
        section_points = None

    # Visualization traces
    midpoint_trace = go.Scatter3d(
        x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]],
        mode='markers', marker=dict(size=8, color='black'), name='Midpoint (centres)'
    )
    traces = [midpoint_trace]
    if section_points is not None:
        traces.append(go.Scatter3d(
            x=section_points[:, 0], y=section_points[:, 1], z=section_points[:, 2],
            mode='markers', marker=dict(size=5, color='orange'), name='Midpoint Cross Section (centres)'
        ))

    local_axes = np.stack([wall_vec, left_right_vec, normal_vec], axis=0)

    return midpoint, traces, section_points, local_axes

## Get the areas of the midsections
def cross_section_area_2d(points):
        pca = PCA(n_components=2)
        pts2 = pca.fit_transform(points)
        hull = ConvexHull(pts2)
        return hull.volume

def get_left_right_midsections(section_points, midpoint, local_axes):

    # The left-right axis is the second vector in local_axes (from get_midpoint_cross_section_from_centres)
    left_right_vec = local_axes[1]

    # Project section points onto the left-right axis (relative to the midpoint)
    relative_points = section_points - midpoint
    side_values = np.dot(relative_points, left_right_vec)

    left_section = section_points[side_values < 0]
    right_section = section_points[side_values >= 0]

    left_section_centre = left_section.mean(axis=0)
    right_section_centre = right_section.mean(axis=0)

    # Create our left and right traces
    left_midsection_trace = go.Scatter3d(
        x=left_section[:, 0], y=left_section[:, 1], z=left_section[:, 2],
        mode='markers', marker=dict(size=5, color='orange'), name='Left Midsection'
    )

    right_midsection_trace = go.Scatter3d(
        x=right_section[:, 0], y=right_section[:, 1], z=right_section[:, 2],
        mode='markers', marker=dict(size=5, color='blue'), name='Right Midsection'
    )

    left_section_centre_trace = go.Scatter3d(
        x=[left_section_centre[0]], y=[left_section_centre[1]], z=[left_section_centre[2]],
        mode='markers', marker=dict(size=8, color='orange'), name='Left Section Centre'
    )

    right_section_centre_trace = go.Scatter3d(
        x=[right_section_centre[0]], y=[right_section_centre[1]], z=[right_section_centre[2]],
        mode='markers', marker=dict(size=8, color='blue'), name='Right Section Centre'
    )
    return left_section, right_section, left_section_centre, right_section_centre,left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace

def make_circle(coords, radius):
    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(coords)
    plane_origin = coords.mean(axis=0)
    plane_normal = pca.components_[-1]  # normal vector to the best-fit plane

    # Create circle points
    circle_points = []
    for angle in np.linspace(0, 2 * np.pi, 100):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        circle_points.append(plane_origin + x * pca.components_[0] + y * pca.components_[1] + z * plane_normal)
    return np.array(circle_points)

def get_centreline_estimate(top_circle_centre, bottom_circle_centre, left_section_centre, right_section_centre):
    points = np.vstack([top_circle_centre, left_section_centre, bottom_circle_centre, right_section_centre, top_circle_centre])  # repeat first point to close
    t = np.linspace(0, 1, len(points))
    t_fine = np.linspace(0, 1, 200)
    spline_x = CubicSpline(t, points[:,0], bc_type='periodic')(t_fine)
    spline_y = CubicSpline(t, points[:,1], bc_type='periodic')(t_fine)
    spline_z = CubicSpline(t, points[:,2], bc_type='periodic')(t_fine)
    spline_trace = go.Scatter3d(
        x=spline_x, y=spline_y, z=spline_z,
        mode='lines',
        line=dict(color='black', width=6),
        name='Central Spline (Closed)'
        )
    return spline_trace, spline_x, spline_y, spline_z

def get_centreline_estimate_and_split(top_circle_centre, bottom_circle_centre,
                                      left_section_centre, right_section_centre,
                                      n_points=200):
    """
    Build a closed spline centreline and split it into left and right guard cell halves
    based on top and bottom anchors.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline
    import plotly.graph_objects as go

    # Full loop
    points = np.vstack([
        top_circle_centre,
        left_section_centre,
        bottom_circle_centre,
        right_section_centre,
        top_circle_centre
    ])  # repeat to close

    t = np.linspace(0, 1, len(points))
    t_fine = np.linspace(0, 1, n_points)

    spline_x = CubicSpline(t, points[:, 0], bc_type='periodic')(t_fine)
    spline_y = CubicSpline(t, points[:, 1], bc_type='periodic')(t_fine)
    spline_z = CubicSpline(t, points[:, 2], bc_type='periodic')(t_fine)
    spline = np.column_stack([spline_x, spline_y, spline_z])

    # Find closest indices to anchors
    top_idx = np.argmin(np.linalg.norm(spline - top_circle_centre, axis=1))
    bottom_idx = np.argmin(np.linalg.norm(spline - bottom_circle_centre, axis=1))

    if top_idx < bottom_idx:
        left_half = spline[top_idx:bottom_idx+1]
        right_half = np.vstack([spline[bottom_idx:], spline[:top_idx+1]])
    else:
        right_half = spline[bottom_idx:top_idx+1]
        left_half = np.vstack([spline[top_idx:], spline[:bottom_idx+1]])

    # Plot traces
    full_trace = go.Scatter3d(
        x=spline_x, y=spline_y, z=spline_z,
        mode='lines', line=dict(color='black', width=4),
        name='Full Centreline'
    )
    left_trace = go.Scatter3d(
        x=left_half[:, 0], y=left_half[:, 1], z=left_half[:, 2],
        mode='lines', line=dict(color='red', width=6),
        name='Left Guard Cell Centreline'
    )
    right_trace = go.Scatter3d(
        x=right_half[:, 0], y=right_half[:, 1], z=right_half[:, 2],
        mode='lines', line=dict(color='blue', width=6),
        name='Right Guard Cell Centreline'
    )

    return full_trace, left_trace, right_trace, left_half, right_half


from scipy.interpolate import CubicSpline
import numpy as np
import plotly.graph_objects as go

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


def get_barycentric_coords(point, face_vertices):
    # face_vertices: (3, 3) array
    v0 = face_vertices[1] - face_vertices[0]
    v1 = face_vertices[2] - face_vertices[0]
    v2 = point - face_vertices[0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])

def barycentric_to_point(face_vertices, bary):
    return bary[0]*face_vertices[0] + bary[1]*face_vertices[1] + bary[2]*face_vertices[2]

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

def split_mesh_at_wall_vertices(mesh, wall_vertices, left_centroid, right_centroid):
    """
    Split mesh into left and right guard cells using the wall vertices,
    without using a slicing plane.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    wall_vertices : (M,) indices of the wall
    left_centroid, right_centroid : (3,) np.arrays
        Approximate centers of left and right guard cells

    Returns
    -------
    left_mesh, right_mesh : trimesh.Trimesh
    """
    vertices = mesh.vertices

    # Compute distance of all vertices to left/right centroid
    dist_left = np.linalg.norm(vertices - left_centroid, axis=1)
    dist_right = np.linalg.norm(vertices - right_centroid, axis=1)

    # Assign vertices to the closer guard cell
    left_mask = dist_left < dist_right
    right_mask = dist_right <= dist_left

    # Include wall in both
    left_mask[wall_vertices] = True
    right_mask[wall_vertices] = True

    # Filter faces
    left_faces_mask = np.all(left_mask[mesh.faces], axis=1)
    right_faces_mask = np.all(right_mask[mesh.faces], axis=1)

    left_faces = mesh.faces[left_faces_mask]
    right_faces = mesh.faces[right_faces_mask]

    # Remap vertex indices
    def remap_vertices(mask, faces):
        old_to_new = np.full(len(vertices), -1, dtype=int)
        old_to_new[np.where(mask)[0]] = np.arange(np.sum(mask))
        return trimesh.Trimesh(vertices=vertices[mask], faces=old_to_new[faces], process=True)

    left_mesh = remap_vertices(left_mask, left_faces)
    right_mesh = remap_vertices(right_mask, right_faces)

    return left_mesh, right_mesh

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











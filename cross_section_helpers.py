## Helper functions for the cross section analysis

import trimesh
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_cross_sections_grid_overlay(sections_points_list1, sections_points_list2, n_cols=5, figsize=(15, 10), filename=None, colors=('k-', 'r-')):
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
    axes = axes.flatten()

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
            pca = PCA(n_components=2)
            section1_2d = pca.fit_transform(section1)
            # Sort section1 points by angle around centroid
            centroid1 = section1_2d.mean(axis=0)
            rel1 = section1_2d - centroid1
            angles1 = np.arctan2(rel1[:, 1], rel1[:, 0])
            sort_idx1 = np.argsort(angles1)
            section1_2d_sorted = section1_2d[sort_idx1]
            section1_2d_sorted = np.vstack([section1_2d_sorted, section1_2d_sorted[0]])
            ax.plot(section1_2d_sorted[:, 0], section1_2d_sorted[:, 1], colors[0], label='Mesh 1')
            if section2 is not None and len(section2) >= 3:
                section2 = np.asarray(section2)
                section2_2d = pca.transform(section2)
                centroid2 = section2_2d.mean(axis=0)
                rel2 = section2_2d - centroid2
                angles2 = np.arctan2(rel2[:, 1], rel2[:, 0])
                sort_idx2 = np.argsort(angles2)
                section2_2d_sorted = section2_2d[sort_idx2]
                section2_2d_sorted = np.vstack([section2_2d_sorted, section2_2d_sorted[0]])
                ax.plot(section2_2d_sorted[:, 0], section2_2d_sorted[:, 1], colors[1], label='Mesh 2')
        elif section2 is not None and len(section2) >= 3:
            section2 = np.asarray(section2)
            pca = PCA(n_components=2)
            section2_2d = pca.fit_transform(section2)
            centroid2 = section2_2d.mean(axis=0)
            rel2 = section2_2d - centroid2
            angles2 = np.arctan2(rel2[:, 1], rel2[:, 0])
            sort_idx2 = np.argsort(angles2)
            section2_2d_sorted = section2_2d[sort_idx2]
            section2_2d_sorted = np.vstack([section2_2d_sorted, section2_2d_sorted[0]])
            ax.plot(section2_2d_sorted[:, 0], section2_2d_sorted[:, 1], colors[1], label='Mesh 2')

        ax.set_title(f'Section {i+1}')
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(n_sections, len(axes)):
        axes[j].set_axis_off()

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_cross_sections_grid(sections_points_list, n_cols=5, figsize=(15, 10), filename=None):
    """
    Plot each cross section (Nx3 array) in a grid of 2D subplots.
    Projects each section to its best-fit 2D plane using PCA.
    """
    n_sections = len(sections_points_list)
    n_rows = int(np.ceil(n_sections / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, section in enumerate(sections_points_list):
        if section is None or len(section) < 3:
            axes[i].set_axis_off()
            continue
        section = np.asarray(section)
        # Project to 2D using PCA
        pca = PCA(n_components=2)
        section_2d = pca.fit_transform(section)
        # Sort points by angle around centroid for consistent ordering
        centroid = section_2d.mean(axis=0)
        rel = section_2d - centroid
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        sort_idx = np.argsort(angles)
        section_2d_sorted = section_2d[sort_idx]
        # Close the loop for plotting
        section_2d_sorted = np.vstack([section_2d_sorted, section_2d_sorted[0]])
        axes[i].plot(section_2d_sorted[:, 0], section_2d_sorted[:, 1], 'k-')
        axes[i].set_title(f'Section {i+1}')
        axes[i].axis('equal')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_axis_off()

    plt.tight_layout()
    plt.savefig(filename) if filename else plt.show()
    plt.show()

def create_centreline_trace(centreline):

    centreline_trace = go.Scatter3d(
        x=centreline[:, 0],
        y=centreline[:, 1],
        z=centreline[:, 2],
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(width=3),
        name='Bezier Centreline'
    )
    return centreline_trace

def create_outer_curve_trace(outer_curve):

    outer_curve_trace = go.Scatter3d(
        x=outer_curve[:, 0],
        y=outer_curve[:, 1],
        z=outer_curve[:, 2],
        mode='lines',
        line=dict(width=2, color='blue'),
        name='Projected Outer Curve'
    )
    return outer_curve_trace

## Create mesh trace
def create_mesh_trace(mesh):

    mesh_trace = go.Mesh3d(
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
    return mesh_trace

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


# def create_projected_curve(mesh, outer_points, smooth=50, n_points=120, end_segment_size=10):
#     """
#     Create a curve that extends to the min/max x boundaries of the cell with
#     parallel end segments perpendicular to circle axes.

#     Fixes:
#       - uses both near_min_x and near_max_x when choosing boundary candidates
#       - chooses the candidate closest to each endpoint
#       - computes end directions from transition->endpoint so signs are correct
#     """
#     from scipy.interpolate import splprep, splev, CubicSpline
#     import numpy as np
#     import trimesh

#     # 1. Fit spline to original curve
#     tck, u = splprep(outer_points.T, s=smooth)

#     # 2. Make a copy of the input points for the central part
#     central_curve = outer_points.copy()

#     # 3. Find the min/max x value in the mesh (with a small buffer)
#     min_x = mesh.vertices[:, 0].min() + 0.5  # buffer
#     max_x = mesh.vertices[:, 0].max() - 0.5  # buffer

#     # 4. Find mesh vertices near minimum and maximum x
#     x_tol = 2.0  # tolerance for being "near" a boundary
#     near_min_x = mesh.vertices[np.abs(mesh.vertices[:, 0] - min_x) < x_tol]
#     near_max_x = mesh.vertices[np.abs(mesh.vertices[:, 0] - max_x) < x_tol]

#     # Helper to pick best boundary point (closest in full 3D distance) from a candidate set
#     def pick_best_candidate(candidates, ref_point):
#         if len(candidates) == 0:
#             return None
#         dists = np.linalg.norm(candidates - ref_point.reshape(1, 3), axis=1)
#         return candidates[np.argmin(dists)]

#     if len(near_min_x) > 0 or len(near_max_x) > 0:
#         # starting and ending endpoints from original outer_points
#         start_pt = outer_points[0]
#         end_pt = outer_points[-1]

#         # pick best candidates near min and max
#         start_candidate_min = pick_best_candidate(near_min_x, start_pt) if len(near_min_x) > 0 else None
#         start_candidate_max = pick_best_candidate(near_max_x, start_pt) if len(near_max_x) > 0 else None

#         # choose the closer of available candidates for the start
#         candidates_for_start = []
#         if start_candidate_min is not None:
#             candidates_for_start.append(start_candidate_min)
#         if start_candidate_max is not None:
#             candidates_for_start.append(start_candidate_max)
#         if len(candidates_for_start) > 0:
#             start_boundary = pick_best_candidate(np.vstack(candidates_for_start), start_pt)
#         else:
#             start_boundary = start_pt.copy()

#         # pick best candidates for end
#         end_candidate_min = pick_best_candidate(near_min_x, end_pt) if len(near_min_x) > 0 else None
#         end_candidate_max = pick_best_candidate(near_max_x, end_pt) if len(near_max_x) > 0 else None

#         candidates_for_end = []
#         if end_candidate_min is not None:
#             candidates_for_end.append(end_candidate_min)
#         if end_candidate_max is not None:
#             candidates_for_end.append(end_candidate_max)
#         if len(candidates_for_end) > 0:
#             end_boundary = pick_best_candidate(np.vstack(candidates_for_end), end_pt)
#         else:
#             end_boundary = end_pt.copy()

#         # If the two boundaries ended up identical (rare), fall back to mirrored behavior:
#         if np.allclose(start_boundary, end_boundary):
#             # try to force one to the opposite side by using the farthest candidate for end if available
#             if len(near_max_x) > 0 and len(near_min_x) > 0:
#                 # pick the farthest candidate from start for end (ensures opposite side)
#                 all_candidates = np.vstack([near_min_x, near_max_x])
#                 dists = np.linalg.norm(all_candidates - start_boundary.reshape(1,3), axis=1)
#                 end_boundary = all_candidates[np.argmax(dists)]

#         # 7. Create new curve with these endpoints
#         extended_curve = np.vstack([start_boundary, central_curve, end_boundary])
#     else:
#         # Fallback to original curve if no min/max x points found
#         extended_curve = central_curve

#     # 8. Project onto mesh surface to ensure all points are on the mesh
#     extended_curve = trimesh.proximity.closest_point(mesh, extended_curve)[0]

#     # 9. Resample for smoothness with higher density at ends
#     if len(extended_curve) >= 3:
#         t = np.linspace(0, 1, len(extended_curve))
#         t_new = np.linspace(0, 1, n_points)

#         # Use cubic interpolation to maintain shape
#         cs_x = CubicSpline(t, extended_curve[:, 0])
#         cs_y = CubicSpline(t, extended_curve[:, 1])
#         cs_z = CubicSpline(t, extended_curve[:, 2])

#         final_curve = np.column_stack([
#             cs_x(t_new), cs_y(t_new), cs_z(t_new)
#         ])

#         # Ensure endpoints are exactly preserved
#         final_curve[0] = extended_curve[0]
#         final_curve[-1] = extended_curve[-1]
#     else:
#         # Fall back to original method if insufficient points
#         tck2, u2 = splprep(extended_curve.T, s=smooth)
#         final_curve = np.array(splev(np.linspace(0, 1, n_points), tck2)).T

#     # 10. Apply additional smoothing
#     tck3, u3 = splprep(final_curve.T, s=smooth * 0.5)
#     final_curve_smooth = np.array(splev(np.linspace(0, 1, n_points), tck3)).T

#     # 11. Make end segments parallel and perpendicular to circle axes
#     if end_segment_size > n_points // 4:
#         end_segment_size = n_points // 4

#     # Transition indices
#     start_transition_idx = end_segment_size
#     end_transition_idx = n_points - end_segment_size - 1

#     # Robust approach: compute direction from transition point toward the true endpoint,
#     # then project that into the X-Z plane (i.e. remove Y component) to get the direction to use.
#     def safe_unit(v):
#         norm = np.linalg.norm(v)
#         return v / norm if norm > 1e-12 else v

#     y_axis = np.array([0.0, 1.0, 0.0])

#     # For start: desired direction = from transition to actual start endpoint (final index 0 of final_curve_smooth)
#     desired_start_dir = final_curve_smooth[0] - final_curve_smooth[start_transition_idx]
#     start_dir = desired_start_dir - np.dot(desired_start_dir, y_axis) * y_axis
#     start_dir = safe_unit(start_dir)
#     if np.linalg.norm(start_dir) < 1e-8:
#         # fallback to tangent-based projection (older behavior) if degenerate
#         start_tangent = final_curve_smooth[start_transition_idx + 1] - final_curve_smooth[start_transition_idx - 1]
#         start_dir = start_tangent - np.dot(start_tangent, y_axis) * y_axis
#         start_dir = safe_unit(start_dir)

#     # For end: direction from transition to final endpoint
#     desired_end_dir = final_curve_smooth[-1] - final_curve_smooth[end_transition_idx]
#     end_dir = desired_end_dir - np.dot(desired_end_dir, y_axis) * y_axis
#     end_dir = safe_unit(end_dir)
#     if np.linalg.norm(end_dir) < 1e-8:
#         end_tangent = final_curve_smooth[end_transition_idx + 1] - final_curve_smooth[end_transition_idx - 1]
#         end_dir = end_tangent - np.dot(end_tangent, y_axis) * y_axis
#         end_dir = safe_unit(end_dir)

#     # Adjust the start segment points (interpolate outward from transition point)
#     for i in range(end_segment_size):
#         # index i runs from 0..end_segment_size-1
#         factor = (end_segment_size - i) / float(end_segment_size)  # 1 at transition -> 0 at endpoint? we want 0 at transition -> 1 at endpoint
#         # we want factor = (i+1)/end_segment_size so 0 at transition_idx and 1 at endpoint
#         factor = (i + 1) / float(end_segment_size)
#         orig_dir = final_curve_smooth[i] - final_curve_smooth[start_transition_idx]
#         dist = np.linalg.norm(orig_dir)
#         final_curve_smooth[i] = final_curve_smooth[start_transition_idx] + dist * start_dir * factor

#     # Adjust the end segment points
#     for i in range(end_segment_size):
#         idx = n_points - end_segment_size + i  # start at end_transition_idx+1 up to last
#         factor = (i + 1) / float(end_segment_size)
#         orig_dir = final_curve_smooth[idx] - final_curve_smooth[end_transition_idx]
#         dist = np.linalg.norm(orig_dir)
#         final_curve_smooth[idx] = final_curve_smooth[end_transition_idx] + dist * end_dir * factor

#     # 12. Project back onto mesh surface
#     final_curve_smooth = trimesh.proximity.closest_point(mesh, final_curve_smooth)[0]

#     # 13. Apply one final light smoothing to blend the adjusted ends
#     tck4, u4 = splprep(final_curve_smooth.T, s=smooth * 0.3)
#     final_curve_smooth = np.array(splev(np.linspace(0, 1, n_points), tck4)).T

#     # Ensure endpoints remain fixed
#     final_curve_smooth[0] = final_curve[0]
#     final_curve_smooth[-1] = final_curve[-1]

#     return final_curve_smooth


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

## Also obsolete - uses the wrong cross section code. However, we may need to use the code here to fit the ellipses, and extract the aspect ratios.
# def extract_cross_section_data(file, path, side="single_a", n_sections=20):
#     """
#     Extract cross-section data from a mesh without creating visualizations.
    
#     Parameters:
#     -----------
#     file_index : int
#         Index of the file to analyze from the files list
#     side : str
#         Side to analyze ("single_a" or "single_b")
#     n_sections : int
#         Number of cross-sections to generate along the centreline
        
#     Returns:
#     --------
#     data : dict
#         Dictionary containing all computed data
#     """
#     from sklearn.decomposition import PCA
#     from skimage.measure import EllipseModel
    
#     # Initialize results dictionary
#     results = {
#         'file_info': {
#             'file_name': file,
#             'side': side,
#             'mesh_path': f"{path}/{file}_{side}.obj"
#         },
#         'mesh_data': {},
#         'cross_sections': {
#             'indices': [],
#             '3d_points': [],
#             '2d_points': [],
#             'pca_components': [],
#             'std_devs': [],
#             'aspect_ratios': [],
#             'ellipse_fits': []
#         }
#     }
    
#     # Load and process mesh
#     mesh_path = results['file_info']['mesh_path']
#     mesh = trimesh.load(mesh_path, process=True)
    
#     # Align the mesh to the Y-axis and center it
#     mesh = align_mesh_to_y_axis(mesh)
    
#     # Ensure consistent orientation
#     outer_points_check = find_outer_edge(mesh, smoothing=50)
#     if np.mean(outer_points_check[:, 0]) < 0:
#         mesh.vertices[:, 0] *= -1
    
#     # Process mesh to find centreline
#     outer_points = find_outer_edge(mesh, smoothing=50)
#     outer_curve_projected = create_projected_curve(
#         mesh, outer_points, smooth=50, n_points=120, end_segment_size=10
#     )

#     # Get midpoint cross-section
#     _, _, section_points = get_midpoint_trace(outer_points, mesh)

#     # Get the circle centers
#     _, _, _, _, _, center_start, center_end, circle_start, circle_end = define_end_point_circles(outer_curve_projected, section_points, mesh=mesh)

#     # Create the centreline using the actual circle centers
#     section_midpoint = np.mean(section_points, axis=0)
#     centreline, _ = create_bezier_centreline(
#         center_start, section_midpoint, center_end, curve_points=50
#     )

#     # Store centreline in results
#     results['mesh_data']['centreline'] = centreline
#     results['mesh_data']['outer_points'] = outer_points
#     results['mesh_data']['outer_curve_projected'] = outer_curve_projected
    
#     # Define cross-section indices
#     end_margin_points = 3  # Avoid unstable sections at the very ends
#     start_index = end_margin_points
#     end_index = len(centreline) - 1 - end_margin_points
#     indices = np.linspace(start_index, end_index, n_sections, dtype=int)
#     results['cross_sections']['indices'] = indices.tolist()
    
#     # Get all cross-sections
#     all_section_points = get_cross_section_points(mesh, centreline, indices)
    
#     # Calculate tangents along the centreline
#     tangents = np.gradient(centreline, axis=0)
#     tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
#     results['mesh_data']['tangents'] = tangents
    
#     # Define a consistent reference for orientation
#     pattern_reference = np.array([1.0, 0.0, 1.0])  # Diagonal pattern in X-Z plane
#     pattern_reference = pattern_reference / np.linalg.norm(pattern_reference)  # Normalize
#     x_axis = np.array([1, 0, 0])
    
#     # Process each cross-section
#     for i, section_points in enumerate(all_section_points):
#         if section_points is None or len(section_points) < 3:
#             # Store empty data for missing sections
#             results['cross_sections']['3d_points'].append(None)
#             results['cross_sections']['2d_points'].append(None)
#             results['cross_sections']['pca_components'].append(None)
#             results['cross_sections']['std_devs'].append(None)
#             results['cross_sections']['aspect_ratios'].append(np.nan)
#             results['cross_sections']['ellipse_fits'].append(None)
#             continue
            
#         idx = indices[i]
#         tangent = tangents[idx]
        
#         # Store 3D points
#         results['cross_sections']['3d_points'].append(section_points.tolist())
        
#         # Perform PCA
#         pca = PCA(n_components=2)
#         pca.fit(section_points)
#         std_devs = np.sqrt(pca.explained_variance_)
        
#         # Store PCA components and std_devs
#         results['cross_sections']['pca_components'].append(pca.components_.tolist())
#         results['cross_sections']['std_devs'].append(std_devs.tolist())

#         # Get PCA components
#         component_0 = pca.components_[0]
#         component_1 = pca.components_[1]

#         # CONSISTENT HORIZONTAL ALIGNMENT:
#         # Check which component is more aligned with X-axis
#         dot_0_x = np.abs(np.dot(component_0, x_axis))
#         dot_1_x = np.abs(np.dot(component_1, x_axis))

#         # The component more aligned with X is the major axis
#         if dot_0_x >= dot_1_x:
#             main_axis_vec = component_0 if np.dot(component_0, x_axis) >= 0 else -component_0
#             minor_axis_vec = component_1 if np.cross(main_axis_vec, component_1)[1] >= 0 else -component_1
#             main_axis_std, minor_axis_std = std_devs[0], std_devs[1]
#         else:
#             main_axis_vec = component_1 if np.dot(component_1, x_axis) >= 0 else -component_1
#             minor_axis_vec = component_0 if np.cross(main_axis_vec, component_0)[1] >= 0 else -component_0
#             main_axis_std, minor_axis_std = std_devs[1], std_devs[0]
        
#         # Calculate aspect ratio
#         if minor_axis_std > 1e-9:
#             aspect_ratio = main_axis_std / minor_axis_std
#         else:
#             aspect_ratio = np.nan
            
#         # Store aspect ratio
#         results['cross_sections']['aspect_ratios'].append(float(aspect_ratio))
        
#         # Project points to 2D using the principal axes
#         centroid = np.mean(section_points, axis=0)
#         centered = section_points - centroid
#         x_coords = centered @ main_axis_vec
#         y_coords = centered @ minor_axis_vec
#         points_2d = np.column_stack([x_coords, y_coords])
        
#         # Store 2D points
#         results['cross_sections']['2d_points'].append(points_2d.tolist())
        
#         # Fit an ellipse to the 2D points
#         ellipse_data = None
#         try:
#             ellipse_model = EllipseModel()
#             if len(points_2d) >= 5 and ellipse_model.estimate(points_2d):
#                 xc, yc, a, b, theta = ellipse_model.params
                
#                 # Generate points along the ellipse
#                 t = np.linspace(0, 2*np.pi, 100)
#                 ellipse_x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
#                 ellipse_y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
#                 ellipse_points = np.column_stack([ellipse_x, ellipse_y])
                
#                 # Calculate ellipse aspect ratio
#                 ellipse_ar = max(a, b) / min(a, b)
                
#                 # Store ellipse data
#                 ellipse_data = {
#                     'center': [float(xc), float(yc)],
#                     'axes_lengths': [float(a), float(b)],
#                     'angle': float(theta),
#                     'aspect_ratio': float(ellipse_ar),
#                     'points': ellipse_points.tolist()
#                 }
#         except Exception as e:
#             pass
            
#         # Store ellipse data
#         results['cross_sections']['ellipse_fits'].append(ellipse_data)
    
#     return results

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
                sections_points_list.append(section.vertices)

    return sections_points_list

def visualize_mesh_with_cross_sections(mesh=None, outer_points = None, centreline = None, n_sections=20, colormap='viridis'):
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
    end_margin_points = 3  # Avoid unstable sections at the very ends
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

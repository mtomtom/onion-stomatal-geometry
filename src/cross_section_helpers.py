################################################### Imports ###################################################
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, cKDTree
from scipy.interpolate import CubicSpline
import trimesh
import plotly.graph_objects as go

################################################### Plotting and visualisation ###################################################

def plot_cross_sections_grid_overlay(sections_points_list1, sections_points_list2, n_cols=5, figsize=(15, 10), filename=None, colors=('k-', 'r-'), align_to_x=True, linewidth=2, ylim=5, mesh1 = "Mesh 1", mesh2 = "Mesh 2"):
    """Plot paired cross sections in a grid of 2D subplots with overlay.
    
    Projects both sections to the best-fit 2D plane using PCA and displays them
    overlaid in a grid layout. Uses a reference PCA basis for consistent alignment.
    
    Parameters
    ----------
    sections_points_list1 : list of ndarray
        First set of cross sections, each an (N, 3) array of 3D points.
    sections_points_list2 : list of ndarray
        Second set of cross sections to overlay.
    n_cols : int, optional
        Number of columns in the grid (default: 5).
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (15, 10)).
    filename : str, optional
        If provided, save the figure to this path.
    colors : tuple of str, optional
        Line colors for (mesh1, mesh2) (default: ('k-', 'r-')).
    align_to_x : bool, optional
        If True, rotate sections to align major axis with x-axis (default: True).
    linewidth : float, optional
        Line width for plotting (default: 2).
    ylim : float, optional
        Y-axis limits as ±ylim (default: 5).
    mesh1 : str, optional
        Label for first mesh in legend (default: "Mesh 1").
    mesh2 : str, optional
        Label for second mesh in legend (default: "Mesh 2").
    """

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
            ax.plot(s1_plot[:, 0], s1_plot[:, 1], colors[0], label=mesh1, linewidth=linewidth)
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
                ax.plot(s2_plot[:, 0], s2_plot[:, 1], colors[1], label=mesh2, linewidth=linewidth)
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

        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(-ylim, ylim)
        #ax.axis('equal')
        plt.legend(loc='upper right')
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

def get_midsection_area(mesh):

    wall_vertices = find_wall_vertices_vertex_normals(mesh, dot_thresh=0.2)
    centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace = get_top_bottom_wall_centres(mesh, wall_vertices)
    midpoint, traces, section_points, local_axes = get_midpoint_cross_section_from_centres(mesh, centre_top, centre_bottom)
    left_section, right_section, left_section_centre, right_section_centre, left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace = get_left_right_midsections(section_points, midpoint, local_axes)

    ## The radius of the circles to place at the top and bottom walls is taken from the radius of the midsections
    area1 = cross_section_area_2d(left_section)
    area2 = cross_section_area_2d(right_section)

    return area1, area2

def visualize_mesh(mesh, extra_details=None, title="Mesh Visualization", opacity = 0.65):
    """Visualize a 3D mesh using Plotly.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to visualize.
    extra_details : list or plotly trace, optional
        Additional Plotly traces to include in the visualization.
    title : str, optional
        Figure title (default: "Mesh Visualization").
    opacity : float, optional
        Mesh opacity between 0 and 1 (default: 0.65).
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D figure.
    """
    traces = [
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color="#0072B2",
            opacity=opacity,
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

################################################### Helper functions ###################################################

def curve_length(x, y, z):
    """Calculate total length of a 3D curve.
    
    Parameters
    ----------
    x : array_like
        X-coordinates of curve points.
    y : array_like
        Y-coordinates of curve points.
    z : array_like
        Z-coordinates of curve points.
    
    Returns
    -------
    float
        Total arc length of the curve.
    """
    # Stack coordinates into (N, 3) array
    points = np.column_stack((x, y, z))
    # Compute distances between consecutive points
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    # Sum to get total length
    return segment_lengths.sum()

def get_circle_trace(circle, name="Circle", colour="red"):
    """Create a Plotly trace for a 3D circle.
    
    Parameters
    ----------
    circle : ndarray
        Circle points as (N, 3) array.
    name : str, optional
        Name for the trace in the legend (default: "Circle").
    colour : str, optional
        Line color (default: "red").
    
    Returns
    -------
    plotly.graph_objects.Scatter3d
        Plotly trace for the circle.
    """
    circle_trace = go.Scatter3d(
        x=circle[:, 0],
        y=circle[:, 1],
        z=circle[:, 2],
        mode='lines',
        line=dict(color=colour, width=2),
        name=name
    )
    return circle_trace

def order_points_consistently(points, normal, midpoint):
    """Order cross-section points consistently by angle around centroid.
    
    Projects points onto a 2D plane perpendicular to the normal vector and
    sorts them by angular position around the midpoint.
    
    Parameters
    ----------
    points : ndarray
        Points to order as (N, 3) array.
    normal : ndarray
        Normal vector defining the slicing plane.
    midpoint : ndarray
        Center point for angular ordering.
    
    Returns
    -------
    ndarray
        Reordered points array.
    """
    ref = np.array([0,0,1]) if abs(np.dot(normal,[0,0,1])) < 0.9 else np.array([1,0,0])
    v1 = np.cross(normal, ref); v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    R = points - midpoint
    coords2d = np.column_stack([R @ v1, R @ v2])
    angles = np.arctan2(coords2d[:,1], coords2d[:,0])
    order = np.argsort(angles)
    return points[order]

def get_barycentric_coords(point, face_vertices):
    """Calculate barycentric coordinates of a point within a triangle.
    
    Parameters
    ----------
    point : ndarray
        3D point coordinates.
    face_vertices : ndarray
        Triangle vertices as (3, 3) array.
    
    Returns
    -------
    ndarray
        Barycentric coordinates (u, v, w) where u + v + w = 1.
    """
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


def cross_section_area_2d(points):
    """Compute 2D area of cross-section points using convex hull.
    
    Parameters
    ----------
    points : ndarray
        Cross-section points as (N, 3) array.
    
    Returns
    -------
    float
        Area of the convex hull in 2D projection.
    """
    pca = PCA(n_components=2)
    pts2 = pca.fit_transform(points)
    hull = ConvexHull(pts2)
    return hull.volume

################################################### Analyze mesh functions ###################################################

def find_wall_vertices_vertex_normals(mesh: trimesh.Trimesh, dot_thresh=0.2):
    """Identify wall vertices based on opposing face normals.
    
    A vertex is classified as wall if its incident face normals contain at least
    one pair with strong opposition (dot product below threshold).
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to analyze.
    dot_thresh : float, optional
        Threshold for detecting opposing normals (default: 0.2).
        Vertices with normals having dot < -(1 - dot_thresh) are marked as wall.
    
    Returns
    -------
    ndarray
        Integer array of wall vertex indices.
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

def find_wall_vertices_axis_extrema(mesh: trimesh.Trimesh, axis=1, quantile=0.08, max_quantile=0.30, min_vertices=20):
    """Identify candidate wall vertices from coordinate extrema (no normals).

    This fallback is useful when mesh generation changes smooth local normals
    enough that opposition-based wall detection returns no vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to analyze.
    axis : int, optional
        Coordinate axis used for top-bottom separation (default: 1 for y-axis).
    quantile : float, optional
        Initial tail quantile for selecting extreme vertices (default: 0.08).
    max_quantile : float, optional
        Maximum tail quantile used during adaptive widening (default: 0.30).
    min_vertices : int, optional
        Minimum number of vertices required to consider fallback valid.

    Returns
    -------
    ndarray
        Integer array of candidate wall vertex indices.
    """
    verts = mesh.vertices
    if verts is None or len(verts) == 0:
        return np.array([], dtype=int)

    coords = verts[:, axis]
    q = float(quantile)
    while q <= float(max_quantile):
        low_cut = np.quantile(coords, q)
        high_cut = np.quantile(coords, 1.0 - q)
        idx = np.where((coords <= low_cut) | (coords >= high_cut))[0]
        if idx.size >= int(min_vertices):
            return idx.astype(int)
        q += 0.04

    return np.array([], dtype=int)


def find_wall_vertices_interface_band(
    mesh: trimesh.Trimesh,
    distance_quantile=0.08,
    opposite_quantile=0.20,
    min_vertices=20,
    interface_axis=None,
):
    """Identify wall vertices near the interface plane between two guard-cell lobes.

    Vertices are split into two coarse lobes with KMeans (k=2). The interface
    is approximated as the mid-plane orthogonal to the lobe-centroid axis.
    Vertices with smallest distance to this plane are treated as wall
    candidates.
    """
    verts = mesh.vertices
    if verts is None or len(verts) < 10:
        return np.array([], dtype=int)

    if interface_axis is not None:
        axis_idx = int(interface_axis)
        axis_vec = np.zeros(3, dtype=float)
        axis_vec[axis_idx] = 1.0
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
            verts[:, axis_idx].reshape(-1, 1)
        )
        c0 = verts[labels == 0].mean(axis=0)
        c1 = verts[labels == 1].mean(axis=0)
    else:
        # Dynamically evaluate PCA components using relaxed wall-vertex normals
        # to robustly distinguish left/right guard cells based on the inner pore wall.
        pca = PCA(n_components=2)
        pca.fit(verts - verts.mean(axis=0))
        
        # 1. Find SOME wall vertices by gently relaxing the threshold
        # The true inner wall always lies firmly on the left/right dividing line
        wall_cand = []
        for t in [0.2, 0.35, 0.5, 0.65]:
            wv = find_wall_vertices_vertex_normals(mesh, dot_thresh=t)
            if len(wv) >= 10:
                wall_cand = wv
                break
                
        best_labels = None
        best_i = 1 # default to standard width
        
        if len(wall_cand) >= 10:
            # 2. Project the wall vertices onto PC0 and PC1.
            # The Left/Right split axis (Width) is the one where the wall has ~0 variance!
            vars = []
            for i in range(2):
                proj = (verts[wall_cand] - verts.mean(axis=0)) @ pca.components_[i]
                vars.append(np.var(proj))
            best_i = np.argmin(vars)
            
            axis = pca.components_[best_i]
            proj = (verts - verts.mean(axis=0)) @ axis
            best_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(proj.reshape(-1, 1))
        else:
            # Fallback to pure geometry if no wall found even after heavy relaxation
            # best_score = float('inf')
            best_i = 1
            axis = pca.components_[best_i]
            proj = (verts - verts.mean(axis=0)) @ axis
            best_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(proj.reshape(-1, 1))
            
            """
            for i in range(2):
                axis = pca.components_[i]
                proj = (verts - verts.mean(axis=0)) @ axis
                lbls = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(proj.reshape(-1, 1))
                c0_test = verts[lbls == 0].mean(axis=0)
                c1_test = verts[lbls == 1].mean(axis=0)
                if len(verts[lbls == 0]) > 0 and len(verts[lbls == 1]) > 0:
                    d0 = np.min(np.linalg.norm(verts[lbls == 0] - c0_test, axis=1))
                    d1 = np.min(np.linalg.norm(verts[lbls == 1] - c1_test, axis=1))
                    score = d0 + d1
                    if score < best_score:
                        best_score = score
                        best_i = i
                        best_labels = lbls
                    
        """
        labels = best_labels
        c0 = verts[labels == 0].mean(axis=0)
        c1 = verts[labels == 1].mean(axis=0)
        axis_vec = c1 - c0

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    if idx0.size < 5 or idx1.size < 5:
        return np.array([], dtype=int)

    norm = np.linalg.norm(axis_vec)
    if norm < 1e-10:
        return np.array([], dtype=int)
    axis_vec = axis_vec / norm

    # Calculate midpoint purely from geometric bounds to guarantee symmetry
    midpoint = 0.5 * (verts.max(axis=0) + verts.min(axis=0))
    signed = (verts - midpoint) @ axis_vec
    dist_to_interface = np.abs(signed)

    plane_thresh = np.quantile(dist_to_interface, float(distance_quantile))
    plane_idx = np.where(dist_to_interface <= plane_thresh)[0]

    pts0 = verts[idx0]
    pts1 = verts[idx1]
    d0, _ = cKDTree(pts1).query(pts0, k=1)
    d1, _ = cKDTree(pts0).query(pts1, k=1)
    t0 = np.quantile(d0, float(opposite_quantile))
    t1 = np.quantile(d1, float(opposite_quantile))
    close_idx = np.concatenate([idx0[d0 <= t0], idx1[d1 <= t1]])

    idx = np.intersect1d(plane_idx, close_idx).astype(int)
    if idx.size < int(min_vertices):
        idx = plane_idx.astype(int)

    if idx.size < int(min_vertices):
        return np.array([], dtype=int)
    return idx


def _get_interface_plane_geometry(mesh: trimesh.Trimesh, interface_axis=None):
    """Estimate interface plane between two coarse guard-cell lobes.

    Returns
    -------
    tuple
        (midpoint, axis_vec, dist_to_interface)
        where dist_to_interface is abs signed distance of each vertex to plane.
    """
    verts = mesh.vertices
    if verts is None or len(verts) < 10:
        return None, None, None

    if interface_axis is not None:
        axis_idx = int(interface_axis)
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
            verts[:, axis_idx].reshape(-1, 1)
        )
        c0 = verts[labels == 0].mean(axis=0)
        c1 = verts[labels == 1].mean(axis=0)
        axis_vec = np.zeros(3, dtype=float)
        axis_vec[axis_idx] = 1.0
    else:
        # Dynamically evaluate PCA components using relaxed wall-vertex normals
        # to robustly distinguish left/right guard cells based on the inner pore wall.
        pca = PCA(n_components=2)
        pca.fit(verts - verts.mean(axis=0))
        
        # 1. Find SOME wall vertices by gently relaxing the threshold
        # The true inner wall always lies firmly on the left/right dividing line
        wall_cand = []
        for t in [0.2, 0.35, 0.5, 0.65]:
            wv = find_wall_vertices_vertex_normals(mesh, dot_thresh=t)
            if len(wv) >= 10:
                wall_cand = wv
                break
                
        best_labels = None
        best_i = 1 # default to standard width
        
        if len(wall_cand) >= 10:
            # 2. Project the wall vertices onto PC0 and PC1.
            # The Left/Right split axis (Width) is the one where the wall has ~0 variance!
            vars = []
            for i in range(2):
                proj = (verts[wall_cand] - verts.mean(axis=0)) @ pca.components_[i]
                vars.append(np.var(proj))
            best_i = np.argmin(vars)
            
            axis = pca.components_[best_i]
            proj = (verts - verts.mean(axis=0)) @ axis
            best_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(proj.reshape(-1, 1))
        else:
            # Fallback to pure geometry if no wall found even after heavy relaxation
            # best_score = float('inf')
            best_i = 1
            axis = pca.components_[best_i]
            proj = (verts - verts.mean(axis=0)) @ axis
            best_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(proj.reshape(-1, 1))
            
            """
            for i in range(2):
                axis = pca.components_[i]
                proj = (verts - verts.mean(axis=0)) @ axis
                lbls = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(proj.reshape(-1, 1))
                c0_test = verts[lbls == 0].mean(axis=0)
                c1_test = verts[lbls == 1].mean(axis=0)
                if len(verts[lbls == 0]) > 0 and len(verts[lbls == 1]) > 0:
                    d0 = np.min(np.linalg.norm(verts[lbls == 0] - c0_test, axis=1))
                    d1 = np.min(np.linalg.norm(verts[lbls == 1] - c1_test, axis=1))
                    score = d0 + d1
                    if score < best_score:
                        best_score = score
                        best_i = i
                        best_labels = lbls
                    
        """
        labels = best_labels
        c0 = verts[labels == 0].mean(axis=0)
        c1 = verts[labels == 1].mean(axis=0)
        axis_vec = c1 - c0

    norm = np.linalg.norm(axis_vec)
    if norm < 1e-10:
        return None, None, None
    axis_vec = axis_vec / norm
    midpoint = 0.5 * (c0 + c1)
    signed = (verts - midpoint) @ axis_vec
    dist_to_interface = np.abs(signed)
    return midpoint, axis_vec, dist_to_interface


def _wall_vertices_match_interface(
    mesh: trimesh.Trimesh,
    wall_vertices,
    interface_axis=None,
    near_quantile=0.20,
    min_near_fraction=0.50,
):
    """Check whether candidate wall vertices lie near the inter-cell interface.

    This guards against normals-based candidates landing on external poles.
    """
    wall_vertices = np.asarray(wall_vertices, dtype=int)
    if wall_vertices.size == 0:
        return False

    _mid, _axis, dist_to_interface = _get_interface_plane_geometry(
        mesh, interface_axis=interface_axis
    )
    if dist_to_interface is None:
        return False

    near_thresh = np.quantile(dist_to_interface, float(near_quantile))
    wall_dists = dist_to_interface[wall_vertices]
    near_fraction = float(np.mean(wall_dists <= near_thresh)) if wall_dists.size else 0.0
    return near_fraction >= float(min_near_fraction)



def visualize_detected_midsection(
    mesh_entry,
    dot_thresh=0.2,
    min_normals_wall_vertices=20,
    wall_split_axis="auto",
    wall_interface_axis=None,
):
    """Visualize where the pipeline detects the midsection on a mesh.

    Parameters
    ----------
    mesh_entry : str, Path, or trimesh.Trimesh
        Mesh path or loaded mesh.
    dot_thresh : float, optional
        Threshold for normals-based wall detection.
    min_normals_wall_vertices : int, optional
        Minimum wall-vertex count needed to trust normals before fallback.

    Returns
    -------
    dict
        Diagnostic summary including method, counts, centers, and dimensions.
    """
    mesh, label = _coerce_mesh_entry(mesh_entry)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("visualize_detected_midsection requires a trimesh.Trimesh input.")

    mesh_name = label or "in_memory_mesh"

    wall_vertices = find_wall_vertices_vertex_normals(mesh, dot_thresh=dot_thresh)
    wall_method = "normals"
    normals_ok = (
        wall_vertices.size >= int(min_normals_wall_vertices)
        and _wall_vertices_match_interface(
            mesh,
            wall_vertices,
            interface_axis=wall_interface_axis,
        )
    )
    if not normals_ok:
        wall_vertices = find_wall_vertices_interface_band(mesh, interface_axis=wall_interface_axis)
        wall_method = "interface_band"

    if wall_vertices.size == 0:
        wall_vertices = find_wall_vertices_axis_extrema(mesh)
        wall_method = "axis_extrema"

    if wall_vertices.size == 0:
        raise ValueError("Could not identify wall vertices for midsection visualization.")

    centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, *_ = get_top_bottom_wall_centres(
        mesh,
        wall_vertices,
        split_axis=wall_split_axis,
    )
    wall_axis_override = None
    if wall_split_axis not in (None, "auto"):
        wall_axis_override = np.zeros(3, dtype=float)
        wall_axis_override[int(wall_split_axis)] = 1.0
    midpoint, _traces, section_points, local_axes = get_midpoint_cross_section_from_centres(
        mesh, centre_top, centre_bottom, wall_axis=wall_axis_override
    )
    if section_points is None or len(section_points) < 3:
        raise ValueError("Midpoint cross section returned insufficient points.")

    left_section, right_section, left_centre, right_centre, *_ = get_left_right_midsections(
        section_points, midpoint, local_axes
    )

    lw, lh = measure_cross_section_width_height(left_section)
    rw, rh = measure_cross_section_width_height(right_section)

    traces = [
        go.Scatter3d(
            x=top_wall_coords[:, 0], y=top_wall_coords[:, 1], z=top_wall_coords[:, 2],
            mode='markers', marker=dict(size=2.5, color='#D55E00'), name='Top wall vertices'
        ),
        go.Scatter3d(
            x=bottom_wall_coords[:, 0], y=bottom_wall_coords[:, 1], z=bottom_wall_coords[:, 2],
            mode='markers', marker=dict(size=2.5, color='#56B4E9'), name='Bottom wall vertices'
        ),
        go.Scatter3d(
            x=[centre_top[0]], y=[centre_top[1]], z=[centre_top[2]],
            mode='markers', marker=dict(size=8, color='#D55E00'), name='Top wall centre'
        ),
        go.Scatter3d(
            x=[centre_bottom[0]], y=[centre_bottom[1]], z=[centre_bottom[2]],
            mode='markers', marker=dict(size=8, color='#56B4E9'), name='Bottom wall centre'
        ),
        go.Scatter3d(
            x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]],
            mode='markers', marker=dict(size=9, color='black'), name='Midpoint'
        ),
        go.Scatter3d(
            x=left_section[:, 0], y=left_section[:, 1], z=left_section[:, 2],
            mode='markers', marker=dict(size=3, color='#E69F00'), name='Left midsection points'
        ),
        go.Scatter3d(
            x=right_section[:, 0], y=right_section[:, 1], z=right_section[:, 2],
            mode='markers', marker=dict(size=3, color='#0072B2'), name='Right midsection points'
        ),
        go.Scatter3d(
            x=[left_centre[0]], y=[left_centre[1]], z=[left_centre[2]],
            mode='markers', marker=dict(size=7, color='#E69F00', symbol='diamond'), name='Left section centre'
        ),
        go.Scatter3d(
            x=[right_centre[0]], y=[right_centre[1]], z=[right_centre[2]],
            mode='markers', marker=dict(size=7, color='#0072B2', symbol='diamond'), name='Right section centre'
        ),
    ]

    title = (
        f"Detected midsection: {Path(mesh_name).name} "
        f"(method={wall_method}, n_wall={len(wall_vertices)})"
    )
    visualize_mesh(mesh, extra_details=traces, title=title, opacity=0.20)

    return {
        "mesh": mesh_name,
        "wall_method": wall_method,
        "n_wall_vertices": int(len(wall_vertices)),
        "n_midsection_points": int(len(section_points)),
        "n_left_points": int(len(left_section)),
        "n_right_points": int(len(right_section)),
        "centre_top": centre_top,
        "centre_bottom": centre_bottom,
        "midpoint": midpoint,
        "left_width": float(lw),
        "left_height": float(lh),
        "right_width": float(rw),
        "right_height": float(rh),
    }

def get_top_bottom_wall_centres(mesh, wall_vertices, split_axis="auto"):
    """Split wall vertices into two internal-junction endpoints and compute centers.

    For current stomata workflows, `wall_vertices` are expected to represent the
    inter-guard-cell wall band. This function separates that band into two
    endpoint groups (historically named top/bottom) used to define the axis for
    midpoint sectioning.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    wall_vertices : ndarray
        Indices of wall vertices.
    
    Returns
    -------
    tuple
        (centre_top, centre_bottom, top_wall_coords, bottom_wall_coords,
         top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace)
        Centers and coordinates of top/bottom walls plus Plotly visualization traces.
    """
    verts = mesh.vertices
    wall_coords = verts[wall_vertices]
    # Split the wall band into two endpoint groups.
    # Default uses deterministic axis split (legacy behavior is y-axis).
    if len(wall_coords) >= 2:
        if split_axis not in (None, "auto"):
            axis_idx = int(split_axis)
            projections = wall_coords[:, axis_idx]
            split_value = np.median(projections)
            top_mask = projections >= split_value
            bottom_mask = ~top_mask

            # Degenerate fallback if all points land on one side of the median.
            if top_mask.sum() == 0 or bottom_mask.sum() == 0:
                q_low, q_high = np.quantile(projections, [0.45, 0.55])
                top_mask = projections >= q_high
                bottom_mask = projections <= q_low

            # Final fallback for extremely degenerate distributions.
            if top_mask.sum() == 0 or bottom_mask.sum() == 0:
                labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
                    projections.reshape(-1, 1)
                )
                group1 = wall_coords[labels == 0]
                group2 = wall_coords[labels == 1]
                if projections[labels == 0].mean() >= projections[labels == 1].mean():
                    top_wall_coords = group1
                    bottom_wall_coords = group2
                else:
                    top_wall_coords = group2
                    bottom_wall_coords = group1
            else:
                top_wall_coords = wall_coords[top_mask]
                bottom_wall_coords = wall_coords[bottom_mask]
        else:
            mesh_centered = mesh.vertices - mesh.vertices.mean(axis=0)
            cov_mesh = np.cov(mesh_centered.T)
            evals_mesh, evecs_mesh = np.linalg.eigh(cov_mesh)
            long_axis = evecs_mesh[:, int(np.argmax(evals_mesh))]

            _mid, interface_axis, _dist = _get_interface_plane_geometry(mesh, interface_axis=None)
            if interface_axis is not None:
                endpoint_axis = long_axis - np.dot(long_axis, interface_axis) * interface_axis
            else:
                endpoint_axis = long_axis

            axis_norm = np.linalg.norm(endpoint_axis)
            if axis_norm < 1e-10:
                wc_center = wall_coords - wall_coords.mean(axis=0)
                cov = np.cov(wc_center.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                endpoint_axis = eigvecs[:, int(np.argmax(eigvals))]
            else:
                endpoint_axis = endpoint_axis / axis_norm

            projections = (wall_coords - wall_coords.mean(axis=0)) @ endpoint_axis
            split_value = np.median(projections)
            top_mask = projections >= split_value
            bottom_mask = ~top_mask

            if top_mask.sum() == 0 or bottom_mask.sum() == 0:
                labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
                    projections.reshape(-1, 1)
                )
                group1 = wall_coords[labels == 0]
                group2 = wall_coords[labels == 1]
                if projections[labels == 0].mean() >= projections[labels == 1].mean():
                    top_wall_coords = group1
                    bottom_wall_coords = group2
                else:
                    top_wall_coords = group2
                    bottom_wall_coords = group1
            else:
                top_wall_coords = wall_coords[top_mask]
                bottom_wall_coords = wall_coords[bottom_mask]
    else:
        # Fallback: use all as top, none as bottom
        top_wall_coords = wall_coords
        bottom_wall_coords = np.empty((0, 3))

    # Create the wall traces (endpoint groups on the internal wall band)
    top_wall_trace = go.Scatter3d(
        x=top_wall_coords[:, 0],
        y=top_wall_coords[:, 1],
        z=top_wall_coords[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Internal Wall Endpoint A'
    )
    bottom_wall_trace = go.Scatter3d(
        x=bottom_wall_coords[:, 0],
        y=bottom_wall_coords[:, 1],
        z=bottom_wall_coords[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Internal Wall Endpoint B'
    )

    centre_top = top_wall_coords.mean(axis=0)
    centre_bottom = bottom_wall_coords.mean(axis=0)

    centre_top_trace = go.Scatter3d(
        x=[centre_top[0]],
        y=[centre_top[1]],
        z=[centre_top[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Internal Wall Centre A'
    )
    centre_bottom_trace = go.Scatter3d(
        x=[centre_bottom[0]],
        y=[centre_bottom[1]],
        z=[centre_bottom[2]],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Internal Wall Centre B'
    )

    return centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace

def get_midpoint_cross_section_from_centres(mesh, centre_top, centre_bottom, wall_axis=None):
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

    # Define wall axis (from bottom to top), unless explicitly provided.
    if wall_axis is not None:
        wall_vec = np.asarray(wall_axis, dtype=float)
        wall_norm = np.linalg.norm(wall_vec)
        if wall_norm < 1e-10:
            raise ValueError("wall_axis has near-zero magnitude.")
        wall_vec = wall_vec / wall_norm
    else:
        wall_vec = centre_top - centre_bottom
        wall_norm = np.linalg.norm(wall_vec)
        if wall_norm < 1e-10:
            raise ValueError("Top/bottom centres are too close to define a wall axis.")
        wall_vec = wall_vec / wall_norm

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

def get_left_right_midsections(section_points, midpoint, local_axes):
    """Split cross-section points into left and right guard cells.
    
    Projects section points onto the left-right axis (second axis in local_axes)
    and separates them based on their position relative to the midpoint.
    
    Parameters
    ----------
    section_points : ndarray
        Cross-section points as (N, 3) array.
    midpoint : ndarray
        Central point for splitting.
    local_axes : ndarray
        3x3 array of local coordinate axes [wall_vec, left_right_vec, normal_vec].
    
    Returns
    -------
    tuple
        (left_section, right_section, left_section_centre, right_section_centre,
         left_midsection_trace, right_midsection_trace, 
         left_section_centre_trace, right_section_centre_trace)
        Section points, centers, and Plotly visualization traces for both sides.
    """

    # The left-right axis is the second vector in local_axes (from get_midpoint_cross_section_from_centres)
    left_right_vec = local_axes[1]

    # Split section points using straight projection onto the left-right axis.
    left_section = np.empty((0, 3))
    right_section = np.empty((0, 3))

    if section_points is not None and len(section_points) >= 3:
        projs = (section_points - midpoint) @ left_right_vec
        left_section = section_points[projs <= 0]
        right_section = section_points[projs > 0]
        relative_points = section_points - midpoint
        side_values = np.dot(relative_points, left_right_vec)
        left_section = section_points[side_values < 0]
        right_section = section_points[side_values >= 0]

    # Robust centre estimate in local PCA box coordinates to avoid density bias.
    def _robust_section_centre(points):
        if points is None or len(points) == 0:
            return np.asarray(midpoint, dtype=float)
        if len(points) < 3:
            return points.mean(axis=0)
        pca_local = PCA(n_components=2)
        pts2 = pca_local.fit_transform(points)
        centre2 = np.array([
            0.5 * (pts2[:, 0].min() + pts2[:, 0].max()),
            0.5 * (pts2[:, 1].min() + pts2[:, 1].max()),
        ])
        return pca_local.mean_ + centre2[0] * pca_local.components_[0] + centre2[1] * pca_local.components_[1]

    left_section_centre = _robust_section_centre(left_section)
    right_section_centre = _robust_section_centre(right_section)

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
    """Create a 3D circle fitted to a point cloud.
    
    Fits a plane to the input coordinates using PCA and creates a circle
    of specified radius in that plane.
    
    Parameters
    ----------
    coords : ndarray
        Points defining the plane as (N, 3) array.
    radius : float
        Radius of the circle to create.
    
    Returns
    -------
    ndarray
        Circle points as (100, 3) array.
    """
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
    """Estimate stomata centreline using cubic spline interpolation.
    
    Creates a closed cubic spline passing through the four key points:
    top, left, bottom, and right centers.
    
    Parameters
    ----------
    top_circle_centre : ndarray
        Center of top wall circle.
    bottom_circle_centre : ndarray
        Center of bottom wall circle.
    left_section_centre : ndarray
        Center of left midsection.
    right_section_centre : ndarray
        Center of right midsection.
    
    Returns
    -------
    tuple
        (spline_trace, spline_x, spline_y, spline_z)
        Plotly trace and coordinate arrays for the centreline.
    """
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

def split_mesh_at_wall_vertices(mesh, wall_vertices, left_centroid, right_centroid):
    """Split stomata mesh into left and right guard cells.
    
    Assigns vertices to guard cells based on proximity to left or right centroids,
    with wall vertices included in both meshes.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input stomata mesh.
    wall_vertices : ndarray
        Indices of wall vertices (M,).
    left_centroid : ndarray
        Approximate center of left guard cell (3,).
    right_centroid : ndarray
        Approximate center of right guard cell (3,).
    
    Returns
    -------
    tuple of (trimesh.Trimesh, trimesh.Trimesh)
        Left and right guard cell meshes.
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

def get_centreline_estimate_and_split(top_circle_centre, bottom_circle_centre,
                                      left_section_centre, right_section_centre,
                                      n_points=200):
    """Build closed spline centreline and split into left and right halves.
    
    Creates a periodic cubic spline through the four key centers and divides
    it into separate centrelines for left and right guard cells.
    
    Parameters
    ----------
    top_circle_centre : ndarray
        Center of top wall circle.
    bottom_circle_centre : ndarray
        Center of bottom wall circle.
    left_section_centre : ndarray
        Center of left midsection.
    right_section_centre : ndarray
        Center of right midsection.
    n_points : int, optional
        Number of points for spline interpolation (default: 200).
    
    Returns
    -------
    tuple
        (full_trace, left_trace, right_trace, left_half, right_half)
        Plotly traces and coordinate arrays for full and split centrelines.
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

def get_regularly_spaced_cross_sections_batch(mesh, smoothed, centre_top, centre_bottom, num_sections=30):
    """Extract regularly spaced cross sections along a centreline.
    
    Optimized version that samples cross sections at regular arc-length intervals,
    avoiding regions near top and bottom walls. Uses batched face search for
    efficient barycentric coordinate calculation.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to section.
    smoothed : ndarray
        Smoothed centreline as (N, 3) array.
    centre_top : ndarray
        Top wall center coordinates.
    centre_bottom : ndarray
        Bottom wall center coordinates.
    num_sections : int, optional
        Number of cross sections to extract (default: 30).
    
    Returns
    -------
    tuple
        (section_points_list, section_traces, section_bary_data)
        Lists of section points, Plotly traces, and barycentric data.
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

################################################## Main Mesh Analysis Function ###################################################

def analyze_stomata_mesh(mesh_path, num_sections=20, n_points=40, visualize=False, mid_area_left_0 = None, mid_area_right_0 = None):
    """Comprehensive analysis of stomata mesh geometry.
    
    Performs complete geometric analysis including wall detection, guard cell
    separation, centreline estimation, and cross-section extraction.
    
    Parameters
    ----------
    mesh_path : str or Path
        Path to mesh file.
    num_sections : int, optional
        Number of cross sections to extract per guard cell (default: 20).
    n_points : int, optional
        Number of points for centreline interpolation (default: 40).
    visualize : bool, optional
        If True, display interactive 3D visualization (default: False).
    mid_area_left_0 : float, optional
        Reference mid-area for left guard cell (used for circle radius).
    mid_area_right_0 : float, optional
        Reference mid-area for right guard cell (used for circle radius).
    
    Returns
    -------
    tuple
        (section_points_right, section_points_left, 
         section_traces_left, section_traces_right, 
         [spline_x, spline_y, spline_z])
        Cross-section points, visualization traces, and centreline coordinates.
    """
    mesh = trimesh.load(mesh_path, process=False)
    ## Get the wall vertices
    wall_vertices = find_wall_vertices_vertex_normals(mesh, dot_thresh=0.2)


    centre_top, centre_bottom, top_wall_coords, bottom_wall_coords, top_wall_trace, bottom_wall_trace, centre_top_trace, centre_bottom_trace = get_top_bottom_wall_centres(mesh, wall_vertices)

    midpoint, traces, section_points, local_axes = get_midpoint_cross_section_from_centres(mesh, centre_top, centre_bottom)

    ## Label the left and right cross sections

    left_section, right_section, left_section_centre, right_section_centre, left_midsection_trace, right_midsection_trace, left_section_centre_trace, right_section_centre_trace = get_left_right_midsections(section_points, midpoint, local_axes)

    ## The radius of the circles to place at the top and bottom walls is taken from the radius of the midsections of the uninflated stomata ( mid_area_left_0, mid_area_right_0)
    area1 = mid_area_left_0
    area2 = mid_area_right_0
    try:
        avg_area = 0.5 * (area1 + area2)
    except Exception as e:
        print(f"Area cannot be none: {e}") 

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

################################################## Additional cross section functions ###############################################

def get_cross_section_points(mesh, centreline, indices):
    """Extract cross-section points at specified centreline positions.
    
    For each index, computes a cross section perpendicular to the centreline
    tangent. When multiple segments exist, returns only the nearest segment.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to section.
    centreline : ndarray
        Centreline points as (N, 3) array.
    indices : array_like
        Indices along centreline where cross sections are desired.
    
    Returns
    -------
    list of ndarray
        Cross-section points for each index.
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

def calculate_cross_section_aspect_ratios_and_lengths(sections_points_list):
    """Compute aspect ratios and axis lengths for cross sections.
    
    For each cross section, calculates the aspect ratio (major/minor) along with
    the major and minor axis lengths using PCA-based alignment.
    
    Parameters
    ----------
    sections_points_list : list of ndarray or ndarray
        Either a list where each element is an (N_i, 3) array of points,
        or a single (N, 3) array.
    
    Returns
    -------
    tuple of (list, list, list)
        (aspect_ratios, major_lengths, minor_lengths) for each section.
        Returns 0.0 for invalid or degenerate sections.
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


################################################## Midsection measurement utilities ###################################################

def measure_cross_section_width_height(section_points):
    """Measure width and height of a single cross section.
    
    Parameters
    ----------
    section_points : ndarray
        Cross-section points as (N, 3) array.
    
    Returns
    -------
    tuple of (float, float)
        (width, height) corresponding to major and minor axis lengths.
        Returns (0.0, 0.0) for invalid sections.
    """
    if section_points is None or len(section_points) < 3:
        return 0.0, 0.0

    _, major_lengths, minor_lengths = calculate_cross_section_aspect_ratios_and_lengths(section_points)
    width = float(major_lengths[0]) if major_lengths else 0.0
    height = float(minor_lengths[0]) if minor_lengths else 0.0
    return width, height


def _coerce_mesh_entry(mesh_entry):
    """Internal helper to convert mesh input to standardized format.
    
    Parameters
    ----------
    mesh_entry : str, Path, or trimesh.Trimesh
        Mesh to process.
    
    Returns
    -------
    tuple of (trimesh.Trimesh, str or None)
        Loaded mesh and its label/path.
    
    Raises
    ------
    FileNotFoundError
        If mesh path does not exist.
    """
    if isinstance(mesh_entry, trimesh.Trimesh):
        label = None
        metadata = getattr(mesh_entry, "metadata", None)
        if isinstance(metadata, dict):
            label = metadata.get("file_name") or metadata.get("name")
        return mesh_entry, label

    mesh_path = Path(mesh_entry)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh path does not exist: {mesh_entry}")
    mesh = trimesh.load(mesh_path, process=False)
    return mesh, str(mesh_path)


def _extract_midsections(
    mesh,
    dot_thresh=0.2,
    return_wall_method=False,
    min_normals_wall_vertices=20,
    wall_split_axis="auto",
    wall_interface_axis=None,
):
    """Internal helper to extract left and right midsections from a mesh.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input stomata mesh.
    dot_thresh : float, optional
        Wall vertex detection threshold (default: 0.2).
    min_normals_wall_vertices : int, optional
        Minimum number of wall vertices required to trust the normals-based
        method before falling back to the geometry-based method (default: 20).
    
    Returns
    -------
    tuple
        By default returns (left_section, right_section).
        If return_wall_method=True, returns
        (left_section, right_section, wall_method).
    
    Raises
    ------
    ValueError
        If wall vertices cannot be identified or midsection has insufficient points.
    """
    wall_vertices = find_wall_vertices_vertex_normals(mesh, dot_thresh=dot_thresh)
    wall_method = "normals"
    normals_ok = (
        wall_vertices.size >= int(min_normals_wall_vertices)
        and _wall_vertices_match_interface(
            mesh,
            wall_vertices,
            interface_axis=wall_interface_axis,
        )
    )
    if not normals_ok:
        wall_vertices = find_wall_vertices_interface_band(mesh, interface_axis=wall_interface_axis)
        wall_method = "interface_band"

    if wall_vertices.size == 0:
        wall_vertices = find_wall_vertices_axis_extrema(mesh)
        wall_method = "axis_extrema"

    if wall_vertices.size == 0:
        raise ValueError("Could not identify wall vertices for midsection measurement.")

    centre_top, centre_bottom, *_ = get_top_bottom_wall_centres(
        mesh,
        wall_vertices,
        split_axis=wall_split_axis,
    )
    wall_axis_override = None
    if wall_split_axis not in (None, "auto"):
        wall_axis_override = np.zeros(3, dtype=float)
        wall_axis_override[int(wall_split_axis)] = 1.0
    midpoint, _traces, section_points, local_axes = get_midpoint_cross_section_from_centres(
        mesh, centre_top, centre_bottom, wall_axis=wall_axis_override
    )
    if section_points is None or len(section_points) < 3:
        raise ValueError("Midpoint cross section returned insufficient points.")

    left_section, right_section, *_ = get_left_right_midsections(section_points, midpoint, local_axes)
    if return_wall_method:
        return left_section, right_section, wall_method
    return left_section, right_section


def batch_midsection_width_height(
    meshes,
    guard_cell="both",
    dot_thresh=0.2,
    wall_split_axis="auto",
    wall_interface_axis=None,
):
    """Measure midsection width and height for multiple meshes.
    
    Batch processes an iterable of meshes to extract midsection dimensions
    for one or both guard cells.
    
    Parameters
    ----------
    meshes : Sequence[Union[str, Path, trimesh.Trimesh]]
        Paths to mesh files or loaded mesh objects.
    guard_cell : {'left', 'right', 'both'}, optional
        Which guard cell(s) to measure (default: 'both').
    dot_thresh : float, optional
        Threshold for wall-vertex detection (default: 0.2).
        Adjust if wall detection is noisy.
    
    Returns
    -------
    list of dict
        Each entry contains the mesh label plus requested width/height pairs.
        Failed analyses include an 'error' field.
    """

    guard_cell = guard_cell.lower()
    if guard_cell not in {"left", "right", "both"}:
        raise ValueError("guard_cell must be 'left', 'right', or 'both'.")

    results = []
    for idx, mesh_entry in enumerate(meshes):
        mesh_obj, label = _coerce_mesh_entry(mesh_entry)
        entry_label = label or f"in_memory_mesh_{idx}"
        record = {"mesh": entry_label}
        try:
            midsections = _extract_midsections(
                mesh_obj,
                dot_thresh=dot_thresh,
                return_wall_method=True,
                wall_split_axis=wall_split_axis,
                wall_interface_axis=wall_interface_axis,
            )
            left_section, right_section = midsections[0], midsections[1]
            wall_method = midsections[2] if len(midsections) > 2 else "normals"
            record["wall_method"] = wall_method
        except Exception as exc:
            record["error"] = str(exc)
            results.append(record)
            continue

        if guard_cell in ("left", "both"):
            width, height = measure_cross_section_width_height(left_section)
            record["left_width"] = width
            record["left_height"] = height
        if guard_cell in ("right", "both"):
            width, height = measure_cross_section_width_height(right_section)
            record["right_width"] = width
            record["right_height"] = height

        results.append(record)

    return results

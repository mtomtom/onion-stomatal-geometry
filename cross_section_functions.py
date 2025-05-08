import numpy as np
import matplotlib.pyplot as plt
import os
import trimesh
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from matplotlib.path import Path

def ellipse(theta, a, b, phi):
    """Parameterized equation of an ellipse. Returns zero if degenerate."""
    if not (a and b and a > 0 and b > 0):
        return 0.0
    cos_term = (b * np.cos(theta - phi))**2
    sin_term = (a * np.sin(theta - phi))**2
    return (a * b) / np.sqrt(cos_term + sin_term)


def get_2d_points(cross_section):
    """Extract 2D points from a cross-section, handling (points, transform) tuples."""
    if cross_section is None:
        return None
    pts = cross_section[0] if isinstance(cross_section, tuple) and len(cross_section) == 2 else cross_section
    if pts is None:
        return None
    arr = np.asarray(pts)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected 2D point array of shape (N,2), got {arr.shape}")
    return arr


def order_points(points, method="angular", center=None):
    """
    Order points using angular or nearest-neighbor method.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"order_points expects shape (N,2), got {pts.shape}")
    if len(pts) < 2:
        return pts.copy()

    if method == "angular":
        cen = np.mean(pts, axis=0) if center is None else np.asarray(center)
        if cen.shape != (2,):
            raise ValueError(f"Center must be shape (2,), got {cen.shape}")
        angles = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
        return pts[np.argsort(angles)]

    elif method == "nearest":
        remaining = pts.tolist()
        ordered = [min(remaining, key=lambda p: p[0])]
        remaining.remove(ordered[0])
        while remaining:
            dists = [np.hypot(p[0] - ordered[-1][0], p[1] - ordered[-1][1]) for p in remaining]
            ordered.append(remaining.pop(int(np.argmin(dists))))
        return np.asarray(ordered)

    else:
        raise ValueError(f"Unknown ordering method: {method}")


def calculate_polygon_area(points):
    """Calculate polygon area via vectorized shoelace formula."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def calculate_convexity(points):
    """Compute actual area / convex hull area. Returns np.nan if degenerate."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return np.nan
    try:
        hull = ConvexHull(pts)
        if hull.volume < 1e-12:
            return np.nan
        hull_points = pts[hull.vertices]
        hull_ordered = order_points(hull_points, method="angular")
        actual = calculate_polygon_area(order_points(pts, method="angular"))
        convex = calculate_polygon_area(hull_ordered)
        return actual / convex if convex > 0 else np.nan
    except Exception:
        return np.nan


def detect_gaps(points, threshold=2.5, ordered=True):
    """
    Detect large gaps in sequence. Optionally enforce ordering first.
    Returns (has_gap, indices_before_gap, gap_sizes).
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return False, [], []
    if ordered:
        pts = order_points(pts, method="angular")

    diffs = pts - np.roll(pts, -1, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    median = np.median(dists)
    mad = np.median(np.abs(dists - median))
    mad = mad if mad >= 1e-9 else median * 0.1
    thresh = median + threshold * mad

    idxs, sizes = zip(*[(i, d) for i, d in enumerate(dists) if d > thresh]) if np.any(dists > thresh) else ([], [])
    return bool(idxs), list(idxs), list(sizes)


def process_cross_section(cross_section, ordering_method="angular"):
    """
    Given raw cross_section data, returns (segments, ordered_points) or None.
    """
    pts = get_2d_points(cross_section)
    if pts is None or len(pts) < 2:
        return None
    try:
        ordered = order_points(pts, method=ordering_method)
        segs = [(tuple(ordered[i]), tuple(ordered[(i + 1) % len(ordered)])) for i in range(len(ordered))]
        return segs, ordered
    except Exception as e:
        print(f"Error in process_cross_section: {e}")
        return None
    
    ### Plotting functions

# Replace the existing create_combined_cross_section_figure function with this:
def create_combined_cross_section_figure(cross_sections, valid_sections, minor_radius, output_dir, closed_stomata=False):
    """
    Create a grid figure of individual cross-sections.
    Only sections flagged as valid are plotted, with clear Plot→Section mapping.
    """
    # Nothing to do if no valid sections or no output directory
    if not valid_sections or not any(valid_sections) or output_dir is None:
        print("Skipping combined cross-section figure: no valid sections or missing output_dir.")
        return

    # Gather (section_index, points) for valid sections
    valid_items = [
        (i, get_2d_points(cs))
        for i, (cs, ok) in enumerate(zip(cross_sections, valid_sections))
        if ok and cs is not None
    ]
    if not valid_items:
        print("Skipping combined cross-section figure: no usable cross-section data.")
        return

    n = len(valid_items)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows), constrained_layout=True)
    mapping_labels = []

    for subplot_idx, (section_idx, pts) in enumerate(valid_items, start=1):
        ax = fig.add_subplot(rows, cols, subplot_idx)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Section {section_idx}', fontsize=10)

        # Center & angularly order
        center = np.mean(pts, axis=0)
        centered = pts - center
        ordered = order_points(centered, method="angular")

        # Plot closed loop
        loop = np.vstack([ordered, ordered[0]])
        ax.plot(loop[:,0], loop[:,1], 'b-')
        ax.plot(ordered[:,0], ordered[:,1], 'ro', markersize=3)

        mapping_labels.append(f'P{subplot_idx}→S{section_idx}')

    # Super-title and mapping text
    fig.suptitle('Combined Cross-Sections', fontsize=14)
    fig.text(
        0.5, 0.02,
        ' '.join(mapping_labels),
        ha='center', fontsize=8
    )

    # Write out
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'all_individual_cross_sections.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined cross-section grid to {save_path}")


def create_visualizations(mesh, centerline_points, tangent_vectors, section_positions,
                         cross_sections, section_objects, section_transforms, raw_centerline_points,
                         inner_points, outer_points, minor_radius, valid_sections,
                         output_dir=None, closed_stomata=False):
    """Create 3D (Plotly) and 2D (Matplotlib) visualizations for mesh and cross-sections."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.path import Path

    # Ensure output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- 1) 3D Plotly Visualization ---
    try:
        import plotly.graph_objects as go
        use_plotly = True
    except ImportError:
        use_plotly = False
        print("Plotly not available; skipping 3D visualization.")

    if use_plotly:
        # Mesh trace
        verts = mesh.vertices
        faces = mesh.faces
        mesh_trace = go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            opacity=0.6, color='lightgreen', name='Mesh'
        )
        # Centerline traces
        raw_trace = go.Scatter3d(
            x=raw_centerline_points[:,0],
            y=raw_centerline_points[:,1],
            z=raw_centerline_points[:,2],
            mode='markers', marker=dict(size=4, color='orange'),
            name='Raw Centerline'
        )
        fit_trace = go.Scatter3d(
            x=centerline_points[:,0],
            y=centerline_points[:,1],
            z=centerline_points[:,2],
            mode='lines+markers', marker=dict(size=3, color='red'),
            line=dict(width=2, color='red'),
            name='Fitted Centerline'
        )
        # Section planes and cross-section outlines
        plane_traces, pts_traces = [], []
        for idx, (pt, tan, valid) in enumerate(zip(centerline_points, tangent_vectors, valid_sections)):
            if not valid:
                continue
            # Create ring in plane
            theta = np.linspace(0, 2*np.pi, 48)
            xs = minor_radius * np.cos(theta)
            ys = minor_radius * np.sin(theta)
            # basis
            v_norm = tan / np.linalg.norm(tan)
            ref = np.array([1,0,0]) if abs(v_norm[0])<0.9 else np.array([0,1,0])
            v1 = ref - v_norm*(ref.dot(v_norm)); v1 /= np.linalg.norm(v1)
            v2 = np.cross(v_norm, v1)
            ring3d = np.array([pt + x*v1 + y*v2 for x,y in zip(xs,ys)])
            plane_traces.append(
                go.Scatter3d(
                    x=ring3d[:,0], y=ring3d[:,1], z=ring3d[:,2],
                    mode='lines', line=dict(color='purple', width=2), name=f'Plane {idx}'
                )
            )
            # Cross-section outline
            cs = cross_sections[idx]
            pts2d = get_2d_points(cs)
            if pts2d is None:
                continue
            centered2d = pts2d - pts2d.mean(axis=0)
            ordered2d = order_points(centered2d)
            outline3d = np.array([pt + p[0]*v1 + p[1]*v2 for p in ordered2d])
            closed3d = np.vstack([outline3d, outline3d[0]])
            pts_traces.append(
                go.Scatter3d(
                    x=closed3d[:,0], y=closed3d[:,1], z=closed3d[:,2],
                    mode='lines', line=dict(color='blue', width=4), name=f'CS {idx}'
                )
            )
        fig3d = go.Figure(data=[mesh_trace, raw_trace, fit_trace] + plane_traces + pts_traces)
        fig3d.update_layout(
            title='3D Stomata with Cross-Sections',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0,r=0,t=40,b=0)
        )
        if output_dir:
            fig3d.write_html(os.path.join(output_dir, '3d_visualization.html'))

    # --- 2) 2D Matplotlib Grid ---
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)

    # A) Top view XY with section lines
    axA = fig.add_subplot(2,3,1)
    axA.set_title('Top View (XY)')
    axA.plot(centerline_points[:,0], centerline_points[:,1], 'ro-')
    for pt, tan, valid in zip(centerline_points, tangent_vectors, valid_sections):
        if not valid:
            continue
        dir2d = tan[:2]
        if np.linalg.norm(dir2d) < 1e-6:
            continue
        dir2d /= np.linalg.norm(dir2d)
        ortho = np.array([-dir2d[1], dir2d[0]])
        L = minor_radius
        seg = np.vstack([pt[:2] + L*ortho, pt[:2] - L*ortho])
        axA.plot(seg[:,0], seg[:,1], 'g-')
    axA.set_aspect('equal'); axA.grid(True)

    # B) Raw cross-sections
    axB = fig.add_subplot(2,3,2)
    axB.set_title('Raw Cross-Sections')
    for obj, valid in zip(section_objects, valid_sections):
        if not valid or obj is None:
            continue
        p2d, _ = obj.to_planar()
        axB.plot(p2d.vertices[:,0], p2d.vertices[:,1], alpha=0.7)
    axB.set_aspect('equal'); axB.grid(True)

    # C) Processed cross-sections
    axC = fig.add_subplot(2,3,3)
    axC.set_title('Processed Cross-Sections')
    for cs, valid in zip(cross_sections, valid_sections):
        if not valid or cs is None:
            continue
        pts = get_2d_points(cs)
        result = process_cross_section(pts)
        if result:
            _, ord_pts = result
            loop = np.vstack([ord_pts, ord_pts[0]])
            axC.plot(loop[:,0], loop[:,1])
    axC.set_aspect('equal'); axC.grid(True)

    # D) Centered overlays
    axD = fig.add_subplot(2,3,4)
    axD.set_title('Centered Overlays')
    for cs, valid in zip(cross_sections, valid_sections):
        if not valid or cs is None:
            continue
        pts = get_2d_points(cs)
        ord_pts = order_points(pts - pts.mean(axis=0))
        loop = np.vstack([ord_pts, ord_pts[0]])
        axD.plot(loop[:,0], loop[:,1], alpha=0.6)
    axD.set_aspect('equal'); axD.grid(True)

    # E) Aspect ratio distribution
    ar_vals = []
    for cs, valid in zip(cross_sections, valid_sections):
        if not valid or cs is None:
            continue
        pts = get_2d_points(cs)
        cov = np.cov(pts.T)
        eig = np.linalg.eigvalsh(cov)
        if eig[1] > 1e-6:
            ar_vals.append(eig[1] / eig[0])
    axE = fig.add_subplot(2,3,5)
    axE.set_title('Aspect Ratios')
    if ar_vals:
        axE.boxplot(ar_vals, vert=True, showfliers=True)
    axE.grid(True)

    # F) Example detailed view
    axF = fig.add_subplot(2,3,6)
    valid_idxs = [i for i,v in enumerate(valid_sections) if v]
    if valid_idxs:
        ex = valid_idxs[len(valid_idxs)//2]
        pts = get_2d_points(cross_sections[ex])
        ord_pts = order_points(pts - pts.mean(axis=0))
        loop = np.vstack([ord_pts, ord_pts[0]])
        axF.plot(loop[:,0], loop[:,1], 'b-')
        axF.set_title(f'Section {ex}')
    axF.set_aspect('equal'); axF.grid(True)

    # Save or show
    if output_dir:
        fig.savefig(os.path.join(output_dir, 'all_cross_sections.png'), dpi=150)
        plt.close(fig)
    else:
        plt.show()

def _apply_proximity_fallback(centered_points, minor_radius, initial_point_count):
    """
    Region-growing fallback: returns filtered points only.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fclusterdata  # fallback alternative

    pts = np.asarray(centered_points)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return pts.copy()

    # Step 1: DBSCAN noise removal
    try:
        from scipy.spatial import KDTree
        from sklearn.cluster import DBSCAN
        eps = minor_radius * 0.4
        db = DBSCAN(eps=eps, min_samples=3).fit(pts)
        mask = db.labels_ != -1
        pts_db = pts[mask]
    except Exception:
        pts_db = pts.copy()

    # If DBSCAN removed too many, keep the larger set
    if len(pts_db) < 3:
        pts_db = pts.copy()

    # Step 2: Region growing around origin
    try:
        tree = KDTree(pts_db)
        dists, _ = tree.query(pts_db, k=min(6, len(pts_db)))
        med = np.median(dists[:,1] if dists.ndim>1 else dists)
        thresh = med * 3.5
        # BFS
        from collections import deque
        visited = set([0])
        q = deque([0])
        while q:
            i = q.popleft()
            for j in tree.query_ball_point(pts_db[i], r=thresh):
                if j not in visited:
                    visited.add(j); q.append(j)
        pts_grown = pts_db[list(visited)]
        if len(pts_grown) >= 3:
            return pts_grown
    except Exception:
        pass

    # fallback to DBSCAN result
    return pts_db


def load_and_align_mesh(file_path, align_axis='Y'):
    """Load mesh, apply PCA inertia + optional axis swap, return mesh and full transform."""
    try:
        mesh = trimesh.load_mesh(file_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        # --- START: MODIFICATIONS TO DISABLE ALIGNMENT ---
        # # center
        # mesh.apply_translation(-mesh.centroid)
        # # inertia alignment
        # T = mesh.principal_inertia_transform
        # mesh.apply_transform(T)
        # # axis swap if needed
        # axes = {'X':0,'Y':1,'Z':2}
        # target = axes.get(align_axis, 1)
        # ext = mesh.extents
        # curr = int(np.argmax(ext))
        import trimesh.transformations as tf
        # R = np.eye(4) # Initialize R as identity if other transforms are skipped
        # if curr!=target:
        #     # rotate around cross axis
        #     axis_map = {(0,1):('z',np.pi/2),(1,0):('z',-np.pi/2),
        #                 (0,2):('y',-np.pi/2),(2,0):('y',np.pi/2),
        #                 (1,2):('x',np.pi/2),(2,1):('x',-np.pi/2)}
        #     key=(curr,target)
        #     if key in axis_map:
        #         ax,ang = axis_map[key]
        #         vec={'x':[1,0,0],'y':[0,1,0],'z':[0,0,1]}[ax]
        #         R = tf.rotation_matrix(ang, vec)
        #         mesh.apply_transform(R)
                
        # # NEW: Secondary alignment to standardize rotation around Y-axis
        # if mesh is not None and not mesh.is_empty:
        #     # Take multiple cross-sections along Y to find the most elliptical one
        #     best_angle = 0
        #     best_ratio = 0
        #     test_angles = np.linspace(0, np.pi, 18)  # Test 18 different angles (10° increments)
            
        #     # Sampling position near the center (can be adjusted)
        #     sample_pos = np.array([0.0, 0.0, 0.0])
            
        #     for angle in test_angles:
        #         # Create rotation matrix around Y-axis
        #         rot = tf.rotation_matrix(angle, [0, 1, 0])
        #         test_mesh = mesh.copy()
        #         test_mesh.apply_transform(rot)
                
        #         # Take cross section
        #         section = test_mesh.section(
        #             plane_origin=sample_pos,
        #             plane_normal=[0, 1, 0]
        #         )
                
        #         if section is not None and len(section.entities) > 0:
        #             # Get 2D representation
        #             path_2D, _ = section.to_2D()
        #             if path_2D.vertices is not None and len(path_2D.vertices) >= 3:
        #                 # Calculate aspect ratio using PCA
        #                 from sklearn.decomposition import PCA # Ensure PCA is imported if this block is active
        #                 pca = PCA(n_components=2).fit(path_2D.vertices)
        #                 var = pca.explained_variance_
        #                 if len(var) == 2 and var[1] > 1e-6:  # Avoid division by zero
        #                     ratio = var[0] / var[1]
        #                     # Choose orientation that gives most elliptical cross-section
        #                     if ratio > best_ratio:
        #                         best_ratio = ratio
        #                         best_angle = angle
            
        #     if best_ratio > 1.0:  # Only rotate if we found a good orientation
        #         # Apply the best rotation
        #         final_rot = tf.rotation_matrix(best_angle, [0, 1, 0])
        #         mesh.apply_transform(final_rot)
        #         # Update the full transform
        #         R = final_rot.dot(R)
        
        # # recenter
        # mesh.apply_translation(-mesh.centroid)
        # # combine transforms
        # full_T = R.dot(T)

        # If all alignment is disabled, full_T should be an identity matrix
        full_T = np.eye(4)
        # --- END: MODIFICATIONS TO DISABLE ALIGNMENT ---

        return mesh, full_T
    except Exception as e:
        print(f"Error in load_and_align_mesh: {e}")
        return None, None


def get_radial_dimensions(mesh, center=None, ray_count=36):
    """Return inner, outer, centerline, avg_minor_radius or (None...)
    """
    import numpy as np
    if mesh is None or mesh.is_empty:
        return None, None, None, None
    c = center if center is not None else mesh.centroid
    angles = np.linspace(0,2*np.pi,ray_count,endpoint=False)
    inner, outer = [], []
    for a in angles:
        dir = np.array([np.cos(a),np.sin(a),0.])
        pts, _, _ = mesh.ray.intersects_location([c],[dir])
        if len(pts)>=2:
            d = np.linalg.norm(pts - c,axis=1)
            idx = np.argsort(d)
            inner.append(pts[idx[0]]); outer.append(pts[idx[-1]])
    if not inner:
        return None, None, None, None
    inner, outer = np.array(inner), np.array(outer)
    raw_center = (inner+outer)/2    
    avg = np.mean(np.linalg.norm(outer-inner,axis=1))/2
    return inner, outer, raw_center, avg


def fit_centerline_ellipse(raw_centerline_points, center):
    """Fit ellipse to 2D projection, return (a,b,phi) or (None,...)."""
    import numpy as np
    from scipy.optimize import curve_fit
    if raw_centerline_points is None or len(raw_centerline_points)<3:
        return None, None, None
    xy = raw_centerline_points[:,:2]
    cen = center[:2]
    r = np.linalg.norm(xy-cen,axis=1)
    th = np.arctan2(xy[:,1]-cen[1], xy[:,0]-cen[0])
    guess=[np.mean(r),np.mean(r),0.]
    try:
        params,_ = curve_fit(ellipse, th, r, p0=guess)
        a,b,phi=params
        if a<b:
            a,b=b,a; phi=(phi+np.pi/2)%np.pi
        return a,b,phi
    except Exception:
        return None, None, None


def filter_section_points(points_2D, minor_radius, origin_2d_target, eps_factor=0.20, min_samples=3):
    """Return filtered 2D points array only, with largest-cluster fallback."""
    import numpy as np
    from sklearn.cluster import DBSCAN
    pts = np.asarray(points_2D)
    if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<min_samples:
        return np.empty((0,2))
    db = DBSCAN(eps=minor_radius*eps_factor, min_samples=min_samples).fit(pts)
    labels = db.labels_
    unique = [l for l in set(labels) if l!=-1]
    if not unique:
        return pts
    # pick cluster containing origin or largest
    best=None; best_size=0
    for l in unique:
        cluster=pts[labels==l]
        size=len(cluster)
        if best is None or size>best_size:
            best, best_size = l, size
    return pts[labels==best]



import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from sklearn.cluster import DBSCAN
import trimesh
from typing import Optional, Tuple, List, Dict, Union
import edge_detection as ed
from scipy.signal import savgol_filter



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ellipse(theta: float, a: float, b: float, phi: float) -> float:
    """Return radius at angle theta for an ellipse with axes a, b rotated by phi."""
    numerator = a * b
    denom = np.hypot(b * np.cos(theta - phi), a * np.sin(theta - phi))
    return numerator / denom


def get_2d_points(cross_section) -> Optional[np.ndarray]:
    """Extract 2D points from a cross-section, handling both tuple and direct array formats."""
    if cross_section is None:
        return None

    pts = cross_section[0] if isinstance(cross_section, tuple) else cross_section
    return np.asarray(pts)


def order_points(
    points: np.ndarray,
    method: str = "nearest",
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Order a set of 2D points by "nearest" neighbor or "angular" sorting around a center.
    """
    pts = np.asarray(points)
    n = len(pts)
    if n <= 1:
        return pts

    if method == "angular":
        if center is None:
            center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]

    # Nearest-neighbor ordering
    remaining = set(range(n))
    current = int(np.argmin(pts[:, 0]))
    order = [current]
    remaining.remove(current)

    while remaining:
        last = pts[current]
        # compute distances to all remaining
        dists = np.linalg.norm(pts[list(remaining)] - last, axis=1)
        idx = list(remaining)[int(np.argmin(dists))]
        order.append(idx)
        remaining.remove(idx)
        current = idx

    return pts[order]


def process_cross_section(
    cross_section
) -> Optional[Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]]:
    """Order cross-section points and build segment list."""
    pts = get_2d_points(cross_section)
    if pts is None or len(pts) < 3:
        logger.warning("Invalid or too few cross-section points.")
        return None

    ordered = order_points(pts, method="nearest")
    segments = [
        (ordered[i], ordered[(i + 1) % len(ordered)])
        for i in range(len(ordered))
    ]
    return segments, ordered


def get_pore_center_vertical_surface_points(
    inner_pts: Optional[np.ndarray],
    x_rel: float = 0.15,
    y_rel: float = 0.25,
    z_rel: float = 0.25
) -> np.ndarray:
    """
    Filter 3D inner boundary points to isolate a vertical sheet at the pore center.
    Returns empty array if no points match.
    """
    if inner_pts is None or len(inner_pts) < 3:
        logger.warning("Not enough inner points provided.")
        return np.empty((0, 3))

    med = np.median(inner_pts, axis=0)
    ext = np.ptp(inner_pts, axis=0)
    thresh = np.where(ext < 1e-6, [x_rel, y_rel, z_rel], ext * np.array([x_rel, y_rel, z_rel]))

    logger.info(
        f"Applying thresholds dx<{thresh[0]:.3f}, dy<{thresh[1]:.3f}, dz<{thresh[2]:.3f}"\
    )

    mask = np.all(np.abs(inner_pts - med) < thresh, axis=1)
    filtered = inner_pts[mask]

    logger.info(f"Isolated {len(filtered)} pore center surface points.")
    return filtered


def get_radial_dimensions(
    mesh: trimesh.Trimesh,
    center: Optional[np.ndarray] = None,
    ray_count: int = 36
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Cast rays in XY-plane around the center to find inner/outer intersections.
    """
    center = center if center is not None else mesh.centroid
    angles = np.linspace(0, 2*np.pi, ray_count, endpoint=False)

    inner, outer = [], []
    for ang in angles:
        dir_vec = np.array([np.cos(ang), np.sin(ang), 0.0])
        try:
            pts, _, _ = mesh.ray.intersects_location([center], [dir_vec])
            if len(pts) >= 2:
                d = np.linalg.norm(pts - center, axis=1)
                order = np.argsort(d)
                inner.append(pts[order[0]])
                outer.append(pts[order[-1]])
        except Exception as e:
            logger.warning(f"Ray error at {np.degrees(ang):.1f}°: {e}")

    if not inner or not outer:
        logger.error("Insufficient ray intersections to compute dimensions.")
        return None, None, None, None

    inner_arr = np.vstack(inner)
    outer_arr = np.vstack(outer)
    centerline = (inner_arr + outer_arr) / 2
    avg_minor = np.linalg.norm(outer_arr - inner_arr, axis=1).mean() / 2

    logger.info(f"Avg minor radius: {avg_minor:.4f}")
    return inner_arr, outer_arr, centerline, avg_minor


def filter_section_points(
    pts_2d: np.ndarray,
    minor_radius: float,
    origin: np.ndarray,
    eps_factor: float = 0.2,
    min_samples: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster 2D points, pick cluster containing origin or closest to it.
    """
    if pts_2d is None or len(pts_2d) < min_samples:
        return np.empty((0, 2)), np.array([], dtype=bool)

    eps = minor_radius * eps_factor
    try:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_2d)
    except Exception as e:
        logger.warning(f"DBSCAN failed: {e}")
        return pts_2d, np.ones(len(pts_2d), dtype=bool)

    masks = {}
    for lbl in set(labels) - {-1}:
        cluster = pts_2d[labels == lbl]
        if len(cluster) < min_samples:
            continue
        centroid = cluster.mean(axis=0)
        dist = np.linalg.norm(centroid - origin)
        masks[lbl] = (centroid, dist)

    if not masks:
        logger.warning("No valid clusters.")
        return np.empty((0, 2)), np.zeros(len(pts_2d), dtype=bool)

    # sort by distance
    best_lbl = min(masks.items(), key=lambda kv: kv[1][1])[0]
    sel_mask = (labels == best_lbl)
    # check containment
    ordered = order_points(pts_2d[sel_mask], method="angular")
    poly = MplPath(ordered)
    if poly.contains_point(origin):
        logger.info(f"Origin inside cluster {best_lbl}.")
    else:
        logger.info(f"Origin outside cluster {best_lbl}, but selected as closest.")

    return pts_2d[sel_mask], sel_mask


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _setup_output_dir(base_dir: Optional[Path], section: str) -> Optional[Path]:
    if base_dir is None:
        return None
    out = base_dir / section
    out.mkdir(parents=True, exist_ok=True)
    return out


def _smooth_centerline(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n < 5:
        return points
    # Choose window (odd, <=11)
    window = min(11, n - (1 if n % 2 == 0 else 0))
    window = max(window, 3)
    order = min(2, window - 1)
    try:
        x = savgol_filter(points[:, 0], window, order)
        y = savgol_filter(points[:, 1], window, order)
        z = savgol_filter(points[:, 2], window, order)
        logger.info(f"Centerline smoothed (window={window}, order={order}).")
        return np.vstack((x, y, z)).T
    except Exception as e:
        logger.warning(f"Smoothing failed: {e}")
        return points


def _determine_plane_midpoint(
    centerline: np.ndarray,
    pore_center_y: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    y_vals = centerline[:, 1]
    target_y = pore_center_y if pore_center_y is not None else 0.0
    idx = int(np.argmin(np.abs(y_vals - target_y)))
    origin = centerline[idx]
    # Compute local tangent
    if len(centerline) < 2:
        tangent = np.array([0.0, 1.0, 0.0])
    else:
        if idx == 0:
            vec = centerline[1] - centerline[0]
        elif idx == len(centerline) - 1:
            vec = centerline[-1] - centerline[-2]
        else:
            vec = centerline[idx + 1] - centerline[idx - 1]
        norm = np.linalg.norm(vec)
        tangent = vec / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    logger.info(f"Midpoint plane at idx {idx}, origin={origin.round(3)}, tangent={tangent.round(3)}")
    return origin, tangent


def _process_file(
    file_path: Path,
    section: str,
    output_dir: Optional[Path]
) -> Dict:
    logger.info(f"Processing: {file_path.name} [{section}]")
    result: Dict = {}

    # Edge detection
    ed_out = ed.find_seam_by_raycasting(str(file_path), visualize=False)
    if not ed_out or not ed_out.get('mesh_object'):
        logger.error("Edge detection failed.")
        return {'error': 'Edge detection failed'}

    mesh = ed_out['mesh_object']
    pore_center_ed = ed_out.get('pore_center_coords')

    # Radial dimensions
    inner, outer, raw_cl, minor = get_radial_dimensions(mesh, center=pore_center_ed)
    if inner is None or outer is None or raw_cl is None or minor is None:
        return {'error': 'Radial ray casting failed'}
    result.update({'minor_radius': minor})

    # Pore vertical surface
    vsp = get_pore_center_vertical_surface_points(inner)
    result['vertical_surface_points'] = vsp

    # Smooth centerline
    cl = _smooth_centerline(raw_cl)
    if len(cl) < 2:
        return {'error': 'Insufficient centerline'}

    # Determine section plane
    if section == 'midpoint':
        origin, tangent = _determine_plane_midpoint(cl, pore_center_ed[1] if pore_center_ed is not None else None)
    else:
        # Tip logic similar extracted into helper if needed
        origin, tangent = _determine_plane_midpoint(cl, None)
    result.update({'plane_origin': origin, 'plane_tangent': tangent})

    # Cross-section slicing and analysis
    sec3d = mesh.section(plane_origin=origin, plane_normal=tangent)
    if sec3d is None:
        logger.error("Sectioning failed")
        return {'error': 'Section failed'}
    path2d, tfm2d = sec3d.to_2D()
    pts_2d = path2d.vertices
    segments, ordered_pts_2d = process_cross_section(pts_2d) or ([], np.empty((0,2)))
    result['segments'] = segments

    # Expose for combined plotting
    result['points_2d'] = ordered_pts_2d

    # Aspect ratio via PCA
    if pts_2d.size:
        from sklearn.decomposition import PCA
        comps = PCA(n_components=2).fit_transform(pts_2d)
        ar = np.ptp(comps[:, 0]) / np.ptp(comps[:, 1]) if np.ptp(comps[:, 1]) else np.nan
        result['aspect_ratio'] = ar
        logger.info(f"Aspect ratio: {ar:.3f}")

    # Optional visualization (PNG, HTML)
    if output_dir:
        # Implement export logic here
        pass

    return result


def analyze_cross_sections(
    file_paths: List[str],
    section: str = 'midpoint',
    output_dir: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Batch analyze a list of mesh files, returning per-file metrics.

    Args:
        file_paths: list of OBJ file strings
        section: 'midpoint' or 'tip'
        output_dir: path to store optional visualizations

    Returns:
        Dictionary mapping file path to result dict or error.
    """
    base_dir = Path(output_dir) if output_dir else None
    out_dir = _setup_output_dir(base_dir, section)

    results: Dict[str, Dict] = {}
    for fp in file_paths:
        path = Path(fp)
        results[fp] = _process_file(path, section, out_dir)
    return results

def _setup_output_dir(base_dir: Optional[Path], section: str) -> Optional[Path]:
    if base_dir is None:
        return None
    out = base_dir / section
    out.mkdir(parents=True, exist_ok=True)
    return out


def _smooth_centerline(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n < 5:
        return points
    window = min(11, n - (1 if n % 2 == 0 else 0))
    window = max(window, 3)
    order = min(2, window - 1)
    try:
        x = savgol_filter(points[:, 0], window, order)
        y = savgol_filter(points[:, 1], window, order)
        z = savgol_filter(points[:, 2], window, order)
        logger.info(f"Centerline smoothed (window={window}, order={order}).")
        return np.vstack((x, y, z)).T
    except Exception as e:
        logger.warning(f"Smoothing failed: {e}")
        return points


def create_combined_2d_plot(
    results: Dict[str, Dict],
    output_path: Union[str, Path],
    location: str = 'Unknown'
) -> None:
    """
    Overlay 2D cross-sections for all valid entries in `results` and save the figure.

    Args:
        results: Mapping from file path to result dict.
        output_path: File path to save the PNG.
        location: Descriptor (e.g., 'Midpoint' or 'Tip').
    """
    output_path = Path(output_path)
    # Filter entries with valid 2D data
    valid = [p for p, v in results.items() if v and 'points_2d' in v and v['points_2d'].ndim == 2 and v['points_2d'].shape[0] >= 3]
    if not valid:
        logger.warning("No valid 2D sections for '%s', skipping combined plot.", location)
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Combined {location} Cross-Sections")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(valid)))

    max_extent = 0.0
    for idx, path in enumerate(valid):
        pts_2d = results[path]['points_2d']
        # Center and order
        center = pts_2d.mean(axis=0)
        pts_centered = pts_2d - center
        ordered = order_points(pts_centered, method='angular')
        xs, ys = ordered[:, 0], ordered[:, 1]
        # Close the polygon
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        ax.plot(xs, ys, color=colors[idx], alpha=0.7, linewidth=1.5, label=Path(path).name)
        max_extent = max(max_extent, np.max(np.abs(pts_centered)))

    lim = max_extent * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if len(valid) <= 10:
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved combined 2D plot for %s at %s", location, output_path)


def create_data_boxplot(
    values: List[float],
    output_path: Union[str, Path],
    title: str = 'Data',
    label: str = 'Value'
) -> None:
    """
    Create and save a box plot for a list of float `values`.

    Args:
        values: Data points for the box plot.
        output_path: Path to save the PNG file.
        title: Title prefix for the plot.
        label: Y-axis label.
    """
    if not values:
        logger.warning("No data provided for boxplot '%s', skipping.", title)
        return

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(values, patch_artist=True)
    # Overlay raw points
    jitter = np.random.normal(1, 0.05, size=len(values))
    ax.scatter(jitter, values, color='black', alpha=0.6)

    ax.set_title(f"{title} (N={len(values)})")
    ax.set_ylabel(label)
    ax.set_xticks([1])
    ax.set_xticklabels([title])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved box plot '%s' at %s", title, output_path)


def create_mesh_grid_plot(
    file_paths: List[str],
    output_path: Union[str, Path],
    rows: int = 3,
    cols: int = 6
) -> None:
    """
    Generate a grid of 3D mesh renderings (matplotlib) for the first rows*cols files.

    Args:
        file_paths: List of OBJ file paths.
        output_path: Path to save the PNG.
        rows: Number of subplot rows.
        cols: Number of subplot columns.
    """
    output_path = Path(output_path)
    num = min(len(file_paths), rows * cols)
    if num == 0:
        logger.warning("No mesh files provided for grid.")
        return

    # Load meshes
    meshes = []
    for fp in file_paths[:num]:
        try:
            m = trimesh.load_mesh(fp, process=False)
            if isinstance(m, trimesh.Scene):
                m = m.dump(concatenate=True)
            meshes.append(m)
        except Exception as e:
            logger.error("Failed to load mesh '%s': %s", fp, e)
            meshes.append(None)

    # Compute global bounds
    verts = np.vstack([m.vertices for m in meshes if m and hasattr(m, 'vertices')])
    center = verts.mean(axis=0)
    # Use np.ptp since ndarray.ptp is removed in NumPy 2.0
    max_range = np.max(np.ptp(verts, axis=0)) * 0.5
    limits = [(center[i] - max_range, center[i] + max_range) for i in range(3)]

    fig = plt.figure(figsize=(cols*3, rows*3))
    for idx in range(rows*cols):
        ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
        m = meshes[idx] if idx < len(meshes) else None
        if m and hasattr(m, 'vertices') and hasattr(m, 'faces'):
            v = m.vertices - center
            ax.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=m.faces, linewidth=0, antialiased=True)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(output_path, dpi=200, facecolor='white')
    plt.close(fig)
    logger.info("Saved mesh grid plot at %s", output_path)


def main():
    import os
    base_output = Path("refactored_results")
    base_output.mkdir(exist_ok=True)
    files = [
        "Meshes/Onion_OBJ/Ac_DA_1_3.obj", "Meshes/Onion_OBJ/Ac_DA_1_2.obj", "Meshes/Onion_OBJ/Ac_DA_1_5.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_7.obj", "Meshes/Onion_OBJ/Ac_DA_3_6.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_4.obj", "Meshes/Onion_OBJ/Ac_DA_3_3.obj", "Meshes/Onion_OBJ/Ac_DA_3_2.obj",
        "Meshes/Onion_OBJ/Ac_DA_3_1.obj", "Meshes/Onion_OBJ/Ac_DA_2_7.obj",
         "Meshes/Onion_OBJ/Ac_DA_2_4.obj", "Meshes/Onion_OBJ/Ac_DA_2_3.obj",
        "Meshes/Onion_OBJ/Ac_DA_1_8_mesh.obj", "Meshes/Onion_OBJ/Ac_DA_1_6.obj"
    ]

    files = ["Meshes/Onion_OBJ/Ac_DA_1_3.obj"]

    # Midpoint analysis
    mid = analyze_cross_sections(files, section='midpoint', output_dir=str(base_output))
    create_combined_2d_plot(mid, base_output / "combined_midpoint.png", "Midpoint")
    ars_mid = [v['aspect_ratio'] for v in mid.values() if v.get('aspect_ratio') is not None]
    create_data_boxplot(ars_mid, base_output / "ar_mid_boxplot.png", "Aspect Ratio", "Midpoint")

    # Tip analysis
    tip = analyze_cross_sections(files, section='tip', output_dir=str(base_output))
    create_combined_2d_plot(tip, base_output / "combined_tip.png", "Tip")
    ars_tip = [v['aspect_ratio'] for v in tip.values() if v.get('aspect_ratio') is not None]
    create_data_boxplot(ars_tip, base_output / "ar_tip_boxplot.png", "Aspect Ratio", "Tip")

    # Mesh grid
    create_mesh_grid_plot(files, base_output / "mesh_grid.png")

if __name__ == '__main__':
    main()
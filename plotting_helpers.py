import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from helper_functions import order_points
import os

def plot_aspect_ratio_curve(positions, aspect_ratios, base_name, output_dir):
    """
    Plot aspect ratio (ellipse) vs normalized position along centerline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = [(p, ar) for p, ar in zip(positions, aspect_ratios) if ar is not None]
    if valid:
        plot_positions, plot_ratios = zip(*sorted(valid))
        ax.plot(plot_positions, plot_ratios, 'o-', linewidth=2)
    ax.set_xlabel('Normalized Position Along Centerline')
    ax.set_ylabel('Aspect Ratio (Ellipse)')
    ax.set_title(f'Aspect Ratio (Ellipse) Along Centerline\n{base_name}')
    ax.grid(True)
    aspect_plot_path = os.path.join(output_dir, f"{base_name}_aspect_ratio_ellipse_curve.png")
    plt.savefig(aspect_plot_path, dpi=150)
    plt.close(fig)

def plot_width_curve(positions, widths, base_name, output_dir):
    """
    Plot ellipse semi-minor axis width vs normalized position.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = [(p, w) for p, w in zip(positions, widths) if w is not None]
    if valid:
        plot_positions, plot_widths = zip(*sorted(valid))
        ax.plot(plot_positions, plot_widths, 'o-', color='green', linewidth=2)
    ax.set_xlabel('Normalized Position Along Centerline')
    ax.set_ylabel('Width (Ellipse Semi-Minor Axis)')
    ax.set_title(f'Width (Ellipse) Along Centerline\n{base_name}')
    ax.grid(True)
    width_plot_path = os.path.join(output_dir, f"{base_name}_width_ellipse_curve.png")
    plt.savefig(width_plot_path, dpi=150)
    plt.close(fig)

def plot_inlier_ratio_curve(positions, inlier_ratios, base_name, output_dir):
    """
    Plot ellipse inlier ratio vs normalized position.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = [(p, ir) for p, ir in zip(positions, inlier_ratios) if ir is not None]
    if valid:
        plot_positions, plot_ratios = zip(*sorted(valid))
        ax.plot(plot_positions, plot_ratios, 'o-', color='purple', linewidth=2)
    ax.set_xlabel('Normalized Position Along Centerline')
    ax.set_ylabel('Ellipse Inlier Ratio (RANSAC)')
    ax.set_title(f'Ellipse Fit Regularity (Inlier Ratio) Along Centerline\n{base_name}')
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    inlier_ratio_plot_path = os.path.join(output_dir, f"{base_name}_ellipse_inlier_ratio_curve.png")
    plt.savefig(inlier_ratio_plot_path, dpi=150)
    plt.close(fig)

def plot_orientation_curve(positions, orientations_deg, base_name, output_dir):
    """
    Plot ellipse major axis orientation (relative to midpoint) vs normalized position.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = [(p, o) for p, o in zip(positions, orientations_deg) if o is not None and np.isfinite(o)]
    if valid:
        plot_positions, plot_orient = zip(*sorted(valid))
        ax.plot(plot_positions, plot_orient, 'o-', color='cyan', linewidth=2)
    ax.set_xlabel('Normalized Position Along Centerline')
    ax.set_ylabel('Relative Orientation of Major Axis (degrees)')
    ax.set_title(f'Ellipse Major Axis Orientation (Relative to Midpoint)\n{base_name}')
    ax.grid(True)
    ax.set_ylim(-95, 95)
    orientation_plot_path = os.path.join(output_dir, f"{base_name}_ellipse_relative_orientation_curve.png")
    plt.savefig(orientation_plot_path, dpi=150)
    plt.close(fig)

def plot_sections_3d_matplotlib(mesh, smoothed_centerline, section_data_3d, base_name, output_dir):
    """
    Plot 3D mesh, centerline, and cross-sections using matplotlib.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh_triangles = mesh.vertices[mesh.faces]
    mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.3, edgecolor='gray', linewidth=0.05, facecolor='lightgray')
    ax.add_collection3d(mesh_collection)
    ax.plot(smoothed_centerline[:, 0], smoothed_centerline[:, 1], smoothed_centerline[:, 2], 'k-', linewidth=2, label='Centerline')
    cmap_mpl = matplotlib.colormaps['plasma']
    for sd_item in section_data_3d:
        points_2d_vis = sd_item['points_2d']
        transform_vis = sd_item['transform']
        norm_pos_vis = sd_item['norm_pos']
        color_vis = cmap_mpl(norm_pos_vis)
        ordered_points_2d_vis = order_points(points_2d_vis, method="angular")
        points_3d_vis_list = []
        for pt_2d_vis in ordered_points_2d_vis:
            pt_2d_h_vis = np.array([pt_2d_vis[0], pt_2d_vis[1], 0.0, 1.0])
            pt_3d_h_vis = transform_vis.dot(pt_2d_h_vis)
            points_3d_vis_list.append(pt_3d_h_vis[:3])
        if not points_3d_vis_list: continue
        points_3d_array_vis = np.array(points_3d_vis_list)
        pts_closed_vis = np.vstack([points_3d_array_vis, points_3d_array_vis[0]])
        ax.plot(pts_closed_vis[:, 0], pts_closed_vis[:, 1], pts_closed_vis[:, 2], color=color_vis, linewidth=2, alpha=0.9)
        verts_vis = [list(zip(pts_closed_vis[:, 0], pts_closed_vis[:, 1], pts_closed_vis[:, 2]))]
        poly_vis = Poly3DCollection(verts_vis, alpha=0.6, facecolor=color_vis)
        ax.add_collection3d(poly_vis)
    sampled_positions_plot = np.array([s['position'] for s in section_data_3d])
    if len(sampled_positions_plot) > 0:
        ax.scatter(sampled_positions_plot[:, 0], sampled_positions_plot[:, 1], sampled_positions_plot[:, 2], c='blue', marker='o', s=30, alpha=0.7, depthshade=False)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'3D Cross-Sections - {base_name}')
    ax.view_init(elev=30, azim=45)
    max_val = np.max(mesh.bounds)
    min_val = np.min(mesh.bounds)
    ax.auto_scale_xyz([min_val, max_val], [min_val, max_val], [min_val, max_val])
    in_situ_path = os.path.join(output_dir, f"{base_name}_3d_sections.png")
    plt.savefig(in_situ_path, dpi=200)
    plt.close(fig)
    print(f"  Created 3D cross-section visualization (Matplotlib): {in_situ_path}")

def plot_sections_3d_plotly(mesh, smoothed_centerline, section_data_3d, base_name, output_dir, seam_points_for_plot=None):
    """
    Plot 3D mesh, centerline, and cross-sections using Plotly.
    """
    import plotly.graph_objects as go
    plotly_traces = []
    plotly_traces.append(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        opacity=0.6, color='lightgrey', name='Mesh'
    ))
    plotly_traces.append(go.Scatter3d(
        x=smoothed_centerline[:,0], y=smoothed_centerline[:,1], z=smoothed_centerline[:,2],
        mode='lines+markers',
        line=dict(color='black', width=6),
        marker=dict(size=4, color='black'),
        name='Centerline'
    ))
    if seam_points_for_plot is not None and len(seam_points_for_plot)>0:
        plotly_traces.append(go.Scatter3d(
            x=seam_points_for_plot[:,0], y=seam_points_for_plot[:,1], z=seam_points_for_plot[:,2],
            mode='markers', marker=dict(size=3,color='red',symbol='x'),
            name='Seam'
        ))
    cmap = matplotlib.colormaps['plasma']
    annotations = []
    for i, sd in enumerate(section_data_3d):
        pts3d = sd['points_3d']
        pts2d = sd['points_2d']
        if pts3d is None or len(pts3d) == 0 or pts2d is None or len(pts2d) != len(pts3d):
            continue

        # Find the order indices
        center_2d = np.mean(pts2d, axis=0)
        angles = np.arctan2(pts2d[:, 1] - center_2d[1], pts2d[:, 0] - center_2d[0])
        order = np.argsort(angles)
        ordered_3d = pts3d[order]
        # Optionally close the polygon
        ordered_3d = np.vstack([ordered_3d, ordered_3d[0]])
        normp = sd['norm_pos']
        color = cmap(normp)
        rgba = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'
        plotly_traces.append(go.Scatter3d(
            x=ordered_3d[:,0], y=ordered_3d[:,1], z=ordered_3d[:,2],
            mode='lines',
            line=dict(color=rgba, width=4),
            name=f'Section {i} (pos {normp:.2f})'
        ))
        pos3 = sd['position']
        annotations.append(dict(
            showarrow=False,
            x=pos3[0], y=pos3[1], z=pos3[2],
            text=f"{normp:.2f}",
            font=dict(color='white',size=10),
            bgcolor='rgba(0,0,0,0.5)'
        ))
    fig = go.Figure(data=plotly_traces)
    fig.update_layout(
        title=f'Interactive 3D Cross-Sections - {base_name}',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data', annotations=annotations,
            camera=dict(eye=dict(x=1.2,y=1.2,z=1.2))
        ),
        margin=dict(l=0,r=0,b=0,t=40)
    )
    plotly_html_path = os.path.join(output_dir, f"{base_name}_3d_sections_interactive.html")
    fig.write_html(plotly_html_path)
    print(f"Created Plotly 3D HTML: {plotly_html_path}")

def get_plotly_figure_for_app(mesh, centerline_data, section_data_3d, seam_points_for_plot=None):
    """
    Generates the Plotly figure object for the main 3D visualization in the Dash app.
    """
    import plotly.graph_objects as go

    smoothed_centerline = centerline_data.get('centerline')
    if smoothed_centerline is None:
        return go.Figure() # Return empty figure if no centerline
    smoothed_centerline = np.array(smoothed_centerline)

    plotly_traces = []
    # Mesh trace
    plotly_traces.append(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        opacity=0.6, color='lightgrey', name='Mesh', hoverinfo='none'
    ))
    # Centerline trace
    plotly_traces.append(go.Scatter3d(
        x=smoothed_centerline[:,0], y=smoothed_centerline[:,1], z=smoothed_centerline[:,2],
        mode='lines',
        line=dict(color='black', width=5),
        name='Centerline'
    ))
    # Seam points trace
    if seam_points_for_plot is not None and len(seam_points_for_plot) > 0:
        seam_points_np = np.array(seam_points_for_plot)
        plotly_traces.append(go.Scatter3d(
            x=seam_points_np[:,0], y=seam_points_np[:,1], z=seam_points_np[:,2],
            mode='markers', marker=dict(size=3, color='red', symbol='x'),
            name='Seam'
        ))
    
    # Section traces
    cmap = matplotlib.colormaps['plasma']
    if section_data_3d:
        for i, sd in enumerate(section_data_3d):
            pts3d = sd.get('points_3d')
            if pts3d is None or len(pts3d) == 0:
                continue
            
            pts3d = np.array(pts3d)
            # Close the polygon for a continuous line
            ordered_3d = np.vstack([pts3d, pts3d[0]])
            
            normp = sd['norm_pos']
            color = cmap(normp)
            rgba = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'
            
            plotly_traces.append(go.Scatter3d(
                x=ordered_3d[:,0], y=ordered_3d[:,1], z=ordered_3d[:,2],
                mode='lines',
                line=dict(color=rgba, width=8),
                name=f'Section {i} (pos {normp:.2f})'
            ))

    fig = go.Figure(data=plotly_traces)
    fig.update_layout(
        title='Interactive 3D View',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_section_montage(section_points_list, ellipse_points_for_plot_list, 
                           positions, aspect_ratios_ellipse, output_path,
                           original_points_3d_list=None,
                           pore_center_3d=None,
                           transform_matrices_list=None,
                           section_origins_3d_list=None,
                           section_normals_3d_list=None):
    """
    Create a montage of cross-sections with their fitted ellipses.
    Orients sections with their highest 2D point at the top and, if possible, seam-proximal point on the right.
    All sections are centered at (0,0) with symmetric axis limits.
    """
    from helper_functions import order_points
    import matplotlib.pyplot as plt
    import numpy as np

    valid_sections_info = []
    
    # Helper function to rotate 2D points
    def rotate_points_2d(points_2d, angle):
        """Rotate 2D points by the given angle in radians."""
        if points_2d is None or len(points_2d) == 0:
            return points_2d
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # Calculate centroid for rotation around center
        centroid = np.mean(points_2d, axis=0)
        # Center, rotate, then translate back
        centered = points_2d - centroid
        rotated = np.dot(centered, rotation_matrix.T)
        return rotated + centroid
    
    # Process each section
    for i, (orig_pts, ellipse_pts) in enumerate(zip(section_points_list, ellipse_points_for_plot_list)):
        # Skip invalid sections
        current_ar = aspect_ratios_ellipse[i]
        if hasattr(current_ar, '__iter__') and len(current_ar) > 0:
            current_ar = current_ar[0] if len(current_ar) > 0 else None
        
        if orig_pts is None or current_ar is None or not np.isfinite(current_ar) or abs(current_ar) > 30.0:
            continue
            
        # Convert to numpy arrays if they aren't already
        orig_pts = np.array(orig_pts)
        ellipse_pts = np.array(ellipse_pts) if ellipse_pts is not None else None
        
        # Center points at (0,0)
        orig_pts_processed = orig_pts.copy()
        ellipse_pts_processed = ellipse_pts.copy() if ellipse_pts is not None else None

        if len(orig_pts_processed) > 0:
            centroid_2d = np.mean(orig_pts_processed, axis=0)
            orig_pts_processed = orig_pts_processed - centroid_2d
            if ellipse_pts_processed is not None and len(ellipse_pts_processed) > 0:
                ellipse_pts_processed = ellipse_pts_processed - centroid_2d

        # --- Primary Orientation: Highest 2D point up ---
        # Find highest y point in 2D (after centering)
        highest_y_idx = np.argmax(orig_pts_processed[:, 1])
        landmark_for_primary_rotation = orig_pts_processed[highest_y_idx]
        current_angle_primary = np.arctan2(landmark_for_primary_rotation[1], landmark_for_primary_rotation[0])
        target_angle_primary = np.pi/2  # 90 degrees = top (positive Y)
        rotation_angle_primary = target_angle_primary - current_angle_primary

        # Create rotation matrix
        cos_theta_p = np.cos(rotation_angle_primary)
        sin_theta_p = np.sin(rotation_angle_primary)
        primary_rotation_matrix = np.array([
            [cos_theta_p, -sin_theta_p],
            [sin_theta_p, cos_theta_p]
        ])

        # Apply primary rotation
        orig_pts_processed = np.dot(orig_pts_processed, primary_rotation_matrix.T)
        if ellipse_pts_processed is not None:
            ellipse_pts_processed = np.dot(ellipse_pts_processed, primary_rotation_matrix.T)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(orig_pts_processed[:,0], orig_pts_processed[:,1], c=np.arange(len(orig_pts_processed)), cmap='viridis')
        plt.title(f"Section {i} after rotation")
        plt.gca().set_aspect('equal')
        plt.show()
        
        # After primary rotation, enforce consistent left/right orientation using 2D points only
        # Find the leftmost and rightmost points
        leftmost_idx = np.argmin(orig_pts_processed[:, 0])
        rightmost_idx = np.argmax(orig_pts_processed[:, 0])
        # If the leftmost point is higher than the rightmost, flip horizontally
        if orig_pts_processed[leftmost_idx, 1] > orig_pts_processed[rightmost_idx, 1]:
            orig_pts_processed[:, 0] *= -1
            if ellipse_pts_processed is not None:
                ellipse_pts_processed[:, 0] *= -1

        # --- Secondary Orientation: If possible, seam-proximal point on the right (X > 0) ---
        # Only attempt if 3D info is available and lengths match
        if (
            original_points_3d_list is not None and
            i < len(original_points_3d_list) and
            original_points_3d_list[i] is not None and
            pore_center_3d is not None
        ):
            orig_3d = original_points_3d_list[i]
            if len(orig_3d) == len(orig_pts):
                dists_to_seam = np.linalg.norm(orig_3d - pore_center_3d, axis=1)
                seam_idx = np.argmin(dists_to_seam)
                seam_point_2d = orig_pts_processed[seam_idx]
                if seam_point_2d[0] < 0:
                    orig_pts_processed[:, 0] *= -1
                    if ellipse_pts_processed is not None:
                        ellipse_pts_processed[:, 0] *= -1

        valid_sections_info.append({
            'original_points_processed': orig_pts_processed,
            'ellipse_points_processed': ellipse_pts_processed,
            'position': positions[i],
            'aspect_ratio': current_ar,
            'original_index': i
        })
    
    if not valid_sections_info:
        print("  No valid sections with ellipse data to create montage")
        return
    
    # --- Calculate global max extent for consistent axis limits ---
    all_points_for_extent = []
    for section_info in valid_sections_info:
        if section_info['original_points_processed'] is not None:
            all_points_for_extent.append(section_info['original_points_processed'])
        if section_info['ellipse_points_processed'] is not None:
            all_points_for_extent.append(section_info['ellipse_points_processed'])
    
    max_extent = 1.0  # Default if no points
    if all_points_for_extent:
        stacked_all = np.vstack(all_points_for_extent)
        max_extent = np.max(np.abs(stacked_all)) * 1.1  # 10% padding
    
    # --- Create the plot ---
    n_sections = len(valid_sections_info)
    cols = min(5, n_sections)
    rows = (n_sections + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    fig.suptitle("Cross-Sections (Oriented by Highest 2D Point, Centered)", fontsize=16)
    
    if rows == 1 and cols == 1: axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, section_info in enumerate(valid_sections_info):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        
        orig_pts = section_info['original_points_processed']
        ellipse_pts = section_info['ellipse_points_processed']
        
        if orig_pts is not None and len(orig_pts) >= 3:
            # Order points before plotting
            ordered_orig_pts = order_points(orig_pts, method="angular")
            
            # Create closed polygon
            closed_orig = np.vstack([ordered_orig_pts, ordered_orig_pts[0:1]])
            ax.plot(closed_orig[:, 0], closed_orig[:, 1], 'b-', linewidth=1.5, label='Original Section')
            ax.fill(ordered_orig_pts[:, 0], ordered_orig_pts[:, 1], alpha=0.2, color='blue')

            if ellipse_pts is not None and len(ellipse_pts) > 0:
                # Order ellipse points
                ordered_ellipse_pts = order_points(ellipse_pts, method="angular")
                closed_ellipse = np.vstack([ordered_ellipse_pts, ordered_ellipse_pts[0:1]])
                ax.plot(closed_ellipse[:, 0], closed_ellipse[:, 1], 'r--', linewidth=1.2, label='Fitted Ellipse')
            
            ax.set_title(f"Pos: {section_info['position']:.2f}\nAR (Ell): {section_info['aspect_ratio']:.2f}", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
        
        # Set symmetric axis limits
        ax.set_aspect('equal')
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.grid(True, alpha=0.3)
        
        # Add coordinate axes for reference
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        if i == 0: ax.legend(fontsize='small', loc='upper right')
    
    # Turn off unused subplots
    for i in range(n_sections, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Created section montage (oriented by highest 2D point, centered): {output_path}")

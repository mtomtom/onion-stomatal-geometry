import matplotlib
# Set a non-interactive backend BEFORE importing pyplot to prevent app crashes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io # Needed for in-memory image generation

def get_plotly_figure_for_app(mesh, centerline_data, section_data_3d, seam_points_for_plot=None):
    """
    Generates the Plotly figure object for the main 3D visualization in the Dash app.
    Handles cases where centerline or section data may not exist yet.
    """
    import plotly.graph_objects as go

    plotly_traces = []
    # Always add the mesh trace
    plotly_traces.append(go.Mesh3d(
        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
        opacity=0.6, color='lightgrey', name='Mesh', hoverinfo='none'
    ))

    # --- FIX: Only add other traces if centerline_data is available ---
    if centerline_data is not None:
        smoothed_centerline = np.array(centerline_data.get('centerline'))
        
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
                pts2d = sd.get('points_2d')
                if pts3d is None or len(pts3d) < 3 or pts2d is None or len(pts2d) != len(pts3d):
                    continue
                
                pts3d = np.array(pts3d)
                pts2d = np.array(pts2d)

                center_2d = np.mean(pts2d, axis=0)
                angles = np.arctan2(pts2d[:, 1] - center_2d[1], pts2d[:, 0] - center_2d[0])
                order = np.argsort(angles)
                ordered_3d = pts3d[order]
                ordered_3d = np.vstack([ordered_3d, ordered_3d[0]])
                
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

def create_section_montage(section_points_list, positions):
    """
    Creates a montage of 2D cross-sections for app download.
    This is the simplified version that does not require or plot ellipses.
    """
    valid_sections_info = [
        {'points': pts, 'pos': pos}
        for pts, pos in zip(section_points_list, positions)
        if pts is not None and len(pts) > 0
    ]

    if not valid_sections_info:
        return None

    n_sections = len(valid_sections_info)
    n_cols = 5
    n_rows = int(np.ceil(n_sections / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), squeeze=False)
    axes_flat = axes.flatten()

    for i, section_info in enumerate(valid_sections_info):
        ax = axes_flat[i]
        points = np.array(section_info['points'])
        
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        ordered_points = points[np.argsort(angles)]
        
        ax.plot(ordered_points[:, 0], ordered_points[:, 1], 'b-', marker='o', markersize=2)
        ax.plot([ordered_points[-1, 0], ordered_points[0, 0]], [ordered_points[-1, 1], ordered_points[0, 1]], 'b-')

        ax.set_title(f"Pos: {section_info['pos']:.2f}")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

    for i in range(n_sections, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
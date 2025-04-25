import os
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA # Import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# --- Import make_subplots ---
from plotly.subplots import make_subplots
import cross_section_functions as csf
from matplotlib.colors import LightSource # Import LightSource

# --- analyze_midpoint_cross_section function ---
def analyze_midpoint_cross_section(file_paths, output_dir=None, visualize=False):
    """
    Analyzes multiple OBJ files to extract the cross-section at the midpoint.
    Calculates the aspect ratio of the 2D cross-section.
    Generates individual 2D PNG and 3D HTML visualizations if visualize=True.

    Args:
        file_paths (list): A list of strings, each a path to an .obj file.
        output_dir (str, optional): Directory to save visualizations. Defaults to None.
        visualize (bool, optional): Whether to generate plots and scenes. Defaults to False.

    Returns:
        dict: A dictionary where keys are file paths and values are tuples
              (points_2d, points_3d, transform, aspect_ratio, pca_minor_std_dev)
              for the midpoint cross-section, or None if analysis failed.
              - points_2d: Filtered 2D points of the cross-section.
              - points_3d: Corresponding original 3D points.
              - transform: 3D to 2D transformation matrix.
              - aspect_ratio: Calculated aspect ratio (long axis / short axis).
              - pca_minor_std_dev: Standard deviation along the minor PCA axis (use as cs_b).
    """
    results = {}

    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")
        aspect_ratio = None
        pca_minor_std_dev = None # Initialize width measure
        try:
            # --- Analysis Steps 1-4 (Load, Dimensions, Midpoint, Section) ---
            # ... (Keep existing code for steps 1-4) ...
            # 1. Load Mesh and Find Center
            mesh = trimesh.load_mesh(file_path)
            center = mesh.centroid
            # 2. Determine Dimensions (Ray Casting)
            ray_count = 36
            ray_angles = np.linspace(0, 2*np.pi, ray_count, endpoint=False)
            inner_points = []
            outer_points = []
            for angle in ray_angles:
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                locations, _, _ = mesh.ray.intersects_location([center], [direction])
                if len(locations) >= 2:
                    dists = np.linalg.norm(locations - center, axis=1)
                    sorted_idx = np.argsort(dists)
                    inner_points.append(locations[sorted_idx[0]])
                    outer_points.append(locations[sorted_idx[-1]])
            if not inner_points or not outer_points:
                 print("  Error: Could not determine dimensions via ray casting.")
                 results[file_path] = None
                 continue
            inner_points = np.array(inner_points)
            outer_points = np.array(outer_points)
            minor_radius = np.mean(np.linalg.norm(outer_points - inner_points, axis=1)) / 2
            # 3. Estimate Midpoint and Tangent
            zero_angle_indices = np.where(np.abs(ray_angles) < np.pi/ray_count)[0]
            if len(zero_angle_indices) > 0 and len(inner_points) > 0 and len(outer_points) > 0:
                 mid_inner = np.mean(inner_points[zero_angle_indices], axis=0)
                 mid_outer = np.mean(outer_points[zero_angle_indices], axis=0)
                 midpoint = (mid_inner + mid_outer) / 2
            else:
                 major_radius_est = np.mean(np.linalg.norm(inner_points[:,:2] - center[:2], axis=1)) + minor_radius if len(inner_points) > 0 else minor_radius * 2
                 midpoint = center + np.array([major_radius_est, 0, 0])
            tangent = np.array([0.0, 1.0, 0.0])
            # 4. Take Cross-Section
            section = mesh.section(plane_origin=midpoint, plane_normal=tangent)
            original_section_points_3d = section.vertices.copy() if section is not None else None
            if section is None or len(section.entities) == 0:
                print("  Midpoint section failed or is empty.")
                results[file_path] = None
                continue

            # 5. Process and Filter Section
            path_2D, transform = section.to_2D()
            points_2D = path_2D.vertices
            midpoint_h = np.append(midpoint, 1)
            midpoint_transformed_h = transform @ midpoint_h
            midpoint_2d_target = midpoint_transformed_h[:2]
            # eps_value = minor_radius * 0.3 # Original
            eps_value = minor_radius * 0.15 # Adjusted value
            clustering = DBSCAN(eps=eps_value, min_samples=3).fit(points_2D)
            labels = clustering.labels_
            unique_labels = np.unique(labels)
            valid_labels = unique_labels[unique_labels != -1]
            final_points_2D = np.empty((0, 2))
            final_original_points_3D = np.empty((0, 3))
            best_label = -1
            # ... (DBSCAN filtering logic remains the same) ...
            if len(valid_labels) > 1:
                min_avg_dist = float('inf')
                for label in valid_labels:
                    label_mask = (labels == label)
                    cluster_points_2d = points_2D[label_mask]
                    if len(cluster_points_2d) < 3: continue
                    distances = np.linalg.norm(cluster_points_2d - midpoint_2d_target, axis=1)
                    avg_distance = np.mean(distances)
                    if avg_distance < min_avg_dist:
                        min_avg_dist = avg_distance
                        best_label = label
                if best_label != -1:
                     final_mask = (labels == best_label)
                     final_points_2D = points_2D[final_mask]
                     if original_section_points_3d is not None and len(original_section_points_3d) == len(points_2D):
                         final_original_points_3D = original_section_points_3d[final_mask]
                     else: final_original_points_3D = np.empty((0, 3))
            elif len(valid_labels) == 1:
                best_label = valid_labels[0]
                final_mask = (labels == best_label)
                final_points_2D = points_2D[final_mask]
                if original_section_points_3d is not None and len(original_section_points_3d) == len(points_2D):
                    final_original_points_3D = original_section_points_3d[final_mask]
                else: final_original_points_3D = np.empty((0, 3))
            else:
                print("  No valid clusters found in midpoint section.")


            if len(final_points_2D) != len(final_original_points_3D) and len(final_original_points_3D) > 0:
                 print(f"  WARNING: Final 2D ({len(final_points_2D)}) and 3D ({len(final_original_points_3D)}) point counts differ.")

            # --- Calculate Aspect Ratio and PCA Minor Axis Std Dev ---
            if len(final_points_2D) >= 3: # Need at least 3 points for PCA
                try:
                    pca = PCA(n_components=2)
                    pca.fit(final_points_2D)
                    # Explained variance gives variance along principal axes
                    # Standard deviation is sqrt of variance
                    std_devs = np.sqrt(pca.explained_variance_)
                    # Ensure std_devs[1] is not zero to avoid division error
                    if std_devs[1] > 1e-6: # Check against a small epsilon
                        aspect_ratio = std_devs[0] / std_devs[1]
                        pca_minor_std_dev = std_devs[1] # <<< STORE THE MINOR AXIS STD DEV
                        print(f"  PCA Results: Aspect Ratio={aspect_ratio:.3f}, Minor Axis StdDev={pca_minor_std_dev:.3f}")
                    else:
                        print("  Warning: Short axis standard deviation is near zero. Cannot calculate AR or width.")
                        aspect_ratio = np.inf # Or None, depending on how you want to handle it
                        pca_minor_std_dev = 0.0 # Or None
                except Exception as pca_err:
                    print(f"  Error calculating PCA results: {pca_err}")
                    aspect_ratio = None
                    pca_minor_std_dev = None
            else:
                 if len(final_points_2D) > 0:
                     print("  Not enough points (<3) to calculate aspect ratio.")
                 aspect_ratio = None
                 pca_minor_std_dev = None


            # --- Store results (including pca_minor_std_dev) ---
            if len(final_points_2D) > 0:
                print(f"  Successfully extracted midpoint section with {len(final_points_2D)} points.")
                # <<< Return pca_minor_std_dev instead of minor_radius >>>
                results[file_path] = (final_points_2D, final_original_points_3D, transform, aspect_ratio, pca_minor_std_dev)

                # --- Individual Visualization (Step 6) ---
                if visualize:
                    # ... (Keep the existing visualization code here) ...
                    # (No changes needed in the visualization part itself)
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    # --- Create and Save Individual 2D Plot ---
                    fig_2d, ax2d = plt.subplots(figsize=(6, 6))
                    ax2d.set_title(f'2D Midpoint Cross-Section\n{base_name}')
                    ax2d.set_aspect('equal')
                    ax2d.grid(True)
                    ordered_plot_points = csf.order_points(final_points_2D, method="angular")
                    ax2d.plot(np.append(ordered_plot_points[:, 0], ordered_plot_points[0, 0]), np.append(ordered_plot_points[:, 1], ordered_plot_points[0, 1]), 'b-', linewidth=1.5)
                    ax2d.plot(ordered_plot_points[:, 0], ordered_plot_points[:, 1], 'b.', markersize=4)
                    ax2d.plot(midpoint_2d_target[0], midpoint_2d_target[1], 'ro', markersize=6, label='Target Center (2D)')
                    # Add aspect ratio to 2D plot title if calculated
                    title_str = f'2D Midpoint Cross-Section\n{base_name}'
                    if aspect_ratio is not None and np.isfinite(aspect_ratio):
                        title_str += f'\nAR: {aspect_ratio:.3f}'
                    if pca_minor_std_dev is not None:
                         title_str += f', Width(b): {pca_minor_std_dev:.3f}' # Add width info
                    ax2d.set_title(title_str)
                    ax2d.legend()
                    plt.tight_layout()
                    if output_dir:
                        save_path_2d = os.path.join(output_dir, f'{base_name}_midpoint_section_2D.png')
                        plt.savefig(save_path_2d, dpi=150)
                        print(f"  Saved 2D visualization to {save_path_2d}")
                    plt.close(fig_2d)
                    # --- Create and Save Individual 3D Plotly Scene ---
                    plotly_traces = []
                    vertices = mesh.vertices
                    faces = mesh.faces
                    mesh_trace = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=0.5, color='green', name='Mesh')
                    plotly_traces.append(mesh_trace)
                    midpoint_trace = go.Scatter3d(x=[midpoint[0]], y=[midpoint[1]], z=[midpoint[2]], mode='markers', marker=dict(size=5, color='red'), name='Est. Midpoint')
                    plotly_traces.append(midpoint_trace)
                    if len(final_original_points_3D) > 0:
                        center_2d_final = np.mean(final_points_2D, axis=0)
                        centered_2d_final = final_points_2D - center_2d_final
                        angles_final = np.arctan2(centered_2d_final[:, 1], centered_2d_final[:, 0])
                        sorted_indices_final = np.argsort(angles_final)
                        ordered_points_3d = final_original_points_3D[sorted_indices_final]
                        closed_points_3d = np.vstack([ordered_points_3d, ordered_points_3d[0]])
                        section_line_trace = go.Scatter3d(x=closed_points_3d[:, 0], y=closed_points_3d[:, 1], z=closed_points_3d[:, 2], mode='lines', line=dict(color='blue', width=5), name='Section Outline')
                        plotly_traces.append(section_line_trace)
                        section_points_trace = go.Scatter3d(x=final_original_points_3D[:, 0], y=final_original_points_3D[:, 1], z=final_original_points_3D[:, 2], mode='markers', marker=dict(size=3, color='blue'), name='Section Points', showlegend=False)
                        plotly_traces.append(section_points_trace)
                    plane_size = minor_radius * 1.5
                    v1 = np.cross(tangent, np.array([0, 0, 1]) if not np.allclose(tangent, [0,0,1]) else np.array([0,1,0]))
                    v1 /= np.linalg.norm(v1)
                    v2 = np.cross(tangent, v1)
                    v2 /= np.linalg.norm(v2)
                    xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 5), np.linspace(-plane_size, plane_size, 5))
                    plane_points_x = midpoint[0] + v1[0] * xx + v2[0] * yy
                    plane_points_y = midpoint[1] + v1[1] * xx + v2[1] * yy
                    plane_points_z = midpoint[2] + v1[2] * xx + v2[2] * yy
                    plane_trace = go.Surface(x=plane_points_x, y=plane_points_y, z=plane_points_z, colorscale=[[0, 'rgba(255,0,255,0.5)'], [1, 'rgba(255,0,255,0.5)']], showscale=False, opacity=0.5, name='Section Plane')
                    plotly_traces.append(plane_trace)
                    fig_3d = go.Figure(data=plotly_traces)
                    fig_3d.update_layout(title=f'3D Visualization - Midpoint Section<br>{base_name}', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=40))
                    if output_dir:
                         save_path_3d_html = os.path.join(output_dir, f'{base_name}_midpoint_scene.html')
                         try: fig_3d.write_html(save_path_3d_html); print(f"  Saved 3D HTML scene to {save_path_3d_html}")
                         except Exception as export_err: print(f"  Failed to export 3D HTML scene: {export_err}")

            else: # If len(final_points_2D) == 0 and aspect_ratio is None
                if aspect_ratio is None and len(points_2D) > 0: # Check if filtering failed but section existed
                     print("  Midpoint section valid but resulted in 0 points after filtering.")
                # Store None if no valid points were found
                results[file_path] = None


        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            results[file_path] = None

    return results

# --- create_combined_midpoint_2d_plot function ---
# (No changes needed here, it uses the results dictionary)
def create_combined_midpoint_2d_plot(results_data, output_path):
    # ... (Keep existing code for combined plot) ...
    print(f"\nCreating combined 2D overlay plot at: {output_path}")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Combined Midpoint Cross-Sections (Centered)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("X (centered)")
    ax.set_ylabel("Y (centered)")
    valid_files = [fp for fp, data in results_data.items() if data is not None and len(data[0]) > 0]
    if not valid_files:
        print("  No valid midpoint sections found to plot.")
        plt.close(fig)
        return
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_files)))
    max_extent = 0
    for i, file_path in enumerate(valid_files):
        points_2d = results_data[file_path][0]
        center_pt = np.mean(points_2d, axis=0)
        centered_points = points_2d - center_pt
        ordered_points = csf.order_points(centered_points, method="angular")
        ax.plot(np.append(ordered_points[:, 0], ordered_points[0, 0]), np.append(ordered_points[:, 1], ordered_points[0, 1]), '-', color=colors[i], linewidth=2, alpha=0.7, label=os.path.basename(file_path))
        max_extent = max(max_extent, np.max(np.abs(ordered_points)))
    limit = max_extent * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    if len(valid_files) <= 10:
        ax.legend(loc='upper right', fontsize='small')
    else:
        print("  Legend omitted due to large number of files.")
    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved combined 2D plot.")
    except Exception as e:
        print(f"  Error saving combined 2D plot: {e}")
    finally:
        plt.close(fig)


# --- NEW Function to create aspect ratio box plot ---
def create_aspect_ratio_boxplot(aspect_ratios, output_path):
    """
    Creates a box plot of the calculated aspect ratios, overlaying individual points.

    Args:
        aspect_ratios (list): A list of valid, finite aspect ratio values.
        output_path (str): Full path to save the box plot PNG image.
    """
    if not aspect_ratios:
        print("  No valid aspect ratios to create a box plot.")
        return

    print(f"\nCreating aspect ratio box plot with points at: {output_path}")
    fig, ax = plt.subplots(figsize=(6, 6))

    # --- Add jittered scatter plot for individual points ---
    # Generate x-coordinates with jitter around position 1
    jitter_strength = 0.08 # Adjust as needed
    x_jitter = np.random.normal(1, jitter_strength, size=len(aspect_ratios))

    # --- Create the box plot ---
    # showfliers=False can be used if scatter shows all points, but keep True for standard box plot look
    bp = ax.boxplot(aspect_ratios, vert=True, patch_artist=True, showmeans=False,
                    positions=[1], # Explicitly set position to 1
                    widths=0.5,    # Adjust width if needed
                    showfliers=True, # Keep showing standard outliers
                    boxprops=dict(facecolor='lightblue', alpha=0.8, zorder=2), # alpha for transparency, zorder=2 to plot over scatter
                    medianprops=dict(color='red', linewidth=2, zorder=3),
                    whiskerprops=dict(zorder=2),
                    capprops=dict(zorder=2),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='orange', markeredgecolor='grey', zorder=2) # Style outliers
                   )
     # Plot the points
    ax.scatter(x_jitter, aspect_ratios, alpha=1.0, s=20, color='red', zorder=4, label='Individual Sections') # zorder=1 to plot behind box
    ax.set_ylabel('Aspect Ratio (Long Axis / Short Axis)')
    ax.set_title(f'Distribution of Midpoint Aspect Ratios (N={len(aspect_ratios)})')
    ax.set_xticks([1]) # Set tick position
    ax.set_xticklabels(['Midpoint Sections']) # Label the tick
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Optional: Add legend if desired, though often clear from context
    # ax.legend()

    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved aspect ratio box plot with points.")
    except Exception as e:
        print(f"  Error saving aspect ratio box plot: {e}")
    finally:
        plt.close(fig)


# --- Function to create mesh grid plot (MATPLOTLIB Version) ---
def create_mesh_grid_plot(file_paths, output_path, rows=3, cols=6):
    """
    Creates a single image file (e.g., PNG) with all meshes in a grid layout
    using Matplotlib's 3D plotting.
    """
    num_meshes = len(file_paths)
    if num_meshes == 0:
        print("No mesh files provided...")
        return
    num_to_process = min(num_meshes, rows * cols)
    if num_meshes > rows * cols:
        print(f"Warning: Number of meshes ({num_meshes}) exceeds grid size ({rows}x{cols}). Displaying first {num_to_process}.")

    print(f"\nCreating Matplotlib mesh grid plot ({rows}x{cols}) at: {output_path}")

    # --- Data Preparation: Load, Align, Collect Vertices ---
    mesh_data_for_plotting = []
    all_vertices_aligned = []
    print("  Preprocessing meshes for Matplotlib plot...")
    for i in range(rows * cols): # Prepare data for all potential cells
        mesh_info = {'vertices': None, 'faces': None, 'error': 'Empty Cell', 'basename': None}
        if i < num_meshes:
            file_path = file_paths[i]
            base_name = os.path.basename(file_path)
            mesh_info['basename'] = base_name
            mesh_info['error'] = f'Error: {base_name}' # Default error

            try:
                # Load mesh
                mesh = trimesh.load_mesh(file_path, process=False)
                if isinstance(mesh, trimesh.Scene): mesh = mesh.dump(concatenate=True)

                if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces') or len(mesh.vertices) == 0:
                    print(f"    ERROR: Invalid mesh data for {base_name}")
                    mesh_info['error'] = f'Invalid Mesh: {base_name}'
                else:
                    vertices = mesh.vertices; faces = mesh.faces
                    # Center mesh
                    center = vertices.mean(axis=0); vertices_centered = vertices - center
                    # Align mesh using PCA
                    vertices_aligned = vertices_centered
                    if len(vertices_centered) >= 3:
                        try:
                            pca = PCA(n_components=3); pca.fit(vertices_centered)
                            principal_axes = pca.components_; longest_axis = principal_axes[0]
                            target_axis = np.array([0.0, 1.0, 0.0]) # Align longest axis with Y
                            rotation_matrix = trimesh.geometry.align_vectors(longest_axis, target_axis)
                            vertices_aligned = trimesh.transform_points(vertices_centered, rotation_matrix)
                        except Exception as e: print(f"    Warning: PCA alignment failed for {base_name}: {e}")

                    mesh_info['vertices'] = vertices_aligned
                    mesh_info['faces'] = faces
                    mesh_info['error'] = None # Success for this mesh
                    all_vertices_aligned.append(vertices_aligned) # Collect for bounds calculation
                    print(f"    Prepared {base_name} successfully")

            except Exception as e:
                print(f"    ERROR processing {base_name} for plot: {e}")
                mesh_info['error'] = f'Processing Error: {base_name}'

        mesh_data_for_plotting.append(mesh_info)
    # --- End Data Preparation ---

    if not all_vertices_aligned:
        print("  No valid mesh data collected to create Matplotlib plot.")
        return

    # --- Calculate Overall Bounds for Consistent Axis Limits ---
    all_verts_np = np.concatenate(all_vertices_aligned, axis=0)
    min_bounds = np.min(all_verts_np, axis=0)
    max_bounds = np.max(all_verts_np, axis=0)
    center_bounds = (min_bounds + max_bounds) / 2
    # Determine the largest range along any axis and add padding
    max_range = np.max(max_bounds - min_bounds) * 0.3 # Use 60% of max range for padding

    plot_limits = [
        (center_bounds[0] - max_range, center_bounds[0] + max_range),
        (center_bounds[1] - max_range, center_bounds[1] + max_range),
        (center_bounds[2] - max_range, center_bounds[2] + max_range),
    ]
    print(f"  Calculated plot limits: X={plot_limits[0]}, Y={plot_limits[1]}, Z={plot_limits[2]}")

    # --- Create Matplotlib Figure and Subplots ---
    fig = plt.figure(figsize=(cols * 5, rows * 5)) # Increased multiplier from 2.5 to 3
    fig.patch.set_facecolor('white') # Set background of the whole figure to white

    ls = LightSource(azdeg=315, altdeg=60)

    processed_count = 0
    print("  Generating Matplotlib subplots...")
    for i in range(rows * cols):
        plot_index = i + 1
        # Add subplot with 3D projection
        ax = fig.add_subplot(rows, cols, plot_index, projection='3d', computed_zorder=False)
        ax.set_facecolor('white') # Set background of each subplot

        mesh_info = mesh_data_for_plotting[i]

        if mesh_info and mesh_info['error'] is None:
            vertices = mesh_info['vertices']
            faces = mesh_info['faces']
            # --- Apply light source and change color ---
            # Define color
            rgb = plt.cm.Greys(0.85) # Use a slightly darker grey from colormap
            # Plot the mesh using plot_trisurf with lightsource

            # Plot the mesh using plot_trisurf
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces,
                color=rgb,         # Use the defined color
                lightsource=ls,    
                shade=True, 
                edgecolor=None,
                linewidth=0,       # Line width for edges if shown
                antialiased=True, # Smoother rendering 
                zorder=1
            )
            processed_count += 1
        elif mesh_info and mesh_info['error'] and mesh_info['error'] != 'Empty Cell':
             # Display error message ONLY if it's not the default 'Empty Cell'
             ax.text(0.5, 0.5, 0.5, mesh_info['error'], ha='center', va='center', transform=ax.transAxes, fontsize=8, color='red', wrap=True)

        # --- Configure Axes Appearance ---
        # Set consistent limits for all axes
        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])
        ax.set_zlim(plot_limits[2])

        # Set aspect ratio to be equal based on limits (important!)
        ax.set_box_aspect([1,1,1]) # Forces cubic aspect ratio for the axes box

        # Set view angle (elevation 90 = top-down, azim -90 aligns Y vertically)
        ax.view_init(elev=90, azim=-90)

        # Hide grid, ticks, and labels for a cleaner look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Optionally hide the axis panes/box completely
        ax.set_axis_off()

    # Add overall title and adjust layout
    #fig.suptitle("Onion confocal images", fontsize=16) # Slightly larger title
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95, wspace=0, hspace=0) # Minimal margins, keep space for title

    # --- Save the Figure ---
    try:
        plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor()) # Added bbox_inches and pad_inches
        print(f"\nProcessed {processed_count} meshes successfully for Matplotlib grid plot.")
        print(f"Saved Matplotlib mesh grid visualization to {output_path}")
    except Exception as e:
        print(f"\nError writing Matplotlib file {output_path}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the list of mesh files to process
    files_to_process = [
        "Meshes/OBJ/Ac_DA_1_3.obj", "Meshes/OBJ/Ac_DA_1_2.obj", "Meshes/OBJ/Ac_DA_1_5.obj",
        "Meshes/OBJ/Ac_DA_1_4.obj", "Meshes/OBJ/Ac_DA_3_7.obj", "Meshes/OBJ/Ac_DA_3_6.obj",
        "Meshes/OBJ/Ac_DA_3_4.obj", "Meshes/OBJ/Ac_DA_3_3.obj", "Meshes/OBJ/Ac_DA_3_2.obj",
        "Meshes/OBJ/Ac_DA_3_1.obj", "Meshes/OBJ/Ac_DA_2_7.obj", "Meshes/OBJ/Ac_DA_2_6b.obj",
        "Meshes/OBJ/Ac_DA_2_6a.obj", "Meshes/OBJ/Ac_DA_2_4.obj", "Meshes/OBJ/Ac_DA_2_3.obj",
        "Meshes/OBJ/Ac_DA_1_8_mesh.obj", "Meshes/OBJ/Ac_DA_1_6.obj"
    ]

    # Check if files exist
    if not all(os.path.exists(f) for f in files_to_process):
         print("Error: One or more specified files do not exist. Please check paths.")
         print("Files expected:", files_to_process)
    else:
        # Define output directory
        output_directory = "midpoint_analysis_results"
        os.makedirs(output_directory, exist_ok=True) # Ensure directory exists

        # --- 1. Run Midpoint Analysis ---
        # Set visualize=True to generate individual plots during analysis
        midpoint_data = analyze_midpoint_cross_section(
            files_to_process,
            output_dir=output_directory,
            visualize=True
        )

        # --- 2. Create Combined 2D Plot ---
        combined_plot_path = os.path.join(output_directory, "combined_midpoint_overlay_2D.png")
        create_combined_midpoint_2d_plot(midpoint_data, combined_plot_path)

        # --- 3. Create Aspect Ratio Box Plot ---
        # Extract valid aspect ratios from results
        valid_aspect_ratios = []
        # --- ALSO EXTRACT WIDTHS FOR A SEPARATE PLOT (Optional) ---
        valid_widths = []
        if midpoint_data:
            for file_path, data in midpoint_data.items():
                # Data is tuple: (pts2d, pts3d, transform, aspect_ratio, pca_minor_std_dev)
                if data and len(data) > 4: # Check length >= 5
                    ar = data[3]
                    width = data[4]
                    if ar is not None and np.isfinite(ar):
                        valid_aspect_ratios.append(ar)
                    if width is not None and np.isfinite(width) and width > 1e-9:
                         valid_widths.append(width)

        if valid_aspect_ratios:
            boxplot_path_ar = os.path.join(output_directory, "aspect_ratio_boxplot.png")
            create_aspect_ratio_boxplot(valid_aspect_ratios, boxplot_path_ar) # Pass only ARs
        else: print("\nNo valid aspect ratios found.")

        # --- Optional: Create Box Plot for Widths ---
        if valid_widths:
             boxplot_path_width = os.path.join(output_directory, "pca_width_boxplot.png")
             # You might need a modified boxplot function or just reuse it
             # Reusing create_aspect_ratio_boxplot for widths:
             create_aspect_ratio_boxplot(valid_widths, boxplot_path_width)
             # Adjust title/labels inside the function if modifying, or accept generic labels
             print(f"Saved PCA width box plot to {boxplot_path_width}")
        else: print("\nNo valid PCA widths found.")
        # --- End Optional Width Plot ---


        # --- 4. Create Input Mesh Grid Plot ---
        mesh_grid_path = os.path.join(output_directory, "input_mesh_grid_matplotlib.png") # Change extension
        # Call the MATPLOTLIB version of the grid plot function
        create_mesh_grid_plot(files_to_process, mesh_grid_path, rows=3, cols=6)

        print("\n--- Analysis and Visualization Complete ---")
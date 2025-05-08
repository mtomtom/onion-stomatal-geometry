import numpy as np
import os
import traceback
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import trimesh
from sklearn.cluster import DBSCAN # Kept for DummyCSF, though main usage is via csf
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots # Not explicitly used, but good to keep if future plans
from matplotlib.colors import LightSource


# Assuming cross_section_functions (csf) is in the same directory or Python path
try:
    import cross_section_functions as csf
except ImportError:
    print("CRITICAL ERROR: cross_section_functions.py (csf) not found. This script relies heavily on it.")
    print("Please ensure cross_section_functions.py is in the same directory or accessible in PYTHONPATH.")
    # Define dummy csf to allow script to be parsed, but it will not function correctly.
    class DummyCSF:
        def order_points(self, points, method="angular", center=None):
            print("Dummy csf.order_points called. THIS WILL NOT WORK CORRECTLY.")
            if not isinstance(points, np.ndarray): points = np.array(points)
            if points.ndim == 1: points = points.reshape(-1,2) # Basic handling
            if len(points) == 0: return np.array([])
            if center is None and len(points) > 0: center = np.mean(points, axis=0)
            elif center is None: center = np.array([0,0])
            angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
            return points[np.argsort(angles)]

        def load_and_align_mesh(self, fp, align_axis='Y'):
            print("Dummy csf.load_and_align_mesh called. THIS WILL NOT WORK CORRECTLY.")
            try: # Attempt a basic load for minimal script execution
                m = trimesh.load_mesh(fp)
                if isinstance(m, trimesh.Scene): m = m.dump(concatenate=True)
                if m.is_empty: return None, None
                m.vertices -= m.centroid # Center it
                # Simplified PCA alignment for dummy
                if len(m.vertices) > 3:
                    pca = PCA(n_components=3)
                    pca.fit(m.vertices)
                    # Align longest axis (first principal component) with target_axis
                    longest_axis_vec = pca.components_[0]
                    target_axis_map = {'X': [1,0,0], 'Y': [0,1,0], 'Z': [0,0,1]}
                    target_axis_vec = np.array(target_axis_map.get(align_axis, [0,1,0]))
                    
                    # Rotation matrix to align longest_axis_vec to target_axis_vec
                    # This is a simplified version, real alignment is more complex
                    # For dummy, just ensure it runs without error
                    # For a robust dummy, trimesh.geometry.align_vectors would be better
                    # but trying to keep dummy simple.
                    # This dummy alignment might not be perfect or even correct.
                return m, np.eye(4) # Return identity transform
            except Exception as e:
                print(f"Dummy csf.load_and_align_mesh error: {e}")
                return None, None

        def get_radial_dimensions(self, m, center=None, ray_count=36):
            print("Dummy csf.get_radial_dimensions called. THIS WILL NOT WORK CORRECTLY.")
            # Return plausible dummy data
            if m is None or m.is_empty: return None,None,None,None
            raw_c_pts = np.array([[0,y,0] for y in np.linspace(m.bounds[0,1], m.bounds[1,1], 10)])
            avg_mr = np.mean(m.extents[m.extents > 1e-6]) / 4 if m.extents is not None else 0.1
            return None, None, raw_c_pts, avg_mr

        def filter_section_points(self, points_2D, minor_radius, origin_2d_target, eps_factor=0.20, min_samples=3):
            print("Dummy csf.filter_section_points called. THIS WILL NOT WORK CORRECTLY.")
            if points_2D is None or len(points_2D) < min_samples:
                return np.empty((0,2))
            # Simplistic pass-through for dummy, maybe select largest cluster if DBSCAN runs
            try:
                if len(points_2D) >= min_samples: # DBSCAN needs enough samples
                    clustering = DBSCAN(eps=minor_radius * eps_factor, min_samples=min_samples).fit(points_2D)
                    labels = clustering.labels_
                    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                    if len(unique_labels) > 0:
                        best_label = unique_labels[np.argmax(counts)]
                        return points_2D[labels == best_label]
            except Exception: # Ignore errors in dummy DBSCAN
                pass
            return points_2D # Fallback pass-through

    csf = DummyCSF()
    # Consider re-raising the error if csf is absolutely essential for any execution:
    # raise ImportError("cross_section_functions.py (csf) not found.")


# --- analyze_midpoint_cross_section function ---
def analyze_midpoint_cross_section(file_paths, output_dir=None, visualize=False):
    """
    Analyzes multiple OBJ files to extract the cross-section at the midpoint
    of an aligned mesh. Calculates aspect ratio and PCA-based width.
    Generates individual 2D PNG and 3D HTML visualizations if visualize=True.

    Returns:
        dict: Keys are file paths, values are tuples
              (final_points_2D, final_section_vertices_3d_aligned,
               transform_2d_to_aligned_3d, aspect_ratio, pca_minor_std_dev)
              or None if analysis failed.
              - final_points_2D: Filtered 2D points of the cross-section.
              - final_section_vertices_3d_aligned: Corresponding 3D points in aligned mesh coordinates.
              - transform_2d_to_aligned_3d: Transform from 2D section to 3D aligned space.
    """
    results = {}
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}")
        aspect_ratio = None
        pca_minor_std_dev = None
        try:
            # 1. Load and Align Mesh
            aligned_mesh, _ = csf.load_and_align_mesh(file_path, align_axis='Y')
            if aligned_mesh is None or aligned_mesh.is_empty:
                print(f"  Error: Failed to load/align mesh or mesh is empty: {file_path}.")
                results[file_path] = None
                continue

            # 2. Determine Dimensions (e.g., minor_radius for filtering)
            _, _, _, minor_radius = csf.get_radial_dimensions(
                aligned_mesh, center=np.array([0.0, 0.0, 0.0]), ray_count=36
            )

            if minor_radius is None or not np.isfinite(minor_radius) or minor_radius <= 1e-6:
                print(f"  Warning: csf.get_radial_dimensions failed for {file_path}. Attempting fallback minor_radius.")
                if aligned_mesh.extents is not None and \
                   np.all(np.isfinite(aligned_mesh.extents)) and \
                   len(aligned_mesh.extents) == 3 and \
                   np.all(aligned_mesh.extents > 1e-6) :
                    estimated_minor_radius_fallback = min(aligned_mesh.extents[0], aligned_mesh.extents[2]) / 2.0
                    if estimated_minor_radius_fallback > 1e-6:
                        minor_radius = estimated_minor_radius_fallback
                        print(f"  Used fallback minor_radius from extents: {minor_radius:.3f}")
                    else:
                        print(f"  Error: Fallback minor_radius from extents is also invalid for {file_path}.")
                        results[file_path] = None
                        continue
                else:
                    print(f"  Error: Cannot determine minor_radius for {file_path} (extents: {aligned_mesh.extents}).")
                    results[file_path] = None
                    continue

            # 3. Define Midpoint Plane for Section
            midpoint_plane_origin = np.array([0.0, 0.0, 0.0])
            midpoint_plane_normal = np.array([0.0, 1.0, 0.0])

            # 4. Take Cross-Section from the ALIGNED mesh
            section_3d_obj_aligned = aligned_mesh.section(plane_origin=midpoint_plane_origin, plane_normal=midpoint_plane_normal)

            if section_3d_obj_aligned is None or len(section_3d_obj_aligned.entities) == 0:
                print("  Midpoint section (after alignment) failed or is empty.")
                results[file_path] = None
                continue

            # 5. Process Section to get 2D points
            path_2D, transform_2d_to_aligned_3d = section_3d_obj_aligned.to_2D()

            # --- Start: New logic to extract initial 2D points for filtering ---
            points_2D_for_filtering = np.empty((0,2))
            processed_paths_list = []

            if path_2D.polygons_closed is not None and len(path_2D.polygons_closed) > 0:
                for poly_verts in path_2D.polygons_closed:
                    if isinstance(poly_verts, np.ndarray) and \
                       poly_verts.ndim == 2 and poly_verts.shape[1] == 2 and \
                       poly_verts.shape[0] >= 3:
                        processed_paths_list.append(poly_verts)
            
            if not processed_paths_list and path_2D.discrete is not None and len(path_2D.discrete) > 0:
                for line_verts in path_2D.discrete:
                    if isinstance(line_verts, np.ndarray) and \
                       line_verts.ndim == 2 and line_verts.shape[1] == 2 and \
                       line_verts.shape[0] >= 3:
                        processed_paths_list.append(line_verts)
                if processed_paths_list:
                    print("  Used discrete paths as no closed polygons were found for initial selection.")
            
            if processed_paths_list:
                points_2D_for_filtering = max(processed_paths_list, key=len)
                print(f"  Selected longest path/polygon with {len(points_2D_for_filtering)} vertices from {len(processed_paths_list)} candidates for filtering.")
            elif path_2D.vertices is not None and len(path_2D.vertices) >=3 :
                print("  Warning: No closed polygons or discrete paths found. Falling back to raw path_2D.vertices for filtering.")
                points_2D_for_filtering = path_2D.vertices
            else:
                print("  Section to_2D resulted in no usable polygons, discrete paths, or raw vertices.")
                results[file_path] = None
                continue
            # --- End: New logic to extract initial 2D points ---

            if points_2D_for_filtering is None or len(points_2D_for_filtering) < 3:
                print("  Initial selected 2D points for filtering are insufficient.")
                results[file_path] = None
                continue

            # 6. Filter the selected 2D points using csf.filter_section_points
            midpoint_2d_target = np.array([0.0, 0.0]) # Section origin projects to (0,0) in 2D
            final_points_2D = csf.filter_section_points(
                points_2D_for_filtering,
                minor_radius,
                midpoint_2d_target,
                eps_factor=0.25, # Keep the 0.15 factor as previously used in this script
                min_samples=3
            )

            if final_points_2D is None or len(final_points_2D) < 3:
                print(f"  Filtered section has too few points ({len(final_points_2D if final_points_2D is not None else 0)}).")
                results[file_path] = None
                continue

            # 7. Transform chosen 2D points back to 3D aligned space
            final_section_vertices_3d_aligned = np.empty((0,3))
            if transform_2d_to_aligned_3d is not None and len(final_points_2D) > 0:
                points_in_2d_plane_for_transform = np.column_stack((final_points_2D, np.zeros(len(final_points_2D))))
                final_section_vertices_3d_aligned = trimesh.transform_points(points_in_2d_plane_for_transform, transform_2d_to_aligned_3d)
            else:
                if len(final_points_2D) > 0: # Only warn if there were points to transform
                    print("  Warning: Cannot transform final_points_2D to 3D due to missing transform_2d_to_aligned_3d.")
            
            if len(final_section_vertices_3d_aligned) == 0 and len(final_points_2D) > 0:
                 print(f"  WARNING: Failed to get 3D points for the {len(final_points_2D)} 2D points.")


            # 8. Calculate Aspect Ratio and PCA Minor Axis Std Dev from final_points_2D
            if len(final_points_2D) >= 3:
                try:
                    pca = PCA(n_components=2)
                    pca.fit(final_points_2D)
                    std_devs = np.sqrt(pca.explained_variance_)
                    # Ensure std_devs has two components and minor axis is positive
                    if len(std_devs) == 2 and std_devs[1] > 1e-9: # Increased tolerance slightly
                        aspect_ratio = std_devs[0] / std_devs[1]
                        pca_minor_std_dev = std_devs[1]
                    elif len(std_devs) == 1: # Only one principal component (e.g. points are collinear)
                        print("  PCA: Points appear collinear, cannot calculate aspect ratio from two std devs.")
                        aspect_ratio = float('inf') # Or some other indicator
                        pca_minor_std_dev = 0.0
                    else: # std_devs[1] is too small or other issues
                        print(f"  PCA: Minor axis standard deviation is too small ({std_devs[1] if len(std_devs)==2 else 'N/A'}) or PCA failed to yield two components.")
                        aspect_ratio = None # Keep as None
                        pca_minor_std_dev = std_devs[1] if len(std_devs)==2 else None
                except Exception as pca_err:
                    print(f"  Error calculating PCA results: {pca_err}")
                    aspect_ratio = None; pca_minor_std_dev = None
            else:
                 aspect_ratio = None; pca_minor_std_dev = None


            # 9. Store results
            print(f"  Successfully processed midpoint section with {len(final_points_2D)} 2D points.")
            results[file_path] = (final_points_2D, final_section_vertices_3d_aligned,
                                  transform_2d_to_aligned_3d, aspect_ratio, pca_minor_std_dev)

            # 10. Visualization
            if visualize and output_dir:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                # --- Create and Save Individual 2D Plot ---
                fig_2d, ax2d = plt.subplots(figsize=(6, 6))
                ax2d.set_aspect('equal'); ax2d.grid(True)
                # Order points for plotting using their own 2D coordinate system's origin/centroid
                ordered_plot_points_2d = csf.order_points(final_points_2D, method="nearest", center=midpoint_2d_target)
                
                if len(ordered_plot_points_2d) > 0: # Check if there are points to plot
                    ax2d.plot(np.append(ordered_plot_points_2d[:, 0], ordered_plot_points_2d[0, 0]),
                              np.append(ordered_plot_points_2d[:, 1], ordered_plot_points_2d[0, 1]), 'b-', linewidth=1.5)
                    ax2d.plot(ordered_plot_points_2d[:, 0], ordered_plot_points_2d[:, 1], 'b.', markersize=4)
                ax2d.plot(midpoint_2d_target[0], midpoint_2d_target[1], 'ro', markersize=6, label='Section Center (2D Projection)')
                
                title_str = f'2D Midpoint Cross-Section (Aligned Mesh)\n{base_name}'
                if aspect_ratio is not None and np.isfinite(aspect_ratio): title_str += f'\nAR: {aspect_ratio:.3f}'
                if pca_minor_std_dev is not None and np.isfinite(pca_minor_std_dev): title_str += f', Width(b): {pca_minor_std_dev:.3f}'
                ax2d.set_title(title_str); ax2d.legend(); plt.tight_layout()
                save_path_2d = os.path.join(output_dir, f'{base_name}_midpoint_section_2D_aligned.png')
                plt.savefig(save_path_2d, dpi=150); plt.close(fig_2d)
                print(f"  Saved 2D visualization to {save_path_2d}")

                # --- Create and Save Individual 3D Plotly Scene ---
                plotly_traces = []
                mesh_trace = go.Mesh3d(x=aligned_mesh.vertices[:, 0], y=aligned_mesh.vertices[:, 1], z=aligned_mesh.vertices[:, 2],
                                       i=aligned_mesh.faces[:, 0], j=aligned_mesh.faces[:, 1], k=aligned_mesh.faces[:, 2],
                                       opacity=0.5, color='lightgreen', name='Aligned Mesh')
                plotly_traces.append(mesh_trace)
                
                midpoint_viz_trace = go.Scatter3d(x=[midpoint_plane_origin[0]], y=[midpoint_plane_origin[1]], z=[midpoint_plane_origin[2]],
                                              mode='markers', marker=dict(size=8, color='red'), name='Section Plane Center')
                plotly_traces.append(midpoint_viz_trace)

                if len(final_section_vertices_3d_aligned) > 0:
                    # Order 3D points based on the angular sort of their 2D counterparts for consistent plotting
                    # final_points_2D and final_section_vertices_3d_aligned should correspond row-wise before ordering
                    center_2d_final = np.mean(final_points_2D, axis=0) # Centroid of the final 2D points
                    angles_final = np.arctan2(final_points_2D[:, 1] - center_2d_final[1], final_points_2D[:, 0] - center_2d_final[0])
                    sorted_indices_final = np.argsort(angles_final)
                    
                    ordered_points_3d_viz = final_section_vertices_3d_aligned[sorted_indices_final]
                    closed_points_3d_viz = np.vstack([ordered_points_3d_viz, ordered_points_3d_viz[0]]) # Close the loop
                    
                    section_line_trace = go.Scatter3d(x=closed_points_3d_viz[:, 0], y=closed_points_3d_viz[:, 1], z=closed_points_3d_viz[:, 2],
                                                      mode='lines', line=dict(color='blue', width=5), name='Section Outline (3D)')
                    plotly_traces.append(section_line_trace)
                    section_points_trace = go.Scatter3d(x=final_section_vertices_3d_aligned[:, 0], y=final_section_vertices_3d_aligned[:, 1], z=final_section_vertices_3d_aligned[:, 2],
                                                        mode='markers', marker=dict(size=3, color='darkblue'), name='Section Points (3D)', showlegend=False)
                    plotly_traces.append(section_points_trace)

                plane_viz_radius = minor_radius * 1.5 if minor_radius > 1e-6 else 1.0 # Ensure positive radius
                v1_plane = np.array([1.0, 0.0, 0.0]); v2_plane = np.array([0.0, 0.0, 1.0]) # For XZ plane (normal is Y)
                xx, yy = np.meshgrid(np.linspace(-plane_viz_radius, plane_viz_radius, 5), np.linspace(-plane_viz_radius, plane_viz_radius, 5))
                plane_points_x = midpoint_plane_origin[0] + v1_plane[0] * xx + v2_plane[0] * yy
                plane_points_y = midpoint_plane_origin[1] + v1_plane[1] * xx + v2_plane[1] * yy
                plane_points_z = midpoint_plane_origin[2] + v1_plane[2] * xx + v2_plane[2] * yy
                
                plane_trace = go.Surface(x=plane_points_x, y=plane_points_y, z=plane_points_z,
                                         colorscale=[[0, 'rgba(255,0,255,0.3)'], [1, 'rgba(255,0,255,0.3)']],
                                         showscale=False, opacity=0.3, name='Section Plane')
                plotly_traces.append(plane_trace)
                
                fig_3d = go.Figure(data=plotly_traces)
                fig_3d.update_layout(title=f'3D Midpoint Section (Aligned Mesh)<br>{base_name}',
                                     scene=dict(xaxis_title='X_aligned', yaxis_title='Y_aligned (Length Axis)', zaxis_title='Z_aligned', aspectmode='data'),
                                     margin=dict(l=0, r=0, b=0, t=40))
                save_path_3d_html = os.path.join(output_dir, f'{base_name}_midpoint_scene_aligned.html')
                try:
                    fig_3d.write_html(save_path_3d_html)
                    print(f"  Saved 3D HTML scene to {save_path_3d_html}")
                except Exception as export_err:
                    print(f"  Failed to export 3D HTML scene: {export_err}")
        
        except Exception as e:
            print(f"  Unhandled error processing {file_path}: {e}")
            traceback.print_exc()
            results[file_path] = None
            
    return results

# --- create_combined_midpoint_2d_plot function ---
def create_combined_midpoint_2d_plot(results_data, output_path):
    print(f"\nCreating combined 2D overlay plot at: {output_path}")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Combined Midpoint Cross-Sections (Horizontally Aligned)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("X (section 2D coords)")
    ax.set_ylabel("Y (section 2D coords)")
    
    valid_files_data = {fp: data for fp, data in results_data.items() if data is not None and data[0] is not None and len(data[0]) > 0}
    
    if not valid_files_data:
        print("  No valid midpoint sections found to plot.")
        plt.close(fig)
        return

    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_files_data)))
    max_extent = 0
    
    for i, (file_path, data) in enumerate(valid_files_data.items()):
        points_2d = data[0]  # final_points_2D
        
        # Center the points
        center_pt = np.mean(points_2d, axis=0)
        centered_points = points_2d - center_pt
        
        # Find principal components to determine rotation angle
        pca = PCA(n_components=2)
        pca.fit(centered_points)
        
        # Get the angle of the first principal component (major axis)
        # arctan2 gives angle in range [-pi, pi]
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        
        # Create rotation matrix to align major axis with x-axis (horizontal)
        # We want to rotate by -angle to make it horizontal
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])
        
        # Apply rotation
        rotated_points = centered_points @ rotation_matrix.T
        
        # Order points for clean perimeter visualization
        ordered = csf.order_points(rotated_points, method="nearest")
        
        # Close the loop
        if len(ordered) > 0:
            loop = np.vstack([ordered, ordered[0]])
            ax.plot(loop[:, 0], loop[:, 1], '-', color=colors[i],
                    linewidth=2, alpha=0.7, label=os.path.basename(file_path))
            
            # Update max_extent for plot limits
            current_max_extent = np.max(np.abs(ordered))
            if current_max_extent > max_extent:
                max_extent = current_max_extent
                
    if max_extent == 0: max_extent = 1.0
    limit = max_extent * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    if len(valid_files_data) <= 10:
        ax.legend(loc='upper right', fontsize='small')
    else:
        print("  Legend omitted for combined plot due to large number of files.")
        
    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved horizontally aligned combined 2D plot.")
    except Exception as e:
        print(f"  Error saving combined 2D plot: {e}")
    finally:
        plt.close(fig)


# --- NEW Function to create aspect ratio box plot ---
def create_aspect_ratio_boxplot(aspect_ratios_map, output_path, data_label="Aspect Ratio", value_type_label="Aspect Ratio (Long/Short Axis)"):
    valid_values = [val for val in aspect_ratios_map.values() if val is not None and np.isfinite(val)]

    if not valid_values:
        print(f"  No valid {data_label.lower()} to create a box plot.")
        return

    print(f"\nCreating {data_label.lower()} box plot with points at: {output_path}")
    fig, ax = plt.subplots(figsize=(6, 6))

    jitter_strength = 0.08
    x_jitter = np.random.normal(1, jitter_strength, size=len(valid_values))

    ax.boxplot(valid_values, vert=True, patch_artist=True, showmeans=False,
               positions=[1], widths=0.5, showfliers=True,
               boxprops=dict(facecolor='lightblue', alpha=0.8, zorder=2),
               medianprops=dict(color='red', linewidth=2, zorder=3),
               whiskerprops=dict(zorder=2), capprops=dict(zorder=2),
               flierprops=dict(marker='o', markersize=5, markerfacecolor='orange', markeredgecolor='grey', zorder=2))
    
    ax.scatter(x_jitter, valid_values, alpha=1.0, s=20, color='red', zorder=4, label='Individual Sections')
    
    ax.set_ylabel(value_type_label)
    ax.set_title(f'Distribution of Midpoint {data_label} (N={len(valid_values)})')
    ax.set_xticks([1])
    ax.set_xticklabels(['Midpoint Sections'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved {data_label.lower()} box plot with points.")
    except Exception as e:
        print(f"  Error saving {data_label.lower()} box plot: {e}")
    finally:
        plt.close(fig)


# --- Function to create mesh grid plot (MATPLOTLIB Version) ---
def create_mesh_grid_plot(file_paths, output_path, rows=3, cols=6):
    """
    Creates a grid of PCA-aligned meshes using Matplotlib.
    Each mesh is loaded, aligned via load_and_align_mesh, and displayed in a 3D subplot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    num_meshes = len(file_paths)
    if num_meshes == 0:
        print("No mesh files provided for grid plot.")
        return

    # Load and align all meshes
    aligned = []
    names = []
    for fp in file_paths:
        mesh, _ = csf.load_and_align_mesh(fp, align_axis='Y')
        if mesh is None or mesh.is_empty:
            print(f"  Skipping invalid or empty mesh: {fp}")
            continue
        aligned.append(mesh)
        names.append(os.path.basename(fp))

    if not aligned:
        print("No valid meshes to plot after alignment.")
        return

    # Compute global bounds for consistent axes
    all_verts = np.vstack([m.vertices for m in aligned])
    mins, maxs = all_verts.min(axis=0), all_verts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) * 0.6, 0.5)
    limits = [(center[i] - radius, center[i] + radius) for i in range(3)]

    fig = plt.figure(figsize=(cols * 3, rows * 3), constrained_layout=True)
    ls = LightSource(azdeg=225, altdeg=45)

    for idx in range(rows * cols):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.set_axis_off()

        if idx < len(aligned):
            m = aligned[idx]
            verts, faces = m.vertices, m.faces
            color = plt.cm.Greys(0.75)
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=faces,
                shade=True,
                color=color,
                lightsource=ls,
                linewidth=0,
                antialiased=True
            )
            ax.set_title(names[idx][:15], fontsize=8)

        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=0, azim=0)

    # Super-title
    if len(aligned) < num_meshes:
        fig.suptitle(f"Mesh Grid (First {len(aligned)} of {num_meshes} Aligned)", fontsize=14)
    else:
        fig.suptitle(f"Mesh Grid ({len(aligned)} Aligned Meshes)", fontsize=14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        fig.savefig(output_path, dpi=200)
        print(f"Saved Matplotlib mesh grid to {output_path}")
    except Exception as e:
        print(f"Error saving grid plot: {e}")
    finally:
        plt.close(fig)



# --- Main Execution Block ---
if __name__ == "__main__":
    # Example: find all .obj files in a directory
    mesh_dir = "Meshes/Onion_OBJ/"
    if os.path.exists(mesh_dir):
        files_to_process = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj")]
    else:
        print(f"Directory not found: {mesh_dir}. Using fallback list.")
        files_to_process = [
            "Meshes/OBJ/Ac_DA_1_3.obj", "Meshes/OBJ/Ac_DA_1_2.obj", # Add more if needed
        ]

    files_to_process = [f for f in files_to_process if os.path.exists(f)] # Filter for existing
    
    if not files_to_process:
        print("Error: No valid input files found. Please check paths or mesh_dir.")
    else:
        print(f"Found {len(files_to_process)} existing files to process.")
        output_directory = "midpoint_analysis_results_aligned"
        os.makedirs(output_directory, exist_ok=True)

        midpoint_data = analyze_midpoint_cross_section(
            files_to_process, output_dir=output_directory, visualize=True
        )

        if midpoint_data: # Check if any data was successfully processed
            combined_plot_path = os.path.join(output_directory, "combined_midpoint_overlay_2D_aligned.png")
            create_combined_midpoint_2d_plot(midpoint_data, combined_plot_path)

            aspect_ratios_from_results = {}
            pca_widths_from_results = {}
            for file_path, data_tuple in midpoint_data.items():
                if data_tuple and len(data_tuple) == 5:
                    ar_val = data_tuple[3]
                    width_val = data_tuple[4]
                    base_fn = os.path.basename(file_path)
                    if ar_val is not None: aspect_ratios_from_results[base_fn] = ar_val
                    if width_val is not None: pca_widths_from_results[base_fn] = width_val
            
            if aspect_ratios_from_results:
                boxplot_path_ar = os.path.join(output_directory, "aspect_ratio_boxplot_aligned.png")
                create_aspect_ratio_boxplot(aspect_ratios_from_results, boxplot_path_ar,
                                            data_label="Aspect Ratio", value_type_label="Aspect Ratio (PCA Major/Minor)")
            else: print("\nNo valid aspect ratios found for boxplot.")

            if pca_widths_from_results:
                 boxplot_path_width = os.path.join(output_directory, "pca_width_boxplot_aligned.png")
                 create_aspect_ratio_boxplot(pca_widths_from_results, boxplot_path_width,
                                             data_label="PCA Width (b)", value_type_label="PCA Minor Axis StdDev (Width 'b')")
            else: print("\nNo valid PCA widths found for boxplot.")
        else:
            print("\nNo data successfully processed by analyze_midpoint_cross_section. Skipping combined plots and boxplots.")

        mesh_grid_path = os.path.join(output_directory, "input_mesh_grid_matplotlib_aligned.png")
        create_mesh_grid_plot(files_to_process, mesh_grid_path, rows=3, cols=6) # Grid plot can still run

        print("\n--- Analysis and Visualization Complete (Aligned Method) ---")

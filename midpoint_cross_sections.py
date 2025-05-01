import os
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cross_section_functions as csf
from matplotlib.colors import LightSource
# --- Added import ---
from matplotlib.path import Path


def analyze_cross_section(file_paths,
                          section_location='midpoint',
                          output_dir=None,
                          visualize=False):
    """
    Analyzes multiple OBJ files to extract the cross-section at a specified location.
    Aligns mesh using PCA before analysis. Calculates aspect ratio of the 2D cross-section.
    Generates individual 2D PNG and 3D HTML visualizations if visualize=True.
    """
    results = {}
    print(f"\n--- Starting Cross-Section Analysis (Location: {section_location.upper()}) ---")

    # Prepare output directory for visualizations
    if visualize and output_dir:
        location_output_dir = os.path.join(output_dir, section_location)
        os.makedirs(location_output_dir, exist_ok=True)
    else:
        location_output_dir = output_dir

    for file_path in file_paths:
        print(f"\nProcessing file: {file_path} for {section_location} section")
        aspect_ratio = None
        pca_minor_std_dev = None
        plane_origin = None
        tangent = None
        mesh, tranform_4x4 = csf.load_and_align_mesh(file_path, align_axis = 'Y')

        if mesh is None:
            print(f"  Skipping file {file_path} due to loading/alignment error.")
            results[file_path] = None
            continue

        # --- Step 3: Determine Ray Casting Origin and Dimensions (on ALIGNED mesh) ---
        ray_origin = np.array([0.0, 0.0, 0.0]) # Default origin (centroid of aligned mesh)

        if section_location == 'midpoint':
            # --- Find Pore Center for Midpoint Ray Casting ---
            print("  Finding pore center for midpoint analysis...")
            try:
                # Cast rays along +/- X axis from the aligned origin
                origins_x = [ray_origin, ray_origin]
                dirs_x = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])]
                locations_x, index_ray_x, _ = mesh.ray.intersects_location(origins_x, dirs_x)

                if len(locations_x) >= 2:
                    pts_plus_x = locations_x[index_ray_x == 0]
                    pts_minus_x = locations_x[index_ray_x == 1]

                    if len(pts_plus_x) > 0 and len(pts_minus_x) > 0:
                        inner_plus_x = pts_plus_x[np.argmin(np.linalg.norm(pts_plus_x - ray_origin, axis=1))]
                        inner_minus_x = pts_minus_x[np.argmin(np.linalg.norm(pts_minus_x - ray_origin, axis=1))]
                        pore_center = (inner_plus_x + inner_minus_x) / 2.0
                        ray_origin = pore_center # Use this as the origin for radial rays
                        print(f"  Using estimated pore center as ray origin: {ray_origin.round(3)}")
                    else:
                        print("  Warning: Could not find inner points along X-axis. Using aligned origin [0,0,0].")
                else:
                    print("  Warning: Ray casting along X-axis failed. Using aligned origin [0,0,0].")
            except Exception as pore_err:
                print(f"  Warning: Error finding pore center: {pore_err}. Using aligned origin [0,0,0].")
            # --- End Finding Pore Center ---

        ## Comment this out to use the above code
        ray_origin = np.array([0.0, 0.0, 0.0])

        # --- Radial Ray Casting ---
        print(f"  Performing radial ray casting from origin: {ray_origin.round(3)}")
        ray_count = 36 # Or adjust as needed
        inner_points, outer_points, raw_centerline_points, minor_radius = csf.get_radial_dimensions(
            mesh, center=ray_origin, ray_count=ray_count
        )

        # --- Add Debug Print Here ---
        print(f"  DEBUG RayCast Results: inner={len(inner_points) if inner_points is not None else 'None'}, outer={len(outer_points) if outer_points is not None else 'None'}, cl_raw={len(raw_centerline_points) if raw_centerline_points is not None else 'None'}, minor_rad={minor_radius}")
        # --- End Debug Print ---

        # Check if ray casting was successful
        if inner_points is None or outer_points is None or raw_centerline_points is None or minor_radius is None:
                print("  Error: Could not determine dimensions via radial ray casting (get_radial_dimensions failed).")
                results[file_path] = None
                continue # Skip to the next file

        print(f"  Estimated Minor Radius (aligned): {minor_radius:.3f}")
        # --- End Ray Casting ---

        # --- Step 4: Estimate Plane Origin and Normal (Tangent) on ALIGNED centerline ---
        print(f"  DEBUG Check Centerline Calc: inner_len={len(inner_points)}, outer_len={len(outer_points)}")
        # --- End Debug Print ---
        if len(inner_points) > 0 and len(outer_points) > 0 and len(inner_points) == len(outer_points):
                raw_centerline_points = (inner_points + outer_points) / 2; num_cl_points = len(raw_centerline_points)
        else:
                print("  Error: Cannot calculate centerline from ray casting results."); results[file_path] = None; continue

        if section_location == 'midpoint':
            # Find point on centerline closest to the ray_origin used
            mid_idx = np.argmin(np.linalg.norm(raw_centerline_points - ray_origin, axis=1))
            plane_origin = raw_centerline_points[mid_idx]
            # Calculate tangent at midpoint
            prev_idx = (mid_idx - 1 + num_cl_points) % num_cl_points; next_idx = (mid_idx + 1) % num_cl_points
            tangent_vec = raw_centerline_points[next_idx] - raw_centerline_points[prev_idx]
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 1e-6: tangent = tangent_vec / tangent_norm
            else: tangent = np.array([0.0, 1.0, 0.0]) # Fallback
            print(f"  Calculated Midpoint Plane Origin (aligned): {plane_origin.round(3)}")
            print(f"  Using Midpoint Tangent (aligned): {tangent.round(3)}")

        elif section_location == 'tip':
            # --- MODIFIED TIP LOGIC ---
            # Find the vertex index with the minimum Y coordinate on the aligned mesh
            min_y_vertex_idx = np.argmin(mesh.vertices[:, 1])
            min_y_vertex = mesh.vertices[min_y_vertex_idx]
            print(f"  Found mesh vertex with min Y (pole estimate): {min_y_vertex.round(3)}")

            # Find the index of the centerline point closest to this minimum Y vertex
            distances_to_min_y_vertex = np.linalg.norm(raw_centerline_points - min_y_vertex, axis=1)
            initial_tip_cl_idx = np.argmin(distances_to_min_y_vertex)
            print(f"  Initial centerline point index {initial_tip_cl_idx} (closest to min Y vertex).")

            # --- Offset by stepping along centerline points ---
            num_steps_inward = 0 # Define how many points to step inwards (ADJUST AS NEEDED)

            # Determine the 'inward' direction (usually towards larger Y)
            prev_idx_initial = (initial_tip_cl_idx - 1 + num_cl_points) % num_cl_points
            next_idx_initial = (initial_tip_cl_idx + 1) % num_cl_points
            step_direction = 0
            if raw_centerline_points[next_idx_initial][1] > raw_centerline_points[prev_idx_initial][1]:
                step_direction = 1 # next_idx is inwards
            elif raw_centerline_points[prev_idx_initial][1] > raw_centerline_points[next_idx_initial][1]:
                step_direction = -1 # prev_idx is inwards
            else:
                # If Y is the same, default to stepping based on index order (arbitrary but consistent)
                step_direction = 1

            if step_direction == 0:
                print("  Warning: Could not determine inward direction reliably. Using initial tip index.")
                new_tip_cl_idx = initial_tip_cl_idx
            else:
                # Calculate the new index by stepping
                new_tip_cl_idx = (initial_tip_cl_idx + num_steps_inward * step_direction + num_cl_points) % num_cl_points
                print(f"  Stepped {num_steps_inward} points inward to index {new_tip_cl_idx}.")

            # Use the new index for plane origin
            plane_origin = raw_centerline_points[new_tip_cl_idx]

            # Calculate Tangent at the NEW centerline index
            prev_idx_new = (new_tip_cl_idx - 1 + num_cl_points) % num_cl_points
            next_idx_new = (new_tip_cl_idx + 1) % num_cl_points
            tangent_vec = raw_centerline_points[next_idx_new] - raw_centerline_points[prev_idx_new]
            tangent_norm = np.linalg.norm(tangent_vec)

            # --- Add Debug Prints Here ---
            print(f"  DEBUG Tip Tangent: Index={new_tip_cl_idx}, Vec={tangent_vec.round(5)}, Norm={tangent_norm:.6e}")
            # --- End Debug Prints ---

            if tangent_norm > 1e-6:
                tangent = tangent_vec / tangent_norm # This is the plane normal
                print(f"  Using Plane Origin at CL Index {new_tip_cl_idx} (aligned): {plane_origin.round(3)}")
                print(f"  Using Tangent calculated at CL Index {new_tip_cl_idx} (aligned): {tangent.round(3)}")
            else:
                # --- Add Debug Print Here ---
                print(f"  DEBUG Tip Tangent: *** Using Fallback Tangent because Norm <= 1e-6 ***")
                # --- End Debug Print ---
                print(f"  Error: Could not calculate valid tangent at new tip CL index {new_tip_cl_idx} (zero norm). Using fallback.")
                # Fallback: Use the origin but a default tangent
                tangent = np.array([0.0, 1.0, 0.0])
                print(f"  Using Plane Origin at CL Index {new_tip_cl_idx} (aligned): {plane_origin.round(3)}")
                print(f"  Using Fallback Tangent: {tangent.round(3)}")
            # --- END MODIFIED TIP LOGIC ---
        else:
            print(f"  Error: Unknown section_location '{section_location}'."); results[file_path] = None; continue

        # --- Step 5: Take Cross-Section (on ALIGNED mesh) ---
        if plane_origin is None or tangent is None:
                print("  Error: Plane origin or normal not determined."); results[file_path] = None; continue
        section = mesh.section(plane_origin=plane_origin, plane_normal=tangent)
        # Store the original 3D points of the section *before* filtering
        original_section_points_3d = section.vertices.copy() if section is not None else None
        if section is None or len(section.entities) == 0:
            print(f"  {section_location.capitalize()} section failed or is empty."); results[file_path] = None; continue

        # --- Step 6: Process and Filter Section (DBSCAN, PCA on 2D section) ---
        path_2D, transform_2d_to_3d = section.to_2D() # Get 2D path and the transform matrix
        points_2D = path_2D.vertices

        # Check if the number of original 3D points matches the number of 2D points
        if original_section_points_3d is None or len(original_section_points_3d) != len(points_2D):
            print(f"  Warning: Mismatch between original 3D section points ({len(original_section_points_3d) if original_section_points_3d is not None else 'None'}) and 2D points ({len(points_2D)}). Cannot reliably map back.")
            original_section_points_3d = None # Invalidate if mismatch

        # Transform 3D plane origin to 2D for distance/containment checks
        try:
            transform_3d_to_2d = np.linalg.inv(transform_2d_to_3d)
        except np.linalg.LinAlgError:
            print("  Error: Cannot invert section transformation matrix. Skipping containment check.")
            transform_3d_to_2d = None

        plane_origin_2d_target = None
        if transform_3d_to_2d is not None:
            plane_origin_h = np.append(plane_origin, 1) # Homogeneous coordinates
            plane_origin_transformed_h = transform_3d_to_2d @ plane_origin_h
            plane_origin_2d_target = plane_origin_transformed_h[:2] # Target point in 2D space
            print(f"  Plane origin projected to 2D: {plane_origin_2d_target.round(3)}")
        else:
            print("  Cannot project plane origin to 2D. Filtering based on distance only.")
            plane_origin_2d_target = np.mean(points_2D, axis=0) if len(points_2D) > 0 else np.array([0.0, 0.0])

        # --- DBSCAN Clustering ---
        eps_factor = 0.15 if section_location == 'midpoint' else 0.20
        min_samples = 3
        final_points_2D, final_mask = csf.filter_section_points(
            points_2D,
            minor_radius,
            plane_origin_2d_target,
            eps_factor=eps_factor,
            min_samples=min_samples
        )

        # Map back to 3D points (ALIGNED space) using the mask
        final_original_points_3D = np.empty((0, 3))
        if original_section_points_3d is not None and len(final_mask) == len(original_section_points_3d):
            final_original_points_3D = original_section_points_3d[final_mask]
        elif len(final_points_2D) > 0:
            # This condition might occur if original_section_points_3d was invalidated earlier
            # or if the mask length somehow doesn't match (shouldn't happen with current logic)
            print(f"  Warning: Could not map filtered 2D points back to 3D points.")

        # --- End Filtering ---

        # --- Step 7: Calculate Aspect Ratio and PCA Minor Axis Std Dev ---
        if len(final_points_2D) >= 3:
            try:
                pca = PCA(n_components=2); pca.fit(final_points_2D)
                std_devs = np.sqrt(pca.explained_variance_)
                if std_devs[1] > 1e-6:
                    aspect_ratio = std_devs[0] / std_devs[1]; pca_minor_std_dev = std_devs[1]
                    print(f"  PCA Results ({section_location}): AR={aspect_ratio:.3f}, Width(b)={pca_minor_std_dev:.3f}")
                else:
                    print(f"  Warning ({section_location}): Minor axis std dev near zero."); aspect_ratio = np.inf; pca_minor_std_dev = 0.0
            except Exception as pca_err:
                print(f"  Error calculating PCA results ({section_location}): {pca_err}"); aspect_ratio = None; pca_minor_std_dev = None
        else:
                if len(final_points_2D) > 0: print(f"  Not enough points (<3) for PCA in {section_location} section.")
                aspect_ratio = None; pca_minor_std_dev = None

        # --- Step 8: Store results ---
        if len(final_points_2D) > 0 and len(final_original_points_3D) > 0:
            print(f"  Successfully extracted {section_location} section with {len(final_points_2D)} points.")
            # Store the 2D points, the 3D points *in the aligned space*, the 2D->3D transform, AR, and width
            results[file_path] = (final_points_2D, final_original_points_3D, transform_2d_to_3d, aspect_ratio, pca_minor_std_dev)
        else:
            if aspect_ratio is None and len(points_2D) > 0: print(f"  {section_location.capitalize()} section valid but resulted in 0 points after filtering or 3D mapping failed.")
            results[file_path] = None

        # --- Step 9: Individual Visualization ---
        if visualize and location_output_dir and results[file_path] is not None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            # --- Create and Save Individual 2D Plot ---
            fig_2d, ax2d = plt.subplots(figsize=(6, 6))
            title_prefix = f'2D {section_location.capitalize()} Cross-Section'
            ax2d.set_aspect('equal'); ax2d.grid(True)
            ordered_plot_points = csf.order_points(final_points_2D, method="angular")
            ax2d.plot(np.append(ordered_plot_points[:, 0], ordered_plot_points[0, 0]), np.append(ordered_plot_points[:, 1], ordered_plot_points[0, 1]), 'b-', linewidth=1.5)
            ax2d.plot(ordered_plot_points[:, 0], ordered_plot_points[:, 1], 'b.', markersize=4)
            if plane_origin_2d_target is not None:
                ax2d.plot(plane_origin_2d_target[0], plane_origin_2d_target[1], 'ro', markersize=6, label=f'Target Center ({section_location}, 2D)')
            title_str = f'{title_prefix}\n{base_name}'
            if aspect_ratio is not None and np.isfinite(aspect_ratio): title_str += f'\nAR: {aspect_ratio:.3f}'
            if pca_minor_std_dev is not None: title_str += f', Width(b): {pca_minor_std_dev:.3f}'
            ax2d.set_title(title_str); ax2d.legend()
            plt.tight_layout()
            save_path_2d = os.path.join(location_output_dir, f'{base_name}_{section_location}_section_2D.png')
            plt.savefig(save_path_2d, dpi=150); print(f"  Saved 2D visualization to {save_path_2d}"); plt.close(fig_2d)

            # --- Create and Save Individual 3D Plotly Scene (Aligned) ---
            plotly_traces = []
            vertices_vis = mesh.vertices; faces_vis = mesh.faces
            mesh_trace = go.Mesh3d(x=vertices_vis[:, 0], y=vertices_vis[:, 1], z=vertices_vis[:, 2], i=faces_vis[:, 0], j=faces_vis[:, 1], k=faces_vis[:, 2], opacity=0.5, color='lightgrey', name='Aligned Mesh')
            plotly_traces.append(mesh_trace)

            origin_trace = go.Scatter3d(x=[plane_origin[0]], y=[plane_origin[1]], z=[plane_origin[2]], mode='markers', marker=dict(size=5, color='red'), name=f'{section_location.capitalize()} Origin (Aligned)')
            plotly_traces.append(origin_trace)

            # Plot the Ray Casting Origin ---
            ray_origin_trace = go.Scatter3d(x=[ray_origin[0]], y=[ray_origin[1]], z=[ray_origin[2]], mode='markers', marker=dict(size=5, color='green', symbol='cross'), name='Ray Casting Origin')
            plotly_traces.append(ray_origin_trace)

            # Use final_original_points_3D for plotting the section in 3D
            if len(final_original_points_3D) > 0:
                # Order points for line plot (using 2D angles for consistency)
                center_2d_final = np.mean(final_points_2D, axis=0); centered_2d_final = final_points_2D - center_2d_final
                angles_final = np.arctan2(centered_2d_final[:, 1], centered_2d_final[:, 0]); sorted_indices_final = np.argsort(angles_final)
                ordered_points_3d_aligned = final_original_points_3D[sorted_indices_final]
                closed_points_3d_aligned = np.vstack([ordered_points_3d_aligned, ordered_points_3d_aligned[0]])
                section_line_trace = go.Scatter3d(x=closed_points_3d_aligned[:, 0], y=closed_points_3d_aligned[:, 1], z=closed_points_3d_aligned[:, 2], mode='lines', line=dict(color='blue', width=5), name='Section Outline (Aligned)')
                plotly_traces.append(section_line_trace)
                section_points_trace = go.Scatter3d(x=final_original_points_3D[:, 0], y=final_original_points_3D[:, 1], z=final_original_points_3D[:, 2], mode='markers', marker=dict(size=3, color='blue'), name='Section Points (Aligned)', showlegend=False)
                plotly_traces.append(section_points_trace)
            # Section Plane Visualization
            plane_size = minor_radius * 1.5
            if np.abs(np.dot(tangent, np.array([0,0,1]))) < 0.99: v1 = np.cross(tangent, np.array([0, 0, 1]))
            else: v1 = np.cross(tangent, np.array([0, 1, 0]))
            v1 /= np.linalg.norm(v1); v2 = np.cross(tangent, v1)
            xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 5), np.linspace(-plane_size, plane_size, 5))
            plane_points_x = plane_origin[0] + v1[0] * xx + v2[0] * yy; plane_points_y = plane_origin[1] + v1[1] * xx + v2[1] * yy; plane_points_z = plane_origin[2] + v1[2] * xx + v2[2] * yy
            plane_trace = go.Surface(x=plane_points_x, y=plane_points_y, z=plane_points_z, colorscale=[[0, 'rgba(255,0,255,0.4)'], [1, 'rgba(255,0,255,0.4)']], showscale=False, opacity=0.4, name='Section Plane (Aligned)')
            plotly_traces.append(plane_trace)
            # Create Figure
            fig_3d = go.Figure(data=plotly_traces)
            fig_3d.update_layout(title=f'3D Visualization (Aligned) - {section_location.capitalize()} Section<br>{base_name}', scene=dict(xaxis_title='X', yaxis_title='Y (Aligned Length)', zaxis_title='Z', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=40))
            save_path_3d_html = os.path.join(location_output_dir, f'{base_name}_{section_location}_scene_aligned.html')
            try: fig_3d.write_html(save_path_3d_html); print(f"  Saved 3D HTML scene (aligned) to {save_path_3d_html}")
            except Exception as export_err: print(f"  Failed to export 3D HTML scene: {export_err}")


    print(f"--- Finished Cross-Section Analysis (Location: {section_location.upper()}) ---")
    return results

def create_combined_2d_plot(results_data, output_path, location_name='Midpoint'):
    """ Creates a combined 2D overlay plot for a given set of section results. """
    print(f"\nCreating combined 2D overlay plot ({location_name}) at: {output_path}")
    fig, ax = plt.subplots(figsize=(8, 8))
    # --- Increase title font size ---
    ax.set_title(f'Combined {location_name} Cross-Sections (Centered)', fontsize=16)

    ax.set_aspect('equal')
    ax.grid(True)
    # --- Swap axis labels and increase font size ---
    ax.set_xlabel("Y (centered)", fontsize=12)
    ax.set_ylabel("X (centered)", fontsize=12)

    valid_files = [fp for fp, data in results_data.items() if data is not None and len(data[0]) > 0]
    if not valid_files:
        print(f"  No valid {location_name} sections found to plot.")
        plt.close(fig)
        return

    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_files)))
    max_extent = 0
    for i, file_path in enumerate(valid_files):
        points_2d = results_data[file_path][0]
        center_pt = np.mean(points_2d, axis=0)
        centered_points = points_2d - center_pt
        ordered_points = csf.order_points(centered_points, method="angular")
        # --- Swap X and Y in plot call ---
        ax.plot(np.append(ordered_points[:, 1], ordered_points[0, 1]), # Y values first
                np.append(ordered_points[:, 0], ordered_points[0, 0]), # X values second
                '-', color=colors[i], linewidth=2, alpha=0.7, label=os.path.basename(file_path))
        max_extent = max(max_extent, np.max(np.abs(ordered_points)))

    limit = max_extent * 1.1
    # --- Swap axis limits ---
    ax.set_xlim(-limit, limit) # Y-axis limits
    ax.set_ylim(-limit, limit) # X-axis limits

    # --- Increase legend font size ---
    if len(valid_files) <= 10:
        ax.legend(loc='upper right', fontsize=10) # Adjusted font size
    else:
        print("  Legend omitted due to large number of files.")

    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved combined 2D plot ({location_name}).")
    except Exception as e:
        print(f"  Error saving combined 2D plot ({location_name}): {e}")
    finally:
        plt.close(fig)

# --- NEW Function to create aspect ratio box plot ---
def create_data_boxplot(data_values, output_path, data_name='Aspect Ratio', location_name='Midpoint'):
    """
    Creates a box plot for a list of data values (e.g., aspect ratios, widths).
    """
    if not data_values:
        print(f"  No valid {data_name} values ({location_name}) to create a box plot.")
        return

    print(f"\nCreating {data_name} box plot ({location_name}) at: {output_path}")
    fig, ax = plt.subplots(figsize=(6, 6))

    jitter_strength = 0.08
    x_jitter = np.random.normal(1, jitter_strength, size=len(data_values))

    bp = ax.boxplot(data_values, vert=True, patch_artist=True, showmeans=False,
                    positions=[1], widths=0.5, showfliers=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.8, zorder=2),
                    medianprops=dict(color='red', linewidth=2, zorder=3),
                    whiskerprops=dict(zorder=2), capprops=dict(zorder=2),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='orange', markeredgecolor='grey', zorder=2)
                   )
    ax.scatter(x_jitter, data_values, alpha=1.0, s=20, color='red', zorder=4, label='Individual Sections')
    ax.set_ylabel(data_name)
    ax.set_title(f'Distribution of {location_name} {data_name} (N={len(data_values)})')
    ax.set_xticks([1])
    ax.set_xticklabels([f'{location_name} Sections'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # ax.legend() # Optional

    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"  Successfully saved {data_name} box plot ({location_name}).")
    except Exception as e:
        print(f"  Error saving {data_name} box plot ({location_name}): {e}")
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

    # files_to_process = [
    #     "Meshes/OBJ/Ac_DA_1_3.obj", "Meshes/OBJ/Ac_DA_1_2.obj", "Meshes/OBJ/Ac_DA_1_5.obj",
    #     "Meshes/OBJ/Ac_DA_1_4.obj",  "Meshes/OBJ/Ac_DA_3_6.obj",
    #     "Meshes/OBJ/Ac_DA_3_4.obj",  
    #      "Meshes/OBJ/Ac_DA_2_7.obj", "Meshes/OBJ/Ac_DA_2_6b.obj",
    #     "Meshes/OBJ/Ac_DA_2_6a.obj", "Meshes/OBJ/Ac_DA_2_4.obj", "Meshes/OBJ/Ac_DA_2_3.obj",
    #     "Meshes/OBJ/Ac_DA_1_8_mesh.obj", "Meshes/OBJ/Ac_DA_1_6.obj"
    # ]

    # Check if files exist
    if not all(os.path.exists(f) for f in files_to_process):
         print("Error: One or more specified files do not exist. Please check paths.")
         print("Files expected:", files_to_process)
    else:
        # Define base output directory
        base_output_directory = "cross_section_analysis_results" # Renamed for clarity
        os.makedirs(base_output_directory, exist_ok=True)

        # --- Run Analysis for MIDPOINT ---
        midpoint_data = analyze_cross_section(
            files_to_process,
            section_location='midpoint',
            output_dir=base_output_directory, # Pass base directory
            visualize=True
        )

        # --- Run Analysis for TIP ---
        tip_data = analyze_cross_section(
            files_to_process,
            section_location='tip',
            output_dir=base_output_directory, # Pass base directory
            visualize=True
        )

        # --- Process and Plot MIDPOINT Results ---
        if midpoint_data:
            # Combined 2D Plot
            combined_plot_path_mid = os.path.join(base_output_directory, "combined_midpoint_overlay_2D.png")
            create_combined_2d_plot(midpoint_data, combined_plot_path_mid, location_name='Midpoint')

            # Extract ARs and Widths
            midpoint_ars = []
            midpoint_widths = []
            for data in midpoint_data.values():
                if data and len(data) > 4:
                    ar, width = data[3], data[4]
                    if ar is not None and np.isfinite(ar): midpoint_ars.append(ar)
                    if width is not None and np.isfinite(width) and width > 1e-9: midpoint_widths.append(width)

            # Box Plots
            if midpoint_ars:
                boxplot_path_ar_mid = os.path.join(base_output_directory, "midpoint_aspect_ratio_boxplot.png")
                create_data_boxplot(midpoint_ars, boxplot_path_ar_mid, data_name='Aspect Ratio', location_name='Midpoint')
            if midpoint_widths:
                boxplot_path_width_mid = os.path.join(base_output_directory, "midpoint_pca_width_boxplot.png")
                create_data_boxplot(midpoint_widths, boxplot_path_width_mid, data_name='PCA Width (b)', location_name='Midpoint')
        else:
            print("\nNo valid data generated for midpoint analysis.")


        # --- Process and Plot TIP Results ---
        if tip_data:
            # Combined 2D Plot
            combined_plot_path_tip = os.path.join(base_output_directory, "combined_tip_overlay_2D.png")
            create_combined_2d_plot(tip_data, combined_plot_path_tip, location_name='Tip')

            # Extract ARs and Widths
            tip_ars = []
            tip_widths = []
            for data in tip_data.values():
                if data and len(data) > 4:
                    ar, width = data[3], data[4]
                    if ar is not None and np.isfinite(ar): tip_ars.append(ar)
                    if width is not None and np.isfinite(width) and width > 1e-9: tip_widths.append(width)

            # Box Plots
            if tip_ars:
                boxplot_path_ar_tip = os.path.join(base_output_directory, "tip_aspect_ratio_boxplot.png")
                create_data_boxplot(tip_ars, boxplot_path_ar_tip, data_name='Aspect Ratio', location_name='Tip')
            if tip_widths:
                boxplot_path_width_tip = os.path.join(base_output_directory, "tip_pca_width_boxplot.png")
                create_data_boxplot(tip_widths, boxplot_path_width_tip, data_name='PCA Width (b)', location_name='Tip')
        else:
            print("\nNo valid data generated for tip analysis.")


        # --- Create Input Mesh Grid Plot (Only needs to be done once) ---
        mesh_grid_path = os.path.join(base_output_directory, "input_mesh_grid_matplotlib.png")
        create_mesh_grid_plot(files_to_process, mesh_grid_path, rows=3, cols=6)

        print("\n--- Analysis and Visualization Complete ---")
import numpy as np
import trimesh
import os
from scipy.optimize import curve_fit
from shapely.geometry import Polygon, Point
import traceback
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import cross_section_functions as csf

# --- Import necessary functions ---
try:
    from midpoint_cross_sections import analyze_cross_section
    from cross_section_functions import ellipse as fit_ellipse_func
except ImportError as e:
    print(f"ERROR: Could not import necessary functions: {e}")
    analyze_cross_section = None
    fit_ellipse_func = None

def _get_guard_cell_parameters(input_file_path, visualize_midpoint_analysis=False):
    """
    Helper function to perform Steps 1 & 2:
    - Load mesh, get center.
    - Perform radial ray casting (csf.get_radial_dimensions).
    - Fit centerline ellipse (csf.fit_centerline_ellipse).
    - Analyze midpoint cross-section (analyze_cross_section).
    - Calculate midpoint cross-section semi-axes and target area.

    Returns:
        dict: A dictionary containing parameters or None if errors occur.
              Keys: 'mesh', 'center', 'center_xy', 'cl_a', 'cl_b', 'cl_phi',
                    'cs_a_mid', 'cs_b_mid', 'AR_mid', 'target_area', 'avg_inner_radius'
    """
    print("STEP 1&2: Extracting geometry and analyzing midpoint...")
    params = {}
    try:
        # --- Step 1 Logic ---
        mesh = trimesh.load_mesh(input_file_path)
        if isinstance(mesh, trimesh.Scene):
             mesh = mesh.dump(concatenate=True)
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
             raise ValueError("Invalid mesh data loaded.")
        params['mesh'] = mesh
        params['center'] = mesh.centroid

        inner_points, outer_points, raw_centerline_points, _ = csf.get_radial_dimensions(
            mesh, center=params['center'], ray_count=36
        )
        if raw_centerline_points is None:
             raise ValueError("get_radial_dimensions failed.")

        avg_inner_radius = None
        if inner_points is not None:
            inner_radii = np.linalg.norm(inner_points - params['center'], axis=1)
            avg_inner_radius = np.mean(inner_radii) if len(inner_radii) > 0 else None
            if avg_inner_radius is not None and avg_inner_radius <= 0: print(f"  Warning: Invalid avg inner radius: {avg_inner_radius}")
            else: print(f"  Avg Inner Radius: {avg_inner_radius:.4f}" if avg_inner_radius is not None else "N/A")
        params['avg_inner_radius'] = avg_inner_radius

        cl_a, cl_b, cl_phi = csf.fit_centerline_ellipse(raw_centerline_points, params['center'])
        if cl_a is None: raise ValueError("fit_centerline_ellipse failed.")
        params['cl_a'], params['cl_b'], params['cl_phi'] = cl_a, cl_b, cl_phi
        params['center_xy'] = params['center'][:2]
        print(f"  Centerline: a={cl_a:.3f}, b={cl_b:.3f}, phi={np.degrees(cl_phi):.1f}°")
        print(f"  Center: {params['center']}")

        if avg_inner_radius is not None and cl_b <= avg_inner_radius:
            print(f"  Warning: Centerline semi-minor axis ({cl_b:.4f}) <= avg inner radius ({avg_inner_radius:.4f}).")

        # --- Step 2 Logic ---
        cs_a_mid, cs_b_mid = 0.0, 0.0
        AR_mid = 1.0
        pca_minor_std_dev = None

        midpoint_results = analyze_cross_section(
            [input_file_path],
            visualize=visualize_midpoint_analysis # Use passed flag
        )

        if midpoint_results and input_file_path in midpoint_results and midpoint_results[input_file_path]:
             analysis_data_tuple = midpoint_results[input_file_path]
             if isinstance(analysis_data_tuple, tuple) and len(analysis_data_tuple) >= 5:
                 ar = analysis_data_tuple[3]
                 width_measure = analysis_data_tuple[4]
                 if ar is not None and np.isfinite(ar):
                     AR_mid = ar
                     if AR_mid < 1.0: AR_mid = 1.0 / AR_mid
                 if width_measure is not None and np.isfinite(width_measure) and width_measure > 1e-9:
                     pca_minor_std_dev = width_measure
             else: print("  Warning: Midpoint analysis returned unexpected data format.")
        else: print("  Warning: Midpoint analysis failed or returned no data.")

        if pca_minor_std_dev is None:
             raise ValueError("Could not determine cross-section width from midpoint analysis.")

        pca_width_scale_factor = 1.43 # Keep consistent
        cs_b_mid = pca_minor_std_dev * pca_width_scale_factor
        cs_a_mid = cs_b_mid * AR_mid
        if cs_a_mid < cs_b_mid: cs_a_mid, cs_b_mid = cs_b_mid, cs_a_mid

        params['cs_a_mid'] = cs_a_mid
        params['cs_b_mid'] = cs_b_mid
        params['AR_mid'] = AR_mid
        params['target_area'] = np.pi * cs_a_mid * cs_b_mid # Calculate target area

        print(f"  Midpoint CS: a={cs_a_mid:.4f}, b={cs_b_mid:.4f}, AR={AR_mid:.4f}")
        print(f"  Target Area: {params['target_area']:.4f}")

        return params

    except Exception as e:
        print(f"  Error during parameter extraction: {e}")
        traceback.print_exc()
        return None
    
# ... (after _get_guard_cell_parameters) ...

def _calculate_half_centerline(cl_a, cl_b, cl_phi, center_xy, center_z, num_segments):
    """Calculates the 3D points for the half-centerline path parallel to the major axis."""
    try:
        # Define angles for the half-centerline parallel to the major axis
        half_start_angle = cl_phi
        half_end_angle = cl_phi + np.pi
        num_half_segments = num_segments // 2 + 1 # Ensure enough points

        # Generate theta values for the half ellipse
        half_theta = np.linspace(half_start_angle, half_end_angle, num_half_segments)

        # Calculate points on the centerline
        half_r = fit_ellipse_func(half_theta, cl_a, cl_b, cl_phi) # Assumes fit_ellipse_func is imported
        half_x = center_xy[0] + half_r * np.cos(half_theta)
        half_y = center_xy[1] + half_r * np.sin(half_theta)
        half_z = np.full_like(half_x, center_z)
        half_centerline = np.column_stack([half_x, half_y, half_z])

        print(f"  Created half-centerline with {len(half_centerline)} points.")
        return half_centerline
    except Exception as e:
        print(f"  Error calculating half centerline: {e}")
        return None

# ... (after _calculate_half_centerline) ...

def _generate_swept_mesh_with_caps(half_centerline, cross_section_polygons, num_cross_section_points, cl_phi):
    """
    Generates the mesh by sweeping/constructing from polygons and adds planar caps.

    Args:
        half_centerline (np.ndarray): N x 3 array of centerline points.
        cross_section_polygons (list or Polygon): Either a list of Shapely Polygons
                                                  (for varying cross-section) or a
                                                  single Shapely Polygon (for constant).
        num_cross_section_points (int): Expected number of vertices per polygon.
        cl_phi (float): Angle of the centerline's major axis (for capping plane normal).

    Returns:
        trimesh.Trimesh or None: The generated and capped mesh.
    """
    print("STEP 5: Generating swept mesh and adding caps...")
    single_cell = None
    try:
        if half_centerline is None or len(half_centerline) < 2:
             raise ValueError("Invalid half_centerline path provided.")
        if cross_section_polygons is None:
             raise ValueError("No cross_section_polygons provided.")

        # --- Generate Swept Body ---
        is_varying = isinstance(cross_section_polygons, list)

        if not is_varying:
            # --- Case 1: Constant Cross-Section (Use trimesh sweep) ---
            print("  Using trimesh.creation.sweep_polygon (constant cross-section)...")
            single_cell = trimesh.creation.sweep_polygon(
                polygon=cross_section_polygons, # Single polygon
                path=half_centerline,
                caps=False # Cap manually later
            )
            print(f"  Generated initial swept mesh: {len(single_cell.vertices)} vertices, {len(single_cell.faces)} faces")
            # Assumption: sweep_polygon generates vertices in order: num_cs_pts per profile.
            num_profiles = len(half_centerline)
            if len(single_cell.vertices) != num_profiles * num_cross_section_points:
                 print(f"  Warning: Vertex count ({len(single_cell.vertices)}) doesn't match expected ({num_profiles * num_cross_section_points}). Capping indices might be wrong.")

        else:
            # --- Case 2: Varying Cross-Section (Manual Construction) ---
            print("  Using manual mesh construction (varying cross-section)...")
            if len(cross_section_polygons) != len(half_centerline):
                raise ValueError("Mismatch between number of polygons and centerline points.")

            all_vertices = []
            all_faces = []
            num_cs_pts = num_cross_section_points # Expected vertices per cross-section

            path_vectors = np.gradient(half_centerline, axis=0)
            path_tangents = path_vectors / np.linalg.norm(path_vectors, axis=1)[:, None]
            if np.linalg.norm(path_tangents[0]) < 1e-9: path_tangents[0] = path_tangents[1]
            if np.linalg.norm(path_tangents[-1]) < 1e-9: path_tangents[-1] = path_tangents[-2]

            base_vertex_count = 0
            for i, path_point in enumerate(half_centerline):
                polygon = cross_section_polygons[i]
                tangent = path_tangents[i]
                if abs(np.dot(tangent, [0, 0, 1])) > 0.999: up_vector = np.cross(tangent, [0, 1, 0])
                else: up_vector = np.array([0, 0, 1])
                normal = tangent
                binormal = np.cross(normal, up_vector); binormal /= np.linalg.norm(binormal)
                local_y = np.cross(binormal, normal); local_y /= np.linalg.norm(local_y)
                local_x = binormal

                cs_verts_2d = np.array(polygon.exterior.coords)[:-1]
                current_num_cs_pts = len(cs_verts_2d)
                if current_num_cs_pts != num_cs_pts:
                     print(f"  Warning: Polygon {i} has {current_num_cs_pts} vertices, expected {num_cs_pts}.")
                     # Adjust num_cs_pts for face generation if needed, or raise error
                     # For now, assume it might cause issues later if counts vary.

                cs_verts_3d = path_point + cs_verts_2d[:, 0][:, None] * local_x + cs_verts_2d[:, 1][:, None] * local_y
                all_vertices.append(cs_verts_3d)

                if i > 0:
                    prev_num_pts = len(all_vertices[i-1])
                    # Use the actual count for face generation
                    if prev_num_pts == current_num_cs_pts:
                        offset = base_vertex_count - prev_num_pts
                        for j in range(current_num_cs_pts):
                            v1 = offset + j
                            v2 = offset + (j + 1) % current_num_cs_pts
                            v3 = base_vertex_count + (j + 1) % current_num_cs_pts
                            v4 = base_vertex_count + j
                            all_faces.append([v1, v2, v3])
                            all_faces.append([v1, v3, v4])
                    else:
                        print(f"  Warning: Vertex count mismatch between section {i-1} ({prev_num_pts}) and {i} ({current_num_cs_pts}). Skipping faces.")

                base_vertex_count += current_num_cs_pts

            if not all_vertices or not all_faces: raise ValueError("No vertices or faces generated.")
            final_vertices = np.vstack(all_vertices)
            final_faces = np.array(all_faces, dtype=np.int32)
            single_cell = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
            print(f"  Generated swept mesh manually: {len(single_cell.vertices)} vertices, {len(single_cell.faces)} faces")

        # --- Capping Logic (Common to both cases) ---
        print("  Creating planar caps by projecting end vertices...")
        start_point = half_centerline[0]
        end_point = half_centerline[-1]

        # Normal vector is PERPENDICULAR to the centerline's major axis direction
        major_axis_direction = np.array([np.cos(cl_phi), np.sin(cl_phi), 0])
        cap_plane_normal = np.array([-major_axis_direction[1], major_axis_direction[0], 0])
        norm_mag = np.linalg.norm(cap_plane_normal)
        if norm_mag > 1e-9: cap_plane_normal /= norm_mag
        else: cap_plane_normal = np.array([0, 1, 0]); print("  Warning: Defaulting cap plane normal.")

        # Identify vertices at start and end profiles
        # This assumes consistent numbering from sweep/manual construction
        num_verts_total = len(single_cell.vertices)
        # Use the *expected* num_cross_section_points for indexing
        if num_verts_total < 2 * num_cross_section_points:
             raise ValueError(f"Mesh has only {num_verts_total} vertices, expected at least {2 * num_cross_section_points} for capping.")

        start_profile_indices = np.arange(num_cross_section_points)
        # Adjust end profile indices based on actual vertex count vs expected stride
        # Find the start index of the last profile
        last_profile_start_index = (len(half_centerline) - 1) * num_cross_section_points
        # Ensure this index is valid
        if last_profile_start_index + num_cross_section_points > num_verts_total:
             # Fallback if vertex count is unexpected - use last N vertices
             print(f"  Warning: Unexpected vertex count ({num_verts_total}). Using last {num_cross_section_points} vertices for end cap.")
             end_profile_indices = np.arange(num_verts_total - num_cross_section_points, num_verts_total)
        else:
             end_profile_indices = np.arange(last_profile_start_index, last_profile_start_index + num_cross_section_points)


        # Project start vertices
        projected_start_vertices = []
        for idx in start_profile_indices:
            v = single_cell.vertices[idx]
            dist = np.dot(v - start_point, cap_plane_normal)
            v_projected = v - dist * cap_plane_normal
            projected_start_vertices.append(v_projected)
            single_cell.vertices[idx] = v_projected

        # Project end vertices
        projected_end_vertices = []
        for idx in end_profile_indices:
            v = single_cell.vertices[idx]
            dist = np.dot(v - end_point, cap_plane_normal)
            v_projected = v - dist * cap_plane_normal
            projected_end_vertices.append(v_projected)
            single_cell.vertices[idx] = v_projected

        print(f"  Projected {len(start_profile_indices)} start and {len(end_profile_indices)} end vertices.")

        # Triangulate caps
        projected_start_vertices = np.array(projected_start_vertices)
        projected_end_vertices = np.array(projected_end_vertices)
        start_center = np.mean(projected_start_vertices, axis=0)
        end_center = np.mean(projected_end_vertices, axis=0)

        center_start_idx = len(single_cell.vertices)
        center_end_idx = center_start_idx + 1
        single_cell.vertices = np.vstack([single_cell.vertices, [start_center, end_center]])

        start_faces, end_faces = [], []
        for i in range(num_cross_section_points):
            v1_idx_start = start_profile_indices[i]
            v2_idx_start = start_profile_indices[(i + 1) % num_cross_section_points]
            start_faces.append([v2_idx_start, v1_idx_start, center_start_idx]) # Consistent winding

            v1_idx_end = end_profile_indices[i]
            v2_idx_end = end_profile_indices[(i + 1) % num_cross_section_points]
            end_faces.append([v1_idx_end, v2_idx_end, center_end_idx]) # Consistent winding

        if start_faces and end_faces:
            all_new_faces = np.vstack(start_faces + end_faces)
            single_cell.faces = np.vstack([single_cell.faces, all_new_faces])
            print(f"  Added {len(all_new_faces)} cap faces.")

        # --- Final Validation ---
        single_cell.fix_normals()
        if not single_cell.is_watertight:
            print("  Warning: Mesh not watertight after capping. Trying fill_holes().")
            single_cell.fill_holes()
            if not single_cell.is_watertight: print("  ERROR: Could not make mesh watertight.")
            else: print("  Made watertight with fill_holes().")
        else:
            print("  Mesh is watertight with planar caps.")

        return single_cell

    except Exception as e:
        print(f"  Error during mesh generation or capping: {e}")
        traceback.print_exc()
        return None

def generate_single_guard_cell(input_file_path, output_path, num_centerline_segments=64,
                              num_cross_section_points=64, visualize_steps=True):
    """
    Generates a SINGLE guard cell with a CONSTANT cross-section using helper functions.
    """
    print(f"\nGenerating SINGLE guard cell (Standard) for: {input_file_path}")
    single_cell = None
    center = None

    try:
        # --- Steps 1 & 2: Get Parameters ---
        params = _get_guard_cell_parameters(input_file_path, visualize_midpoint_analysis=visualize_steps)
        if params is None: return None, None
        center = params['center'] # Store center for return

        # --- Step 3: Visualization (Optional - simplified) ---
        if visualize_steps:
            print("STEP 3: Visualizing centerline and MIDPOINT cross-section...")
            # (Add simplified visualization code here if desired, similar to bulging Step 3)
            try:
                fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
                centerline_theta = np.linspace(0, 2*np.pi, 100)
                centerline_r = fit_ellipse_func(centerline_theta, params['cl_a'], params['cl_b'], params['cl_phi'])
                centerline_x = params['center_xy'][0] + centerline_r * np.cos(centerline_theta)
                centerline_y = params['center_xy'][1] + centerline_r * np.sin(centerline_theta)
                ax1.plot(centerline_x, centerline_y, 'b-', label='Full centerline')
                major_axis_x = [params['center_xy'][0] - 1.1*params['cl_a']*np.cos(params['cl_phi']), params['center_xy'][0] + 1.1*params['cl_a']*np.cos(params['cl_phi'])]
                major_axis_y = [params['center_xy'][1] - 1.1*params['cl_a']*np.sin(params['cl_phi']), params['center_xy'][1] + 1.1*params['cl_a']*np.sin(params['cl_phi'])]
                ax1.plot(major_axis_x, major_axis_y, 'g--', linewidth=1, label='Major axis')
                ax1.scatter(params['center_xy'][0], params['center_xy'][1], color='black', s=50, label='Center')
                ax1.set_title('Centerline Ellipse (Standard)'); ax1.set_xlabel('X'); ax1.set_ylabel('Y')
                ax1.axis('equal'); ax1.grid(True); ax1.legend()
                viz_dir = os.path.dirname(output_path);
                if viz_dir and not os.path.exists(viz_dir): os.makedirs(viz_dir)
                viz_path = os.path.join(viz_dir, f'visualization_standard_step3_{os.path.basename(output_path)}.png')
                plt.tight_layout(); plt.savefig(viz_path); plt.close(fig)
                print(f"  Centerline visualization saved to: {viz_path}")
            except Exception as e: print(f"  Error during visualization: {e}")

        # --- Step 4: Define HALF centerline and CONSTANT cross-section ---
        print("STEP 4: Defining HALF centerline path and CONSTANT cross-section...")
        half_centerline = _calculate_half_centerline(
            params['cl_a'], params['cl_b'], params['cl_phi'],
            params['center_xy'], params['center'][2],
            num_centerline_segments
        )
        if half_centerline is None: raise ValueError("Failed to calculate half centerline.")

        # Create single cross-section polygon using midpoint values
        cs_angles = np.linspace(0, 2*np.pi, num_cross_section_points, endpoint=False)
        cs_x = params['cs_a_mid'] * np.cos(cs_angles)
        cs_y = params['cs_b_mid'] * np.sin(cs_angles)
        cs_vertices = np.column_stack([cs_x, cs_y])
        cs_polygon = Polygon(cs_vertices)
        if not cs_polygon.is_valid: cs_polygon = cs_polygon.buffer(0) # Fix potential self-intersection
        if not cs_polygon.is_valid: raise ValueError("Created cross-section polygon is invalid.")
        print(f"  Created constant cross-section polygon with {len(cs_vertices)} vertices.")

        # --- Step 5: Generate Mesh and Caps ---
        single_cell = _generate_swept_mesh_with_caps(
            half_centerline,
            cs_polygon, # Pass single polygon
            num_cross_section_points,
            params['cl_phi']
        )
        if single_cell is None: raise ValueError("Failed to generate swept mesh with caps.")

        # --- Step 6: Export ---
        print("STEP 6: Exporting single guard cell (Standard)...")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        single_cell.export(file_obj=output_path, file_type='ply', encoding='ascii')
        print(f"  Successfully saved single guard cell to: {output_path}")

    except Exception as e:
        print(f"  Error generating standard single guard cell: {e}")
        traceback.print_exc()
        return None, None # Return None, None on error

    return single_cell, center


def get_modulated_AR(theta_cl, cl_phi, AR_mid, AR_min, transition_power=2.0): # transition_power is now less relevant here
    """
    Calculates a modulated aspect ratio that maintains AR_mid along most of the guard cell
    with a transition to AR_min only at the tips.
    """
    if AR_mid <= AR_min:
        return AR_mid  # No modulation needed

    # Calculate angular distance from nearest major axis point (0 to pi/2)
    relative_angle = (theta_cl - cl_phi) % (2 * np.pi)
    dist_to_major = min(relative_angle % np.pi, np.pi - (relative_angle % np.pi))  # 0 at major, pi/2 at minor

    # Normalize to 0-1 range (0 at sides, 1 at tips)
    normalized_dist = dist_to_major / (np.pi/2)

    # EXPLICIT CONTROL: Define exact threshold where transition begins
    # --- INCREASE FLAT ZONE SIGNIFICANTLY ---
    flat_zone = 0.6  # Stay at AR_mid for 95% of the distance from side to tip
                      # Adjust this value (0.9 - 0.98)

    if normalized_dist <= flat_zone:
        # Within the flat zone - maintain AR_mid exactly
        modulation = 0.0
    else:
        # In the transition zone - smooth curve
        # Rescale position in transition zone to 0-1
        transition_pos = (normalized_dist - flat_zone) / (1.0 - flat_zone)
        # --- USE POWER > 1 FOR SLOWER INITIAL TRANSITION ---
        # This makes modulation stay near 0 longer within the transition zone
        modulation = transition_pos ** 4.0 # Try high power (e.g., 2.0, 4.0, 6.0)

    # Calculate final AR
    current_AR = AR_mid - (AR_mid - AR_min) * modulation

    # Debug output (keep this active)
    print(f"  theta={np.degrees(theta_cl):.1f}, dist_major={np.degrees(dist_to_major):.1f}, " +
          f"norm={normalized_dist:.3f}, mod={modulation:.4f}, AR={current_AR:.4f}")

    return current_AR

def generate_single_bulging_guard_cell(input_file_path, output_path, num_centerline_segments=64,
                                       num_cross_section_points=64, visualize_steps=True,
                                       min_aspect_ratio=1.1, transition_power=2.0):
    """
    Generates a SINGLE guard cell with VARYING cross-section using helper functions.
    """
    print(f"\nGenerating SINGLE BULGING guard cell (Min AR={min_aspect_ratio:.2f}, Power={transition_power:.2f}) for: {input_file_path}")
    single_cell = None
    center = None

    try:
        # --- Steps 1 & 2: Get Parameters ---
        params = _get_guard_cell_parameters(input_file_path, visualize_midpoint_analysis=visualize_steps)
        if params is None: return None, None
        center = params['center'] # Store center for return

        if params['AR_mid'] <= min_aspect_ratio:
            print(f"  Warning: Midpoint AR ({params['AR_mid']:.4f}) <= Min AR ({min_aspect_ratio:.2f}). No bulging will occur.")

        # --- Step 3: Visualization (Optional - simplified) ---
        if visualize_steps:
            print("STEP 3: Visualizing centerline (Bulging)...")
            # (Add simplified visualization code here if desired, similar to standard Step 3)
            try:
                fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
                centerline_theta = np.linspace(0, 2*np.pi, 100)
                centerline_r = fit_ellipse_func(centerline_theta, params['cl_a'], params['cl_b'], params['cl_phi'])
                centerline_x = params['center_xy'][0] + centerline_r * np.cos(centerline_theta)
                centerline_y = params['center_xy'][1] + centerline_r * np.sin(centerline_theta)
                ax1.plot(centerline_x, centerline_y, 'b-', label='Full centerline')
                major_axis_x = [params['center_xy'][0] - 1.1*params['cl_a']*np.cos(params['cl_phi']), params['center_xy'][0] + 1.1*params['cl_a']*np.cos(params['cl_phi'])]
                major_axis_y = [params['center_xy'][1] - 1.1*params['cl_a']*np.sin(params['cl_phi']), params['center_xy'][1] + 1.1*params['cl_a']*np.sin(params['cl_phi'])]
                ax1.plot(major_axis_x, major_axis_y, 'g--', linewidth=1, label='Major axis')
                ax1.scatter(params['center_xy'][0], params['center_xy'][1], color='black', s=50, label='Center')
                ax1.set_title('Centerline Ellipse (Bulging)'); ax1.set_xlabel('X'); ax1.set_ylabel('Y')
                ax1.axis('equal'); ax1.grid(True); ax1.legend()
                viz_dir = os.path.dirname(output_path);
                if viz_dir and not os.path.exists(viz_dir): os.makedirs(viz_dir)
                viz_path = os.path.join(viz_dir, f'visualization_bulging_step3_{os.path.basename(output_path)}.png')
                plt.tight_layout(); plt.savefig(viz_path); plt.close(fig)
                print(f"  Centerline visualization saved to: {viz_path}")
            except Exception as e: print(f"  Error during visualization: {e}")


        # --- Step 4: Define HALF centerline and VARYING cross-sections ---
        print("STEP 4: Defining HALF centerline path and VARYING cross-sections...")
        half_centerline = _calculate_half_centerline(
            params['cl_a'], params['cl_b'], params['cl_phi'],
            params['center_xy'], params['center'][2],
            num_centerline_segments
        )
        if half_centerline is None: raise ValueError("Failed to calculate half centerline.")

        # Generate list of varying cross-section polygons
        cross_section_polygons = []
        cs_base_angles = np.linspace(0, 2*np.pi, num_cross_section_points, endpoint=False)
        half_theta_angles = np.linspace(params['cl_phi'], params['cl_phi'] + np.pi, len(half_centerline)) # Recalculate angles for modulation

        for i, theta_cl in enumerate(half_theta_angles):
            current_AR = get_modulated_AR(theta_cl, params['cl_phi'] + np.pi/2, params['AR_mid'], min_aspect_ratio, transition_power)
            current_cs_b = np.sqrt(params['target_area'] / (np.pi * current_AR))
            current_cs_a = current_AR * current_cs_b

            cs_x = current_cs_a * np.cos(cs_base_angles)
            cs_y = current_cs_b * np.sin(cs_base_angles)
            cs_vertices = np.column_stack([cs_x, cs_y])

            poly = Polygon(cs_vertices)
            if not poly.is_valid: poly = poly.buffer(0)
            if not poly.is_valid: raise ValueError(f"Created cross-section polygon at index {i} is invalid.")
            cross_section_polygons.append(poly)
        print(f"  Generated {len(cross_section_polygons)} varying cross-section polygons.")

        # --- Step 5: Generate Mesh and Caps ---
        single_cell = _generate_swept_mesh_with_caps(
            half_centerline,
            cross_section_polygons, # Pass list of polygons
            num_cross_section_points,
            params['cl_phi']
        )
        if single_cell is None: raise ValueError("Failed to generate swept mesh with caps.")

        # --- Step 6: Export ---
        print("STEP 6: Exporting single guard cell (Bulging)...")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        single_cell.export(file_obj=output_path, file_type='ply', encoding='ascii')
        print(f"  Successfully saved single guard cell to: {output_path}")

    except Exception as e:
        print(f"  Error generating bulging single guard cell: {e}")
        traceback.print_exc()
        return None, None # Return None, None on error

    return single_cell, center

# ... (rest of the file, including create_full_stomata_from_half and __main__) ...


def create_full_stomata_from_half(single_cell_mesh, center_point, output_path):
    """
    Takes a single guard cell mesh (with planar caps), duplicates it, rotates
    the duplicate 180 degrees around the Z-axis at the center_point, combines
    them, adds vertex AND face labels/signals, and exports the final mesh.

    Args:
        single_cell_mesh (trimesh.Trimesh): The input mesh for one guard cell.
        center_point (np.ndarray): The 3D coordinates of the center for rotation.
        output_path (str): Full path to save the final combined .ply mesh file.
    """
    print("\nSTEP 7: Creating full stomata from single guard cell...")
    if single_cell_mesh is None or not isinstance(single_cell_mesh, trimesh.Trimesh):
        print("  Error: Invalid single_cell_mesh provided.")
        return None
    if center_point is None or len(center_point) != 3:
        print("  Error: Invalid center_point provided.")
        return None

    try:
        # --- 1. Duplicate the mesh ---
        mesh1 = single_cell_mesh # Keep the original
        mesh2 = single_cell_mesh.copy()
        print(f"  Duplicated single cell mesh.")

        # --- 2. Rotate the duplicate ---
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.pi,
            direction=[0, 0, 1],
            point=center_point
        )
        mesh2.apply_transform(rotation_matrix)
        print(f"  Rotated duplicate mesh 180 degrees around Z-axis at {center_point}.")

        # --- 3. Combine the meshes ---
        # Store original counts before combining
        num_vertices_mesh1 = len(mesh1.vertices)
        num_faces_mesh1 = len(mesh1.faces)

        # Concatenate meshes
        combined_mesh = trimesh.util.concatenate([mesh1, mesh2])
        print(f"  Concatenated meshes. Vertices before merge: {len(combined_mesh.vertices)}, Faces: {len(combined_mesh.faces)}")

        # --- 4. Create Face Labels and Signals BEFORE Merging ---
        # Labels: 1 for mesh1 faces, 2 for mesh2 faces
        num_faces_total_pre_merge = len(combined_mesh.faces)
        face_labels_pre_merge = np.ones(num_faces_total_pre_merge, dtype=np.int32) # Use int32
        face_labels_pre_merge[num_faces_mesh1:] = 2
        # Dummy signal: Let's use 1.0 for all faces
        face_signals_pre_merge = np.ones(num_faces_total_pre_merge, dtype=np.float32) # Use float32

        # --- 5. Create Vertex Labels and Signals BEFORE Merging ---
        # Labels: 0 for all vertices (as per example)
        vertex_labels_pre_merge = np.zeros(len(combined_mesh.vertices), dtype=np.int32)
        # Signal: 1.0 for all vertices (as per example)
        vertex_signals_pre_merge = np.ones(len(combined_mesh.vertices), dtype=np.float32)

        # --- 6. Merge Vertices ---
        # Trimesh's merge_vertices might affect attribute handling.
        # We need to see if attributes persist or need re-assignment.
        # Store pre-merge attributes in case they are needed for re-mapping later.
        temp_vertex_attributes = {
            'label': vertex_labels_pre_merge,
            'signal': vertex_signals_pre_merge
        }
        temp_face_attributes = {
            'label': face_labels_pre_merge,
            'signal': face_signals_pre_merge
        }

        # Perform the merge
        combined_mesh.merge_vertices()
        print(f"  Merged vertices. Final vertex count: {len(combined_mesh.vertices)}")
        # Note: Merging vertices doesn't usually remove faces, but can make some degenerate.
        # Let's assume face count remains the same unless specific issues arise.
        num_faces_total_post_merge = len(combined_mesh.faces)
        print(f"  Final face count: {num_faces_total_post_merge}")


        # --- 7. Assign Vertex Attributes ---
        # Assign vertex attributes matching the *final* number of vertices.
        # Using label 0 and signal 1 as per example. Using standard types.
        final_vertex_labels = np.zeros(len(combined_mesh.vertices), dtype=np.int32) # Back to int32
        final_vertex_signals = np.ones(len(combined_mesh.vertices), dtype=np.float32) # Back to float32

        combined_mesh.vertex_attributes['label'] = final_vertex_labels
        combined_mesh.vertex_attributes['signal'] = final_vertex_signals # Re-added signal
        print(f"  Assigned vertex attributes 'label' (all 0) and 'signal' (all 1).") # Updated message

        # --- 8. Assign Face Attributes ---
        # Assign face attributes matching the *final* number of faces.
        # Use standard types.
        if num_faces_total_post_merge != num_faces_total_pre_merge:
             print("  Warning: Face count changed after merging vertices. Face labels might be inaccurate.")
             final_face_labels = np.ones(num_faces_total_post_merge, dtype=np.int32) # Back to int32
             split_point = min(num_faces_mesh1, num_faces_total_post_merge)
             final_face_labels[split_point:] = 2
             final_face_signals = np.ones(num_faces_total_post_merge, dtype=np.float32) # Back to float32
        else:
             # Ensure correct type even if count didn't change
             final_face_labels = face_labels_pre_merge.astype(np.int32) # Back to int32
             final_face_signals = face_signals_pre_merge.astype(np.float32) # Back to float32

        # Store in face_data
        if not hasattr(combined_mesh, 'face_data'):
             combined_mesh.face_data = trimesh.caching.DataStore()
        combined_mesh.face_data['label'] = final_face_labels
        combined_mesh.face_data['signal'] = final_face_signals # Re-added signal
        count1 = np.sum(final_face_labels == 1)
        count2 = np.sum(final_face_labels == 2)
        print(f"  Assigned face attributes 'label' ({count1} faces=1, {count2} faces=2) and 'signal' (all 1).") # Updated message


        # --- 9. Export the final mesh ---
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  Created output directory: {output_dir}")

        # Export with vertex and face attributes
        export_result = combined_mesh.export(
            file_obj=output_path,
            file_type='ply',
            encoding='ascii',
            vertex_normal=False
        )
        print(f"  Successfully saved combined mesh with vertex & face attributes to: {output_path}")

        return combined_mesh

    except Exception as e:
        print(f"  Error during duplication, rotation, combination, or export: {e}")
        traceback.print_exc()
        return None

# --- Modified Example Usage ---
if __name__ == '__main__':
    # --- Select ONE file to process ---

    mesh_list = ["Meshes/OBJ/Ac_DA_1_3.obj", "Meshes/OBJ/Ac_DA_1_2.obj", "Meshes/OBJ/Ac_DA_1_5.obj",
        "Meshes/OBJ/Ac_DA_1_4.obj", "Meshes/OBJ/Ac_DA_3_7.obj", "Meshes/OBJ/Ac_DA_3_6.obj",
        "Meshes/OBJ/Ac_DA_3_4.obj", "Meshes/OBJ/Ac_DA_3_3.obj", "Meshes/OBJ/Ac_DA_3_2.obj",
        "Meshes/OBJ/Ac_DA_3_1.obj", "Meshes/OBJ/Ac_DA_2_7.obj", "Meshes/OBJ/Ac_DA_2_6b.obj",
        "Meshes/OBJ/Ac_DA_2_6a.obj", "Meshes/OBJ/Ac_DA_2_4.obj", "Meshes/OBJ/Ac_DA_2_3.obj",
        "Meshes/OBJ/Ac_DA_1_8_mesh.obj", "Meshes/OBJ/Ac_DA_1_6.obj"]
    
    # Define output paths for BOTH standard and bulging meshes
    results_dir = "results" # Define results directory

    for file_to_process in mesh_list:
        #file_to_process = "Meshes/OBJ/Ac_DA_1_3.obj" # Example file
        base_name = os.path.splitext(os.path.basename(file_to_process))[0]

        single_cell_output_std = os.path.join(results_dir, f"single_guard_cell_{base_name}_std.ply")
        full_stomata_output_std = os.path.join(results_dir, f"full_stomata_{base_name}_std.ply")
        single_cell_output_bulge = os.path.join(results_dir, f"single_guard_cell_{base_name}_bulge.ply")
        full_stomata_output_bulge = os.path.join(results_dir, f"full_stomata_{base_name}_bulge.ply")

        # Create results directory if needed
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created directory: {results_dir}")


        if not os.path.exists(file_to_process):
            print(f"Error: Input file not found: {file_to_process}")
        else:
            # --- Generate the STANDARD single guard cell first ---
            print("=" * 30)
            print("GENERATING STANDARD MESH")
            print("=" * 30)
            single_cell_mesh_std, center_point_std = generate_single_guard_cell(
                file_to_process,
                single_cell_output_std, # Use the correct variable
                num_centerline_segments=128, # Use higher resolution for smoother curves
                num_cross_section_points=64,
                visualize_steps=True
            )
            print("-" * 20)

            # --- Create the STANDARD full stomata ---
            if single_cell_mesh_std is not None and center_point_std is not None:
                create_full_stomata_from_half(
                    single_cell_mesh_std,
                    center_point_std,
                    full_stomata_output_std # Use the correct variable
                )
            else:
                print("\nSkipping STANDARD full stomata creation due to errors in single cell generation.")


            # --- Generate the BULGING single guard cell ---
            print("\n" + "=" * 30)
            print("GENERATING BULGING MESH")
            print("=" * 30)
            single_cell_mesh_bulge, center_point_bulge = generate_single_bulging_guard_cell(
                input_file_path=file_to_process,
                output_path=single_cell_output_bulge, # Now this variable is defined
                num_centerline_segments=128, # Keep consistent resolution
                num_cross_section_points=64,
                visualize_steps=True,
                min_aspect_ratio=1.0, # Adjust this value as needed (e.g., 1.0 for circular ends)
                transition_power=1.0
            )
            print("-" * 20)

            # --- Create the BULGING full stomata ---
            if single_cell_mesh_bulge is not None and center_point_bulge is not None:
                create_full_stomata_from_half(
                    single_cell_mesh_bulge,
                    center_point_bulge, # Use the center calculated during its generation
                    full_stomata_output_bulge # Now this variable is defined
                )
            else:
                print("\nSkipping BULGING full stomata creation due to errors in single cell generation.")

        print("\nProcessing complete.")
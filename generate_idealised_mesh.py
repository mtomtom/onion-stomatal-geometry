import numpy as np
import trimesh
import os
from scipy.optimize import curve_fit
from shapely.geometry import Polygon, Point
import traceback
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# --- Import necessary functions ---
try:
    from midpoint_cross_sections import analyze_midpoint_cross_section
    from cross_section_functions import ellipse as fit_ellipse_func
except ImportError as e:
    print(f"ERROR: Could not import necessary functions: {e}")
    analyze_midpoint_cross_section = None
    fit_ellipse_func = None

# --- (Keep generate_average_mesh_reconstruction if you might use it later) ---
def generate_average_mesh_reconstruction(file_paths, output_path, reference_index=0, icp_threshold=0.1, poisson_depth=8):
    # ... (existing reconstruction code) ...
    pass

# --- Helper function for Aspect Ratio Modulation ---
def get_modulated_AR(theta, cl_phi, AR_mid, AR_min):
    """
    Calculates a modulated aspect ratio based on the angle along the centerline.
    AR is AR_mid at the major axis points (theta = cl_phi or cl_phi+pi)
    and transitions smoothly to AR_min at the ends/tips (minor axis points: theta = cl_phi+pi/2 or cl_phi+3pi/2).
    """
    if AR_mid <= AR_min:
        return AR_mid # No modulation needed

    # Calculate angular distance from the *nearest* major axis point (0 to pi/2)
    relative_angle = (theta - cl_phi) % (2 * np.pi)
    dist_to_major = min(relative_angle % np.pi, np.pi - (relative_angle % np.pi)) # 0 at major, pi/2 at minor

    # Use cosine modulation: factor is 1 at major axis (dist=0), 0 at minor axis (dist=pi/2)
    modulation_factor = (np.cos(dist_to_major * 2) + 1) / 2.0

    # Interpolate between AR_mid (at major axis, factor=1) and AR_min (at minor axis, factor=0)
    # current_AR = AR_min + (AR_mid - AR_min) * modulation_factor # <<< OLD LOGIC (AR_min at major)
    current_AR = AR_min + (AR_mid - AR_min) * modulation_factor # <<< CORRECTED LOGIC (AR_mid at major)
    # Let's re-verify the interpolation:
    # When modulation_factor = 1 (major axis): AR_min + (AR_mid - AR_min) * 1 = AR_mid
    # When modulation_factor = 0 (minor axis): AR_min + (AR_mid - AR_min) * 0 = AR_min
    # This seems correct now.

    return current_AR

# --- Modified Function (Sweep Half + Duplicate/Rotate - WITH CAPPING for Wall) ---
def generate_single_ideal_mesh_two_cells_ply(input_file_path, output_path, num_centerline_segments=64, num_cross_section_points=32):
    """
    Generates an idealized mesh composed of two separate but touching "guard cell"
    meshes WITH AN INTERNAL DIVIDING WALL. It sweeps one half-cross-section,
    ensures it's capped (creating the wall surface), duplicates it, rotates
    the duplicate, and combines them.
    Exports a single PLY file with a custom face property 'label' (1 for cell 1,
    2 for cell 2) stored in mesh.face_data.

    Args:
        input_file_path (str): Path to the input mesh file (.obj).
        output_path (str): Full path to save the generated .ply mesh file.
        num_centerline_segments (int): Number of points defining the centerline path.
        num_cross_section_points (int): Total number of points desired for the *full*
                                        cross-section ellipse perimeter before splitting.
                                        (should be an even number).
    """
    print(f"\nGenerating idealized mesh (2 cells, PLY, sweep half + rotate, with wall) for: {input_file_path}")
    if analyze_midpoint_cross_section is None or fit_ellipse_func is None:
        print("  Error: Required analysis functions not imported.")
        return
    if num_cross_section_points % 2 != 0:
        print("  Warning: num_cross_section_points should be even. Adjusting...")
        num_cross_section_points += 1
    # Number of points for the half-ellipse arc + 2 endpoints = num_cross_section_points // 2 + 1
    num_half_points = num_cross_section_points // 2 + 1


    # --- 1. Analyze Midpoint Cross-Section (Same as before) ---
    print("  Analyzing midpoint cross-section...")
    # ... (analysis code remains the same) ...
    midpoint_results = analyze_midpoint_cross_section([input_file_path], visualize=False)
    if not midpoint_results or input_file_path not in midpoint_results or midpoint_results[input_file_path] is None:
        print("  Error: Midpoint analysis failed or returned no data.")
        return
    analysis_data = midpoint_results[input_file_path]
    if len(analysis_data) < 5 or analysis_data[3] is None or analysis_data[4] is None or not np.isfinite(analysis_data[3]):
        print("  Error: Midpoint analysis did not return valid aspect ratio or minor radius.")
        return
    cross_section_aspect_ratio = analysis_data[3]
    minor_radius_est = analysis_data[4]
    print(f"  Derived Parameters: Aspect Ratio={cross_section_aspect_ratio:.3f}, Minor Radius={minor_radius_est:.3f}")


    # --- 2. Determine Centerline Parameters (Ellipse Fit) (Same as before) ---
    print("  Determining centerline parameters...")
    center = None # Initialize center
    cl_phi = 0.0 # Initialize cl_phi
    center_xy = None # Initialize center_xy
    try:
        mesh = trimesh.load_mesh(input_file_path)
        center = mesh.centroid # Store center for rotation later
        # ... (rest of ray casting and ellipse fitting logic remains the same) ...
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
             raise ValueError("Could not determine dimensions via ray casting for centerline.")
        inner_points = np.array(inner_points)
        outer_points = np.array(outer_points)
        raw_centerline_points = (inner_points + outer_points) / 2
        xy_centerline = raw_centerline_points[:, :2]
        center_xy = center[:2] # Store center_xy for path calculation
        r = np.linalg.norm(xy_centerline - center_xy, axis=1)
        theta = np.arctan2(xy_centerline[:, 1] - center_xy[1], xy_centerline[:, 0] - center_xy[0])
        major_radius_est = np.mean(r)
        initial_guess = [major_radius_est, major_radius_est, 0]
        params, _ = curve_fit(fit_ellipse_func, theta, r, p0=initial_guess)
        cl_a, cl_b, cl_phi = params
        cl_phi = cl_phi % np.pi
        if cl_a < cl_b:
            cl_a, cl_b = cl_b, cl_a
            cl_phi += np.pi/2
            cl_phi = cl_phi % np.pi
        print(f"  Centerline Ellipse Fit: a={cl_a:.3f}, b={cl_b:.3f}, phi={np.degrees(cl_phi):.1f} deg")
    except Exception as e:
        print(f"  Error determining centerline parameters: {e}")
        return
    if center is None or center_xy is None:
        print("  Error: Center point not determined.")
        return


    # --- 3. Define ONE Half Cross-Section Shape (Direct Angle Method) ---
    # (Using the refined definition from the previous attempt)
    print(f"  Defining half-ellipse polygon with {num_half_points} vertices...")
    cs_a = minor_radius_est * np.sqrt(cross_section_aspect_ratio)
    cs_b = minor_radius_est / np.sqrt(cross_section_aspect_ratio)
    half_angles = np.linspace(-np.pi / 2, np.pi / 2, num_half_points)
    half_cs_x = cs_a * np.cos(half_angles)
    half_cs_y = cs_b * np.sin(half_angles)
    vertices1 = np.column_stack([half_cs_x, half_cs_y])
    try:
        polygon1_shapely = Polygon(vertices1)
        if polygon1_shapely.is_empty or not polygon1_shapely.is_valid:
             raise ValueError("Created cross-section half-polygon is invalid or empty.")
        expected_area = np.pi * cs_a * cs_b / 2
        print(f"  Defined one half-polygon. Area: {polygon1_shapely.area:.4f} (Expected ~{expected_area:.4f})")
    except Exception as e:
        print(f"  Error creating Shapely polygon for cross-section half: {e}")
        return
    # --- End Modification ---

    # --- 4. Define Centerline Path (3D Ellipse) (Same as before) ---
    cl_theta_path = np.linspace(0, 2 * np.pi, num_centerline_segments)
    cl_r_path = fit_ellipse_func(cl_theta_path, cl_a, cl_b, cl_phi)
    cl_x_path = center_xy[0] + cl_r_path * np.cos(cl_theta_path)
    cl_y_path = center_xy[1] + cl_r_path * np.sin(cl_theta_path)
    cl_z_path = np.full_like(cl_x_path, center[2])
    centerline_path_3d = np.column_stack([cl_x_path, cl_y_path, cl_z_path])


    # --- 5. Perform Sweep for ONE Half (Ensure Capping) ---
    print("  Sweeping cross-section half along centerline...")
    try:
        mesh1 = trimesh.creation.sweep_polygon(
            polygon=polygon1_shapely,
            path=centerline_path_3d
            # sweep_polygon with a closed path *should* automatically cap
        )
        print(f"    Generated mesh for cell 1 ({len(mesh1.vertices)} vertices, {len(mesh1.faces)} faces).")

        # Explicitly check and attempt to cap if needed
        if not mesh1.is_watertight:
            print("    Mesh 1 not watertight after sweep, attempting fill_holes() to create caps.")
            mesh1.fill_holes()
            if mesh1.is_watertight:
                 print("    Mesh 1 successfully capped using fill_holes().")
            else:
                 # If still not watertight, something is wrong with the geometry/sweep
                 raise ValueError("Mesh 1 could not be made watertight after sweep and fill_holes(). Cannot guarantee dividing wall.")
        else:
            print("    Mesh 1 is watertight after sweep (caps should exist).")

        if not mesh1.faces.shape[0] > 0:
            raise ValueError("Generated mesh1 has no faces.")

    except Exception as e:
        print(f"  Error during mesh sweep or capping: {e}")
        traceback.print_exc()
        return
    # --- End Modification ---

    # --- 6. Duplicate, Rotate, Combine, Add Face Data, and Export ---
    # (Same as before, but now mesh1 should have caps)
    try:
        print("  Duplicating and rotating capped mesh...")
        mesh2 = mesh1.copy()
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.pi,
            direction=[0, 0, 1],
            point=center
        )
        mesh2.apply_transform(rotation_matrix)
        print(f"    Created mesh 2 by rotating mesh 1.")

        # Concatenate meshes - the caps should now form the internal wall
        combined_mesh = trimesh.util.concatenate([mesh1, mesh2])
        print(f"  Combined mesh has {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces.")

        # --- Create Face Label Attribute and store in face_data ---
        num_faces_mesh1 = len(mesh1.faces)
        num_faces_total = len(combined_mesh.faces)
        face_labels = np.ones(num_faces_total, dtype=np.uint8)
        face_labels[num_faces_mesh1:] = 2

        # --- Store labels in mesh.face_data ---
        if not hasattr(combined_mesh, 'face_data'):
             print("  Warning: combined_mesh object missing face_data attribute. Attempting to create.")
             combined_mesh.face_data = trimesh.caching.DataStore()
        combined_mesh.face_data['label'] = face_labels
        print(f"  Stored face labels (1 or 2) in mesh.face_data['label']")
        # --- End Face Label Creation ---

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  Created output directory: {output_dir}")

        # Export as PLY (ASCII)
        export_result = combined_mesh.export(
            file_obj=output_path,
            file_type='ply',
            encoding='ascii' # Keep ASCII for now
        )
        print(f"  Successfully saved combined mesh with internal wall to {output_path} (PLY ASCII format)")
        print(f"  NOTE: Mesh should have an internal dividing wall. Faces also have a 'label' property (1 or 2).")

    except Exception as e:
        print(f"  Error duplicating, combining meshes or exporting PLY: {e}")
        traceback.print_exc()
    # --- End Modification ---


def generate_single_guard_cell(input_file_path, output_path, num_centerline_segments=64, 
                              num_cross_section_points=64, visualize_steps=True):
    """
    Step-by-step generation of a SINGLE guard cell by:
    1. Getting the centerline ellipse
    2. Getting the cross-section ellipse
    3. Optionally visualizing these elements
    4. Sweeping the cross-section along HALF of the centerline, divided parallel to the major axis
    5. Adding VERTICAL caps to the open ends to ensure proper alignment when duplicated
    6. Saving as PLY for verification
    
    Args:
        input_file_path (str): Path to the input mesh file (.obj)
        output_path (str): Path to save the generated single guard cell (.ply)
        num_centerline_segments (int): Number of points defining the centerline path
        num_cross_section_points (int): Number of points defining the cross-section
        visualize_steps (bool): Whether to create plots for verification
    """
    print(f"\nGenerating SINGLE guard cell for: {input_file_path}")
    
    # --- STEP 1: Get Centerline, Inner Points, and Average Inner Radius ---
    print("STEP 1: Extracting centerline, inner points, and average inner radius...")
    center = None
    cl_phi = 0.0
    center_xy = None
    cl_a, cl_b = 0.0, 0.0
    avg_inner_radius = 0.0 # Initialize average inner radius

    try:
        mesh = trimesh.load_mesh(input_file_path)
        center = mesh.centroid

        # Ray casting to find inner and outer points
        ray_count = 36
        ray_angles = np.linspace(0, 2*np.pi, ray_count, endpoint=False)
        inner_points = []
        outer_points = [] # Still need outer for centerline calculation

        for angle in ray_angles:
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            locations, _, _ = mesh.ray.intersects_location([center], [direction])
            if len(locations) >= 2:
                dists = np.linalg.norm(locations - center, axis=1)
                sorted_idx = np.argsort(dists)
                inner_points.append(locations[sorted_idx[0]])
                outer_points.append(locations[sorted_idx[-1]])

        if not inner_points or not outer_points:
            raise ValueError("Could not determine dimensions via ray casting.")

        inner_points = np.array(inner_points)
        outer_points = np.array(outer_points)

        # --- Calculate Average Inner Radius ---
        inner_radii = np.linalg.norm(inner_points - center, axis=1)
        avg_inner_radius = np.mean(inner_radii)
        if avg_inner_radius <= 0:
             raise ValueError(f"Calculated average inner radius is invalid: {avg_inner_radius}")
        print(f"  Calculated Average Inner Radius from Ray Casting: {avg_inner_radius:.4f}")
        # --- End Inner Radius Calculation ---

        # Fit centerline ellipse (using midpoint between inner/outer)
        raw_centerline_points = (inner_points + outer_points) / 2
        xy_centerline = raw_centerline_points[:, :2]
        center_xy = center[:2]
        r = np.linalg.norm(xy_centerline - center_xy, axis=1)
        theta = np.arctan2(xy_centerline[:, 1] - center_xy[1], xy_centerline[:, 0] - center_xy[0])
        major_radius_est = np.mean(r)
        initial_guess = [major_radius_est, major_radius_est, 0]
        params, _ = curve_fit(fit_ellipse_func, theta, r, p0=initial_guess)
        cl_a, cl_b, cl_phi = params
        cl_phi = cl_phi % np.pi
        if cl_a < cl_b: # Ensure cl_a is major axis
            cl_a, cl_b = cl_b, cl_a
            cl_phi += np.pi/2
            cl_phi = cl_phi % np.pi

        print(f"  Centerline parameters: a={cl_a:.3f}, b={cl_b:.3f}, phi={np.degrees(cl_phi):.1f}°")
        print(f"  Center point: {center}")

        # Check if centerline minor axis is larger than inner radius
        if cl_b <= avg_inner_radius:
            print(f"  Warning: Centerline semi-minor axis ({cl_b:.4f}) is not larger than average inner radius ({avg_inner_radius:.4f}). Cross-section calculation might be inaccurate.")
            # Fallback: Use a small fraction of cl_b for cs_b? Or revert to previous method?
            # Let's proceed but be aware the result might be odd.

    except Exception as e:
        print(f"  Error determining centerline parameters or inner radius: {e}")
        traceback.print_exc()
        return None, None

    # --- STEP 2: Get Cross-Section AR and Width from Midpoint Analysis ---
    print("STEP 2: Analyzing midpoint cross-section for AR and Width...")
    cs_a, cs_b = 0.0, 0.0 # Initialize
    analysis_aspect_ratio = 1.0 # Default
    pca_minor_std_dev = None # Default width measure

    try:
        # Call the updated midpoint analysis function
        # Pass cl_phi if the function expects it (depends on which version of midpoint_cross_sections you settled on)
        # Assuming the version that DOES NOT use cl_phi:
        midpoint_results = analyze_midpoint_cross_section(
            [input_file_path],
            visualize=visualize_steps # Pass output_dir if needed by the function
        )

        # Extract results from the tuple: (pts2d, pts3d, transform, aspect_ratio, pca_minor_std_dev)
        if midpoint_results and input_file_path in midpoint_results and midpoint_results[input_file_path]:
             analysis_data_tuple = midpoint_results[input_file_path]
             if isinstance(analysis_data_tuple, tuple) and len(analysis_data_tuple) >= 5:
                 ar = analysis_data_tuple[3] # Index 3 is aspect ratio
                 width_measure = analysis_data_tuple[4] # Index 4 is pca_minor_std_dev

                 if ar is not None and np.isfinite(ar):
                     analysis_aspect_ratio = ar
                     if analysis_aspect_ratio < 1.0:
                         print(f"  Inverting analysis aspect ratio ({analysis_aspect_ratio:.4f}) to >= 1.")
                         analysis_aspect_ratio = 1.0 / analysis_aspect_ratio
                 else: print("  Warning: Midpoint analysis returned invalid aspect ratio.")

                 if width_measure is not None and np.isfinite(width_measure) and width_measure > 1e-9:
                     pca_minor_std_dev = width_measure
                     print(f"  PCA Width (Minor Axis StdDev): {pca_minor_std_dev:.4f}")
                 else: print("  Warning: Midpoint analysis returned invalid PCA width measure.")

             else: print("  Warning: Midpoint analysis returned unexpected data format or length.")
        else: print("  Warning: Midpoint analysis failed or returned no data.")

        print(f"  Using Analysis Results: AR={analysis_aspect_ratio:.4f}, Width(b)={pca_minor_std_dev if pca_minor_std_dev else 'N/A'}")

    except Exception as analysis_err:
         print(f"  Error during midpoint analysis call: {analysis_err}. Cannot determine cross-section params.")
         traceback.print_exc()
         return None, None # Cannot proceed without analysis results

    # --- Calculation using width and AR from midpoint analysis ---
    if pca_minor_std_dev is None:
         print("  FATAL: Could not determine cross-section width from midpoint analysis. Aborting.")
         return None, None
    
    pca_width_scale_factor = 1.43 # <<< ADJUST THIS FACTOR AS NEEDED
    print(f"  Applying PCA width scale factor: {pca_width_scale_factor:.3f}")

    # 1. Set cs_b directly from the PCA width measure
    cs_b = pca_minor_std_dev * pca_width_scale_factor
    print(f"  Set cs_b (semi-minor axis) directly from PCA width: {cs_b:.4f}")

    
    # 2. Calculate cs_a using the analysis aspect ratio
    cs_a = cs_b * analysis_aspect_ratio

    # Ensure cs_a is major axis
    if cs_a < cs_b:
         # This shouldn't happen if analysis_aspect_ratio >= 1, but check anyway
         print(f"  Warning: Calculated cs_a ({cs_a:.4f}) < cs_b ({cs_b:.4f}). Check AR calculation. Swapping.")
         cs_a, cs_b = cs_b, cs_a

    print(f"  Final Cross-section parameters (from Midpoint Analysis): a={cs_a:.4f}, b={cs_b:.4f}")

# --- End of STEP 2 block ---

    # --- STEP 3: Visualize centerline and cross-section ---
    if visualize_steps:
        print("STEP 3: Visualizing centerline and cross-section...")
        try:
            # Create figure with two subplots
            fig = plt.figure(figsize=(12, 6))
            
            # Plot 1: Centerline ellipse
            ax1 = fig.add_subplot(121)
            centerline_theta = np.linspace(0, 2*np.pi, 100)
            centerline_r = fit_ellipse_func(centerline_theta, cl_a, cl_b, cl_phi)
            centerline_x = center_xy[0] + centerline_r * np.cos(centerline_theta)
            centerline_y = center_xy[1] + centerline_r * np.sin(centerline_theta)
            
            # Plot full centerline
            ax1.plot(centerline_x, centerline_y, 'b-', label='Full centerline')
            
            # Calculate the angles for half the ellipse PARALLEL to the major axis
            # The major axis is at angle cl_phi, so we take points from cl_phi to cl_phi+π
            half_start = cl_phi  # Start at the major axis
            half_end = cl_phi + np.pi  # End at the opposite side of the major axis
            
            # Find indices where theta is within our half-ellipse angle range
            half_indices = []
            for i, theta in enumerate(centerline_theta):
                # Normalize angles to [0, 2π) for comparison
                theta_norm = theta % (2*np.pi)
                half_start_norm = half_start % (2*np.pi)
                half_end_norm = half_end % (2*np.pi)
                
                # Handle the case where our range crosses the 0/2π boundary
                if half_end_norm < half_start_norm:
                    if theta_norm >= half_start_norm or theta_norm <= half_end_norm:
                        half_indices.append(i)
                else:
                    if half_start_norm <= theta_norm <= half_end_norm:
                        half_indices.append(i)
            
            # Plot the half along the major axis (symmetry line)
            ax1.plot(centerline_x[half_indices], centerline_y[half_indices], 'r-', linewidth=3, 
                    label='Half centerline (parallel to major axis)')
            
            # Plot major axis line
            major_axis_x = [center_xy[0] - 1.5*cl_a*np.cos(cl_phi), center_xy[0] + 1.5*cl_a*np.cos(cl_phi)]
            major_axis_y = [center_xy[1] - 1.5*cl_a*np.sin(cl_phi), center_xy[1] + 1.5*cl_a*np.sin(cl_phi)]
            ax1.plot(major_axis_x, major_axis_y, 'g--', linewidth=2, label='Major axis')
            
            # Plot division line (perpendicular to major axis)
            division_vec = np.array([-np.sin(cl_phi), np.cos(cl_phi)])  # Perpendicular to major axis
            division_x = [center_xy[0] - 1.5*cl_b*division_vec[0], center_xy[0] + 1.5*cl_b*division_vec[0]]
            division_y = [center_xy[1] - 1.5*cl_b*division_vec[1], center_xy[1] + 1.5*cl_b*division_vec[1]]
            ax1.plot(division_x, division_y, 'm--', linewidth=2, label='Division line')
            
            # Plot center and original points
            ax1.scatter(center_xy[0], center_xy[1], color='black', s=50, label='Center')
            ax1.scatter(xy_centerline[:, 0], xy_centerline[:, 1], color='green', s=30, alpha=0.7, 
                       label='Extracted points')
            
            ax1.set_title('Centerline Ellipse with Major Axis Division')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.axis('equal')
            ax1.grid(True)
            ax1.legend()
            
            # Save figure
            viz_dir = os.path.dirname(output_path)
            if viz_dir and not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            
            viz_path = os.path.join(viz_dir, 'visualization_step3.png')
            plt.tight_layout()
            plt.savefig(viz_path)
            plt.close(fig)
            print(f"  Visualization saved to: {viz_path}")
            
        except Exception as e:
            print(f"  Error during visualization: {e}")
            traceback.print_exc()
            # Continue execution despite visualization errors
    
    # --- STEP 4: Define HALF centerline parallel to major axis and create cross-section polygon ---
    print("STEP 4: Defining HALF centerline path (parallel to major axis) and cross-section...")
    try:
        # Define angles for the half-centerline parallel to the major axis
        # Take the range of angles from cl_phi to cl_phi+π
        half_start_angle = cl_phi
        half_end_angle = cl_phi + np.pi
        
        # Ensure the angles are in [0, 2π] range
        while half_start_angle < 0:
            half_start_angle += 2*np.pi
        while half_end_angle > 2*np.pi:
            half_end_angle -= 2*np.pi
        
        # If half_end_angle < half_start_angle, we need to handle the wrap-around
        # by dividing into two segments
        if half_end_angle < half_start_angle:
            # First segment: half_start_angle to 2π
            theta1 = np.linspace(half_start_angle, 2*np.pi, num_centerline_segments//4 + 1)[:-1]
            # Second segment: 0 to half_end_angle
            theta2 = np.linspace(0, half_end_angle, num_centerline_segments//4 + 1)
            half_theta = np.concatenate([theta1, theta2])
        else:
            # No wrap-around, just take angles from half_start_angle to half_end_angle
            half_theta = np.linspace(half_start_angle, half_end_angle, num_centerline_segments//2 + 1)
        
        # Calculate points on the centerline
        half_r = fit_ellipse_func(half_theta, cl_a, cl_b, cl_phi)
        half_x = center_xy[0] + half_r * np.cos(half_theta)
        half_y = center_xy[1] + half_r * np.sin(half_theta)
        half_z = np.full_like(half_x, center[2])
        half_centerline = np.column_stack([half_x, half_y, half_z])
        
        print(f"  Created half-centerline with {len(half_centerline)} points, divided parallel to major axis.")
        
        # Create cross-section polygon (full ellipse using cs_a, cs_b from Step 2)
        cs_angles = np.linspace(0, 2*np.pi, num_cross_section_points, endpoint=False)
        cs_x = cs_a * np.cos(cs_angles) # Uses cs_a calculated in Step 2
        cs_y = cs_b * np.sin(cs_angles) # Uses cs_b calculated in Step 2
        cs_vertices = np.column_stack([cs_x, cs_y])
        cs_polygon = Polygon(cs_vertices)
        
        if not cs_polygon.is_valid:
            raise ValueError("Created cross-section polygon is invalid.")
            
        print(f"  Created idealized cross-section polygon with {len(cs_vertices)} vertices.")
        
    except Exception as e:
        print(f"  Error creating half-centerline or cross-section: {e}")
        traceback.print_exc()
        return None, None
    
    # --- STEP 5: Sweep, Project End Vertices, and Create Planar Caps ---
    print("STEP 5: Sweep, project ends, and create planar caps...")
    try:
        # Perform sweep WITHOUT automatic capping
        single_cell = trimesh.creation.sweep_polygon(
            polygon=cs_polygon,
            path=half_centerline,
            caps=False
        )
        print(f"  Generated initial swept mesh: {len(single_cell.vertices)} vertices, {len(single_cell.faces)} faces")

        # Get endpoint coordinates and plane normal
        start_point = half_centerline[0]
        end_point = half_centerline[-1]
        vertical_normal = np.array([-np.sin(cl_phi), np.cos(cl_phi), 0])
        vertical_normal /= np.linalg.norm(vertical_normal)

        print("  Identifying and projecting end vertices onto planar caps...")

        # --- Identify Vertices at Start and End Profiles ---
        # We assume the sweep function generates vertices in order along the path.
        # The first num_cross_section_points vertices belong to the start profile.
        # The last num_cross_section_points vertices belong to the end profile.
        # Note: This assumption might be fragile with complex sweeps, but likely holds here.
        if len(single_cell.vertices) < 2 * num_cross_section_points:
             raise ValueError("Sweep did not generate enough vertices to identify end profiles.")

        start_profile_indices = np.arange(num_cross_section_points)
        end_profile_indices = np.arange(len(single_cell.vertices) - num_cross_section_points, len(single_cell.vertices))

        # --- Project Identified Vertices onto the Correct Planes ---
        projected_start_vertices = []
        for idx in start_profile_indices:
            v = single_cell.vertices[idx]
            # Project v onto the plane defined by start_point and vertical_normal
            dist = np.dot(v - start_point, vertical_normal)
            v_projected = v - dist * vertical_normal
            projected_start_vertices.append(v_projected)
            # Update the vertex position in the mesh directly
            single_cell.vertices[idx] = v_projected

        projected_end_vertices = []
        for idx in end_profile_indices:
            v = single_cell.vertices[idx]
            # Project v onto the plane defined by end_point and vertical_normal
            dist = np.dot(v - end_point, vertical_normal)
            v_projected = v - dist * vertical_normal
            projected_end_vertices.append(v_projected)
            # Update the vertex position in the mesh directly
            single_cell.vertices[idx] = v_projected

        print(f"  Projected {len(start_profile_indices)} start and {len(end_profile_indices)} end vertices.")

        # --- Verification (Optional but Recommended) ---
        for idx in start_profile_indices:
            plane_eq = np.dot(single_cell.vertices[idx] - start_point, vertical_normal)
            if abs(plane_eq) > 1e-9: # Use a slightly larger tolerance after projection
                print(f"  Warning: Projected start vertex {idx} not exactly on plane. Error: {plane_eq}")
        for idx in end_profile_indices:
             plane_eq = np.dot(single_cell.vertices[idx] - end_point, vertical_normal)
             if abs(plane_eq) > 1e-9:
                 print(f"  Warning: Projected end vertex {idx} not exactly on plane. Error: {plane_eq}")
        # --- End Verification ---

        # --- Triangulate Caps using the (Now Projected) Profile Vertices ---
        # Calculate center points using the *projected* vertices
        # Convert lists back to numpy arrays for mean calculation
        projected_start_vertices = np.array(projected_start_vertices)
        projected_end_vertices = np.array(projected_end_vertices)
        start_center = np.mean(projected_start_vertices, axis=0)
        end_center = np.mean(projected_end_vertices, axis=0)

        # Add center points as new vertices
        center_start_idx = len(single_cell.vertices)
        center_end_idx = center_start_idx + 1
        single_cell.vertices = np.vstack([single_cell.vertices, [start_center, end_center]])

        # Create faces using the *original indices* of the projected vertices
        start_faces = []
        for i in range(num_cross_section_points):
            # Indices refer to the vertices already in single_cell.vertices
            v1_idx = start_profile_indices[i]
            v2_idx = start_profile_indices[(i + 1) % num_cross_section_points]
            start_faces.append([v1_idx, v2_idx, center_start_idx])

        end_faces = []
        for i in range(num_cross_section_points):
            # Indices refer to the vertices already in single_cell.vertices
            v1_idx = end_profile_indices[i]
            v2_idx = end_profile_indices[(i + 1) % num_cross_section_points]
            end_faces.append([v1_idx, v2_idx, center_end_idx]) # Same winding as start (NEW)

        # Add cap faces to the mesh
        if start_faces and end_faces:
            all_new_faces = np.vstack(start_faces + end_faces)
            single_cell.faces = np.vstack([single_cell.faces, all_new_faces])
            print(f"  Added {len(all_new_faces)} cap faces using projected end vertices.")

        # --- Validate Mesh ---
        if not single_cell.is_watertight:
            print("  Warning: Mesh is not watertight after projection and capping.")
            single_cell.fill_holes()
            if single_cell.is_watertight:
                print("  Successfully made mesh watertight with fill_holes().")
            else:
                print("  Warning: Could not make mesh watertight. Check model carefully.")
        else:
            print("  Mesh is watertight with projected ends and planar caps.")

    except Exception as e:
        print(f"  Error during sweep, projection, or capping: {e}")
        traceback.print_exc()
        return None, None # Modified return

    # --- STEP 6: Export single cell PLY ---
    print("STEP 6: Exporting single guard cell...")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export as PLY
        export_result = single_cell.export(
            file_obj=output_path,
            file_type='ply',
            encoding='ascii'
        )
        
        print(f"  Successfully saved single guard cell to: {output_path}")
        print(f"  This mesh represents HALF of the final stomata structure.")
        print(f"  Inspect this mesh to verify it has the correct shape before proceeding.")
        
    except Exception as e:
        print(f"  Error exporting PLY: {e}")
        traceback.print_exc()
        return None, None
        
    return single_cell, center  # Return the mesh for potential further processing

def generate_single_bulging_guard_cell(input_file_path, output_path, num_centerline_segments=64,
                                       num_cross_section_points=64, visualize_steps=True,
                                       min_aspect_ratio=1.1):
    """
    Generates a SINGLE guard cell with varying cross-section aspect ratio.
    The AR decreases towards the tips while maintaining constant cross-sectional area,
    creating a bulging effect.

    Args:
        input_file_path (str): Path to the input mesh file (.obj)
        output_path (str): Path to save the generated single guard cell (.ply)
        num_centerline_segments (int): Number of points defining the centerline path
        num_cross_section_points (int): Number of points defining the cross-section
        visualize_steps (bool): Whether to create plots for verification
        min_aspect_ratio (float): The minimum aspect ratio the cross-section reaches at the tips.

    Returns:
        tuple: (trimesh.Trimesh or None, np.ndarray or None) - The generated mesh and the center point.
    """
    print(f"\nGenerating SINGLE BULGING guard cell (Min AR={min_aspect_ratio:.2f}) for: {input_file_path}")

    # --- STEP 1: Get Centerline, Inner Points (Same as generate_single_guard_cell) ---
    print("STEP 1: Extracting centerline, inner points...")
    center = None
    cl_phi = 0.0
    center_xy = None
    cl_a, cl_b = 0.0, 0.0

    try:
        mesh = trimesh.load_mesh(input_file_path)
        center = mesh.centroid

        # Ray casting (same as original function)
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
        if not inner_points or not outer_points: raise ValueError("Could not determine dimensions via ray casting.")
        inner_points = np.array(inner_points)
        outer_points = np.array(outer_points)

        # Fit centerline ellipse (same as original function)
        raw_centerline_points = (inner_points + outer_points) / 2
        xy_centerline = raw_centerline_points[:, :2]
        center_xy = center[:2]
        r = np.linalg.norm(xy_centerline - center_xy, axis=1)
        theta = np.arctan2(xy_centerline[:, 1] - center_xy[1], xy_centerline[:, 0] - center_xy[0])
        major_radius_est = np.mean(r)
        initial_guess = [major_radius_est, major_radius_est, 0]
        params, _ = curve_fit(fit_ellipse_func, theta, r, p0=initial_guess)
        cl_a, cl_b, cl_phi = params
        cl_phi = cl_phi % np.pi
        if cl_a < cl_b: # Ensure cl_a is major axis
            cl_a, cl_b = cl_b, cl_a
            cl_phi += np.pi/2
            cl_phi = cl_phi % np.pi
        print(f"  Centerline parameters: a={cl_a:.3f}, b={cl_b:.3f}, phi={np.degrees(cl_phi):.1f}°")
        print(f"  Center point: {center}")

    except Exception as e:
        print(f"  Error determining centerline parameters: {e}")
        traceback.print_exc()
        return None, None

    # --- STEP 2: Get Midpoint Cross-Section AR/Width & Calculate Target Area ---
    print("STEP 2: Analyzing midpoint cross-section for AR/Width...")
    cs_a_mid, cs_b_mid = 0.0, 0.0 # Midpoint semi-axes
    AR_mid = 1.0 # Midpoint aspect ratio
    pca_minor_std_dev = None

    try:
        # Midpoint analysis call (same as original function)
        midpoint_results = analyze_midpoint_cross_section([input_file_path], visualize=False) # Keep visualize off here
        if midpoint_results and input_file_path in midpoint_results and midpoint_results[input_file_path]:
             analysis_data_tuple = midpoint_results[input_file_path]
             if isinstance(analysis_data_tuple, tuple) and len(analysis_data_tuple) >= 5:
                 ar = analysis_data_tuple[3]
                 width_measure = analysis_data_tuple[4]
                 if ar is not None and np.isfinite(ar):
                     AR_mid = ar
                     if AR_mid < 1.0: AR_mid = 1.0 / AR_mid # Ensure >= 1
                 if width_measure is not None and np.isfinite(width_measure) and width_measure > 1e-9:
                     pca_minor_std_dev = width_measure
             else: print("  Warning: Midpoint analysis returned unexpected data format.")
        else: print("  Warning: Midpoint analysis failed or returned no data.")

        if pca_minor_std_dev is None:
             print("  FATAL: Could not determine cross-section width from midpoint analysis. Aborting.")
             return None, None

        # Calculate midpoint cs_a, cs_b (same as original function)
        pca_width_scale_factor = 1.43 # Keep consistent
        cs_b_mid = pca_minor_std_dev * pca_width_scale_factor
        cs_a_mid = cs_b_mid * AR_mid
        if cs_a_mid < cs_b_mid: cs_a_mid, cs_b_mid = cs_b_mid, cs_a_mid # Ensure a is major

        # --- Calculate Constant Target Area ---
        target_area = np.pi * cs_a_mid * cs_b_mid
        print(f"  Midpoint Cross-section: a={cs_a_mid:.4f}, b={cs_b_mid:.4f}, AR={AR_mid:.4f}")
        print(f"  Target Constant Cross-section Area: {target_area:.4f}")
        if AR_mid <= min_aspect_ratio:
            print(f"  Warning: Midpoint AR ({AR_mid:.4f}) is already <= Min AR ({min_aspect_ratio:.2f}). No bulging will occur.")

    except Exception as analysis_err:
         print(f"  Error during midpoint analysis call: {analysis_err}. Cannot determine cross-section params.")
         traceback.print_exc()
         return None, None
    
    # --- STEP 3: Visualize (Optional - Shows only midpoint cross-section) ---
    # (Visualization step is less informative here as cross-section varies, keep simplified or remove)
    if visualize_steps:
        print("STEP 3: Visualizing centerline and MIDPOINT cross-section...")
        # (Code similar to original Step 3, but maybe only plot centerline)
        try:
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
            centerline_theta = np.linspace(0, 2*np.pi, 100)
            centerline_r = fit_ellipse_func(centerline_theta, cl_a, cl_b, cl_phi)
            centerline_x = center_xy[0] + centerline_r * np.cos(centerline_theta)
            centerline_y = center_xy[1] + centerline_r * np.sin(centerline_theta)
            ax1.plot(centerline_x, centerline_y, 'b-', label='Full centerline')
            # Plot major axis line
            major_axis_x = [center_xy[0] - 1.1*cl_a*np.cos(cl_phi), center_xy[0] + 1.1*cl_a*np.cos(cl_phi)]
            major_axis_y = [center_xy[1] - 1.1*cl_a*np.sin(cl_phi), center_xy[1] + 1.1*cl_a*np.sin(cl_phi)]
            ax1.plot(major_axis_x, major_axis_y, 'g--', linewidth=1, label='Major axis')
            ax1.scatter(center_xy[0], center_xy[1], color='black', s=50, label='Center')
            ax1.set_title('Centerline Ellipse')
            ax1.set_xlabel('X'); ax1.set_ylabel('Y')
            ax1.axis('equal'); ax1.grid(True); ax1.legend()
            viz_dir = os.path.dirname(output_path)
            if viz_dir and not os.path.exists(viz_dir): os.makedirs(viz_dir)
            viz_path = os.path.join(viz_dir, f'visualization_bulging_step3_{os.path.basename(output_path)}.png')
            plt.tight_layout(); plt.savefig(viz_path); plt.close(fig)
            print(f"  Centerline visualization saved to: {viz_path}")
        except Exception as e:
            print(f"  Error during visualization: {e}")
            traceback.print_exc()


    # --- STEP 4: Define HALF centerline and generate VARYING cross-section polygons ---
    print("STEP 4: Defining HALF centerline path and VARYING cross-sections...")
    cross_section_polygons = []
    half_centerline = None
    try:
        # Define angles for the half-centerline parallel to the major axis (same as original)
        half_start_angle = cl_phi
        half_end_angle = cl_phi + np.pi
        # Use enough segments for a smooth sweep
        num_half_segments = num_centerline_segments // 2 + 1
        half_theta = np.linspace(half_start_angle, half_end_angle, num_half_segments)

        # Calculate points on the half centerline (same as original)
        half_r = fit_ellipse_func(half_theta, cl_a, cl_b, cl_phi)
        half_x = center_xy[0] + half_r * np.cos(half_theta)
        half_y = center_xy[1] + half_r * np.sin(half_theta)
        half_z = np.full_like(half_x, center[2])
        half_centerline = np.column_stack([half_x, half_y, half_z])
        print(f"  Created half-centerline with {len(half_centerline)} points.")

        # --- Generate list of cross-section polygons ---
        cs_base_angles = np.linspace(0, 2*np.pi, num_cross_section_points, endpoint=False)
        for i, theta_cl in enumerate(half_theta):
            # Calculate modulated AR for this point on the centerline
            current_AR = get_modulated_AR(theta_cl, cl_phi, AR_mid, min_aspect_ratio)

            # Calculate current semi-axes ensuring constant area
            # area = pi * a * b = pi * (AR * b) * b = pi * AR * b^2
            # b^2 = area / (pi * AR)
            current_cs_b = np.sqrt(target_area / (np.pi * current_AR))
            current_cs_a = current_AR * current_cs_b

            # Create vertices for this specific cross-section
            cs_x = current_cs_a * np.cos(cs_base_angles)
            cs_y = current_cs_b * np.sin(cs_base_angles)
            cs_vertices = np.column_stack([cs_x, cs_y])

            # Create Shapely polygon
            poly = Polygon(cs_vertices)
            if not poly.is_valid:
                # Attempt to fix simple self-intersections if they occur
                poly = poly.buffer(0)
                if not poly.is_valid:
                     raise ValueError(f"Created cross-section polygon at index {i} is invalid even after buffer(0). AR={current_AR:.3f}, a={current_cs_a:.4f}, b={current_cs_b:.4f}")
            cross_section_polygons.append(poly)

        print(f"  Generated {len(cross_section_polygons)} varying cross-section polygons.")

    except Exception as e:
        print(f"  Error creating half-centerline or varying cross-sections: {e}")
        traceback.print_exc()
        return None, None

    # --- STEP 5: Manually Construct Mesh from Discrete Polygons and Cap ---
    print("STEP 5: Manually constructing mesh from discrete polygons and capping...")
    single_cell = None
    try:
        if not cross_section_polygons or half_centerline is None or len(cross_section_polygons) != len(half_centerline):
             raise ValueError("Mismatch between number of polygons and centerline points, or missing data.")
        if len(half_centerline) < 2:
             raise ValueError("Half centerline path needs at least 2 points.")

        all_vertices = []
        all_faces = []
        num_cs_pts = num_cross_section_points # Vertices per cross-section

        # --- Calculate Path Tangents (for orientation) ---
        path_vectors = np.gradient(half_centerline, axis=0)
        path_tangents = path_vectors / np.linalg.norm(path_vectors, axis=1)[:, None]
        # Handle potential zero vectors at ends if gradient is zero
        if np.linalg.norm(path_tangents[0]) < 1e-9: path_tangents[0] = path_tangents[1]
        if np.linalg.norm(path_tangents[-1]) < 1e-9: path_tangents[-1] = path_tangents[-2]

        # --- Generate Vertices for each Transformed Cross-Section ---
        base_vertex_count = 0
        for i, path_point in enumerate(half_centerline):
            polygon = cross_section_polygons[i]
            tangent = path_tangents[i]

            # Define coordinate system at path_point
            # Assume Z is up (0,0,1) unless tangent is vertical
            if abs(np.dot(tangent, [0, 0, 1])) > 0.999:
                # Tangent is nearly vertical, use Y as 'up' for cross-section plane
                up_vector = np.cross(tangent, [0, 1, 0])
            else:
                up_vector = np.array([0, 0, 1])

            # Create basis vectors for the cross-section plane
            normal = tangent # Normal to the cross-section plane is the path tangent
            binormal = np.cross(normal, up_vector)
            binormal /= np.linalg.norm(binormal)
            local_y = np.cross(binormal, normal) # Should be normalized already
            local_y /= np.linalg.norm(local_y)
            local_x = binormal # Renaming for clarity (cs_x maps to binormal)

            # Get 2D vertices from Shapely polygon exterior
            cs_verts_2d = np.array(polygon.exterior.coords)[:-1] # Exclude duplicate end point

            if len(cs_verts_2d) != num_cs_pts:
                 print(f"  Warning: Polygon {i} has {len(cs_verts_2d)} vertices, expected {num_cs_pts}. Using actual count.")
                 # This might cause issues with face generation if counts vary wildly.
                 # For now, we proceed, but ideally, all polygons should have the same vertex count.

            # Transform 2D vertices to 3D
            # cs_x maps to local_x (binormal), cs_y maps to local_y
            cs_verts_3d = path_point + cs_verts_2d[:, 0][:, None] * local_x + cs_verts_2d[:, 1][:, None] * local_y
            all_vertices.append(cs_verts_3d)

            # --- Generate Faces connecting this section to the previous one ---
            if i > 0:
                prev_num_pts = len(all_vertices[i-1])
                curr_num_pts = len(cs_verts_3d)
                # Simple case: assume vertex counts match
                if prev_num_pts == curr_num_pts:
                    offset = base_vertex_count - prev_num_pts
                    for j in range(curr_num_pts):
                        v1 = offset + j
                        v2 = offset + (j + 1) % curr_num_pts
                        v3 = base_vertex_count + (j + 1) % curr_num_pts
                        v4 = base_vertex_count + j
                        all_faces.append([v1, v2, v3]) # Triangle 1
                        all_faces.append([v1, v3, v4]) # Triangle 2
                else:
                    print(f"  Warning: Vertex count mismatch between section {i-1} ({prev_num_pts}) and {i} ({curr_num_pts}). Skipping face generation between them.")
                    # More complex stitching logic would be needed here if counts vary.

            base_vertex_count += len(cs_verts_3d)


        # --- Combine Vertices and Faces ---
        if not all_vertices or not all_faces:
             raise ValueError("No vertices or faces were generated.")

        final_vertices = np.vstack(all_vertices)
        final_faces = np.array(all_faces, dtype=np.int32)

        single_cell = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
        print(f"  Generated swept mesh manually: {len(single_cell.vertices)} vertices, {len(single_cell.faces)} faces")

        # --- Capping: Project End Vertices and Triangulate (Replaces fill_holes) ---
        print("  Creating planar caps by projecting end vertices...")

        # Get endpoint coordinates and the normal of the capping plane
        # The capping plane should be perpendicular to the centerline's major axis
        start_point = half_centerline[0]
        end_point = half_centerline[-1]
        # Normal vector is parallel to the centerline's major axis direction vector
        major_axis_direction = np.array([np.cos(cl_phi), np.sin(cl_phi), 0])
        cap_plane_normal = major_axis_direction # Normal to the plane is along the major axis

        # Identify vertices at start and end profiles (indices 0 to num_cs_pts-1 and the last num_cs_pts)
        if len(single_cell.vertices) < 2 * num_cs_pts:
             raise ValueError("Sweep did not generate enough vertices to identify end profiles for capping.")

        start_profile_indices = np.arange(num_cs_pts)
        end_profile_indices = np.arange(len(single_cell.vertices) - num_cs_pts, len(single_cell.vertices))

        # Project start vertices onto the plane defined by start_point and cap_plane_normal
        projected_start_vertices = []
        for idx in start_profile_indices:
            v = single_cell.vertices[idx]
            dist = np.dot(v - start_point, cap_plane_normal)
            v_projected = v - dist * cap_plane_normal
            projected_start_vertices.append(v_projected)
            single_cell.vertices[idx] = v_projected # Update mesh vertex

        # Project end vertices onto the plane defined by end_point and cap_plane_normal
        projected_end_vertices = []
        for idx in end_profile_indices:
            v = single_cell.vertices[idx]
            dist = np.dot(v - end_point, cap_plane_normal)
            v_projected = v - dist * cap_plane_normal
            projected_end_vertices.append(v_projected)
            single_cell.vertices[idx] = v_projected # Update mesh vertex

        print(f"  Projected {len(start_profile_indices)} start and {len(end_profile_indices)} end vertices onto capping planes.")

        # Triangulate caps using the projected vertices
        projected_start_vertices = np.array(projected_start_vertices)
        projected_end_vertices = np.array(projected_end_vertices)
        start_center = np.mean(projected_start_vertices, axis=0)
        end_center = np.mean(projected_end_vertices, axis=0)

        # Add center points as new vertices
        center_start_idx = len(single_cell.vertices)
        center_end_idx = center_start_idx + 1
        single_cell.vertices = np.vstack([single_cell.vertices, [start_center, end_center]])

        # Create cap faces using original indices of projected vertices
        start_faces = []
        for i in range(num_cs_pts):
            v1_idx = start_profile_indices[i]
            v2_idx = start_profile_indices[(i + 1) % num_cs_pts]
            # Check winding order relative to cap_plane_normal (should point outwards)
            # If normal points along +major_axis, winding should be counter-clockwise when viewed from +major_axis
            start_faces.append([v1_idx, v2_idx, center_start_idx]) # Keep this winding

        end_faces = []
        for i in range(num_cs_pts):
            v1_idx = end_profile_indices[i]
            v2_idx = end_profile_indices[(i + 1) % num_cs_pts]
            # End cap normal should point outwards (-major_axis direction).
            # Let's try the SAME winding as the start cap and let fix_normals handle it.
            end_faces.append([v1_idx, v2_idx, center_end_idx]) # Same winding as start (NEW)

        # Add cap faces
        if start_faces and end_faces:
            all_new_faces = np.vstack(start_faces + end_faces)
            single_cell.faces = np.vstack([single_cell.faces, all_new_faces])
            print(f"  Added {len(all_new_faces)} cap faces using projected end vertices.")

        # Fix normals and validate watertightness
        single_cell.fix_normals() # Important after manual face addition/vertex modification
        if not single_cell.is_watertight:
            print("  Warning: Mesh is not watertight after projection and capping. Trying fill_holes() as fallback.")
            single_cell.fill_holes() # Attempt fallback if projection/triangulation failed
            if single_cell.is_watertight:
                print("  Successfully made mesh watertight with fill_holes().")
            else:
                print("  ERROR: Could not make mesh watertight. Check model carefully.")
                # return None, None # Consider stopping if capping fails critically
        else:
            print("  Mesh is watertight with projected ends and planar caps.")
        # --- End Capping Modification ---


        # Basic validation (remains the same)
        if not single_cell.faces.shape[0] > 0:
            raise ValueError("Generated mesh has no faces after manual construction/capping.")

    except Exception as e:
        print(f"  Error during manual mesh construction or capping: {e}")
        traceback.print_exc()
        return None, None

    # --- STEP 6: Export single cell PLY ---
    print("STEP 6: Exporting single bulging guard cell...")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        export_result = single_cell.export(file_obj=output_path, file_type='ply', encoding='ascii')
        print(f"  Successfully saved single bulging guard cell to: {output_path}")

    except Exception as e:
        print(f"  Error exporting PLY: {e}")
        traceback.print_exc()
        return None, None # Return None on export error

    return single_cell, center # Return mesh and center


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
    file_to_process = "Meshes/OBJ/Ac_DA_1_3.obj" # Example file
    base_name = os.path.splitext(os.path.basename(file_to_process))[0]

    # Define output paths for BOTH standard and bulging meshes
    results_dir = "results" # Define results directory
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
            min_aspect_ratio=1.1 # Adjust this value as needed (e.g., 1.0 for circular ends)
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
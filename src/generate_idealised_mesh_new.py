## To define an ellipse, we need its semi-majpr axis, a, and semi-minor axis, b. The standard parametric equation for an ellipse centred at (0,0) is:

# x(t) = a * cos(t)
# y(t) = b * sin(t)

# Where t is the parameter that ranges from 0 to 2π.

import numpy as np
import matplotlib.pyplot as plt
import trimesh
#import tidy3d

def create_elliptical_torus_with_shared_wall_taper(
    major_radius_a=2.0,
    major_radius_b=1.0,
    minor_radius_a=0.3,
    minor_radius_b=0.4,
    major_segments=120,
    minor_segments=40,
    wall_thickness=0.0,
    cap_inset_frac=0.05           # fraction of minor_radius to inset cap centre along normal
):
    """
    Create two half-oval guard cell meshes forming a continuous ring along the y-axis.
    Each half has tip caps. This version smoothly tapers the cross-section near tips
    and insets the tip centre to avoid a 90-degree corner.
    Returns left_mesh, right_mesh (Trimesh objects).
    """
    import numpy as np
    import trimesh

    # top-level parameters
    ref_major_radius = 2.0
    base_taper_frac = 0.05
    base_taper_length_fraction = 0.05

    # inside generate_half
    


    def generate_half(theta_start, theta_end, offset_sign):
        vertices = []
        faces = []
        major_seg = major_segments // 2
        taper_frac = base_taper_frac * (ref_major_radius / major_radius_a)
        taper_length_fraction = base_taper_length_fraction * (ref_major_radius / major_radius_a)

        # precompute taper region size in indices
        taper_len = max(1, int(round(taper_length_fraction * major_seg)))
        # Construct a taper weight function along i=0..major_seg-1 that is 1.0 in the middle
        # and down to (1 - taper_frac) at the ends (over taper_len indices).
        taper_weights = np.ones(major_seg, dtype=float)
        if taper_len > 0:
            # linear taper; you can change to cosine for smoother slope
            for i in range(taper_len):
                w = (i + 1) / (taper_len + 1)  # 0..1
                # use cosine ease for smoother derivative
                ease = 0.5 * (1 - np.cos(np.pi * w))
                taper_weights[i] = 1.0 - taper_frac * (1 - ease)   # near start -> slightly reduced
                taper_weights[major_seg - 1 - i] = 1.0 - taper_frac * (1 - ease)

        # Generate vertices with taper applied based on i index
        normals_at_section = []  # store normal for each section for cap inset
        for i in range(major_seg):
            theta = theta_start + i * (theta_end - theta_start) / (major_seg - 1)
            path_x = major_radius_a * np.sin(theta)   # X vertical in plane
            path_y = major_radius_b * np.cos(theta)   # Y horizontal in plane

            # Tangent and normal
            tx = major_radius_a * np.cos(theta)
            ty = -major_radius_b * np.sin(theta)
            tangent = np.array([tx, ty, 0.0])
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-ty, tx, 0.0])
            normal /= np.linalg.norm(normal)
            binormal = np.array([0.0, 0.0, 1.0])

            normals_at_section.append(normal.copy())

            # taper factor for this section
            taper_scale = taper_weights[i]

            for j in range(minor_segments):
                phi = j * 2.0 * np.pi / minor_segments
                local_x = minor_radius_a * np.cos(phi) * taper_scale
                local_z = minor_radius_b * np.sin(phi)  # don't taper z much; keep thickness
                vertex = np.array([path_x, path_y, 0.0]) - local_x * normal + local_z * binormal \
                         + (normal * (wall_thickness / 2.0) * offset_sign)
                vertices.append(vertex)

        # Generate side faces
        for i in range(major_seg - 1):
            for j in range(minor_segments):
                nj = (j + 1) % minor_segments
                v1 = i * minor_segments + j
                v2 = (i + 1) * minor_segments + j
                v3 = (i + 1) * minor_segments + nj
                v4 = i * minor_segments + nj
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        # Tip wall at first cross-section (i=0)
        tip_indices_start = np.arange(0, minor_segments)
        # compute center as mean of tip indices then inset along local normal
        center_start_pos = np.mean(np.array(vertices)[tip_indices_start], axis=0)
        normal_start = normals_at_section[0]
        cap_inset = cap_inset_frac * minor_radius_a
        cap_inset_frac_scaled = cap_inset_frac * (ref_major_radius / major_radius_a)
        cap_inset = cap_inset_frac_scaled * minor_radius_a

        center_start = len(vertices)
        vertices.append(center_start_pos + normal_start * (-cap_inset))  # inset towards interior
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            # ordering chosen to keep face normals consistent (may need flip depending on orientation)
            faces.append([int(tip_indices_start[j]), int(tip_indices_start[nj]), center_start])

        # Tip wall at last cross-section (i = major_seg - 1)
        tip_indices_end = np.arange((major_seg - 1) * minor_segments, major_seg * minor_segments)
        center_end_pos = np.mean(np.array(vertices)[tip_indices_end], axis=0)
        normal_end = normals_at_section[-1]
        center_end = len(vertices)
        vertices.append(center_end_pos + normal_end * (-cap_inset))
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            faces.append([int(tip_indices_end[nj]), int(tip_indices_end[j]), center_end])

        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
        mesh.fix_normals()
        return mesh

    left_mesh = generate_half(0.0, np.pi, offset_sign=-1)
    right_mesh = generate_half(np.pi, 2.0 * np.pi, offset_sign=1)

    return left_mesh, right_mesh



def create_elliptical_torus_with_shared_wall(
    major_radius_a=2.0,
    major_radius_b=1.0,
    minor_radius_a=0.3,
    minor_radius_b=0.4,
    major_segments=120,
    minor_segments=40,
    wall_thickness=0.0
):
    """
    Create two half-oval guard cell meshes forming a continuous ring along the y-axis.
    Each half has a wall at both tips (triangular caps).
    
    The two halves are offset by wall_thickness/2 on each side to prevent overlapping
    geometry at the shared wall interface.

    Returns:
        left_mesh, right_mesh: two separate Trimesh objects
    """

    import numpy as np
    import trimesh

    def generate_half(theta_start, theta_end, offset_sign):
        vertices = []
        faces = []
        major_seg = major_segments // 2

        # Generate vertices
        for i in range(major_seg):
            theta = theta_start + i * (theta_end - theta_start) / (major_seg - 1)
            path_x = major_radius_a * np.sin(theta)   # X vertical in plane
            path_y = major_radius_b * np.cos(theta)   # Y horizontal in plane

            # Tangent and normal
            tx = major_radius_a * np.cos(theta)
            ty = -major_radius_b * np.sin(theta)
            tangent = np.array([tx, ty, 0])
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-ty, tx, 0])
            normal /= np.linalg.norm(normal)
            binormal = np.array([0, 0, 1])

            # Optional offset for wall separation
            offset = normal * (wall_thickness / 2.0) * offset_sign

            for j in range(minor_segments):
                phi = j * 2 * np.pi / minor_segments
                local_x = minor_radius_a * np.cos(phi)
                local_z = minor_radius_b * np.sin(phi)
                vertex = np.array([path_x, path_y, 0]) - local_x * normal + local_z * binormal + offset
                vertices.append(vertex)

        # Generate side faces
        for i in range(major_seg - 1):
            for j in range(minor_segments):
                nj = (j + 1) % minor_segments
                v1 = i * minor_segments + j
                v2 = (i + 1) * minor_segments + j
                v3 = (i + 1) * minor_segments + nj
                v4 = i * minor_segments + nj
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        # Tip wall at first cross-section
        tip_indices_start = np.arange(minor_segments)
        center_start = len(vertices)
        vertices.append(np.mean(np.array(vertices)[tip_indices_start], axis=0))  # add center vertex
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            faces.append([tip_indices_start[j], tip_indices_start[nj], center_start])

        # Tip wall at last cross-section
        tip_indices_end = np.arange((major_seg - 1) * minor_segments, major_seg * minor_segments)
        center_end = len(vertices)
        vertices.append(np.mean(np.array(vertices)[tip_indices_end], axis=0))  # add center vertex
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            faces.append([tip_indices_end[nj], tip_indices_end[j], center_end])

        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
        mesh.fix_normals()
        return mesh

    # Left half: theta 0 -> π, offset to left side
    left_mesh = generate_half(0, np.pi, offset_sign=-1)
    # Right half: theta π -> 2π, offset to right side
    right_mesh = generate_half(np.pi, 2 * np.pi, offset_sign=1)

    return left_mesh, right_mesh



def create_elliptical_torus_with_wall(
    major_radius_a=2.0,
    major_radius_b=1.0,
    minor_radius_a=0.3,
    minor_radius_b=0.4,
    major_segments=120,
    minor_segments=40,
    wall_thickness=0.0
):
    """
    Create two half-oval guard cell meshes forming a continuous ring along the y-axis.
    
    Returns:
        left_mesh, right_mesh: two separate Trimesh objects
    """
    
    def generate_half(theta_start, theta_end, offset_sign):
        vertices = []
        faces = []
        major_seg = major_segments // 2
        for i in range(major_seg):
            theta = theta_start + i * (theta_end - theta_start) / (major_seg - 1)
            path_x = major_radius_a * np.sin(theta)   # X now vertical in plane
            path_y = major_radius_b * np.cos(theta)   # Y now horizontal in plane

            
            # Tangent and normal
            tx =  major_radius_a * np.cos(theta)      # derivative of path_x
            ty = -major_radius_b * np.sin(theta)      # derivative of path_y

            tangent = np.array([tx, ty, 0])
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-ty, tx, 0])
            normal /= np.linalg.norm(normal)
            binormal = np.array([0, 0, 1])
            
            # Optional offset for wall separation
            offset = normal * (wall_thickness / 2.0) * offset_sign
            
            for j in range(minor_segments):
                phi = j * 2 * np.pi / minor_segments
                local_x = minor_radius_a * np.cos(phi)
                local_z = minor_radius_b * np.sin(phi)
                vertex = np.array([path_x, path_y, 0]) - local_x * normal + local_z * binormal + offset
                vertices.append(vertex)
        
        # Create faces
        for i in range(major_seg - 1):
            for j in range(minor_segments):
                nj = (j + 1) % minor_segments
                v1 = i * minor_segments + j
                v2 = (i + 1) * minor_segments + j
                v3 = (i + 1) * minor_segments + nj
                v4 = i * minor_segments + nj
                # Counter-clockwise winding
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
        mesh.fix_normals()
        return mesh

    # Left half: theta 0 -> π, offset to left side
    left_mesh = generate_half(0, np.pi, offset_sign=-1)
    # Right half: theta π -> 2π, offset to right side
    right_mesh = generate_half(np.pi, 2*np.pi, offset_sign=1)
    
    return left_mesh, right_mesh


def create_elliptical_torus(
    major_radius_a, major_radius_b, 
    minor_radius_a, minor_radius_b, 
    major_segments, minor_segments
):
    """
    Creates a mesh where an elliptical cross-section travels along an elliptical path.

    Args:
        major_radius_a: Semi-axis of the major (path) ellipse along the x-axis.
        major_radius_b: Semi-axis of the major (path) ellipse along the y-axis.
        minor_radius_a: Semi-axis of the minor (cross-section) ellipse (width).
        minor_radius_b: Semi-axis of the minor (cross-section) ellipse (height).
        major_segments: Number of segments for the major elliptical path.
        minor_segments: Number of segments for the minor elliptical cross-section.
    """
    vertices = []
    faces = []

    # Create vertices
    for i in range(major_segments):
        theta = i * 2 * np.pi / major_segments

        # Position on the major (path) ellipse
        path_x = major_radius_a * np.cos(theta)
        path_y = major_radius_b * np.sin(theta)
        
        # Tangent vector to the major ellipse at this point (for orientation)
        # The derivative of the path gives the tangent direction
        tx = -major_radius_a * np.sin(theta)
        ty =  major_radius_b * np.cos(theta)
        tangent = np.array([tx, ty, 0])
        tangent /= np.linalg.norm(tangent)

        # Normal and binormal vectors to define the plane of the cross-section
        normal = np.array([-ty, tx, 0]) # Perpendicular to tangent in XY plane
        normal /= np.linalg.norm(normal)
        binormal = np.array([0, 0, 1]) # Perpendicular to the XY plane

        for j in range(minor_segments):
            phi = j * 2 * np.pi / minor_segments

            # Position on the minor (cross-section) ellipse
            # These are local coordinates on the plane of the cross-section
            local_x = minor_radius_a * np.cos(phi)
            local_z = minor_radius_b * np.sin(phi)

            # Combine the local coordinates with the orientation vectors
            # to position the vertex in 3D space
            vertex = np.array([path_x, path_y, 0]) + local_x * normal + local_z * binormal
            vertices.append(vertex)

    # Create faces (this logic remains the same)
    for i in range(major_segments):
        for j in range(minor_segments):
            next_i = (i + 1) % major_segments
            next_j = (j + 1) % minor_segments

            v1 = i * minor_segments + j
            v2 = next_i * minor_segments + j
            v3 = next_i * minor_segments + next_j
            v4 = i * minor_segments + next_j

            faces.append([v1, v4, v2])
            faces.append([v2, v4, v3])

    return trimesh.Trimesh(vertices=vertices, faces=faces)

import numpy as np
import trimesh

def create_elliptical_torus_bulged(
    major_radius_a, major_radius_b, 
    mid_radius_a, mid_radius_b,
    tip_radius_a, tip_radius_b, 
    major_segments, minor_segments
):
    """
    Creates a mesh where an elliptical cross-section travels along an elliptical path.

    Args:
        major_radius_a: Semi-axis of the major (path) ellipse along the x-axis.
        major_radius_b: Semi-axis of the major (path) ellipse along the y-axis.
        mid_radius_a: Semi-axis of the minor (cross-section) ellipse at midpoint (sides).
        mid_radius_b: Semi-axis of the minor (cross-section) ellipse at midpoint (sides).
        tip_radius_a: Semi-axis of the minor (cross-section) ellipse at tip (top/bottom).
        tip_radius_b: Semi-axis of the minor (cross-section) ellipse at tip (top/bottom).
        major_segments: Number of segments for the major elliptical path.
        minor_segments: Number of segments for the minor elliptical cross-section.
    """
    vertices = []
    faces = []
    for i in range(major_segments):
        theta = i * 2 * np.pi / major_segments

        # Position on the major (path) ellipse
        path_x = major_radius_a * np.cos(theta)
        path_y = major_radius_b * np.sin(theta)

        # Use vertical position (y) to determine proximity to tip (top/bottom)
        norm_pos_from_tip = abs(path_y) / major_radius_b  # 0 at sides, 1 at tips

        # --- REVERSED INTERPOLATION ---
        # Now: 0 at midpoint (sides), 1 at tips
        # Instead of: 1 at midpoint, 0 at tips
        transition_point = 0.8
        if norm_pos_from_tip < transition_point:
            interp_factor = norm_pos_from_tip / transition_point  # Changed from (1.0 - norm_pos...)
        else:
            interp_factor = 1.0  # Changed from 0.0

        interp_factor = 1.0
        # Corrected interpolation formula - applies mid_radius at midpoints and tip_radius at tips
        current_minor_radius_a = mid_radius_a * (1 - interp_factor) + tip_radius_a * interp_factor
        current_minor_radius_b = mid_radius_b * (1 - interp_factor) + tip_radius_b * interp_factor

        shrink_start = 0.9  # Start shrinking at 95% of the way to the tip
        # if norm_pos_from_tip >= shrink_start:
        #     # Quadratic shrink: increases faster near the tip
        #     t = (norm_pos_from_tip - shrink_start) / (1.0 - shrink_start)
        #     shrink_factor = 1.0 - 0.2 * (t ** 5)
        #     current_minor_radius_a *= shrink_factor
        #     current_minor_radius_b *= shrink_factor
            
        # Tangent vector to the major ellipse at this point (for orientation)
        tx = -major_radius_a * np.sin(theta)
        ty =  major_radius_b * np.cos(theta)
        tangent = np.array([tx, ty, 0])
        tangent /= np.linalg.norm(tangent)

        # Normal and binormal vectors to define the plane of the cross-section
        normal = np.array([-ty, tx, 0])  # Perpendicular to tangent in XY plane
        normal /= np.linalg.norm(normal)
        binormal = np.array([0, 0, 1])   # Perpendicular to XY plane

        for j in range(minor_segments):
            phi = j * 2 * np.pi / minor_segments

            local_x = current_minor_radius_a * np.cos(phi)
            local_z = current_minor_radius_b * np.sin(phi)

            # Only apply shrink near the tip
            if norm_pos_from_tip >= shrink_start:
                t = (norm_pos_from_tip - shrink_start) / (1.0 - shrink_start)
                full_shrink = 1.0 - 0.2 * (t ** 5)

                # Normalize local_x: -max_a (outside) to +max_a (inside)
                # So: 0 (outside) ... 1 (inside)
                x_norm = (local_x + current_minor_radius_a) / (2 * current_minor_radius_a)
                # Blend shrink: full_shrink at outside, 1 at inside
                blend_shrink = full_shrink * (1 - x_norm) + 1.0 * x_norm

                local_x *= blend_shrink
                local_z *= blend_shrink  # If you want to keep aspect ratio

            vertex = np.array([path_x, path_y, 0]) + local_x * normal + local_z * binormal
            vertices.append(vertex)

    # Create faces connecting vertices (quads as two triangles)
    for i in range(major_segments):
        for j in range(minor_segments):
            next_i = (i + 1) % major_segments
            next_j = (j + 1) % minor_segments

            v1 = i * minor_segments + j
            v2 = next_i * minor_segments + j
            v3 = next_i * minor_segments + next_j
            v4 = i * minor_segments + next_j

            faces.append([v1, v4, v2])
            faces.append([v2, v4, v3])

    return trimesh.Trimesh(vertices=vertices, faces=faces)

def calculate_cross_section_radius(theta, side_radius, tip_radius, transition_point=0.8):
    """
    Calculate the cross-section radius at a given angle theta around the elliptical path.
    
    Args:
        theta: Angle in radians (0 = right side, pi/2 = top, pi = left side, 3pi/2 = bottom)
        side_radius: Radius at the sides of the elliptical path (theta=0 or pi)
        tip_radius: Radius at the tips of the elliptical path (theta=pi/2 or 3pi/2)
        transition_point: Fraction of distance between side and tip where transition completes
    
    Returns:
        The radius at the specified angle
    """
    # Map theta to a value between 0 and pi/2 (a quadrant)
    quadrant_theta = theta % np.pi
    if quadrant_theta > np.pi/2:
        quadrant_theta = np.pi - quadrant_theta
    
    # Normalize position from side (0) to tip (1)
    # At theta=0, this equals 0 (side)
    # At theta=pi/2, this equals 1 (tip)
    norm_pos = quadrant_theta / (np.pi/2)
    
    # Apply piecewise linear interpolation
    if norm_pos <= transition_point:
        # Linear change from side to transition point
        interp_factor = norm_pos / transition_point
        radius = side_radius * (1 - interp_factor) + tip_radius * interp_factor
    else:
        # Constant value (tip_radius) for the remainder
        radius = tip_radius
        
    return radius

def plot_radius_distribution(side_radius_a, tip_radius_a, side_radius_b=None, tip_radius_b=None, transition_point=0.8):
    """
    Plots the distribution of cross-section radii around the elliptical path.
    
    Args:
        side_radius_a: The a-axis radius at the sides (horizontal axis points)
        tip_radius_a: The a-axis radius at the tips (vertical axis points)
        side_radius_b: The b-axis radius at the sides (optional, for plotting both dimensions)
        tip_radius_b: The b-axis radius at the tips (optional, for plotting both dimensions)
        transition_point: Fraction of distance where transition completes
    """
    theta_values = np.linspace(0, 2*np.pi, 100)
    
    # Calculate radius_a values
    radius_a_values = [calculate_cross_section_radius(theta, side_radius_a, tip_radius_a, transition_point) 
                      for theta in theta_values]
    
    # Convert theta to degrees for easier reading
    theta_degrees = np.degrees(theta_values)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot radius_a
    plt.plot(theta_degrees, radius_a_values, 'b-', label='radius_a')
    plt.axhline(y=side_radius_a, color='r', linestyle='--', 
               label=f'Side radius_a: {side_radius_a}')
    plt.axhline(y=tip_radius_a, color='g', linestyle='--', 
               label=f'Tip radius_a: {tip_radius_a}')
    
    # If b-axis values are provided, plot those too
    if side_radius_b is not None and tip_radius_b is not None:
        radius_b_values = [calculate_cross_section_radius(theta, side_radius_b, tip_radius_b, transition_point) 
                          for theta in theta_values]
        plt.plot(theta_degrees, radius_b_values, 'c-', label='radius_b')
        plt.axhline(y=side_radius_b, color='m', linestyle='--', 
                   label=f'Side radius_b: {side_radius_b}')
        plt.axhline(y=tip_radius_b, color='y', linestyle='--', 
                   label=f'Tip radius_b: {tip_radius_b}')
    
    # Mark the sides and tips
    plt.axvline(x=0, color='blue', linestyle=':', label='Side (θ=0°)')
    plt.axvline(x=180, color='blue', linestyle=':')
    plt.axvline(x=90, color='orange', linestyle=':', label='Tip (θ=90°)')
    plt.axvline(x=270, color='orange', linestyle=':')
    
    # Mark the transition points
    transition_angle = transition_point * 90
    plt.axvline(x=transition_angle, color='purple', linestyle='-.', 
               label=f'Transition point: {transition_point*100}%')
    plt.axvline(x=180-transition_angle, color='purple', linestyle='-.')
    plt.axvline(x=180+transition_angle, color='purple', linestyle='-.')
    plt.axvline(x=360-transition_angle, color='purple', linestyle='-.')
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cross-section radius')
    plt.title('Distribution of cross-section radii around the elliptical path')
    plt.grid(True)
    plt.legend()
    plt.savefig('radius_distribution.png')
    plt.show()

    # After plotting radius_a_values
    shrink_start = 0.95
    theta_values = np.linspace(0, 2*np.pi, 100)
    norm_pos_from_tip = np.abs(np.sin(theta_values))  # For a circle; for ellipse use path_y/major_radius_b

    plt.plot(np.degrees(theta_values), norm_pos_from_tip, 'k--', alpha=0.3, label='norm_pos_from_tip')

    # Mark where shrink starts
    shrink_angles = np.degrees(theta_values[norm_pos_from_tip >= shrink_start])
    if len(shrink_angles) > 0:
        plt.axvline(x=shrink_angles[0], color='purple', linestyle='-.', label='Shrink start')
        plt.axvline(x=shrink_angles[-1], color='purple', linestyle='-.')

    plt.legend()
    plt.show()
    
    return radius_a_values









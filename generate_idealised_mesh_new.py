## To define an ellipse, we need its semi-majpr axis, a, and semi-minor axis, b. The standard parametric equation for an ellipse centred at (0,0) is:

# x(t) = a * cos(t)
# y(t) = b * sin(t)

# Where t is the parameter that ranges from 0 to 2π.

import numpy as np
import matplotlib.pyplot as plt
import trimesh
#import tidy3d

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
            
        # Corrected interpolation formula - applies mid_radius at midpoints and tip_radius at tips
        current_minor_radius_a = mid_radius_a * (1 - interp_factor) + tip_radius_a * interp_factor
        current_minor_radius_b = mid_radius_b * (1 - interp_factor) + tip_radius_b * interp_factor

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

            # Cross-section ellipse local coordinates
            local_x = current_minor_radius_a * np.cos(phi)
            local_z = current_minor_radius_b * np.sin(phi)

            # Combine local coordinates with orientation vectors to get vertex position
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
    
    return radius_a_values

# --- Example Usage ---
if __name__ == '__main__':
    # Create a torus with an elliptical path and a circular cross-section
    # major_radius_a: radius of path ellipse along x-axis
    # major_radius_b: radius of path ellipse along y-axis
    # minor_radius_a: radius of cross-section ellipse along x-axis 
    # minor_radius_b: radius of cross-section ellipse along y-axis 
    # if minor_radius_a == minor_radius_b, it becomes a circular cross-section

    ## Get the parameters for the elliptical torus from the confocal mesh

    ## Create our tables

    mesh_name =["1_2","1_3","1_4","1_5","1_6","1_8","2_1", "2_3", "2_6a", "2_6b", "2_7","3_1","3_2", "3_3", "3_4","3_6", "3_7"]
    minor_radius_a = [8, 8, 7.5, 9, 7.9, 7.4, 6.7, 7.2, 8, 7.8, 7, 7.5, 8.4, 8.5, 8.1, 8.5, 7.9]
    minor_radius_b = [5, 5, 5.5, 6, 5.5, 5.8, 5.0, 4, 5, 6, 5.3, 5.5, 6, 6, 7, 6.5, 5.9]
    stomata_length = [43, 40,40.5, 48.2, 45.2, 40.3, 37, 39.6, 37.5, 36, 40, 42.1, 41.6, 40.6, 40.5, 45.5, 41] # vertical dimension (y-axis)
    stomata_width = [37, 39, 35.4, 41.5, 37, 37.6, 36.0, 37.6, 37.0, 35, 33.5, 32.9, 37.2, 35.8, 36.3, 38.5, 35] # horizontal dimension (x-axis)
    confocal_pore_area = [40.9,44.7,43.0, 49.6, 53.9, 65.7, 73.8, 74.2, 21.7, 14.2, 49.2, 22, 20.9, 7.3, 23.5, 37.3, 22.3]

    for i in range(len(mesh_name)):
        this_mesh = mesh_name[i]
        major_radius_a = (stomata_width[i] - 2 * minor_radius_a[i]) / 2
        major_radius_b = (stomata_length[i] - 2 * minor_radius_a[i]) / 2

        elliptical_torus_mesh = create_elliptical_torus(
            major_radius_a=major_radius_a,  # Wider path
            major_radius_b=major_radius_b,  # Narrower path
            minor_radius_a=minor_radius_a[i], 
            minor_radius_b=minor_radius_b[i], 
            major_segments=100,
            minor_segments=50
        )
    ## Save the mesh to a PLY file
        elliptical_torus_mesh.export('Meshes/Idealised/idealised_' + this_mesh + '.ply')

    # stomata_length = 39.4 # vertical dimension (y-axis)
    # stomata_width = 38.4 # horizontal dimension (x-axis)
    # major_radius_a = (stomata_width - 2 * minor_radius_a) / 2
    # major_radius_b = (stomata_length - 2 * minor_radius_a) / 2

    # mid_radius_a = minor_radius_a
    # mid_radius_b = minor_radius_b
    # tip_radius_a = 7.5
    # tip_radius_b = 7.5

    # elliptical_torus_mesh = create_elliptical_torus_bulged(
    # major_radius_a, major_radius_b, 
    # mid_radius_a, mid_radius_b,
    # tip_radius_a, tip_radius_b, 
    # major_segments=100,
    # minor_segments=50
# )
    
#      ## Save the mesh to a PLY file
#     elliptical_torus_mesh.export('elliptical_torus_bulged.ply')










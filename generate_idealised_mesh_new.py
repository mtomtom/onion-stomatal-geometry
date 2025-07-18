## To define an ellipse, we need its semi-majpr axis, a, and semi-minor axis, b. The standard parametric equation for an ellipse centred at (0,0) is:

# x(t) = a * cos(t)
# y(t) = b * sin(t)

# Where t is the parameter that ranges from 0 to 2π.

import numpy as np
import matplotlib.pyplot as plt
import trimesh
import tidy3d

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

            faces.append([v1, v2, v4])
            faces.append([v2, v3, v4])

    return trimesh.Trimesh(vertices=vertices, faces=faces)

# --- Example Usage ---
if __name__ == '__main__':
    # Create a torus with an elliptical path and a circular cross-section
    # major_radius_a: radius of path ellipse along x-axis
    # major_radius_b: radius of path ellipse along y-axis
    # minor_radius_a: radius of cross-section ellipse along x-axis 
    # minor_radius_b: radius of cross-section ellipse along y-axis 
    # if minor_radius_a == minor_radius_b, it becomes a circular cross-section

    ## Get the parameters for the elliptical torus from the confocal mesh
    pore_area = 40.4
    pore_length = 13.1
    pore_width = 4.9
    stomata_length = 42
    stomata_width = 38

    minor_radius_a = 8
    minor_radius_b = 5
    major_radius_a = (stomata_length - minor_radius_a) / 2
    major_radius_b = (stomata_width - minor_radius_a) / 2

    elliptical_torus_mesh = create_elliptical_torus(
        major_radius_a=major_radius_a,  # Wider path
        major_radius_b=major_radius_b,  # Narrower path
        minor_radius_a=minor_radius_a, 
        minor_radius_b=minor_radius_b, 
        major_segments=100,
        minor_segments=50
    )
    ## Save the mesh to a PLY file
    elliptical_torus_mesh.export('elliptical_torus.ply')







## To define an ellipse, we need its semi-majpr axis, a, and semi-minor axis, b. The standard parametric equation for an ellipse centred at (0,0) is:

# x(t) = a * cos(t)
# y(t) = b * sin(t)

# Where t is the parameter that ranges from 0 to 2π.

import numpy as np
import matplotlib.pyplot as plt
import trimesh
#import tidy3d




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






import numpy as np
import trimesh












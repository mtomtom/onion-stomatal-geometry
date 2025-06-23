import numpy as np
import trimesh

# Torusoid cross-section parameters
A = 50            # Desired area of ellipse cross-section
L = 10            # Major axis length of ellipse cross-section

# Compute ellipse radii (a = major axis/2, b = minor axis/2) from area and length
a = L / 2
b = (2 * A) / (np.pi * L)

# Elliptical sweep path parameters (major radius in x and y)
R_x = 15
R_y = 10

# Mesh resolution
n_path = 100    # number of steps around the major elliptical path
n_cross = 50    # number of steps around the elliptical cross-section

# Parameterization for major path and cross-section
t = np.linspace(0, 2 * np.pi, n_path, endpoint=False)
phi = np.linspace(0, 2 * np.pi, n_cross, endpoint=False)

# Sweep path (ellipse in XY plane)
path = np.stack([
    R_x * np.cos(t),
    R_y * np.sin(t),
    np.zeros_like(t)
], axis=-1)

# Compute tangent vectors along path
tangent = np.stack([
    -R_x * np.sin(t),
    R_y * np.cos(t),
    np.zeros_like(t)
], axis=-1)
tangent /= np.linalg.norm(tangent, axis=1)[:, None]

# Compute normal and binormal vectors for local frames
ref = np.array([0, 0, 1])  # Up vector
normal = np.cross(tangent, ref)
normal /= np.linalg.norm(normal, axis=1)[:, None]
binormal = np.cross(tangent, normal)

# Build vertices array
vertices = []
for i in range(n_path):
    for j in range(n_cross):
        offset = (
            a * np.cos(phi[j]) * normal[i] +     # major axis offset
            b * np.sin(phi[j]) * binormal[i]     # minor axis offset
        )
        vertex = path[i] + offset
        vertices.append(vertex)

vertices = np.array(vertices)

# Build faces for torusoid surface
faces = []
for i in range(n_path):
    for j in range(n_cross):
        i_next = (i + 1) % n_path
        j_next = (j + 1) % n_cross

        idx0 = i * n_cross + j
        idx1 = i * n_cross + j_next
        idx2 = i_next * n_cross + j
        idx3 = i_next * n_cross + j_next

        faces.append([idx0, idx1, idx2])
        faces.append([idx1, idx3, idx2])

# --- Add the minor-axis dividing wall ---

wall_faces = []
wall_loop_top = []
wall_loop_bottom = []

for i in range(n_path):
    # phi = π/2 (top of minor axis)
    offset_top = a * np.cos(np.pi / 2) * normal[i] + b * np.sin(np.pi / 2) * binormal[i]
    v_top = path[i] + offset_top
    wall_loop_top.append(len(vertices))
    vertices = np.vstack([vertices, v_top])

    # phi = 3π/2 (bottom of minor axis)
    offset_bot = a * np.cos(3 * np.pi / 2) * normal[i] + b * np.sin(3 * np.pi / 2) * binormal[i]
    v_bot = path[i] + offset_bot
    wall_loop_bottom.append(len(vertices))
    vertices = np.vstack([vertices, v_bot])

# Connect the wall vertices with faces
for i in range(n_path):
    i_next = (i + 1) % n_path
    t0 = wall_loop_top[i]
    t1 = wall_loop_top[i_next]
    b0 = wall_loop_bottom[i]
    b1 = wall_loop_bottom[i_next]

    wall_faces.append([t0, b0, t1])
    wall_faces.append([t1, b0, b1])

faces.extend(wall_faces)

# Create mesh and export
mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
mesh.export("torusoid_minor_axis_wall.ply")
print("✅ Exported 'torusoid_minor_axis_wall.ply'")







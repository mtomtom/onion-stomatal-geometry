import numpy as np
import pyvista as pv
import trimesh
from shapely.geometry import Polygon
import argparse

def _create_single_cell_trimesh(centerline_path, cross_section_poly, num_cs_points):
    """
    Creates a single, valid guard cell as a trimesh object.
    This function correctly sweeps, projects ends to be planar, and caps the mesh.
    """
    # 1. Sweep vertices by placing the cross-section at each point on the path
    path_tangents = np.gradient(centerline_path, axis=0)
    path_tangents /= np.linalg.norm(path_tangents, axis=1)[:, None]
    
    all_vertices = []
    for i, path_point in enumerate(centerline_path):
        tangent = path_tangents[i]
        up_vector = np.array([0, 0, 1])
        binormal = np.cross(tangent, up_vector); binormal /= np.linalg.norm(binormal)
        normal = np.cross(binormal, tangent)
        
        cs_verts_2d = np.array(cross_section_poly.exterior.coords)[:-1]
        cs_verts_3d = path_point + cs_verts_2d[:, 0][:, None] * normal + cs_verts_2d[:, 1][:, None] * binormal
        all_vertices.append(cs_verts_3d)
        
    vertices = np.vstack(all_vertices)
    
    # 2. Create faces for the main body of the cell
    faces = []
    num_path_points = len(centerline_path)
    for i in range(num_path_points - 1):
        for j in range(num_cs_points):
            v1 = i * num_cs_points + j
            v2 = i * num_cs_points + (j + 1) % num_cs_points
            v3 = (i + 1) * num_cs_points + (j + 1) % num_cs_points
            v4 = (i + 1) * num_cs_points + j
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
            
    # 3. Project end vertices onto a plane perpendicular to the path tangent
    start_tip_indices = np.arange(num_cs_points)
    end_tip_indices = np.arange(len(vertices) - num_cs_points, len(vertices))
    
    start_plane_normal = centerline_path[1] - centerline_path[0]; start_plane_normal /= np.linalg.norm(start_plane_normal)
    end_plane_normal = centerline_path[-1] - centerline_path[-2]; end_plane_normal /= np.linalg.norm(end_plane_normal)
    
    start_plane_point = centerline_path[0]
    end_plane_point = centerline_path[-1]

    for idx in start_tip_indices:
        v = vertices[idx]
        dist = np.dot(v - start_plane_point, start_plane_normal)
        vertices[idx] = v - dist * start_plane_normal
        
    for idx in end_tip_indices:
        v = vertices[idx]
        dist = np.dot(v - end_plane_point, end_plane_normal)
        vertices[idx] = v - dist * end_plane_normal
        
    # 4. Add caps with correct winding order to ensure an outward-facing solid
    start_center_v = np.mean(vertices[start_tip_indices], axis=0)
    start_center_idx = len(vertices)
    vertices = np.vstack([vertices, [start_center_v]])
    for j in range(num_cs_points):
        faces.append([start_center_idx, start_tip_indices[(j + 1) % num_cs_points], start_tip_indices[j]])
        
    end_center_v = np.mean(vertices[end_tip_indices], axis=0)
    end_center_idx = len(vertices)
    vertices = np.vstack([vertices, [end_center_v]])
    for j in range(num_cs_points):
        faces.append([end_center_idx, end_tip_indices[j], end_tip_indices[(j + 1) % num_cs_points]])
        
    # 5. Create and return the final, valid trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.fix_normals()
    return mesh

def create_stomatal_mesh(
    pore_area: float,
    pore_length: float,
    stomatal_length: float,
    cross_aspect_ratio: float,
    cell_width: float,
    output_path: str = "stomatal_complex.ply",
    is_bulging: bool = False, # Note: Bulging is not implemented in this version
    tip_aspect_ratio: float = None,
):
    """
    Generates a 3D stomatal complex using the robust trimesh-based method.
    """
    # --- Step 1: Define the geometric parameters from inputs ---
    pore_a = pore_length / 2.0
    pore_b = pore_area / (np.pi * pore_a) if pore_a > 1e-9 else 0
    cl_a = pore_a + cell_width
    cl_b = pore_b + cell_width
    
    # --- Step 2: Generate the RIGHT guard cell using the trusted trimesh helper ---
    num_cs_points = 32
    num_centerline_points = 64
    
    # A. Define the flat, elliptical centerline path
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_centerline_points)
    centerline_x = cl_a * np.cos(theta)
    centerline_y = cl_b * np.sin(theta)
    centerline_path = np.column_stack((centerline_x, centerline_y, np.zeros_like(theta)))
    
    # B. Define the 2D cross-section polygon
    cs_a = stomatal_length / 2.0
    cs_b = cs_a / cross_aspect_ratio
    cs_angles = np.linspace(0, 2 * np.pi, num_cs_points, endpoint=False)
    cs_vertices_2d = np.column_stack([cs_b * np.cos(cs_angles), cs_a * np.sin(cs_angles)])
    cs_polygon = Polygon(cs_vertices_2d)

    # C. Create the single cell trimesh object
    right_cell_trimesh = _create_single_cell_trimesh(centerline_path, cs_polygon, num_cs_points)

    # --- Step 3: Create and Combine Two Cells using Trimesh ---
    left_cell_trimesh = right_cell_trimesh.copy()
    # Apply a 180-degree rotation around the Z-axis
    transform_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    left_cell_trimesh.apply_transform(transform_matrix)

    # Combine the two valid cells and merge the vertices at the tips
    combined_trimesh = trimesh.util.concatenate(right_cell_trimesh, left_cell_trimesh)
    combined_trimesh.merge_vertices()
    combined_trimesh.fix_normals()

    # --- Step 4: Convert final Trimesh object to PyVista for saving ---
    faces_as_array = np.hstack((np.full((len(combined_trimesh.faces), 1), 3), combined_trimesh.faces))
    combined_mesh = pv.PolyData(combined_trimesh.vertices, faces_as_array)

    n_faces_right = len(right_cell_trimesh.faces)
    face_labels = np.ones(combined_mesh.n_cells, dtype=int)
    face_labels[n_faces_right:] = 2
    combined_mesh.cell_data['label'] = face_labels
    combined_mesh.cell_data['signal'] = np.ones(combined_mesh.n_cells, dtype=float)
    combined_mesh.point_data['label'] = np.zeros(combined_mesh.n_points, dtype=int)
    combined_mesh.point_data['signal'] = np.ones(combined_mesh.n_points, dtype=float)

    combined_mesh.save(output_path)
    print(f"Stomatal complex mesh saved to {output_path}")
    
    return combined_mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate idealized stomatal guard cell mesh")
    parser.add_argument("--pore-area", type=float, default=100, help="Target pore area")
    parser.add_argument("--pore-length", type=float, default=20, help="Target pore length")
    parser.add_argument("--stomatal-length", type=float, default=5, help="Z-axis height of stomata")
    parser.add_argument("--cross-ar", type=float, default=2.0, help="Cross-section aspect ratio (height/width)")
    parser.add_argument("--cell-width", type=float, default=2.5, help="Cell wall thickness in XY plane")
    parser.add_argument("--output", default="stomatal_complex.ply", help="Output file path")
    
    args = parser.parse_args()
    
    create_stomatal_mesh(
        pore_area=args.pore_area,
        pore_length=args.pore_length,
        stomatal_length=args.stomatal_length,
        cross_aspect_ratio=args.cross_ar,
        cell_width=args.cell_width,
        output_path=args.output
    )

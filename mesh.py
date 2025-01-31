import open3d as o3d
import numpy as np
import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.learning.frechet_mean import FrechetMean
from scipy.spatial import cKDTree
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of filenames for your meshes
mesh_files = ['Meshes/PLY/mesh_1_2.ply', 'Meshes/PLY/mesh_1_3.ply']

# Desired number of vertices for remeshing
desired_num_vertices = 100000

def load_and_preprocess_mesh(file):
    """Load and preprocess a mesh from a file."""
    try:
        mesh = o3d.io.read_triangle_mesh(file)
        if not mesh.is_empty():
            logging.info(f"Successfully loaded {file}")
            # Smooth the mesh
            mesh = mesh.filter_smooth_simple(number_of_iterations=10)
            # Simplify the mesh to the desired number of vertices
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=desired_num_vertices)
            return mesh
        else:
            logging.warning(f"Mesh {file} is empty.")
            return None
    except Exception as e:
        logging.error(f"Error reading {file}: {e}")
        return None

def normalize_vertices(vertices):
    """Normalize vertices by centering and scaling."""
    centroid = np.mean(vertices, axis=0)
    vertices_centered = vertices - centroid
    scale = np.max(np.linalg.norm(vertices_centered, axis=1))
    vertices_normalized = vertices_centered / scale
    return vertices_normalized, centroid, scale

def denormalize_vertices(vertices_normalized, centroid, scale):
    """Reverse the normalization process."""
    return vertices_normalized * scale + centroid

def interpolate_vertices(vertices_list, num_vertices):
    """Interpolate vertices to ensure all meshes have the same number of vertices."""
    interpolated_vertices_list = []
    for vertices in vertices_list:
        if vertices.shape[0] != num_vertices:
            tree = cKDTree(vertices)
            _, idx = tree.query(vertices[:num_vertices])
            interpolated_vertices = vertices[idx]
        else:
            interpolated_vertices = vertices
        interpolated_vertices_list.append(interpolated_vertices)
    return interpolated_vertices_list

def compute_mean_shape(vertices_array, num_vertices):
    """Compute the mean shape using Frechet Mean."""
    preshape_space = PreShapeSpace(k_landmarks=num_vertices, ambient_dim=3)
    frechet_mean = FrechetMean(space=preshape_space)
    frechet_mean.optimizer.max_iter = 5000  # Increased iterations
    frechet_mean.optimizer.tolerance = 1e-3  # Reduced tolerance
    frechet_mean.initialization = vertices_array[0]  # Initialize with the first mesh
    mean_shape = frechet_mean.fit(vertices_array).estimate_
    return mean_shape

import open3d as o3d

import open3d as o3d

import numpy as np
import open3d as o3d

def align_meshes(meshes, desired_num_vertices=5000, max_correspondence_distance=0.05, max_iterations=5000, radius_normal=0.1, radius_feature=0.2):
    """Align all meshes to the first mesh using RANSAC-based feature matching followed by ICP."""
    
    reference_mesh = meshes[0]
    reference_pcd = reference_mesh.sample_points_uniformly(number_of_points=desired_num_vertices)
    aligned_meshes = [reference_mesh]
    
    # Iterate through each mesh to align
    for mesh in meshes[1:]:
        pcd = mesh.sample_points_uniformly(number_of_points=desired_num_vertices)
        
        # Step 1: Global Registration (RANSAC + Feature Matching)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        reference_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        reference_fpfh = o3d.pipelines.registration.compute_fpfh_feature(reference_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        # Perform RANSAC-based global registration with mutual filter disabled
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            reference_pcd, pcd, reference_fpfh, fpfh, mutual_filter=False, 
            max_correspondence_distance=max_correspondence_distance,
            ransac_n=3,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.99)
        )
        initial_transformation = result_ransac.transformation
        # Visualize RANSAC result before ICP refinement
        o3d.visualization.draw_geometries([reference_pcd, pcd, reference_pcd.transform(result_ransac.transformation)])

        
        # Step 2: ICP Refinement (convert initial transformation to numpy array)
        initial_transformation_np = np.asarray(initial_transformation)  # Convert to numpy array
        
        reg_icp = o3d.pipelines.registration.registration_icp(
            pcd, reference_pcd, max_correspondence_distance=max_correspondence_distance,
            init=initial_transformation_np,  # Pass as numpy array
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )

        # Apply the transformation from ICP refinement
        aligned_mesh = mesh.transform(reg_icp.transformation)
        aligned_meshes.append(aligned_mesh)
    
    # Optionally visualize the result
    o3d.visualization.draw_geometries(aligned_meshes)
    
    return aligned_meshes




def visualize_meshes(meshes):
    """Visualize all meshes together as wireframes (edges only)."""
    wireframes = []
    for mesh in meshes:
        # Convert the mesh into a line set (wireframe)
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireframe.paint_uniform_color([0, 0, 0])  # Set wireframe color (black)
        wireframes.append(wireframe)
    
    # Draw the wireframe-only meshes
    o3d.visualization.draw_geometries(wireframes)

def main():
    # Load and preprocess meshes sequentially
    meshes = [load_and_preprocess_mesh(file) for file in mesh_files]
    meshes = [mesh for mesh in meshes if mesh is not None]
    
    if not meshes:
        logging.error("No valid meshes to process.")
        return

    # Align meshes using ICP
    aligned_meshes = align_meshes(meshes)

    # Visualize all aligned meshes together with reduced opacity
    visualize_meshes(aligned_meshes)

    # Convert meshes to numpy arrays of vertices
    vertices_list = [np.asarray(mesh.vertices) for mesh in aligned_meshes]
    num_vertices = min(vertices.shape[0] for vertices in vertices_list)
    interpolated_vertices_list = interpolate_vertices(vertices_list, num_vertices)
    
    # Normalize vertices
    normalized_vertices_list = []
    centroids = []
    scales = []
    for vertices in interpolated_vertices_list:
        vertices_normalized, centroid, scale = normalize_vertices(vertices)
        normalized_vertices_list.append(vertices_normalized)
        centroids.append(centroid)
        scales.append(scale)

    # Convert vertices to Geomstats format
    vertices_array = gs.array(normalized_vertices_list)

    # Compute the mean shape
    mean_shape = compute_mean_shape(vertices_array, num_vertices)

    # Denormalize the mean shape
    mean_shape_denormalized = denormalize_vertices(mean_shape, centroids[0], scales[0])

    # Convert the mean shape back to an Open3D TriangleMesh
    mean_mesh = o3d.geometry.TriangleMesh()
    mean_mesh.vertices = o3d.utility.Vector3dVector(mean_shape_denormalized)
    mean_mesh.triangles = aligned_meshes[0].triangles  # Use the triangles from the first mesh

    # Save and visualize the mean shape mesh
    o3d.io.write_triangle_mesh("mean_shape.ply", mean_mesh)
    logging.info("Mean shape saved to 'mean_shape.ply'.")
    o3d.visualization.draw_geometries([mean_mesh])

if __name__ == "__main__":
    main()
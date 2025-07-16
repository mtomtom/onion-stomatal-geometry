import numpy as np
import pyvista as pv
import trimesh
from shapely.geometry import Polygon
import argparse
from trimesh.creation import sweep_polygon

def create_stomatal_complex(
    pore_area: float,
    pore_length: float,
    stomatal_height: float,
    cross_aspect_ratio: float,
    wall_thickness: float,
    output_path: str = "stomatal_complex.ply",
):
    """
    Generates a single, fully‐watertight stomatal ring by sweeping
    a hollow 2D cross-section around a closed elliptical path.
    This method is robust as it avoids joining or boolean operations.
    """
    # --- 1. Define Cross-Section Geometry ---
    # The cross-section is an ellipse defining the shape of the guard cell wall.
    cs_semi_height = wall_thickness / cross_aspect_ratio  # Semi-major axis (a)
    cs_semi_width = wall_thickness # Semi-minor axis (b)

    # --- 2. Define Pore and Centerline Geometry ---
    # The centerline is the path the cross-section will be swept along.
    pore_semi_length = pore_length / 2.0
    pore_semi_width = pore_area / (np.pi * pore_semi_length) if pore_semi_length > 1e-9 else 0.0
    
    # The centerline is offset from the pore by the width of the cross-section itself.
    centerline_semi_length = pore_semi_length + cs_semi_width
    centerline_semi_width = pore_semi_width + cs_semi_width

    # --- 3. Build the Closed Elliptical Path ---
    n_path_points = 128
    theta = np.linspace(0, 2 * np.pi, n_path_points, endpoint=False) # Full 360 degrees
    path = np.column_stack((
        centerline_semi_length * np.cos(theta),
        centerline_semi_width * np.sin(theta),
        np.zeros_like(theta)
    ))
    # Close the loop for trimesh
    path = np.vstack([path, path[0]])

    # --- 4. Define the Hollow Cross-Section Polygon ---
    n_cs_points = 64
    angles = np.linspace(0, 2 * np.pi, n_cs_points, endpoint=False)
    
    # Outer boundary of the cross-section
    outer_verts = np.column_stack([cs_semi_width * np.cos(angles), cs_semi_height * np.sin(angles)])
    
    # Inner boundary is inset by the wall_thickness
    # We must ensure the thickness doesn't exceed the cell's dimensions
    inner_semi_width = max(0, cs_semi_width - wall_thickness)
    inner_semi_height = max(0, cs_semi_height - wall_thickness)
    inner_verts = np.column_stack([inner_semi_width * np.cos(angles), inner_semi_height * np.sin(angles)])
    
    # Create the shapely polygon with a hole (inner vertices must be reversed)
    cross_section_poly = Polygon(outer_verts, holes=[inner_verts[::-1]])

    # --- 5. Sweep the Polygon Around the Path ---
    mesh = sweep_polygon(polygon=cross_section_poly, path=path)
    mesh.fix_normals()
    
    if not mesh.is_watertight:
        print("Warning: Generated mesh is not watertight.")

    # --- 6. Export via PyVista ---
    faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
    pv_mesh = pv.PolyData(mesh.vertices, faces)
    pv_mesh.save(output_path)
    print(f"Saved stomatal complex ring to {output_path}")

    return pv_mesh

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate an idealized stomatal guard cell ring.")
    p.add_argument("--pore-area", type=float, default=100.0, help="Target area of the stomatal pore.")
    p.add_argument("--pore-length", type=float, default=20.0, help="Target length of the stomatal pore (major axis).")
    p.add_argument("--stomatal-height", type=float, default=5.0, help="The total height of the stomata in the Z-axis.")
    p.add_argument("--cross-ar", type=float, default=2.0, help="Aspect ratio of the guard cell cross-section (height/width).")
    p.add_argument("--wall-thickness", type=float, default=2.5, help="Thickness of the guard cell wall.")
    p.add_argument("--output", default="stomatal_complex.ply", help="Path to save the output .ply file.")
    args = p.parse_args()

    # Use the hardcoded values from the file for demonstration
    pore_area = 40.4
    pore_length = 13.1
    stomatal_height = 42.0 # Renamed from stomatal_length for clarity
    wall_thickness = 17.2 # Renamed from cell_width for clarity
    cross_aspect_ratio = 1.6

    create_stomatal_complex(
        pore_area=pore_area,
        pore_length=pore_length,
        stomatal_height=stomatal_height,
        cross_aspect_ratio=cross_aspect_ratio,
        wall_thickness=wall_thickness,
        output_path=args.output
    )



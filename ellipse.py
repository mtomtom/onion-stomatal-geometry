import numpy as np
import trimesh
import argparse
from shapely.geometry import Polygon

# Attempt to import the preferred solver, but have a fallback
try:
    from scipy.optimize import fsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy library not found. Using a simple iterative solver.")
    print("         For better performance, please install it: pip install scipy")


def create_torusoid(
    pore_area: float,
    pore_length: float,
    cross_section_long_axis: float,
    midsection_cross_section_aspect_ratio: float,
    tip_cross_section_aspect_ratio: float,
    output_path: str,
    n_path: int = 100,
    n_cross: int = 50,
):
    """
    Generates a torusoid mesh, solving for the correct dimensions to precisely
    match the target pore area and length.
    """
    
    # --- Pre-Solver Setup ---
    # Establish a fixed "shape" (aspect ratio) for the centerline based on the initial guess.
    # The solver will then find the correct "size" (L_cs) to fit this shape.
    target_inner_Rx = pore_length / 2.0
    target_inner_Ry = pore_area / (np.pi * target_inner_Rx)

    a_base_guess = cross_section_long_axis / 2.0
    b_base_guess = a_base_guess / midsection_cross_section_aspect_ratio
    const_cs_area_guess = np.pi * a_base_guess * b_base_guess
    b_tip_guess = np.sqrt(const_cs_area_guess / (np.pi * tip_cross_section_aspect_ratio))
    a_tip_guess = tip_cross_section_aspect_ratio * b_tip_guess

    Rx_guess = target_inner_Rx + a_tip_guess
    Ry_guess = target_inner_Ry + a_base_guess
    # This fixed aspect ratio is the key to a stable solver.
    fixed_centerline_ar = Ry_guess / Rx_guess

    # --- Solver Function ---
    def _get_pore_area_for_L_cs(L_cs_candidate):
        """Helper function for the solver. Calculates the actual pore area for a given cross-section long axis (L_cs)."""
        if L_cs_candidate <= 0: return np.inf

        a_base = L_cs_candidate / 2.0
        b_base = a_base / midsection_cross_section_aspect_ratio
        const_cs_area = np.pi * a_base * b_base
        b_tip = np.sqrt(const_cs_area / (np.pi * tip_cross_section_aspect_ratio))
        a_tip = tip_cross_section_aspect_ratio * b_tip

        # Calculate centerline using the fixed aspect ratio
        R_x = target_inner_Rx + a_tip
        R_y = R_x * fixed_centerline_ar

        # Generate the inner pore boundary vertices
        t = np.linspace(0, 2 * np.pi, n_path, endpoint=False)
        path = np.stack([R_x * np.cos(t), R_y * np.sin(t), np.zeros_like(t)], axis=-1)
        tangent = np.stack([-R_x * np.sin(t), R_y * np.cos(t), np.zeros_like(t)], axis=-1)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = np.cross(tangent, np.array([0, 0, 1]))
        normal /= np.linalg.norm(normal, axis=1)[:, None]

        inner_pore_vertices_2d = []
        for i in range(n_path):
            pos_from_midsection = 1 - abs(np.sin(t[i]))
            if pos_from_midsection < 0.8:
                interp_fraction = pos_from_midsection / 0.8
                current_ar = midsection_cross_section_aspect_ratio + interp_fraction * (tip_cross_section_aspect_ratio - midsection_cross_section_aspect_ratio)
            else:
                current_ar = tip_cross_section_aspect_ratio
            
            b_current = np.sqrt(const_cs_area / (np.pi * current_ar))
            a_current = current_ar * b_current
            
            inner_vertex = path[i] - a_current * normal[i]
            inner_pore_vertices_2d.append(inner_vertex[:2])
        
        return Polygon(inner_pore_vertices_2d).area

    # --- Step 1: Solve for the correct cross-section size ---
    print("--- Solving for cross-section size to match target pore area ---")
    
    def _area_error(L_cs_candidate):
        return _get_pore_area_for_L_cs(L_cs_candidate) - pore_area

    if HAS_SCIPY:
        solution, _, ier, msg = fsolve(_area_error, cross_section_long_axis, full_output=True)
        if ier != 1:
            print(f"  WARNING: Solver did not converge. Message: {msg}")
        final_L_cs = solution[0]
    else: # Fallback iterative solver
        L_cs_current = cross_section_long_axis
        for i in range(15):
            err = _area_error(L_cs_current)
            if abs(err) < 0.001 * pore_area: break
            h = L_cs_current * 0.01
            derivative = (_area_error(L_cs_current + h) - err) / h
            if abs(derivative) < 1e-6: break
            L_cs_current -= err / derivative
        final_L_cs = L_cs_current
    
    final_area = _get_pore_area_for_L_cs(final_L_cs)
    print(f"  Target Area: {pore_area:.2f}, Achieved Area: {final_area:.2f}")
    print(f"  Initial L_cs Guess: {cross_section_long_axis:.2f}, Solved L_cs: {final_L_cs:.2f}")

    # --- Step 2: Generate the final mesh using the solved dimensions ---
    a_base = final_L_cs / 2.0
    b_base = a_base / midsection_cross_section_aspect_ratio
    const_cs_area = np.pi * a_base * b_base
    b_tip = np.sqrt(const_cs_area / (np.pi * tip_cross_section_aspect_ratio))
    a_tip = tip_cross_section_aspect_ratio * b_tip

    # Use the same stable logic to calculate final centerline
    R_x = target_inner_Rx + a_tip
    R_y = R_x * fixed_centerline_ar

    t = np.linspace(0, 2 * np.pi, n_path, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_cross, endpoint=False)
    path = np.stack([R_x * np.cos(t), R_y * np.sin(t), np.zeros_like(t)], axis=-1)

    tangent = np.stack([-R_x * np.sin(t), R_y * np.cos(t), np.zeros_like(t)], axis=-1)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    ref = np.array([0, 0, 1])
    normal = np.cross(tangent, ref)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    binormal = np.cross(tangent, normal)

    vertices = []
    for i in range(n_path):
        pos_from_midsection = 1 - abs(np.sin(t[i]))
        if pos_from_midsection < 0.8:
            interp_fraction = pos_from_midsection / 0.8
            current_ar = midsection_cross_section_aspect_ratio + interp_fraction * (tip_cross_section_aspect_ratio - midsection_cross_section_aspect_ratio)
        else:
            current_ar = tip_cross_section_aspect_ratio
        
        b_current = np.sqrt(const_cs_area / (np.pi * current_ar))
        a_current = current_ar * b_current
        
        for j in range(n_cross):
            offset = (a_current * np.cos(phi[j]) * normal[i] + b_current * np.sin(phi[j]) * binormal[i])
            vertex = path[i] + offset
            vertices.append(vertex)
    vertices = np.array(vertices)

    # --- Step 3 & 4: Build faces, walls, and export (logic unchanged) ---
    faces = []
    for i in range(n_path):
        for j in range(n_cross):
            i_next = (i + 1) % n_path
            j_next = (j + 1) % n_cross
            idx0, idx1 = i * n_cross + j, i * n_cross + j_next
            idx2, idx3 = i_next * n_cross + j, i_next * n_cross + j_next
            faces.append([idx0, idx1, idx2])
            faces.append([idx1, idx3, idx2])

    wall_faces = []
    wall_1_v_indices = np.arange(0, n_cross)
    wall_1_center = np.mean(vertices[wall_1_v_indices], axis=0)
    wall_1_center_idx = len(vertices)
    vertices = np.vstack([vertices, wall_1_center])
    for j in range(n_cross):
        v_idx1 = wall_1_v_indices[j]
        v_idx2 = wall_1_v_indices[(j + 1) % n_cross]
        wall_faces.append([wall_1_center_idx, v_idx2, v_idx1])
        wall_faces.append([wall_1_center_idx, v_idx1, v_idx2])

    i_wall_2 = n_path // 2
    wall_2_v_indices = np.arange(i_wall_2 * n_cross, (i_wall_2 + 1) * n_cross)
    wall_2_center = np.mean(vertices[wall_2_v_indices], axis=0)
    wall_2_center_idx = len(vertices)
    vertices = np.vstack([vertices, wall_2_center])
    for j in range(n_cross):
        v_idx1 = wall_2_v_indices[j]
        v_idx2 = wall_2_v_indices[(j + 1) % n_cross]
        wall_faces.append([wall_2_center_idx, v_idx1, v_idx2])
        wall_faces.append([wall_2_center_idx, v_idx2, v_idx1])

    faces.extend(wall_faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    mesh.export(output_path)
    print(f"✅ Exported '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a torusoid mesh with specified dimensions.")
    parser.add_argument("--pore-area", type=float, default=40, help="Area of the central pore.")
    parser.add_argument("--pore-length", type=float, default=13.1, help="Length of the central pore.")
    parser.add_argument("--cross-section-long-axis", type=float, default=16.7, help="Initial GUESS for the cross-section's long axis at the midsection.")
    parser.add_argument("--midsection-ar", type=float, default=1.5, help="Aspect ratio (long/short) of the cross-section at the midsection.")
    parser.add_argument("--tip-ar", type=float, default=1.0, help="Aspect ratio (long/short) of the cross-section at the tips.")
    parser.add_argument("--output", default="torusoid_variable_cs.obj", help="Output file path.")
    
    args = parser.parse_args()

    mesh_1_2 = [[15.8, 9.3],[16.6, 11.1]]
    mesh_1_3 = [[16.1, 10.8],[16.2, 10.1]]
    mesh_1_4 = [[15.6, 11.2],[13.8, 11.6]]
    mesh_1_5 = [[18.1, 12.5], [17.4, 11.5]]
    mesh_1_6 = [[16.3, 11.6],[15.2, 12.4]]
    mesh_1_8 = [[14.3, 11.5], [15.5, 11.3]]
    mesh_2_1 = [[13.4, 10.2], [13.2, 9.3]]
    mesh_2_3 = [[13.6, 7.1],[15.8, 7.9]] ## Issues?
    ##mesh_2_4 ## Issue
    mesh_2_6a = [[17.0, 10.8],[16.9, 10.2]]
    mesh_2_6b = [[16.2, 12.6],[15.6, 10.8]] ## Odd cross section
    mesh_2_7a = [[13.9, 10.5], [13.7, 11.6]]
    mesh_3_1 = [[14.6, 10.9], [14.6, 11.1]]
    mesh_3_2 = [[17.3, 12.8],[17.3, 12.3]]
    mesh_3_3 = [[17.9, 13.0],[17.2, 11.2]]

    meshes_to_process = [mesh_1_2, mesh_1_3, mesh_1_4, mesh_1_5]
    mesh_names = ["mesh_1_2", "mesh_1_3", "mesh_1_4", "mesh_1_5"]
    areas =[40.4, 44.7, 43, 49.6]
    lengths = [13.1, 12.6, 13.6, 13.5]
    tip_aspect_ratios = [1.04, 1.1, 1.16, 1.1]

    for mesh, name in zip(meshes_to_process, mesh_names):

        ## Create std mesh
        output_path = f"{name}_std.ply"
        cross_section_long_axis = mesh[0][0]
        midsection_cross_section_aspect_ratio = mesh[0][0] / mesh[0][1]
        tip_cross_section_aspect_ratio = mesh[1][0] / mesh[1][1]
        pore_length = lengths[meshes_to_process.index(mesh)]
        pore_area = areas[meshes_to_process.index(mesh)]

        print(f"--- Running with parameters from mesh {meshes_to_process.index(mesh) + 1} ---")
        print(f"Target Pore Area: {pore_area}")
        print(f"Target Pore Length: {pore_length}")
        print(f"Initial Cross-section long axis guess: {cross_section_long_axis:.2f}")
        print(f"Midsection aspect ratio: {midsection_cross_section_aspect_ratio:.2f}")
        print(f"Tip aspect ratio: {tip_cross_section_aspect_ratio:.2f}")

        create_torusoid(
            pore_area=pore_area,
            pore_length=pore_length,
            cross_section_long_axis=cross_section_long_axis,
            midsection_cross_section_aspect_ratio=midsection_cross_section_aspect_ratio,
            tip_cross_section_aspect_ratio=tip_cross_section_aspect_ratio,
            output_path=output_path
        )

        ## Create variable cross-section mesh
        output_path = f"{name}_variable_cs.ply"
        tip_cross_section_aspect_ratio = tip_aspect_ratios[meshes_to_process.index(mesh)]
        create_torusoid(
            pore_area=pore_area,
            pore_length=pore_length,
            cross_section_long_axis=cross_section_long_axis,
            midsection_cross_section_aspect_ratio=midsection_cross_section_aspect_ratio,
            tip_cross_section_aspect_ratio=tip_cross_section_aspect_ratio,
            output_path=output_path
        )   

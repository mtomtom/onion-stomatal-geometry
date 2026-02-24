import sys
from pathlib import Path
import trimesh
import matplotlib.pyplot as plt
from scipy.stats import linregress

src_path = str(Path.cwd().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import cross_section_helpers as csh
import ast
import re
import numpy as np

## Function to get the cross section of the idealised mesh at the midpoint
def extract_midpoint_cross_section(mesh, guard_cell="left"):
    """
    Extract a midpoint cross-section and return a list with one (N,3) float array,
    ordered consistently. Uses wall-centre helpers when possible, else falls back
    to PCA-based midplane slicing.
    """
    import numpy as np
    import cross_section_helpers as csh
    from sklearn.decomposition import PCA

    def _order_and_validate(pts, normal, midpoint):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
            raise ValueError(f"Selected section invalid shape {pts.shape}; expected (N,3) with N>=3.")
        ordered = csh.order_points_consistently(pts, normal=normal, midpoint=midpoint)
        return [ordered.astype(float)]

    # Try robust wall-based method
    try:
        wall_vertices = csh.find_wall_vertices(mesh)
        if wall_vertices.size == 0:
            wall_vertices = csh.find_wall_vertices_vertex_normals(mesh)
        if wall_vertices.size > 0:
            centre_top, centre_bottom, *_ = csh.get_top_bottom_wall_centres(mesh, wall_vertices)
            midpoint, _traces, section_points, local_axes = csh.get_midpoint_cross_section_from_centres(
                mesh, centre_top, centre_bottom
            )
            if section_points is None or len(section_points) < 3:
                raise RuntimeError("Midpoint cross-section failed (insufficient points)")
            left_section, right_section, *_ = csh.get_left_right_midsections(section_points, midpoint, local_axes)
            pts = left_section if guard_cell == "left" else right_section
            wall_vec = local_axes[0]
            return _order_and_validate(pts, normal=wall_vec, midpoint=midpoint)
    except Exception:
        # Fall back to PCA-based method
        pass

    # Fallback: PCA-based midplane perpendicular to longest axis
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.shape[0] < 3:
        raise ValueError("Mesh has too few vertices to compute a cross-section.")

    p = PCA(n_components=3).fit(verts)
    longest_axis = p.components_[0]
    # Project onto plane orthogonal to longest axis through mesh centroid
    centroid = verts.mean(axis=0)

    # Take section using trimesh.section
    section = mesh.section(plane_origin=centroid, plane_normal=longest_axis)
    if section is None:
        raise ValueError("Fallback sectioning failed: no intersection with midplane.")

    # Convert section to dense points
    if hasattr(section, 'discrete') and section.discrete:
        section_points = np.vstack([seg for seg in section.discrete if len(seg) > 0])
    else:
        section_points = np.asarray(section.vertices, dtype=float)

    if section_points is None or len(section_points) < 3:
        raise ValueError("Fallback section produced insufficient points (<3).")

    # Build local axes: wall_vec ~ longest_axis, left_right_vec via PCA in plane
    # Use PCA on section points to get principal directions in the section plane
    p2 = PCA(n_components=2).fit(section_points)
    left_right_vec = p2.components_[0]
    wall_vec = longest_axis / (np.linalg.norm(longest_axis) + 1e-12)

    # Midpoint for ordering/splitting
    midpoint = section_points.mean(axis=0)

    # Split into left/right using projection on left_right_vec
    rel = section_points - midpoint
    proj = rel @ left_right_vec
    left_section = section_points[proj < 0]
    right_section = section_points[proj >= 0]

    pts = left_section if guard_cell == "left" else right_section
    return _order_and_validate(pts, normal=wall_vec, midpoint=midpoint)

def create_ar_area_plots(confocal_df=None, df_combined=None, selected_meshes=None, colours=None):
    # Create first figure for Pore Area plots
    plt.figure(figsize=(8, 6))
    
    for mesh in selected_meshes:
        idealised_oval = df_combined[(df_combined["Mesh ID"].str.contains(mesh)) & (df_combined["Cross-section type"] == "oval")]
        idealised_circular = df_combined[(df_combined["Mesh ID"].str.contains(mesh)) & (df_combined["Cross-section type"] == "circular")]
        mesh_data_confocal = confocal_df[confocal_df["Mesh ID"].str.contains(mesh)]
        if idealised_oval.empty or mesh_data_confocal.empty:
            print(f"Skipping mesh {mesh} due to missing data.")
            continue
        idealised_oval_start = idealised_oval[idealised_oval["Pressure"] == 0.0]
        if len(idealised_oval_start) == 0:
            print(f"Skipping mesh {mesh} due to missing start data in idealised oval.")
            continue
        idealised_circular_start = idealised_circular[idealised_circular["Pressure"] == 0.0]
        mesh_data_confocal_start = mesh_data_confocal[mesh_data_confocal["Pressure"] == 0.0]
        if len(mesh_data_confocal_start) == 0:
            print(f"Skipping mesh {mesh} due to missing start data in confocal.")
            continue
        idealised_oval_end = idealised_oval[idealised_oval["Pressure"] == idealised_oval["Pressure"].max()]
        idealised_circular_end = idealised_circular[idealised_circular["Pressure"] == idealised_circular["Pressure"].max()]
        if len(idealised_circular_start) == 0 or len(idealised_circular_end) == 0:
            print(f"Skipping mesh {mesh} due to missing start or end data in idealised circular.")
            continue
        mesh_data_confocal_end = mesh_data_confocal[mesh_data_confocal["Pressure"] == mesh_data_confocal["Pressure"].max()]
        idealised_oval_pore_area_change = (idealised_oval_end["Pore Area"].values[0]/ idealised_oval_start["Pore Area"].values[0])
        idealised_circular_pore_area_change = (idealised_circular_end["Pore Area"].values[0] / idealised_circular_start["Pore Area"].values[0])
        confocal_pore_area_change = (mesh_data_confocal_end["Pore Area"].values[0] / mesh_data_confocal_start["Pore Area"].values[0])

        plt.plot(idealised_oval_start["Pore Area"].values[0], idealised_oval_pore_area_change, 'o', markersize = 10, markeredgecolor='k', label="Idealised Oval", color=colours['idealised_oval'])
        plt.plot(idealised_circular_start["Pore Area"].values[0], idealised_circular_pore_area_change, 'o', markersize = 10, markeredgecolor='k', label="Idealised Circular", color=colours['idealised_circular'])
        plt.plot(mesh_data_confocal_start["Pore Area"].values[0], confocal_pore_area_change, 'o', markersize = 10, markeredgecolor='k',label="Experimental", color=colours['empirical'])

    ## Plot y=1 in red (after loop)
    xs = np.arange(0,80)
    ys = np.ones(len(xs))
    plt.plot(xs, ys, 'r--', label='y=1')
    
    plt.xlabel("Start Pore Area (um^2)")
    plt.ylabel("End Pore Area / Start Pore Area")

    # Example: Restrict legend to the first two elements
    handles, labels = plt.gca().get_legend_handles_labels()  # Get all handles and labels
    plt.legend(handles[:3], labels[:3], loc='lower right')  # Use only the first three elements

    plt.ylim(0.4, 1.8)
    plt.xlim(0, 80)

    plt.savefig("../Figures/Fig3/idealised_confocal_above_one.png", dpi=300)
    plt.show()

    import re

    def extract_float(val):
        # Handles strings like '[np.float64(1.4530883389409186)]'
        import re
        if isinstance(val, str):
            # Match a float inside parentheses after np.float64
            match = re.search(r'np\.float64\(([-+]?[0-9]*\.?[0-9]+)\)', val)
            if match:
                return round(float(match.group(1)), 2)
            # Fallback: match any float number in the string
            match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', val)
            if match:
                return round(float(match.group(1)), 2)
            raise ValueError(f"Could not extract float from {val}")
        else:
            return round(float(val), 2)



    oval_ars = []
    circular_ars = []
    confocal_ars = []
    oval_changes = []
    circular_changes = []
    confocal_changes = []

    # Create second figure for Aspect Ratio plots
    plt.figure(figsize=(10, 6))
    
    # c) Plot starting aspect ratio vs end pore area / start pore area for idealised and confocal (line of best fit)
    for mesh in selected_meshes:
        idealised_oval = df_combined[(df_combined["Mesh ID"].str.contains(mesh)) & (df_combined["Cross-section type"] == "oval")]
        idealised_circular = df_combined[(df_combined["Mesh ID"].str.contains(mesh)) & (df_combined["Cross-section type"] == "circular")]
        mesh_data_confocal = confocal_df[confocal_df["Mesh ID"].str.contains(mesh)]
        if idealised_oval.empty or mesh_data_confocal.empty:
            print(f"Skipping mesh {mesh} due to missing data.")
            continue
        idealised_oval_start = idealised_oval[idealised_oval["Pressure"] == 0.0]
        if len(idealised_oval_start) == 0:
            print(f"Skipping mesh {mesh} due to missing start data in idealised oval.")
            continue
        idealised_circular_start = idealised_circular[idealised_circular["Pressure"] == 0.0]
        mesh_data_confocal_start = mesh_data_confocal[mesh_data_confocal["Pressure"] == 0.0]
        if len(mesh_data_confocal_start) == 0:
            print(f"Skipping mesh {mesh} due to missing start data in confocal.")
            continue
        idealised_oval_end = idealised_oval[idealised_oval["Pressure"] == idealised_oval["Pressure"].max()]
        idealised_circular_end = idealised_circular[idealised_circular["Pressure"] == idealised_circular["Pressure"].max()]
        if len(idealised_circular_start) == 0 or len(idealised_circular_end) == 0:
            print(f"Skipping mesh {mesh} due to missing start or end data in idealised circular.")
            continue
        mesh_data_confocal_end = mesh_data_confocal[mesh_data_confocal["Pressure"] == mesh_data_confocal["Pressure"].max()]
        idealised_oval_pore_area_change = (idealised_oval_end["Pore Area"].values[0]/ idealised_oval_start["Pore Area"].values[0])
        idealised_circular_pore_area_change = (idealised_circular_end["Pore Area"].values[0] / idealised_circular_start["Pore Area"].values[0])
        confocal_pore_area_change = (mesh_data_confocal_end["Pore Area"].values[0] / mesh_data_confocal_start["Pore Area"].values[0])

        idealised_oval_ar = extract_float(idealised_oval_start['Aspect Ratio'].values[0])
        idealised_circular_ar = extract_float(idealised_circular_start['Aspect Ratio'].values[0])
        confocal_ar = mesh_data_confocal_start["Midsection AR left"].values[0]
        plt.plot(idealised_oval_ar, idealised_oval_pore_area_change, 'o', label="Idealised Oval (a)", color=colours['idealised_oval'], markersize = 10)
        plt.plot(idealised_circular_ar, idealised_circular_pore_area_change, 'o', label="Idealised Circular (b)", color=colours['idealised_circular'], markersize = 10, markeredgecolor='k')
        plt.plot(confocal_ar, confocal_pore_area_change, 'o', label="Experimental (c)", color=colours['empirical'], markersize = 10, markeredgecolor='k')

        oval_ars.append(idealised_oval_ar)
        circular_ars.append(idealised_circular_ar)
        confocal_ars.append(confocal_ar)
        oval_changes.append(idealised_oval_pore_area_change)
        circular_changes.append(idealised_circular_pore_area_change)
        confocal_changes.append(confocal_pore_area_change)

    ## Plot y=1 in red (after loop)
    xs = np.arange(0,3)
    ys = np.ones(len(xs))
    plt.plot(xs, ys, 'r--', label='y=1')

    # Example: oval_ars and oval_changes are your x and y values for the oval group
    if len(oval_ars) > 1:
        slope, intercept, r, p, stderr = linregress(oval_ars, oval_changes)
        summary = (
            f"Linear regression: slope = {slope:.3f} ± {stderr:.3f}, "
            f"intercept = {intercept:.3f}, "
            f"Pearson r = {r:.3f}, "
            f"p-value = {p:.3g}"
        )
        print("Oval: " + summary)
        xs = np.linspace(min(oval_ars), max(oval_ars), 100)
        plt.plot(xs, slope*xs + intercept, color=colours['idealised_oval'], linestyle='-', label='Oval fit')

    # if len(circular_ars) > 1:
    #     slope, intercept, r, p, stderr = linregress(circular_ars, circular_changes)
    #     xs = np.linspace(min(circular_ars), max(circular_ars), 100)
    #     plt.plot(xs, slope*xs + intercept, color=colours['idealised_circular'], linestyle='-', label='Circular fit')

    if len(confocal_ars) > 1:
        slope, intercept, r, p, stderr = linregress(confocal_ars, confocal_changes)
        summary = (
            f"Linear regression: slope = {slope:.3f} ± {stderr:.3f}, "
            f"intercept = {intercept:.3f}, "
            f"Pearson r = {r:.3f}, "
            f"p-value = {p:.3g}"
        )
        print("Empirical: " + summary)
        xs = np.linspace(min(confocal_ars), max(confocal_ars), 100)
        plt.plot(xs, slope*xs + intercept, color=colours['empirical'], linestyle='-', label='Realistic fit')

    plt.xlabel("Starting Aspect Ratio")
    plt.ylabel("End Pore Area / Start Pore Area")
    plt.xlim(0,2.5)
    # Example: Restrict legend to the first two elements
    handles, labels = plt.gca().get_legend_handles_labels()  # Get all handles and labels
    plt.legend(handles[:3], labels[:3],loc = 4)  # Use only the first three elements
    plt.xlim(0.9, 1.7)
    plt.ylim(0.4, 1.8)
    plt.savefig("../Figures/Fig3/idealised_confocal_AR_above_one_bf.png", dpi=300)
    plt.show()

def AR_area_change_no_oval(selected_meshes, df_combined=None, confocal_df = None, colours = None):

    def extract_float(val):
        # Handles strings like '[np.float64(1.4530883389409186)]'
        import re
        if isinstance(val, str):
            # Match a float inside parentheses after np.float64
            match = re.search(r'np\.float64\(([-+]?[0-9]*\.?[0-9]+)\)', val)
            if match:
                return round(float(match.group(1)), 2)
            # Fallback: match any float number in the string
            match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', val)
            if match:
                return round(float(match.group(1)), 2)
            raise ValueError(f"Could not extract float from {val}")
        else:
            return round(float(val), 2)

    # Create second figure for Aspect Ratio plots
    plt.figure(figsize=(8, 6))


    circular_ars = []
    confocal_ars = []

    circular_changes = []
    confocal_changes = []
    
    # c) Plot starting aspect ratio vs end pore area / start pore area for idealised and confocal (line of best fit)
    for mesh in selected_meshes:
        idealised_circular = df_combined[(df_combined["Mesh ID"].str.contains(mesh)) & (df_combined["Cross-section type"] == "circular")]
        mesh_data_confocal = confocal_df[confocal_df["Mesh ID"].str.contains(mesh)]
        idealised_circular_start = idealised_circular[idealised_circular["Pressure"] == 0.0]
        mesh_data_confocal_start = mesh_data_confocal[mesh_data_confocal["Pressure"] == 0.0]
        if len(mesh_data_confocal_start) == 0:
            print(f"Skipping mesh {mesh} due to missing start data in confocal.")
            continue
        idealised_circular_end = idealised_circular[idealised_circular["Pressure"] == idealised_circular["Pressure"].max()]
        if len(idealised_circular_start) == 0 or len(idealised_circular_end) == 0:
            print(f"Skipping mesh {mesh} due to missing start or end data in idealised circular.")
            continue
        mesh_data_confocal_end = mesh_data_confocal[mesh_data_confocal["Pressure"] == mesh_data_confocal["Pressure"].max()]
        idealised_circular_pore_area_change = (idealised_circular_end["Pore Area"].values[0] / idealised_circular_start["Pore Area"].values[0])
        confocal_pore_area_change = (mesh_data_confocal_end["Pore Area"].values[0] / mesh_data_confocal_start["Pore Area"].values[0])
        
        idealised_circular_ar = extract_float(idealised_circular_start['Aspect Ratio'].values[0])
        confocal_ar = mesh_data_confocal_start["Midsection AR left"].values[0]

        circular_ars.append(idealised_circular_ar)
        confocal_ars.append(confocal_ar)
        circular_changes.append(idealised_circular_pore_area_change)
        confocal_changes.append(confocal_pore_area_change)
        
        plt.plot(idealised_circular_ar, idealised_circular_pore_area_change, 'o', label="Idealised Circular", color=colours['idealised_circular'], markersize = 10,markeredgecolor='k')
        plt.plot(confocal_ar, confocal_pore_area_change, 'o', label="Experimental", color=colours['empirical'], markersize = 10, markeredgecolor='k')


    ## Plot y=1 in red (after loop)
    xs = np.arange(0,3)
    ys = np.ones(len(xs))
    plt.plot(xs, ys, 'r--', label='y=1')

    #if len(confocal_ars) > 1:
    #    slope, intercept, r, p, stderr = linregress(confocal_ars, confocal_changes)
    #    summary = (
    #        f"Linear regression: slope = {slope:.3f} ± {stderr:.3f}, "
    #        f"intercept = {intercept:.3f}, "
    #        f"Pearson r = {r:.3f}, "
    #        f"p-value = {p:.3g}"
    #    )
    #    print("Empirical: " + summary)
    #    xs = np.linspace(min(confocal_ars), max(confocal_ars), 100)
    #    plt.plot(xs, slope*xs + intercept, color=colours['empirical'], linestyle='-', #label='Realistic fit')

    plt.xlabel("Starting Aspect Ratio")
    plt.ylabel("End Pore Area / Start Pore Area")
    plt.xlim(0,2.5)
    # Example: Restrict legend to the first two elements
    handles, labels = plt.gca().get_legend_handles_labels()  # Get all handles and labels
    plt.legend(handles[:2], labels[:2], loc = 4)  # Use only the first three elements
    plt.xlim(0.9, 2.0)
    plt.ylim(0.4, 1.8)
    plt.savefig("../Figures/Fig3/idealised_confocal_AR_above_one_bf_no_oval.png", dpi=300)
    plt.show()



def run_plot_fig3(idealised_df=None, colours=None, oval_path=None, circular_path=None, confocal_df=None, df_combined=None, selected_meshes=None):

    idealised_oval_mesh = trimesh.load_mesh(oval_path)
    idealised_circular_mesh = trimesh.load_mesh(circular_path)

    confocal_df_sample = confocal_df[(confocal_df["Mesh ID"] == "1_2") & (confocal_df["Pressure"]==0.0)].copy()
    ## Get the cross section list (each entry is (N,3)) for the oval idealised mesh, left guard cell
    idealised_oval_sections_left = extract_midpoint_cross_section(idealised_oval_mesh, guard_cell="left")
    idealised_circular_sections_left = extract_midpoint_cross_section(idealised_circular_mesh, guard_cell="left")

    # Preprocess the strings to ensure proper formatting for confocal (kept for later usage)
    confocal_df_sample["Midsection Points Left"] = confocal_df_sample["Midsection Points Left"].apply(
        lambda x: re.sub(r',\s*\]', ']', re.sub(r'\[\s*,', '[', re.sub(r'\s+', ',', x)))
    )
    confocal_df_sample["Midsection Points Left"] = confocal_df_sample["Midsection Points Left"].apply(lambda x: np.array(ast.literal_eval(x)))

    ## Plot the cross section(s). We pass lists for both inputs per helper's contract
    csh.plot_cross_sections_grid_overlay(list(confocal_df_sample["Midsection Points Left"]), idealised_oval_sections_left, n_cols=1, figsize=(7, 7), colors=(colours["empirical"], colours["idealised_oval"]), filename="../Figures/Fig3/cross_section_oval.png", linewidth = 5, ylim=10, mesh1 = "Experimental", mesh2 = "Idealised Oval")
    csh.plot_cross_sections_grid_overlay(list(confocal_df_sample["Midsection Points Left"]), idealised_circular_sections_left, n_cols=1, figsize=(7, 7), colors=(colours["empirical"], colours["idealised_circular"]), filename="../Figures/Fig3/cross_section_circular.png", linewidth=5, ylim=10,mesh1 = "Experimental", mesh2 = "Idealised Circular")   

    confocal_df_sample_inflated = confocal_df[(confocal_df["Mesh ID"] == "1_2") & (confocal_df["Pressure"]==2.0)].copy()

    confocal_df_sample_inflated["Midsection Points Left"] = confocal_df_sample_inflated["Midsection Points Left"].apply(
        lambda x: re.sub(r',\s*\]', ']', re.sub(r'\[\s*,', '[', re.sub(r'\s+', ',', x)))
    )
    confocal_df_sample_inflated["Midsection Points Left"] = confocal_df_sample_inflated["Midsection Points Left"].apply(lambda x: np.array(ast.literal_eval(x)))

    csh.plot_cross_sections_grid_overlay(list(confocal_df_sample["Midsection Points Left"]), list(confocal_df_sample_inflated["Midsection Points Left"]), n_cols=1, figsize=(7, 7), colors=(colours["empirical"], colours["tip"]), filename="../Figures/Fig3/inflated_deflated_1_2_cross_section.png", linewidth = 5, ylim=10, mesh1 = "Experimental 0.0 MPa", mesh2 = "Experimental 2.0 MPa")
    

    ## Create AR vs Area plots
    create_ar_area_plots(confocal_df = confocal_df, df_combined=df_combined, selected_meshes = selected_meshes, colours=colours)     
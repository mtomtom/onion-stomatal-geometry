import numpy as np
import shutil
import os, sys
from pathlib import Path
import pandas as pd
sys.path.append(os.path.dirname(__file__))
import helper_functions as hf
## Path to mesh files: update as necessary

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Meshes", "Idealised")) + os.sep
mesh_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "confocal_aniso_results_df_batch.csv"))

df = pd.read_csv(mesh_data_path)
df_pressure0 = df[df["Pressure"] == 0.0]

## The subfolder in which the resulting meshes will be saved
outpath_mesh = path + "anisotropy_pressure_results/"
## The subdolder in which the resulting pore area files will be saved
outpath_area = path + "anisotropy_pressure_pore/"

if not os.path.exists(outpath_mesh):
    os.makedirs(outpath_mesh)

if not os.path.exists(outpath_area):
    os.makedirs(outpath_area)


selected_meshes = ['1_2', '1_3', '1_4', '1_5', '1_6', '1_8', '2_1', '2_3', '2_6a', '2_7', '3_2', '3_4', '3_6', '3_7']
selected_meshes = ['1_6','1_8', '2_1', '2_3', '2_6a', '2_7', '3_2', '3_4', '3_6', '3_7']

selected_meshes = ["1_2"]

mesh_names = []

for mesh_id in selected_meshes:
    mesh_names.append(f"idealised_final_mdx_{mesh_id}_oval")
    mesh_names.append(f"idealised_final_mdx_{mesh_id}_circular")

pressure = np.arange(0.0, 2.1, 0.1)
pressure = [f"{p:.1f}" for p in pressure]

E13 = 100
E2 = 500
poisson = 0.3
stack = 2


for mesh_id in selected_meshes:
    ## Get the stomata major and minor length
    df_mesh = df_pressure0[df_pressure0["Mesh ID"] == mesh_id].copy()
    major_length = df_mesh["Measured major length"].values[0]
    minor_length = df_mesh["Measured minor length"].values[0]
    left_width = df_mesh["Major length left"].values[0]

    radius_y = (major_length/2) - (left_width/2)
    radius_x = (minor_length/2) - (left_width/2)

    print(radius_x, radius_y)

    #for shape in ["oval", "circular"]:
    for shape in ["oval"]:
        mesh_name = path + f"idealised_final_mdx_{mesh_id}_{shape}.obj"
        print(mesh_name)
    
        ## Load the mesh
        Process.Mesh__System__Import("Stack" + str(stack), mesh_name, 'OBJ Import','No','No')
        ## Set the correct stack
        Process.Stack__System__Set_Current_Stack("Stack" + str(stack),"")

        ## Set the reference configuration
        hf.ref_cfg(Process)

        ## Set the stress strain
        hf.set_stress_strain(Process)

        ## Set material properties
        hf.material_props(Process, E13, E2, poisson)

        ## Create the Polygon to set the anisotropy
        Process.Tools__Cell_Maker__Mesh_2D__Shapes__Polygon('','Polygon',str(radius_x),str(radius_y),'36','360')

        ## Set the Polygon as the current cell complex
        Process.Model__CCF__999_Set_Current_Cell_Complex('Polygon')

        ## Delete the face
        Process.Mesh__Selection__Action__Select_All('Faces')
        Process.Mesh__System__Delete_Selection('No','Faces')
        Process.Mesh__Selection__Action__Clear_Selection('All')

        ## Set the mesh as the current cell complex
        Process.Model__CCF__999_Set_Current_Cell_Complex('OBJ Import')

        ## Set anisotropy
        hf.set_aniso_from_lines(Process)


        # ## Delete previous temporary area file_path
        filename = "pore_area.txt"

        if os.path.exists(filename):
            os.remove(filename)
            print(f"{filename} deleted")
        else:
            print(f"{filename} does not exist")


        for p in pressure:

            print(p)

            ## Set pressure
            hf.set_pressure(Process, p)

            ## Run simulation
            hf.run_fem(Process)

            ## Get the pore area
            Process.Model__Plugins__Calculate_Pore_Area('0.1','0.1','Yes','No')

            outfile = outpath_mesh + mesh_id + "_" + p + ".obj"

            ## Export obj
            Process.Mesh__System__Export("Stack" + str(stack),outfile,'OBJ','No','Yes')

        ## Rename pore_area.txt and move it to the correct folder
        shutil.move("pore_area.txt",outpath_area +"pore_area_" + mesh_id + ".txt")

        ## Delete the stack
        Process.Stack__System__Delete_Stack("Stack" + str(stack))
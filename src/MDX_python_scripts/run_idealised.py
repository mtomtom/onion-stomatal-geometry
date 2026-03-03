import numpy as np
import shutil
import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
import helper_functions as hf

## Code to create all of the obj files for the pressure vs pore area plots

## Path to mesh files: update as necessary

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Meshes", "Idealised")) + os.sep

## The subfolder in which the resulting meshes will be saved
outpath_mesh = path + "pressure_results/"
## The subdolder in which the resulting pore area files will be saved
outpath_area = path + "pressure_pore/"

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


E13 = "100.0"
E2 = "100.0"
poisson = "0.3"
stack = 2


for mesh in mesh_names:
    mesh_name = path + mesh + ".obj"
    ## Load the mesh
    Process.Mesh__System__Import("Stack" + str(stack), mesh_name, 'Imported','No','No')
    ## Set the correct stack
    Process.Stack__System__Set_Current_Stack("Stack" + str(stack),"")

    ## Set the reference configuration
    hf.ref_cfg(Process)

    ## Set the stress strain
    hf.set_stress_strain(Process)

    ## Set material properties
    hf.material_props(Process, E13, E2, poisson)

    ## Delete previous temporary area file_path
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

        outfile = outpath_mesh + mesh + "_" + p + ".obj"

        ## Export obj
        Process.Mesh__System__Export("Stack" + str(stack),outfile,'OBJ','No','Yes')

    ## Rename pore_area.txt and move it to the correct folder
    shutil.move("pore_area.txt",outpath_area +"pore_area_" + mesh + ".txt")

    ## Delete the stack
    Process.Stack__System__Delete_Stack("Stack" + str(stack))







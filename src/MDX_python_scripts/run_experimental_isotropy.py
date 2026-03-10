import numpy as np
import shutil
import os, sys
sys.path.append(os.path.dirname(__file__))
import helper_functions as hf

## Run the simulations for the experimental meshes

## Set the path to the folder containing the meshes (absolute, based on this script).
# Use an absolute path so the script works when invoked from any CWD.
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Meshes", "Onion meshes")) + os.sep

## The subfolder in which the resulting meshes will be saved
outpath_mesh = path + "pressure_results/"
## The subdolder in which the resulting pore area files will be saved
outpath_area = path + "pressure_pore/"

if not os.path.exists(outpath_mesh):
    os.makedirs(outpath_mesh)

if not os.path.exists(outpath_area):
    os.makedirs(outpath_area)

mesh_names = ["Ac_DA_1_2", "Ac_DA_1_3", "Ac_DA_1_4", "Ac_DA_1_5", "Ac_DA_1_6", "Ac_DA_1_8", "Ac_DA_2_1", "Ac_DA_2_3", "Ac_DA_2_6a", "Ac_DA_2_7", "Ac_DA_3_2", "Ac_DA_3_4", "Ac_DA_3_6", "Ac_DA_3_7"]

pressure = np.arange(0.0, 2.1, 0.1)
pressure = [f"{p:.1f}" for p in pressure]

E13 = 100
E2 = 100
poisson = 0.3

for mesh in mesh_names:
    mesh_name = path + mesh + "/" + mesh + ".mdxm"
    ## Load the mesh
    hf.load_initial_mesh(Process, mesh_name)

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
        hf.set_pressure(Process,p)

        ## Run simulation
        hf.run_fem(Process)

        ## Get the pore area
        Process.Model__Plugins__Calculate_Pore_Area('0.1','0.1','Yes','No')

        outfile_mesh = outpath_mesh + mesh + "_" + p + ".obj"

        ## Export obj
        Process.Mesh__System__Export('',outfile_mesh,'OBJ','No')

    ## Rename pore_area.txt and move it to the correct folder
    shutil.move("pore_area.txt", outpath_area + "pore_area_" + mesh + ".txt")






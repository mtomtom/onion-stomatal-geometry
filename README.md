# onion-stomatal-geometry

*Guard cell cross-sectional shape changes promote stomatal opening in onion*

Melissa Tomkins, Matthew J. Wilson, Jodie V. Armand, Nathanael Y. H. Tan, Briony Parker, Julie E. Gray, Andrew J. Fleming, Richard J. Morris and Richard S. Smith

__jupyter notebooks__:

- air_mattress_pipeline: all of the code for creating the meshes for the simulations, and analysing the results
- main_figures: all of the code for generating the main figures
- SI_figures: all of the code for generating the Supporting Information figures


Most of the functions for obtaining stomatal guard cell mesh measurements can be found in the accompanying streamlit app: [streamlit app](https://stomata-mesh-viewer.streamlit.app/)

Running the simulations:
From vlab: Open the "OnionGeometry" OOF. Right click on the icon. Type "make run". MDX will start up with an experimental mesh (Ac_DA_1_2) loaded and ready to run a simulation with isotropic material. Click run (double green arrows in top right hand corner of application).

To run all of the simulations: use the three python scripts (src/MDX_python_scripts/run_idealised.py, src/MDX_python_scripts/run_experimental_isotropy.py, src/MDX_python_scripts/run_experimental_anisotropy.py). Open MDX, navigate to the python scripts, and add the absolute path of the script you want to run to Tools/Python/Python. Make sure the directories are in the same structure, so that the scripts can find the correct meshes.


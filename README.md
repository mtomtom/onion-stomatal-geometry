# onion-stomatal-geometry

*Guard cell cross-sectional shape changes promote stomatal opening in onion*

Melissa Tomkins, Matthew J. Wilson, Jodie V. Armand, Nathanael Y. H. Tan, Briony Parker, Julie E. Gray, Andrew J. Fleming, Richard J. Morris and Richard S. Smith

## 🔬 Project Overview: Simulation & Analysis Pipeline

This repository contains the full computational pipeline for generating meshes, executing  simulations, and analyzing the resulting data.

### 📓 Jupyter Notebooks
* [`analysis_pipeline.ipynb`](./analysis_pipeline.ipynb): Core logic for mesh generation and simulation result analysis.
* [`main_figures.ipynb`](./main_figures.ipynb): Visualization scripts for all figures presented in the main manuscript.
* [`SI_figures.ipynb`](./SI_figures.ipynb): Visualization scripts for Supporting Information (SI) figures.

> **Note:** For interactive stomatal guard cell mesh measurements, please refer to the accompanying [Streamlit Web App](https://your-streamlit-link-here.com).

---

### 🚀 Running Simulations

#### Single Simulation (Manual via vLab)
1.  **Open OOF:** Launch the **"OnionGeometry"** OOF within the vLab environment.
2.  **Initialize:** Right-click the icon and execute `make run`. 
3.  **Setup:** MDX will launch with an experimental mesh (`Ac_DA_1_2`) pre-loaded with isotropic material properties.
4.  **Execute:** Click the **Run** icon (double green arrows in the top-right corner).

#### Batch Simulations (Automated)
To run large-scale simulations, use the specialized Python drivers. Ensure your directory structure remains intact so the scripts can resolve mesh paths correctly.

1.  Open **MDX**.
2.  Navigate to **Tools > Python > Python Script**.
3.  Provide the **absolute path** to one of the following scripts:
    * [`run_idealised.py`](./src/MDX_python_scripts/run_idealised.py)
    * [`run_experimental_isotropy.py`](./src/MDX_python_scripts/run_experimental_isotropy.py)
    * [`run_experimental_anisotropy.py`](./src/MDX_python_scripts/run_experimental_anisotropy.py)


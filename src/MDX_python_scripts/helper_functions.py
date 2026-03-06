import os

def make_dir(directory):

    file_path = directory
    command = 'mkdir ' + file_path
    os.system(command)

def load_initial_mesh(Process, mesh_name):

    Process.Mesh__System__Open('', mesh_name, 'No', 'No')
    Process.Mesh__Selection__Action__Clear_Selection('All')

def ref_cfg(Process):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__03_Reference_Configuration('', '', 'Linear Triangle', 'Triangle Element', 'False', 1.0)

def set_stress_strain(Process):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__04_StressStrain('Stress', 'Strain', '', '', '', '', '', '', '', '', 'Von Mises Stress', 'Stress', 'Strain', '1.0', 'No', '', '1', 'Linear Triangle', 'Triangle Element', 'StVK TransIso','TransIso Material')


def material_props(Process, E13, E2, poisson):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__05_Set_Material_Properties('', '', E13, E2, poisson, '1', 'Faces', 'TransIso Material')


def set_pressure(Process, press):
    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__07_Set_Pressure('', '', press, '1', 'Fem Pressure' , 'Triangle Element')
    Process.Mesh__Selection__Action__Clear_Selection('All')

def run_fem(Process):
    Process.Model__CCF__01_FEM_Membranes('', '', '10', '.1', '.0001', '1', '1.1', '0.7', '0.1', '10', '0.1', 'Backward Euler', 'Preconditioned Conjugate Gradient', '50', '1e-10', 'Yes', '10e-5', 'Model/CCF/04 StressStrain', '', 'Model/CCF/02 Triangle Derivs', 'Model/CCF/08 Pressure Derivs', 'Model/CCF/10 Dirichlet Derivs')

def set_aniso_from_lines(Process):
    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__32_Set_Ansio_Dir_From_Lines('Linear Triangle','Triangle Element','E2','Orthogonal','1e-6','Polygon')
    Process.Mesh__Selection__Action__Clear_Selection('All')

def create_3d_mesh(Process):
    Process.Mesh__Segmentation__Label_Connected_Cells('', '', 'Faces')
    Process.Mesh__Misc__Convert_to_CCF('', '', 'No', '0.0')
    Process.Tools__Cell_Maker__Mesh_3D__From_2D_Mesh__3D_Volumes_From_2D_Labels('', '')
    Process.Tools__Cell_Maker__Mesh_3D__Tools__Combine_3D_Cells('', '', 'Final', '1e-6')
    Process.Mesh__Selection__Topology__Select_Neighbors_Size('10', 'Vertices')
    Process.Mesh__Selection__Faces__Select_Faces_of_Vertices()
    Process.Mesh__Structure__Merge_Faces()
    Process.Mesh__Structure__Triangulate_Faces('1', 'No', 'Yes', '3', '0')
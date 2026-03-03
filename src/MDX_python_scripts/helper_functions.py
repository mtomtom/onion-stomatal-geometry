import os

def make_dir(directory):

    file_path = directory
    command = 'mkdir ' + file_path
    os.system(command)

def load_initial_mesh(Process, mesh_name):

    Process.Mesh__System__Open('', mesh_name, 'No', 'No')
    Process.Mesh__Selection__Action__Clear_Selection('All')

def ref_cfg(Process):

#    Process.Mesh__Selection__Action__Load_Selection('rods.txt')
#    Process.Model__CCF__03_Reference_Configuration('Linear Triangle', 'Triangle Element', 'False', rod)
#    Process.Mesh__Selection__Action__Invert_Selection('Faces')
#    Process.Model__CCF__03_Reference_Configuration('Linear Triangle', 'Triangle Element', 'False', wall)
    
    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__03_Reference_Configuration('', '', 'Linear Triangle', 'Triangle Element', 'False', 1.0)

def set_stress_strain(Process):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__04_StressStrain('Stress', 'Strain', '', '', '', '', '', '', '', '', 'Von Mises Stress', 'Stress', 'Strain', '1.0', 'No', '', '1', 'Linear Triangle', 'Triangle Element', 'TransIso Material')

def set_aniso_dir(Process):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__06_Set_Ansio_Dir('', '', 'Linear Triangle', 'Triangle Element', 'E2', '0.0 1.0 0.0', 'Parallel', '1e-6')
    Process.Model__CCF__21_Visualize_Directions('Material Direction', 'Growth Direction', '1.0', '', 'Linear Triangle', 'Triangle Element', '1.e-6')
#    Process.Model__CCF__32_Set_Ansio_Dir_From_Lines('Linear Triangle', 'Triangle Element', 'E2', 'Parallel', '1e-6', cc_dir)

def material_props(Process, E13, E2, poisson):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__05_Set_Material_Properties('', '', E13, E2, poisson, '1', 'Faces', 'TransIso Material')


def initial_pressure(Process):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Load_Selection('SCs.txt', 'No', 'No')
    Process.Model__CCF__90_Set_3D_Cell_Pressure('', '', const.p_sc_i, 'Signal Name')
    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Load_Selection('GCs.txt', 'No', 'No')
    Process.Model__CCF__90_Set_3D_Cell_Pressure('', '', const.p_gc_i, 'Signal Name')
    Process.Model__CCF__91_Set_Face_Pressure_From_Volumes('', '', 'Fem Pressure', 'Triangle Element', 'Signal Name')
    Process.Model__CCF__01_FEM_Membranes('', '', '10', '.1', '.0001', '10', '1.1', '0.7', '0.1', '10', '0.1', 'Backward Euler', 'Preconditioned Conjugate Gradient', '50', '1e-10', 'Yes', '10e-5', 'Model/CCF/04 StressStrain', 'Model/CCF/02 Triangle Derivs', 'Model/CCF/08 Pressure Derivs', 'Model/CCF/10 Dirichlet Derivs')

def set_boundary_conds(Process):

    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Load_Selection('boundary-SC.txt', 'No', 'No')
    Process.Model__CCF__09_Set_Dirichlet('', '', '1 1 1', 'Fem Dirichlet')

#    Process.Mesh__Selection__Action__Clear_Selection('All')
#    Process.Mesh__Selection__Action__Load_Selection('fix_x.txt')
#    Process.Model__CCF__09_Set_Dirichlet('1 0 0', 'Fem Dirichlet')
#
#
#    Process.Mesh__Selection__Action__Clear_Selection('All')
#    Process.Mesh__Selection__Action__Load_Selection('fix_y.txt')
#    Process.Model__CCF__09_Set_Dirichlet('0 1 0', 'Fem Dirichlet')
#
#    Process.Mesh__Selection__Action__Clear_Selection('All')
#    Process.Mesh__Selection__Action__Load_Selection('fix_z.txt')
#    Process.Model__CCF__09_Set_Dirichlet('0 0 1', 'Fem Dirichlet')

def set_pressure(Process, press):
    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__07_Set_Pressure('', '', press, '1', 'Fem Pressure' , 'Triangle Element')
    Process.Mesh__Selection__Action__Clear_Selection('All')

def run_fem(Process):
    Process.Model__CCF__01_FEM_Membranes('', '', '10', '.1', '.0001', '10', '1.1', '0.7', '0.1', '10', '0.1', 'Backward Euler', 'Preconditioned Conjugate Gradient', '50', '1e-10', 'Yes', '10e-5', 'Model/CCF/04 StressStrain', '', 'Model/CCF/02 Triangle Derivs', 'Model/CCF/08 Pressure Derivs', 'Model/CCF/10 Dirichlet Derivs')

def write_csv_save_mesh(Process,a, press_gc, press_sc, file_path):
     
#    Process.Mesh__Structure__Center_Mesh()
    Process.Mesh__Heat_Map__Measures_3D__Geometry__Cell_Wall_Area('','','','Cell Wall Area')
    Process.Mesh__Heat_Map__Measures_3D__Geometry__Volume('','','','Volume')
    if (a == 0): 
        Process.Model__CCF__103_Geometry_to_CSV('100',str(press_gc),str(press_sc),'data.csv','True')
    else:
        Process.Model__CCF__103_Geometry_to_CSV('100',str(press_gc),str(press_sc),'data.csv','False')
    Process.Mesh__System__Save('',file_path + '/GC_' + str(int(press_gc*10)) + '.mdxm', 'no')

def move_csv(directory):
    file_path = directory
    command = 'mv data.csv ' + file_path
    os.system(command)

def set_aniso_from_lines(Process):
    Process.Mesh__Selection__Action__Clear_Selection('All')
    Process.Mesh__Selection__Action__Select_All('Faces')
    Process.Model__CCF__32_Set_Ansio_Dir_From_Lines('Linear Triangle','Triangle Element','E2','Orthogonal','1e-6','Polygon')
    Process.Mesh__Selection__Action__Clear_Selection('All')
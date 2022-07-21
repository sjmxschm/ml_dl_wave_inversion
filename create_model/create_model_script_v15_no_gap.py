# import fundamental abaqus files:
from abaqus import *
from abaqusConstants import *

# import abaqus modules:
import sketch
import material
import part
import section
import assembly
import interaction
import mesh
import step
import load
import job
import visualization

import sys
import numpy as np
import json
import time
from datetime import date, datetime

session.journalOptions.setValues(replayGeometry=COORDINATE,
                                 recoverGeometry=COORDINATE)

'''
Script contains function to create Abaqus/CAE model based on the Koreck paper research and applies it to Framatome
specimen

Functions contained in this file:
    -   create_abaqus_model()

This script is called by 'python run_simulation.py'.

If this function should be called directly:
Cd to the folder that contains the script and type into the command line
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
abaqus cae noGUI=create_model_script_v15.py
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
instead because we need to get access to Abaqus CAE or run file from Abaqus GUI: File -> Run Script

Remarks:
    - capitalized commands create the specified entity and lowercase letter commands select the previously created
        entity - e.g. 'assembly.Set()' creates set and 'assembly.sets['...']' selects the specified set

All units used are SI-units, e.g. m, kg, N,...

created by Max Schmitz - mschmitz7@gatech.edu 
created on 04/05/2021

change log:
v10:
    - simulation information file will be created for each model: sim_info.json
    - job name now contains date and time of when it was started so .odb can be related to information file
    - added Tie constraint between coating and base plate
v11:
    - reduce triangle amplitude time t_max
    - reduce step time of excitation step t_period
    - add modification to FieldOutput Request to manually change sampling frequency
    - Rayleight speed in Chrome is automatically calculated from material properties
v12:
    - Layer thickness is changed between 100 and 600 microns in subchanges
v13:
    - Base plate thickness is now constant 3mm
    - Spatial length of model reduced from 18cm to 9cm
    - adjusted and reduced simulation time by a lot to reduce influence of reflections
v14:
    - move whole script into function to call this script with varying parameters
    - obtain parameters from command line arguments (run_simulation.py)
    - parse_input_variables function added to conveniently convert the user input to variables
    - added if __name__ == 'main' function to run run_abaqus_model() if this file is called
v15:
    - added that coating does not need to be uniform anymore

'''


def run_abaqus_model(
        coating_height=600E-6,
        plate_width=0.08,
        base_plate_height=0.003,
        coating_density=7190.0,
        coating_youngs_mod=279E9,
        coating_pois_rat=0.21,
        base_plate_density=6560.0,
        base_plate_youngs_mod=99.3E9,
        base_plate_pois_rat=0.37,
        cb_width=0.005,
        cg_width=0.02,
        cs_width=0.055,
        cg_top_left=0.001,
        cg_top_right=0.001,
        cg_bevel=0.001,
        cg_gap_depth=0.00005,
        ex_amp=2e-06,
        num_mesh=1,
        t_max=2E-8,
        t_s=2E-8,
        t_p=9.2E-05,
        run_simulation=False,
        create_inp=True,
        n_cores=None
):
    """
    create FEA Abaqus/CAE model of a two layered plate (base plate + coating) and excite one end with a triangular
     pulse. Run simulation of this automatically and store results as well as project meta information.
     Some basic variables are transferred, some are defined locally in this function and can be changed into an
     argument if necessary later.

    The default material properties are used for Chrome in the coat.

    run_abaqus_model(...) creates Abaqus/CAE output files and an ..._info.json file containing meta information
     of the model/sim. Within the Abaqus/CAE output files, the .odb-file is the most interesting one which is needed for
     further processing later one. The displacement information will be extracted from this file.

    args:
     geometric properties:
        - coating_height - m
        - plate_width - m - for physical meaningful results needs to be >= 7cm, at least >= 0.3mm for debugging
        - base_plate_height - m
     material properties:
        - coating_density - kg/m^3
        - coating_youngs_mod - Pa
        - coating_pois_rat -
        - base_plate_density - kg/m^3
        - base_plate_youngs_mod - Pa
        - base_plate_pois_rat -
        - cb_width - m - 0.005 (=0.5cm) - width of coating boundary part
        - cg_width - m - 0.02 (=2cm) - width of coating gap part
        - cs_width - m - 0.055 (5.5cm) - width of coating sampling part
        - cg_top_left - m - 0.001 - length until start of gap of gap part on left side
        - cg_top_right = 0.001 - length from gap to end of coating gap part on right side
        - cg_bevel = 0.001 - length for drop from coating_height to coating_height-cg_gap_depth
        - cg_gap_depth = 0.00005 - depth of gap in coating
     utils:
        - ex_amp - metres - amplitude of excitation (maximum displacement during excitation) - excitation downwards
            means negative excitation later on. Put in positive value for Excitation
        - num_mesh - factor of which the mesh size needs to be smaller than the smallest wave length.
        - t_max - seconds - time until triangular impulse has reached maximum
        - t_s - seconds - sampling time for history requests
        - t_p - seconds - period time of simulation (duration of the simulation/excitation step)
        - run_simulation - boolean - Specify if simulation should be run or not
        - create_inp - boolean - define if Input file should be created if simulation should not be run

    """

    # --- collect all material properties here
    # Material properties:
    c_density = coating_density  # kg/m^3 - density
    c_youngs_mod = coating_youngs_mod  # Pa - Youngs Modulus
    c_pois_rat = coating_pois_rat  # Poissons Ratio

    zy4_density = base_plate_density  # kg/m^3 - density
    zy4_youngs_mod = base_plate_youngs_mod  # Pa - Youngs Modulus
    zy4_pois_rat = base_plate_pois_rat  # Poissons Ratio

    # geometric properties of specimen
    c_height = coating_height  # m - coating height
    p_width = plate_width  # m - plate's widht
    b_height = base_plate_height  # m - base plate's height

    # __________________________________________________________________________________________________________________

    print('_____________ new model is going to be created _____________')

    cModel = mdb.Model(name='Cr-Zy_Model')

    # --- create materials:
    cr_material = cModel.Material(name='Chrome')
    cr_material.Density(table=((c_density,),))  # density=7190 kg/m^3
    cr_material.Elastic(table=((c_youngs_mod, c_pois_rat),))  # youngsModulus=279E9 Pa, Poissons ratio=0.21

    zy4_material = cModel.Material(name='Zirconium-4')
    zy4_material.Density(table=((zy4_density,),))  # density=6560 kg/m^3
    zy4_material.Elastic(table=((zy4_youngs_mod, zy4_pois_rat),))  # youngsModulus=993E7 Pa, Poissons ratio=0.37

    # --- create sketch:
    cb_sketch = cModel.ConstrainedSketch(name='cb_Sketch', sheetSize=1)
    cg_sketch = cModel.ConstrainedSketch(name='cg_Sketch', sheetSize=1)
    cs_sketch = cModel.ConstrainedSketch(name='cs_Sketch', sheetSize=1)
    b_sketch = cModel.ConstrainedSketch(name='b_Sketch', sheetSize=1)

    '''
    Pre-indices are meaning the following. Assume that the three part approach is used to easily define the right
    nodes for excitation, boundary conditions and sampling. The idea is to specify the nodes in the easier geometries,
    then merge the nodes. The new merged part still has the link to these nodes and the mapping can just be copied
        - cb -> coating boundary part 
        - cg -> coating gap part
        - cs -> coating sampling part
    Example:
        __________      ________________
        |    |    \\__//   |           |
        | cb |    cg       |  cs       |
    '''

    # cb_width = 0.005  # 0.5cm
    # cg_width = 0.02  # 2cm
    # cs_width = 0.055  # 5.

    # create boundary (cb) part:
    cb_xyCoords = ((0, 0), (0, c_height), (cb_width, c_height), (cb_width, 0), (0, 0))
    # create profile from vertices using the X- and Y-coordinates provided above
    for i in range(len(cb_xyCoords) - 1):
        cb_sketch.Line(point1=cb_xyCoords[i],
                       point2=cb_xyCoords[i + 1])

    # create gap (cg) part:
    # cg_top_left = 0.001
    # cg_top_right = 0.001
    # cg_bevel = 0.001
    # cg_gap_depth = 0.005  # 0.00005

    assert cg_top_left + 2 * cg_bevel + cg_top_right <= cg_width, 'Gap is too long, check cg_bevel,...!!!'
    assert c_height >= cg_gap_depth, 'Coating height needs to be bigger than gap_depth!!!'
    assert cg_top_left + 2 * cg_bevel + cg_top_right <= p_width, 'c_middle needs to be bigger than 0!!!'

    ############################################################
    # # comment this out after no gap simulation was tested:
    # cg_top_left = 0.00001
    # # rectangle
    # cg_xyCoords = (
    #     (cb_width, 0),
    #     (cb_width, c_height),
    #     (cb_width + cg_top_left, c_height),
    #     (cb_width + cg_top_left, 0),
    #     (cb_width, 0)
    # )
    #
    # # triangle
    cg_xyCoords = (
        (cb_width, 0),
        (cb_width, c_height),
        (cb_width + cg_top_left, 0),
        (cb_width, 0)
    )
    # and there is a change in the sampling part!
    ############################################################

    # cg_xyCoords = (
    #     (cb_width, 0),
    #     (cb_width, c_height),
    #     (cb_width + cg_top_left, c_height),
    #     (cb_width + cg_top_left + cg_bevel, c_height - cg_gap_depth),
    #     (cb_width + cg_width - cg_top_right - cg_bevel, c_height - cg_gap_depth),
    #     (cb_width + cg_width - cg_top_right, c_height),
    #     (cb_width + cg_width, c_height),
    #     (cb_width + cg_width, 0),
    #     (cb_width, 0)
    # )

    for i in range(len(cg_xyCoords) - 1):
        cg_sketch.Line(point1=cg_xyCoords[i],
                       point2=cg_xyCoords[i + 1])

    # create sampling (cs) part:
    # cs_xyCoords = (
    #     (cb_width + cg_width, 0),
    #     (cb_width + cg_width, c_height),
    #     (cb_width + cg_width + cs_width, c_height),
    #     (cb_width + cg_width + cs_width, 0),
    #     (0, 0)
    # )
    cs_xyCoords = (
        (0, 0),
        (cg_top_left, c_height),
        # (0, c_height),
        (cs_width, c_height),
        (cs_width, 0),
        (0, 0)
    )

    for i in range(len(cs_xyCoords) - 1):
        cs_sketch.Line(point1=cs_xyCoords[i],
                       point2=cs_xyCoords[i + 1])

    # create base plate part
    b_xyCoords = (
        (0, 0),
        (0, -b_height),
        (p_width, -b_height),
        (p_width, 0),
        (0, 0)
    )

    for i in range(len(b_xyCoords) - 1):
        b_sketch.Line(point1=b_xyCoords[i],
                      point2=b_xyCoords[i + 1])

    # --- create the parts in the model:
    cb_Part = cModel.Part(name='Coating_Boundary_Part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    cg_Part = cModel.Part(name='Coating_Gap_Part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    cs_Part = cModel.Part(name='Coating_Sampling_Part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)

    b_Part = cModel.Part(name='Base_Plate_Part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)

    # link part and sketch:
    cModel.parts['Coating_Boundary_Part'].BaseShell(sketch=cb_sketch)
    cModel.parts['Coating_Gap_Part'].BaseShell(sketch=cg_sketch)
    # cModel.parts['Coating_Sampling_Part'].BaseShell(sketch=cs_sketch)
    cs_Part.BaseShell(sketch=cs_sketch)

    cModel.parts['Base_Plate_Part'].BaseShell(sketch=b_sketch)

    # --- create a section for parts
    cb_Section = cModel.HomogeneousSolidSection(name='Coating_Boundary_Section', material='Chrome', thickness=None)
    cg_Section = cModel.HomogeneousSolidSection(name='Coating_Gap_Section', material='Chrome', thickness=None)
    cs_Section = cModel.HomogeneousSolidSection(name='Coating_Sampling_Section', material='Chrome', thickness=None)

    b_Section = cModel.HomogeneousSolidSection(name='BasePlate_Section', material='Zirconium-4', thickness=None)

    # --- create sets from parts for assignment later
    cb_Part.Set(faces=cb_Part.faces.getSequenceFromMask(('[#1 ]',), ), name='Coating_Boundary_Set')
    cg_Part.Set(faces=cg_Part.faces.getSequenceFromMask(('[#1 ]',), ), name='Coating_Gap_Set')
    cs_Part.Set(faces=cs_Part.faces.getSequenceFromMask(('[#1 ]',), ), name='Coating_Sampling_Set')

    b_Part.Set(faces=b_Part.faces.getSequenceFromMask(('[#1 ]',), ), name='Base_Plate_Set')

    # --- assign section to parts
    cb_Part.SectionAssignment(offset=0.0,
                              offsetField='',
                              offsetType=MIDDLE_SURFACE,
                              region=cb_Part.sets['Coating_Boundary_Set'],
                              sectionName='Coating_Boundary_Section',
                              thicknessAssignment=FROM_SECTION)
    cg_Part.SectionAssignment(offset=0.0,
                              offsetField='',
                              offsetType=MIDDLE_SURFACE,
                              region=cg_Part.sets['Coating_Gap_Set'],
                              sectionName='Coating_Gap_Section',
                              thicknessAssignment=FROM_SECTION)
    cs_Part.SectionAssignment(offset=0.0,
                              offsetField='',
                              offsetType=MIDDLE_SURFACE,
                              region=cs_Part.sets['Coating_Sampling_Set'],
                              sectionName='Coating_Sampling_Section',
                              thicknessAssignment=FROM_SECTION)

    b_Part.SectionAssignment(offset=0.0,
                             offsetField='',
                             offsetType=MIDDLE_SURFACE,
                             region=b_Part.sets['Base_Plate_Set'],
                             sectionName='BasePlate_Section',
                             thicknessAssignment=FROM_SECTION)

    # --- create an instance for both parts
    assembly = cModel.rootAssembly
    cb_Instance = assembly.Instance(name='Coating_Boundary_Instance', part=cb_Part, dependent=ON)
    cg_Instance = assembly.Instance(name='Coating_Gap_Instance', part=cg_Part, dependent=ON)
    cs_Instance = assembly.Instance(name='Coating_Sampling_Instance', part=cs_Part, dependent=ON)

    b_Instance = assembly.Instance(name='Base_Plate_Instance', part=b_Part, dependent=ON)

    # --- fix parts on the left side
    ''' this is just for the visualization - the tie constraints are set up below '''

    # assembly.EdgeToEdge(clearance=0.0,
    #                     fixedAxis=b_Instance.edges[1],
    #                     flip=ON,
    #                     movableAxis=c_Instance.edges[3])
    #
    # assembly.EdgeToEdge(clearance=0.0,
    #                     fixedAxis=b_Instance.edges[0],
    #                     flip=OFF,
    #                     movableAxis=c_Instance.edges[0])

    # --- create amplitude for triangle excitation impulse
    '''
    creates triangular amplitude for the excitation
            /\\
           /  \\
        0-t_max-t_end
    '''
    # t_max = 5e-8  # was initially 5e-8, check 5e-7 and 5e-6- 5e-7 works
    t_end = 2 * t_max
    cModel.TabularAmplitude(name='TRIANGLE_PULSE', data=((0, 0), (t_max, 1), (t_end, 0),), timeSpan=TOTAL)

    # --- create mesh
    ''' 
    - cR_chrome = [m/s] Rayleight Wave speed of Chrome (should be in magnitude of 3000 m/s)
    - f_max      = [Hz] maximal frequency from excitation (get it from FT of excitation).
                    for a triangular excitation it should be 4*pi/t_max
    - n_mesh    = [] factor of which the mesh size needs to be smaller than the smallest wave length.
                    should be around 2 to 20, but should be even number to make sure the right nodes are seleted later
                    n_mesh = 20 means there are 20 mesh elements in the smallest wavelength
                    
    Runtime examples: 
        - t_max=5e-7 and n_mesh=10 results in approx 4h of simulation time - extraction from odb took like 30 min
        - t_max=5e-6 and n_mesh=2 results in a couple of seconds simulation time
    
    '''
    f_max = 4 * np.pi / t_max
    # calculate the Rayleight wave speed:
    c_eta = (0.87 + 1.12 * c_pois_rat) / (1 + c_pois_rat)
    c_mu = c_youngs_mod / (2 + 2 * c_pois_rat)
    cR_chrome = c_eta * np.sqrt(c_mu / c_density)

    n_mesh = int(num_mesh)  # 2  #int(num_mesh) # was 1
    c_elem_size = cR_chrome / (n_mesh * f_max)   # = 0.001455276846282673
    # c_elem_size = round(cR_chrome / (n_mesh * f_max), 6)  # = 0.001455276846282673
    print('c_elem_size = ', c_elem_size)
    b_elem_size = c_elem_size  # set to same element size to reduce reflections at interface

    cb_Part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=c_elem_size)  # elem_size = 1/10 * wavelength
    cb_Part.generateMesh()
    cg_Part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=c_elem_size)  # elem_size = 1/10 * wavelength
    cg_Part.generateMesh()
    cs_Part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=c_elem_size)  # elem_size = 1/10 * wavelength
    cs_Part.generateMesh()

    b_Part.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=b_elem_size)  # elem_size = 1/10 * wavelength
    b_Part.generateMesh()

    assembly.regenerate()

    # --- create sets for boundary conditions
    # (assembly.Set() creates set and assembly.sets['...'] selects the specified set)

    # ---- obtain all nodes on COATING_BOUNDARY left edge and assign them to Coating_Boundary_Left_BC_Nodes
    cb_n_max_nodes_x = int(
        round(cb_width / c_elem_size))  # get number of mesh elements for coating in horizont direction
    cb_n_max_nodes_x += 1  # there are max_nodes_x + 1 nodes in each line
    cb_n_max_nodes_y = int(round(c_height / c_elem_size))  # get number of elements for coating in vertical direction

    print('cb_n_max_nodes_x = ', cb_n_max_nodes_x)

    cb_nodes_sym = eval('cb_Part.nodes[0:1]')  # (0,0) is left bottom corner
    for y in range(1, cb_n_max_nodes_y + 1):
        # print('y * c_max_nodes_x = ', y * cb_n_max_nodes_x)
        cb_nodes_sym += eval('cb_Part.nodes[' + str(y * cb_n_max_nodes_x) + ':' + str(y * cb_n_max_nodes_x + 1) + ']')

    cb_sym_nodes = cb_Part.Set(name='Coating_Boundary_Left_BC_Nodes', nodes=cb_nodes_sym)

    # ------ select loading node on top left corner of COATING_BOUNDARY
    c_loading_node = cb_Part.Set(
        name='Coating_Boundary_Loading_Node',
        nodes=cb_Part.nodes[
              cb_n_max_nodes_y * cb_n_max_nodes_x:cb_n_max_nodes_y * cb_n_max_nodes_x + 1
              ]
    )

    print('loading node ', cb_n_max_nodes_y * cb_n_max_nodes_x)

    # ------ obtain all nodes on BASE PLATE left edge and assign them to BasePlate_Left_Sym_Nodes
    b_n_max_nodes_x = int(
        round(p_width / b_elem_size)
    )  # get number of mesh elements for coating in horizontal direction
    b_n_max_nodes_x += 1  # there are is one node more than mesh elements in each row
    b_n_max_nodes_y = int(round(b_height / b_elem_size))  # get number of nodes for coating in vertical direction
    b_n_max_nodes_y += 1

    # node 0 is on top right corner and node number increases in -x direction (I don't know why but that's how it is)
    # b_nodes_sym = eval('b_Part.nodes['+str(b_max_nodes_x-1)+':'+str(b_max_nodes_x-1+1)+']') # do not fix first node
    b_nodes_sym = eval('b_Part.nodes[' + str(b_n_max_nodes_x + b_n_max_nodes_x - 1) + ':' + str(
        b_n_max_nodes_x + b_n_max_nodes_x - 1 + 1) + ']')

    for y in range(b_n_max_nodes_y):
        '''
        str(y * b_max_nodes_x + y-1):
         - b_max_nodes_x:  means that I access the node at the left end
         - y:              makes sure that I select the end node at each row
         - y-1:            adds y (-1 bc loop is starting at 2) to end node at each row
        '''
        # print('y * b_n_max_nodes_x + y-1 = ', y * b_n_max_nodes_x + b_n_max_nodes_x - 1)
        b_nodes_sym += eval('b_Part.nodes[' + str(y * b_n_max_nodes_x + b_n_max_nodes_x - 1) + ':' + str(
            y * b_n_max_nodes_x + 1 + b_n_max_nodes_x - 1) + ']')

    b_sym_nodes = b_Part.Set(name='Base_Plate_Left_BC_Nodes', nodes=b_nodes_sym)  # c_nodes_sym

    # --- select nodes for sampling
    # node 0 is on bottom left corner
    cs_n_max_nodes_x = int(
        round(cs_width / c_elem_size))  # get number of mesh elements for coating in horizontal direction
    cs_n_max_nodes_x += 1  # there are max_nodes_x + 1 nodes in each line
    cs_n_max_nodes_y = int(round(c_height / c_elem_size))  # get number of elements for coating in vertical direction

    # d_start = 0.03  # distance from excitation node to first sampling node - set 3cm or 2cm arbitrarily
    # d_end = 0.03  # distance from end of plate to last sampling node - set 3cm arbitrarily

    d_start = 0.005  # cb_width + cg_width + 0.0  # change arbitrarily and see if that works
    d_end = 0.01  # change arbitrarily and see if that is good
    if cs_width <= d_start + d_end + 0.0001:
        ''' take care of small plates (might not be meaningful physically,
        but is helpful for debugging -> short plate = short sim dur) '''
        print('DEBUG MODE: p_width is really small')
        d_start, d_end = 0, 0

    msmt_len = cs_width - d_start - d_end  # length of area with msmt nodes
    assert msmt_len > 0, 'length of area with msmt nodes must be positive!'

    node_start = cs_n_max_nodes_x * cs_n_max_nodes_y + int(d_start / c_elem_size)
    node_end = cs_n_max_nodes_x * (cs_n_max_nodes_y + 1) - int(d_end / c_elem_size)

    # node_start = int(d_start / c_elem_size)
    # node_end = cs_n_max_nodes_x - int(d_end / c_elem_size)

    c_sampling_nodes = cModel.parts['Coating_Sampling_Part'].Set(
        name='Sampling_Nodes',
        nodes=cModel.parts['Coating_Sampling_Part'].nodes[node_start:node_end]
    )

    # --- create surfaces for tie constraints
    '''
    surfaces need to be defined in part and not in assembly!!!
    any point on this surface can be used (but no corner, and point needs to be smaller than plate dimension!)
    '''

    # c_Part.Surface(name='Coating_Bottom_Surface',
    #                side1Edges=c_Part.edges.findAt(((0.0003, 0.0, 0.0),)))
    #
    # b_Part.Surface(name='Base_Plate_Top_Surface',
    #                side1Edges=b_Part.edges.findAt(
    #                    ((0.0003, 0.0, 0.0),)))  # # any point on this surface can be used (no corner)

    assembly.regenerate()

    # --- add tie constraints
    ''' 
    make sure the coating and the base plate are stuck together
    - want to use surface to surface since it provides better results even though it is not as
    efficients as node to surface 
    
    Update: do not use tie constraint - simulation results and simulation performance is way better when
    nodes are merged, so use merging at boundary instead!!
    '''

    # cModel.Tie(adjust=ON,
    #            master=c_Instance.surfaces['Coating_Bottom_Surface'],
    #            name='Coating_Base_Plate_TIE',
    #            positionToleranceMethod=COMPUTED,
    #            slave=b_Instance.surfaces['Base_Plate_Top_Surface'],
    #            thickness=ON,
    #            tieRotations=ON)

    # --- create position constraint for merging later
    print('cb_width + cg_width = ', sum(cb_width, cg_width))

    ''''
    Watch out: changing to the other edge to edge constraint fucks the numbering of the nodes,
    and suddenly the sampling nodes might not be on top of the coating part but on the interface 
    with the base plate!!!
    '''

    assembly.EdgeToEdge(
        clearance=0.0,
        fixedAxis=b_Instance.edges.findAt((p_width, -0.000001, 0.0), ),
        flip=ON,
        movableAxis=cs_Instance.edges.findAt((cs_width, 0.000001, 0.0), )
    )

    # assembly.EdgeToEdge(
    #     clearance=0.0,
    #     fixedAxis=b_Instance.edges.findAt((0.08, -0.00075, 0.0), ),
    #     flip=ON,
    #     movableAxis=cs_Instance.edges.findAt((0.055, 0.000225, 0.0), )
    # )

    # assembly.EdgeToEdge(
    #     clearance=0.0,
    #     fixedAxis=cg_Instance.edges.findAt((sum(cb_width, cg_width), 0.000001, 0.0), ),
    #     flip=OFF,
    #     movableAxis=cs_Instance.edges.findAt((0.0, 0.000001, 0.0), )
    # )

    # --- merge mesh from all parts

    mdb.meshEditOptions.setValues(enableUndo=True, maxUndoCacheElements=0.5)
    assembly._previewMergeMeshes(
        instances=(
            cb_Instance,
            cg_Instance,
            cs_Instance,
            b_Instance
        ),
        nodeMergingTolerance=1e-06
    )
    p_merge_Instance = assembly.InstanceFromBooleanMerge(
        domain=MESH,
        instances=(
            cb_Instance,
            cg_Instance,
            cs_Instance,
            b_Instance
        ),
        mergeNodes=BOUNDARY_ONLY,
        name='Plate_complete_MERGE',
        nodeMergingTolerance=1e-06,
        originalInstances=SUPPRESS
    )

    # --- create step (duration of simulation)
    # t_period = 9.2E-05  # s - was 6.5E-04s. Current time looks good from simulation
    # (wave passed + almost no reflection)
    # t_period = 4.5E-05 #4E-05 # 1e-5  # 0.14E-05 # s - with smaller plate there are too many reflections
    # -> use smaller simulation time
    t_period = t_p

    cModel.ExplicitDynamicsStep(
        name='excitation_explicit_analysis',
        previous='Initial',
        timePeriod=t_period,
        description='Apply the excitation to one end of the specimen'
    )

    # --- create boundary conditions
    cModel.DisplacementBC(
        amplitude=UNSET,
        createStepName='excitation_explicit_analysis',
        distributionType=UNIFORM,
        fieldName='',
        fixed=OFF,
        localCsys=None,
        name='FixCoatingLeftSide_BC',
        region=p_merge_Instance.sets['Coating_Boundary_Left_BC_Nodes'],
        u1=0.0,
        u2=UNSET,
        ur3=UNSET
    )
    cModel.DisplacementBC(
        amplitude=UNSET,
        createStepName='excitation_explicit_analysis',
        distributionType=UNIFORM,
        fieldName='',
        fixed=OFF,
        localCsys=None,
        name='FixBasePlateLeftSide_BC',
        region=p_merge_Instance.sets['Base_Plate_Left_BC_Nodes'],
        u1=0.0,
        u2=UNSET,
        ur3=UNSET
    )

    # --- create boundary condition that applies excitation

    cModel.DisplacementBC(
        amplitude='TRIANGLE_PULSE',
        createStepName='excitation_explicit_analysis',
        distributionType=UNIFORM,
        fieldName='',
        fixed=OFF,
        localCsys=None,
        name='LoadingNode_BC',
        region=p_merge_Instance.sets['Coating_Boundary_Loading_Node'],
        u1=UNSET,
        u2=-ex_amp,  # -2e-06,
        u3=UNSET
    )

    # --- specify sampling instances and sampling rate
    n_sampling = 'not used'  # at least 2 due to Nyquist criterion
    # t_sampling = 1/(n_sampling*f_max) # was 2E-7 before, should be around 1E-9
    # t_sampling = 2E-7  # provides ok-like results. 2E-7 is exactly 2*period of highest frequency!
    # t_sampling = 2E-8  # too small? weird behaviour
    # t_sampling = 1.1E-8 # too small too - similar to 2E-8
    # t_sampling = 1.1E-7
    t_sampling = t_s

    cModel.fieldOutputRequests['F-Output-1'].setValues(
        rebar=EXCLUDE,
        region=assembly.allInstances['Plate_complete_MERGE-1'].sets['Sampling_Nodes'],
        sectionPoints=DEFAULT,
        timeInterval=t_sampling,
        variables=('S', 'U')
    )

    # --- create job
    current_time = datetime.now().strftime("%H-%M-%S")
    job_name = str(date.today().strftime('%m-%d')) + '_' + str(current_time) + '_max_analysis_job'
    max_analysis_job = mdb.Job(name=job_name, model='Cr-Zy_Model', description='Runs basic triangle excitation')
    if n_cores is not None:
        max_analysis_job.setValues(
            activateLoadBalancing=False,
            numCpus=n_cores,
            numDomains=n_cores
        )
    else:
        max_analysis_job.setValues(
            activateLoadBalancing=False,
            numCpus=1,
            numDomains=1
        )

    # put following into new script if possible

    # get time job takes for simulation:
    start_time = time.time()
    print('simulation started at ', start_time)

    ####################################################################################################################
    if run_simulation:
        max_analysis_job.submit()
        max_analysis_job.waitForCompletion()
    elif create_inp:
        mdb.jobs[job_name].writeInput()
    ####################################################################################################################

    end_time = time.time()
    sim_dur = end_time - start_time  # time is in seconds
    print('Runtime of the simulation is ', sim_dur)

    # --- create simulation information file with all variables
    geometric_properties = {
        'cb_width': cb_width,
        'cg_width': cg_width,
        'cs_width': cs_width,
        'cg_top_left': cg_top_left,
        'cg_top_right': cg_top_right,
        'cg_bevel': cg_bevel,
        'cg_gap_depth': cg_gap_depth,
    }

    sim_info = {
        'job_name': job_name,
        'c_height': c_height,
        'b_height': b_height,
        'p_width': p_width,
        't_max': t_max,
        't_end': t_end,
        't_period': t_period,
        't_sampling': t_sampling,
        'n_sampling': n_sampling,
        'n_mesh': n_mesh,
        'f_max': f_max,
        'cR_chrome': cR_chrome,
        'c_elem_size': c_elem_size,
        'b_elem_size': b_elem_size,
        'd_start': d_start,
        'd_end': d_end,
        'msmt_len': msmt_len,
        'ex_amp': ex_amp,
        'sim_dur': sim_dur,
        'num_cores': n_cores,
        'geometric_properties': geometric_properties,
    }

    info_name = job_name + '_info.json'

    with open(info_name, 'w') as outfile:
        json.dump(sim_info, outfile, indent=6)


if __name__ == "__main__":
    ''' call main function if this file is called as standalone program '''
    # run_abaqus_model(coating_height=0.0006, num_mesh=2, run_simulation=False)
    # run_abaqus_model(plate_width=0.18, num_mesh=1, run_simulation=True)

    # debug the cluster and go from nonsense values to real values:
    # run_abaqus_model(
    #     plate_width=0.01,
    #     coating_height=0.0001,
    #     base_plate_height=0.0001,
    #     t_p=9.2E-07,
    #     t_max=2E-8,
    #     run_simulation=False
    # )

    run_abaqus_model(
        plate_width=0.08,
        coating_height=0.000300,
        base_plate_height=0.001,
        cg_width=0.02,
        cg_top_left=0.000001,  # 0.000|01 *1E3 for folder on cluster: 0.001 = 1mm (cluster), 0.01 = 10mm
        cg_top_right=0.001,
        cg_bevel=0.002,
        cg_gap_depth=0,  # 0.00001,
        run_simulation=False
    )

    # run_abaqus_model(
    #     coating_height=100E-4,
    #     run_simulation=True,
    # )

    # run_abaqus_model(
    #     coating_height=100E-6,
    #     run_simulation=True,
    #     n_cores=2
    # )

    # # debug the cluster with nonsense but stable values:
    # run_abaqus_model(
    #     plate_width=0.001,
    #     coating_height=0.00001,
    #     base_plate_height=0.00001,
    #     t_p=9.2E-07,
    #     t_max=2E-8,
    #     run_simulation=True
    # )

    # # try to use shorter t_max
    # run_abaqus_model(
    #     # plate_width=0.01,
    #     # coating_height=0.001,
    #     # t_max=5E-9, #0.000000005,
    #     plate_width=0.07,
    #     t_max=5E-8, # 3E-8,
    #     n_cores=2,  # None
    #     run_simulation=True
    # )

    # # Alu-Tape:
    # run_abaqus_model(
    #     coating_height=1E-03,
    #     plate_width=0.133,
    #     base_plate_height=0.25E-03,
    #     coating_density=2700.0,
    #     coating_youngs_mod=70.785E9,
    #     coating_pois_rat=0.3375,
    #     base_plate_density=1106.0,
    #     base_plate_youngs_mod=1E9,
    #     base_plate_pois_rat=0.35,
    #     ex_amp=2e-06,
    #     num_mesh=1,
    #     t_s=5E-8,
    #     t_p=9.2E-05,
    #     run_simulation=True,
    #     create_inp=True,
    # )

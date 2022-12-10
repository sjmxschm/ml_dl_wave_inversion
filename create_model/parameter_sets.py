"""
Contains the parameters sets that have been or should be simulated.
Old and used parameters sets are commented out. This file is imported in run_parallel_simulations.py

Additionally, before starting the simulation with new parameter sets, run this file by itself. This way,
a comparison with previously simulated parameter sets is conducted and already simulated parameter
combinations will be removed.

created by: Maximilian Schmitz
on: 12/05/2022

"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_param_sets():
    """
    Loads and returns param_sets. Make sure that every single param_set is in one
    string and that the string as the following format:
    ' -- param1=value_of_param1 -- param2=value_of_param2 -- ... -- '
    make sure there is an ' -- ' at the end of the file!

    returns:
        - param_sets (list): list with parameters to simulate. Different param_sets
            are separated elements of the list
    """

    # # new simulations starting at 12/06/2022
    # # first, address area between 100 and 200 microns coating thickness
    # # batch 1
    # param_sets = [
    #     # 110
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000110'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000110'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000110'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000110'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000110'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000110'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000110'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     # 120
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000120'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000120'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000120'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000120'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000120'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000120'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000120'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     # 130
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000130'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000080'
    #     ' -- ',
    #     # 140 # 22 simulations until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000080'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000140'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000090'
    #     ' -- ',
    #     # 150
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000150'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000090'
    #     ' -- ',
    #     # 160 # 41 until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     # 45 simulations in total
    # ]

    # param_sets = [
    #     # 160 continued
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000080'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000090'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000100'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000110'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000120'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000130'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000160'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     # 170 - 8 until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000080'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000090'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000100'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000110'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000120'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000130'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000170'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     # 180 - 23 until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000080'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000090'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000100'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000110'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000120'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000130'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000180'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     # 190 - 39 simulations
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000010'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000020'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000030'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000040'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000050'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000070'
    #     ' -- ',
    #     # 45 new simulations until here
    # ]

    # param_sets = [
    #     # 190 continued
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000090'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000100'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000110'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000190'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     # 200
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000200'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000060'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000200'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000080'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000200'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000120'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000200'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000200'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000200'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000170'
    #     ' -- ',
    #     # 210
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000210'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000130'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000210'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000210'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000210'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000210'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000170'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000210'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000180'
    #     ' -- ',
    #     # 220
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000220'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000130'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000220'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000220'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000220'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000220'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000220'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000170'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000220'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000180'
    #     ' -- ',
    #     # 230 - 23 simulations until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000230'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000130'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000230'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000230'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000230'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000230'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000230'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000170'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000230'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000180'
    #     ' -- ',
    #     # 240 - 30 simulations until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000240'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000130'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000240'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000240'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000240'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000240'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000170'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000240'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.002'
    #     ' -- cg_gap_depth=0.000180'
    #     ' -- ',
    #     # 250 - 36 simulations until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000250'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000250'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000170'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000250'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000180'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000250'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000190'
    #     ' -- ',
    #     # 260 - 40 simulations until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000260'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000140'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000260'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000150'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000260'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000260'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000170'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000260'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000180'
    #     ' -- ',
    #     # 270 - 45 simulations until here
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000270'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000160'
    #     ' -- ',
    #     ' -- plate_width=0.08'
    #     ' -- coating_height=0.000270'
    #     ' -- base_plate_height=0.001'
    #     ' -- t_sampling=0.00000002'
    #     ' -- cg_top_left=0.001'
    #     ' -- cg_top_right=0.001'
    #     ' -- cg_bevel=0.003'
    #     ' -- cg_gap_depth=0.000180'
    #     ' -- ',
    #     # 45 simulations until here
    # ]

    param_sets = [
        # 300
        ' -- plate_width=0.08'
        ' -- coating_height=0.000300'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000300'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000300'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000300'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000300'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 310
        ' -- plate_width=0.08'
        ' -- coating_height=0.000310'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000310'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000310'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000310'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000310'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 320
        ' -- plate_width=0.08'
        ' -- coating_height=0.000320'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000320'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000320'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000320'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000320'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 330
        ' -- plate_width=0.08'
        ' -- coating_height=0.000330'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000330'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000330'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000330'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000330'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 340
        ' -- plate_width=0.08'
        ' -- coating_height=0.000340'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000340'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000340'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000340'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000340'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 350
        ' -- plate_width=0.08'
        ' -- coating_height=0.000350'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000350'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000350'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000350'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000350'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 360
        ' -- plate_width=0.08'
        ' -- coating_height=0.000360'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000360'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000360'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000360'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000360'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 370
        ' -- plate_width=0.08'
        ' -- coating_height=0.000370'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000370'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000370'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000370'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000370'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 380
        ' -- plate_width=0.08'
        ' -- coating_height=0.000380'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000380'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000380'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000380'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000380'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
        # 390 - 45 until here
        ' -- plate_width=0.08'
        ' -- coating_height=0.000390'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000040'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000390'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000080'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000390'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.002'
        ' -- cg_gap_depth=0.000120'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000390'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000160'
        ' -- ',
        ' -- plate_width=0.08'
        ' -- coating_height=0.000390'
        ' -- base_plate_height=0.001'
        ' -- t_sampling=0.00000002'
        ' -- cg_top_left=0.001'
        ' -- cg_top_right=0.001'
        ' -- cg_bevel=0.003'
        ' -- cg_gap_depth=0.000200'
        ' -- ',
    ]

    return param_sets


def get_values_from_param_set(
        param_set: str,
        attribute: str = 'coating_height=',
        factor: float = 1E6,
        formatting: str = '.5f',
        rm_trailing_zeros: bool = True
) -> float:
    """
    Function finds attribute in param_set and returns corresponding value in format
    defined by formatting

    args:
        - param_set: string which contains all attributes with numerical values
            e.g. ' -- plate_width=0.08 -- coating_height=0.00010'
        - attribute: attribute to which the numerical value should be returned
            e.g. 'coating_height=' (IMPORTANT: include the '=' sign in string)
        - factor: factor to scale the decimal power to a nicer number
            e.g. 0.0004 * 1E4= 4
        - formatting: specify how the output float should be formatted

    """
    start_idx = str(param_set).find(attribute) + len(attribute)
    end_idx = str(param_set).find(' --', start_idx)
    value = float(str(param_set)[start_idx:end_idx]) * factor
    if rm_trailing_zeros:
        value = format(value, formatting).rstrip('0')
    # print(value)
    return value


def check_duplicate_params(p_sets: list, p_infos_fn: Path):
    """
    check_duplicate_params() is comparing the new parameter list for upcoming simulations and
    compares it to simulations already conducted. If a parameter combination has been simulated
    already before, it does not need to be simulated again.

    :arg:
        -   p_sets: a list of parameters
        -   p_infos_fn: Path object to .csv file which contains dataframe of existing simulations

    :return:
        -   p_sets_updated: the previous list of parameters but without the duplicates
    """
    p_sets_updated = []
    p_sets_duplicates = []

    duplicate_count = 0

    p_infos = pd.read_csv(p_infos_fn)

    for p_set in tqdm(p_sets):
        # 1. extract the numerical values needed from the parameter sets
        c_height = get_values_from_param_set(p_set, attribute='coating_height=',
                                             factor=1, rm_trailing_zeros=False)

        cg_top_left = get_values_from_param_set(p_set, attribute='cg_top_left=',
                                                factor=1, rm_trailing_zeros=False)

        cg_bevel = get_values_from_param_set(p_set, attribute='cg_bevel=',
                                             factor=1, rm_trailing_zeros=False)

        cg_top_right = get_values_from_param_set(p_set, attribute='cg_top_right=',
                                                 factor=1, rm_trailing_zeros=False)

        cg_gap_depth = get_values_from_param_set(p_set, attribute='cg_gap_depth=',
                                                 factor=1, rm_trailing_zeros=False)  # formatting was '.1f'

        print(f"{c_height},{cg_top_left},{cg_bevel},{cg_top_right},{cg_gap_depth}")

        # 2. compare the numerical values with the parameters of simulations already conducted
        if (
                (p_infos['c_height'] == c_height) &
                (p_infos['cg_top_left'] == cg_top_left) &
                (p_infos['cg_bevel'] == cg_bevel) &
                (p_infos['cg_top_right'] == cg_top_right) &
                (p_infos['cg_gap_depth'] == cg_gap_depth)
        ).any():
            duplicate_count += 1
            p_sets_duplicates.append(p_set)
        else:
            p_sets_updated.append(p_set)

        # 3. output an information about if and if yes, which simulations are duplicates and
        #   need to be removed
    if not duplicate_count == 0:
        print(f"There have been >> {duplicate_count} << duplicates found. "
              f"Please remove them before submitting simulations!")
    else:
        print("No duplicates have been found!")

    return p_sets_updated, p_sets_duplicates


def print_param_sets(p_sets) -> None:
    """
    function prints out each parameter set to a new line
    :param p_sets: a list with each element being a simulation parameter set
    :return:
    """
    print('___ ___ ___ ___ ___ ___')
    for p_set in p_sets:
        print(p_set)
        print('___ ___ ___ ___ ___ ___')


if __name__ == '__main__':
    """
    run this script before starting the simulations to check for duplicates in the simulation paramters
    """

    param_infos_file_name = Path(__file__).parent.resolve() / '2dfft_data_selected' \
                            / 'param_infos' / '12-05_18-43-50param_infos.csv'

    param_sets = load_param_sets()

    param_sets_updated, param_sets_dups = check_duplicate_params(param_sets, param_infos_file_name)

    print_param_sets(param_sets_dups)

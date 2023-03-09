import sys
import json

from create_model_script_v15 import run_abaqus_model

'''
This script takes the input arguments from the command line, converts them into usable variables and finally calls 
 create_abaqus_model to run the model
 
Since this script imports the run_abaqus_model command this script needs to be run in Abaqus
->> abaqus python run_simulation.py
and cannot be called within an python environment directly
'''


def parse_input_variables(input_list):
    """
     extracts the input variables for the Abaqus model from the command line input string and outputs them in
      an appropriate format. If no input is given, the respective default value will be used

     output: dictionary with all variables needed for simulation

     args:
        - input_list - list of strings with input arguments. Strings are in the form of 'variable_name=variable_value'

     WATCH OUT: THERE IS A DUPLICATE OF THIS FUNCTION IN UTILS.PY!!
     (duplicate needs to be there because needs to passed directly into Abaqus)
    """
    sd = {
        'coating_height': 600E-6,
        'plate_width': 0.08,
        'base_plate_height': 0.003,
        'coating_density': 7190.0,
        'coating_youngs_mod': 279E9,
        'coating_pois_rat': 0.21,
        'base_plate_density': 6560.0,
        'base_plate_youngs_mod': 99.3E9,
        'base_plate_pois_rat': 0.37,
        'cb_width': 0.005,
        'cg_width': 0.02,
        'cs_width': 0.055,
        'cg_top_left': 0.001,
        'cg_top_right': 0.001,
        'cg_bevel': 0.001,
        'cg_gap_depth': 0.00000, # 0.00005
        'ex_amp': 2e-06,
        'num_mesh': 1,
        't_max': 2E-8,
        't_sampling': 2E-8,  # 5E-8,  # 2E-7,
        't_period': 9.2E-5,
        'run_it': True,
        'save_inp': False,
        'cores': None
    }
    for elem in input_list:
        if elem == 'run_it=True':
            sd['run_it'] = True
        elif elem == 'run_it=False':
            sd['run_it'] = False
        elif elem == 'save_inp=True':
            sd['save_inp'] = True
        elif elem == 'save_inp=False':
            sd['save_inp'] = False
        elif '=' in elem:
            eq_idx = elem.find('=')
            # if elem[0:eq_idx] == 'cores':
            #     if elem[eq_idx+1:] == 'None':
            #         sd[elem[0:eq_idx]] = None
            #     else:
            #         sd[elem[0:eq_idx]] = None  # int(elem[eq_idx+1:])
            if elem[2:eq_idx] in sd:    # elem[0:eq_idx]
                sd[elem[2:eq_idx]] = float(elem[eq_idx+1:])
    return sd


if __name__ == "__main__":
    '''
    Call main Abaqus construction script with updated parameters
    '''
    print(sys.argv)
    # import pdb; pdb.set_trace()

    sd = parse_input_variables(sys.argv)

    print('The input variables in run_simulation are:')
    print(sd)

    run_abaqus_model(
        sd['coating_height'],
        sd['plate_width'],
        sd['base_plate_height'],
        sd['coating_density'],
        sd['coating_youngs_mod'],
        sd['coating_pois_rat'],
        sd['base_plate_density'],
        sd['base_plate_youngs_mod'],
        sd['base_plate_pois_rat'],
        cb_width=sd['cb_width'],
        cg_width=sd['cg_width'],
        cs_width=sd['cs_width'],
        cg_top_left=sd['cg_top_left'],
        cg_top_right=sd['cg_top_right'],
        cg_bevel=sd['cg_bevel'],
        cg_gap_depth=sd['cg_gap_depth'],
        ex_amp=sd['ex_amp'],
        num_mesh=sd['num_mesh'],
        t_max=sd['t_max'],
        t_s=sd['t_sampling'],
        t_p=sd['t_period'],
        run_simulation=sd['run_it'],
        create_inp=sd['save_inp'],
        n_cores=sd['cores']
    )

    # with open('test_info.json', 'w') as outfile:
    #     json.dump(sd, outfile, indent = 6)

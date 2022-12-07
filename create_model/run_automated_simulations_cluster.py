import os
import sys
from subprocess import Popen, PIPE
import subprocess
# from tqdm import tqdm
import time

try:
    from create_model.postprocess_2dfft_max_v15 import postprocessing_2dfft

    from create_model.utils import (
        delete_unwanted_files,
        get_newest_file_name,
        send_slack_message,
        parse_input_variables
    )

except ModuleNotFoundError:
    from postprocess_2dfft_max_v15 import postprocessing_2dfft

    from utils import (
        delete_unwanted_files,
        get_newest_file_name,
        send_slack_message,
        parse_input_variables
    )

'''
Automate simulations in Abaqus/CAE for the data creation
 pipeline/parameter study. This script manages creating a parameter
 list and calling Abaqus/CAE with the specified parameters from the command line
 
To run this script type
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
python run_automated_simulations.py
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
into the command line

Make sure that run_simulation.py, create_model_script_v14.py,
 extract_disp_history_max_v5.py and postprocess_2dfft_max_v9
 are in the same directory as this file and that the arguments
 are separated by whitespaces
 
The package pushbullet gets deinstalled regularly on the desktop 
 pc. Make sure to install it first with 'pip install pushbullet.py==0.9.1'
-> Pushbullet wants me to pay - no, switch to Slack for pop-up messages instead
 
 
This script does the following
1. create parameter list (better: load parameter list from external file)
2. conduct simulation in Abaqus with 'run_simulation()'
3. extract displacement information with 'extract_disp_history_max_v5'
4. apply post processing to disp data with 'postprocess_2dfft_max_v9'
5. delete all files besides python scripts (.py), simulation information
    file (.json) and extracted 2dfft data (.csv)
6. (TODO:) store simulation information file and extracted 2dfft data in meaninful 
            way


 created by: Max Schmitz - mschmitz7@gatech.edu
 created on: 05/10/2021
'''
# track time of execution
start_time = time.time()

# specify all file types in folder which should not be deleted later on
white_list = ['.py',
              '.csv',
              '.json',
              '.dat',
              '.out',
              '.pbs',
              '.png',
              '.txt'
              ]
              # '.odb']

print('______ extract parameters from command line ______')
sd = parse_input_variables(sys.argv)
print('sys.argv of run_automated_cluster.py is %s' % str(sd))
param_sets = ''
for elem in sd:
    param_sets = param_sets + ' -- ' + str(elem) + '=' + str(sd[elem])
param_sets = [param_sets]  # make it to a list
print('param_sets (type: %s): %s\nis passed into loop' % (type(param_sets), param_sets))


sim_specs = str(round(sd['coating_height'] * 1E6)) + '_' \
            + str(sd['cg_top_left'] * 1E3) + '_' + str(sd['cg_bevel'] * 1E3) + '_' \
            + str(sd['cg_top_right'] * 1E3) + '_' + str(round(sd['cg_gap_depth'] * 1E6))

print('______ simulation pipeline was started ______')
send_slack_message('-> Parameters: %s\nAbaqus simulation pipeline started' % sim_specs)

i = 1
for param_set in param_sets:
    print('\n-> current param_set to simulate: \n%s\n' % str(param_set))
    send_slack_message('\n-> current param_set to simulate: \n%s' % str(param_set))

    ''' input to command line needs to be: abaqus cae noGUI=run_simulation.py -- variable_name1=variable_value1 ...
    -> make sure that the arguments are separated by white spaces and prepended dash. Example:
    r = os.system(
        'abaqus cae noGUI=run_simulation.py -- plate_width=0.07 -- base_plate_height=0.0001 -- coating_height=0.0006
        -- run_it -- num_mesh=0.1')
    '''

    # r = Popen(
    r = subprocess.run(
        ['abaqus cae noGUI=run_simulation.py%s ' % param_set],
        # stdout=PIPE,
        # stderr=PIPE,
        shell=True
    )
    # r.wait()
    print('r = ' + str(r))
    # stdout, _ = r.communicate()
    # print('stdout = ' + str(stdout))
    print('--> simulation completed')
    sim_time = time.time() - start_time
    send_slack_message('-> Parameters: %s\n'
                       'simulation completed after %s h\n' % (sim_specs, sim_time/(60**2)))
    send_slack_message('\n|-> current param_set to simulate: \n%s' % str(param_set))

    q = Popen(
        # q = subprocess.run(
        ['abaqus python extract_disp_history_max_v5.py'],
        stdout=PIPE,
        stderr=PIPE,
        shell=True
    )
    q.wait()
    print('q = ' + str(q))
    # stdout, _ = q.communicate()
    # print('stdout = ' + str(stdout))
    print('---> extraction completed')
    extraction_time = time.time() - sim_time
    send_slack_message(str('-> Parameters: %s\n'
                           'extraction completed after %s min' % (sim_specs, extraction_time/60)))

    postprocessing_2dfft(
        plot=True,
        show=True,
        save=True,
        plt_type='contf',
        plt_res=300,
        fitting_style='lin',
        add_analytical=False,
        add_scatter=False,
        add_fit=False,
        clip_threshold=0.0001,
        m_axis=[0, 17500, 0, 2.5E7],  # [0, 17500, 0, 3E7]
        cluster=True
    )
    print('----> postprocessing (2DFFT) completed')
    fft_time = time.time() - extraction_time
    send_slack_message('CLUSTER: Postprocessing completed after %s min' % (fft_time/60))

    delete_unwanted_files(white_list)
    print('-----> unnecessary files deleted')

    # store simulation information file and extracted 2dfft data in meaningful
    #    way (e.g. subfolders with thickness of plate as name) - done with way simulation is called

    # move features and input file to the features folder!

    send_slack_message(
        'Simulation pipeline for %s\n'
        '%s out of %s completed\n' % (sim_specs, i, len(param_sets))
    )
    i += 1

runtime = time.time() - start_time
print('runtime of simulation pipeline was: %s min' % (runtime/60))
print('______ simulation pipeline finished ______ ')
send_slack_message('-> Parameters: %s\n'
                   'Abaqus simulation pipeline finished after %s h\n' % (sim_specs, runtime/(60**2)))

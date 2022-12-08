import os
from subprocess import Popen, PIPE
import subprocess
# from tqdm import tqdm
import time
from pathlib import Path

# try:
#     from utils import send_push_msg
# except ModuleNotFoundError:
#     os.system('pip install pushbullet.py==0.9.1')
#     # os.system('conda install - c auto pushbullet.py')
#     from utils import send_push_msg

try:
    from create_model.utils import delete_unwanted_files, get_newest_file_name, send_slack_message
    from create_model.postprocess_2dfft_max_v15 import postprocessing_2dfft
except ModuleNotFoundError:
    from utils import delete_unwanted_files, get_newest_file_name, send_slack_message
    from postprocess_2dfft_max_v15 import postprocessing_2dfft
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
              '.txt']

# do not need any parameter since no simulation is conducted here
param_sets = ['1']

folders = [elem for elem in os.listdir(Path.cwd())
               if elem.find('old') == -1 and not Path(Path.cwd() / elem).is_file() and elem.find('py') == -1]
print(folders)

print('______ postprocessing pipeline was started ______')
send_slack_message('CLUSTER: postprocessing pipeline started')

i = 1
for folder in folders:
    # --- 1. check if the files that should be created in the following are already existing
    j = 0
    folder_path = Path(__file__).parent.resolve() / folder
    for file in folder_path.glob('*.png'):
        j += 1
        print(f'>> The file \n{file}\nexists already! Move on to the next folder.')
        send_slack_message(f'>> The file \n{file}\nexists already! Move on to the next folder.')
        break

    if not j == 0:
        print(f'>> The folder {folder} includes the requested data already! Move forward!')
        send_slack_message(f'>> The folder {folder} includes the requested data already! Move forward!')
    else:
        # --- 2. conduct simulation in Abaqus with 'run_simulation()
        ''' input to command line needs to be: abaqus cae noGUI=run_simulation.py -- variable_name1=variable_value1 ...
        -> make sure that the arguments are separated by white spaces and prepended dash. Example:
        r = os.system(
            'abaqus cae noGUI=run_simulation.py -- plate_width=0.07 -- base_plate_height=0.0001 -- coating_height=0.0006
            -- run_it -- num_mesh=0.1')
        '''

        # r = Popen(
        #     ['abaqus cae noGUI=run_simulation.py -- %s -- run_it' % (param_set)],
        #     stdout=PIPE,
        #     stderr=PIPE,
        #     shell=True
        # )
        # print('r = ' + str(r))
        # stdout, _ = r.communicate()
        # print('stdout = ' + str(stdout))
        # print('--> simulation completed')
        # sim_time = time.time() - start_time
        # send_push_msg('CLUSTER: simulation completed after %s s\ntime: %s' % (sim_time, start_time))

        # q = Popen(
        #     # q = subprocess.run(
        #     ['abaqus python extract_disp_history_max_v5.py'],
        #     stdout=PIPE,
        #     stderr=PIPE,
        #     shell=True
        # )
        # q.wait()
        # print('q = ' + str(q))
        # # stdout, _ = q.communicate()
        # # print('stdout = ' + str(stdout))
        # extraction_time = time.time() - start_time
        # print('---> extraction completed after %s min' % (extraction_time / 60))
        # send_slack_message(str('CLUSTER: extraction completed after %s min' % (extraction_time / 60)))

        postprocessing_2dfft(
            sim_path=Path.cwd() / folder,
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
        fft_time = time.time() - start_time
        # send_push_msg('CLUSTER: simulation completed after %s s' % (fft_time))
        send_slack_message('CLUSTER: postprocessing completed after %s min' % (fft_time/60))

        delete_unwanted_files(white_list, cur_path=Path.cwd() / folder)
        print('-----> unnecessary files deleted')

    # -- 7. give push update after each single simulation
    send_slack_message('Simulation pipeline %s out of %s completed\n'
                       'start time = %s' % (i, len(folders), start_time))
    i += 1

runtime = time.time() - start_time
print('runtime of pipeline was: %s min' % (runtime/60))
print('______ simulation pipeline finished ______ ')
send_slack_message('CLUSTER: Abaqus simulation pipeline finished after %s min\n'
                   'start time = %s' % (runtime/60, start_time))

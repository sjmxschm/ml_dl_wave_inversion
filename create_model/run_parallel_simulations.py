import os
import sys
import time
from pathlib import Path
import subprocess
from subprocess import Popen, PIPE
from shutil import copyfile


"""

This script allows to run automated parallel simulations on the cluster and is
called within the activated wave_env_cl. This script runs directly on the end node
so make sure the following commands have been used BEFORE running this script:

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
module load anaconda3/2020.11   (load anaconda)
conda activate wave_env_cl      (activate environment)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Then this script can be called with

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
python3 run_parallel_simulations.py
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

And nothing else needs to be done if the param_sets have been specified in the right way.
It should just run!

For this,
    1. a set of parameter sets is loaded/specified, then a loop over the parameter
        sets in param_sets is started,
    2. a folder is created taking the thickness as name (use only coating thickness first)
    3. the script copies the run_parallel.pbs script and all python files needed into folder to
    4. call the run_parallel.pbs script which calls run_automated_simulations_cluster.py
        with the parameter set as an argument (use print f and try_passing_vars2pbs.py first)
    5. move directory up again and do the same with next parameter set

Create another script later which collects all the features either if it is lin,
poly2 or whatever.

Make sure that all the python scripts needed for the pipeline are located in the same
working directory as this script, so they can be copied from there!

created by: Max Schmitz on 09/29/2021

"""

from parameter_sets import load_param_sets, get_values_from_param_set


def copy2folder(file, parent_dir, folder) -> None:
    """
    Copies file from parent_dir to parent_dir/folder.
    For this, file needs to be in parent_dir

    args:
        - file (str): name of file which needs to be copied
        - parent_dir (str or Path.object): current working directory,
            make sure file is contained in parent_dir
        - folder (str): name of folder where file should be copied to
    """
    src = parent_dir / file
    dst = parent_dir / folder / file
    copyfile(src, dst)


# def get_values_from_param_set(
#         param_set: str,
#         attribute: str = 'coating_height=',
#         factor: float = 1E6,
#         formatting: str = '.5f'
# ) -> float:
#     """
#     Function finds attribute in param_set and returns corresponding value in format
#     defined by formatting
#
#     args:
#         - param_set: string which contains all attributes with numerical values
#             e.g. ' -- plate_width=0.08 -- coating_height=0.00010'
#         - attribute: attribute to which the numerical value should be returned
#             e.g. 'coating_height=' (IMPORTANT: include the '=' sign in string)
#         - factor: factor to scale the decimal power to a nicer number
#             e.g. 0.0004 * 1E4= 4
#         - formatting: specify how the output float should be formatted
#
#     """
#     start_idx = str(param_set).find(attribute) + len(attribute)
#     end_idx = str(param_set).find(' --', start_idx)
#     value = float(str(param_set)[start_idx:end_idx]) * factor
#     value = format(value, formatting).rstrip('0')
#     # print(value)
#     return value


def run_parallel_sims():
    """
    Start parallel simulations. For this, load param sets and call scheduler from
    right directory
    """

    ###################################

    copy_files = [
        'create_model_script_v15.py',
        'extract_disp_history_max_v5.py',
        'parameter_sets.py',
        'postprocess_2dfft_max_v15.py',
        'run_automated_simulations_cluster.py',
        'run_parallel_on_cluster.pbs',
        'run_simulation.py',
        'utils.py',
        'slack_url.py'
    ]

    assert all([True if Path.is_file(Path.cwd() / file) else False for file in copy_files]), (
        '!! at least one .py file is missing for pipeline !!'
    )

    param_sets = load_param_sets()

    print(f'There are {len(param_sets)} param_sets to simulate! :)')

    for param_set in param_sets:
        # extract coating_height to obtain naming scheme - e.g. 0.0004 * 1E4= 4
        thick = get_values_from_param_set(param_set, attribute='coating_height=',
                                          factor=1E6, formatting='.1f')
        print(thick)

        # youngs_mod = get_values_from_param_set(param_set, attribute='coating_youngs_mod=',
        #                                        factor=1E-9, formatting='.d')

        cg_top_left = get_values_from_param_set(param_set, attribute='cg_top_left=',
                                                factor=1E3, formatting='.1f')

        cg_top_right = get_values_from_param_set(param_set, attribute='cg_top_right=',
                                                 factor=1E3, formatting='.1f')

        cg_bevel = get_values_from_param_set(param_set, attribute='cg_bevel=',
                                             factor=1E3, formatting='.1f')

        cg_gap_depth = get_values_from_param_set(param_set, attribute='cg_gap_depth=',
                                                 factor=1E6, formatting='3.1f')  # formatting was '.1f'
        print(cg_gap_depth)

        folder = str(thick) + '_' \
                 + str(cg_top_left) + '_' + str(cg_bevel) + '_' \
                 + str(cg_top_right) + '_' + str(cg_gap_depth)

        parent_dir = Path.cwd()  # Path(__file__).parent.resolve()
        print(f'parent_dir / folder')
        try:
            os.mkdir(parent_dir / folder)
        except FileExistsError:
            print('Folder already exists, continue with it and start simulation in it')

        for c_file in copy_files:
            copy2folder(c_file, parent_dir, folder)

        print(f'sys.argv from run_parallel_simulations.py = {sys.argv}')

        # call the .pbs script with the respective param_set as argument
        subprocess.run(['dos2unix', 'run_parallel_on_cluster.pbs'], cwd=Path(parent_dir / folder))
        r = subprocess.Popen(
            ['qsub', '-v', f'P_SET="{param_set}"', 'run_parallel_on_cluster.pbs'],
            cwd=Path(parent_dir / folder),
            stdout=PIPE,
            stderr=PIPE
        )
        print('r = ' + str(r))
        stdout, _ = r.communicate()
        print('stdout = ' + str(stdout))

        # make sure that no simulation is started on same second for naming sim_info_file without ambiguity
        time.sleep(1)


if __name__ == '__main__':
    run_parallel_sims()

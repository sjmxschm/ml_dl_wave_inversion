import os
from os import listdir, remove, rename
from os.path import isfile
from pathlib import Path

from create_model.utils import delete_unwanted_files

"""
This script cleans up my local folder structure after copying the folders
from the simulations from the cluster to my local laptop. Unfortunately,
I was lazy and use a different naming scheme for the data on the cluster 
than the one I use locally on my computer.

After I conducted a batch of simulations and deleted most files, I copy 
all the folders directly to my local drive. In these folders there are 
some files which are too big (i.e. the _disp.csv) file, but I might need
them later on so I want to keep them on the cluster. This file helps to:
    1. delete all unnecessary files
    2. rename the folder on the drive accordingly
    
Created by Max Schmitz on 10/27/2021
"""


def find_and_delete_superfluous_files(analysis_path: Path, rename_folder: bool = False):
    """
    Scan all folders in analysis_path (most probably in analysis_2dfft), select
    the folders which where newly transferred from the cluster and delete all
    superfluous files. Additionally, rename all folders to the appropriate naming
    scheme used locally on my laptpop

    args:
        - analysis_path: Path object - specifies where this scan should be
            conducted.
        - rename_folder: bool = False - define if folder name should be changed
            to current naming scheme (e.g. 040_1_1_1_015 )
    """
    folders = [elem for elem in os.listdir(analysis_path) if elem.find('old') == -1 and elem.find('.') != -1]
    white_list = [
        # '.py',
        '.csv',
        '.json',
        # '.dat',
        # '.out',
        # '.pbs',
        '.png',
        # '.txt'
    ]

    # print(folders)
    print(f'There are {len(folders)} folders to clean up!')
    error_folders = []

    for folder in folders:
        if folder.count('.') >= 4:
            cur_path = analysis_path / folder
            delete_unwanted_files(white_list, cur_path, del_disp=True)

            # reformat folder name
            thick_end_idx = folder.find('._', 0)
            thick = folder[0: thick_end_idx]
            thick = thick.zfill(3)

            left_end_idx = folder.find('._', thick_end_idx + 2)
            left = folder[thick_end_idx + 2: left_end_idx]

            bevel_end_idx = folder.find('._', left_end_idx + 2)
            bevel = folder[left_end_idx + 2: bevel_end_idx]

            right_end_idx = folder.find('._', bevel_end_idx + 2)
            right = folder[bevel_end_idx + 2: right_end_idx]

            depth = folder[right_end_idx + 2::]
            depth = depth.zfill(3)

            new_folder_name = thick + '_' \
                              + left + '_' \
                              + bevel + '_'\
                              + right + '_'\
                              + depth
            if rename_folder:
                try:
                    rename(analysis_path / folder, analysis_path / new_folder_name)
                except FileExistsError:
                    error_folders.append(new_folder_name)
                    print( f'The folder {new_folder_name} exists already,'
                           f'renaming from {folder} was aborted!')
    return error_folders


if __name__ == '__main__':
    # working_path = Path.cwd().parent.resolve() / 'analysis_2dfft'
    # working_path = Path('C:\\Users\\Max\\Documents') / 'analysis_2dfft'

    working_path = Path.cwd().parent.resolve() / 'COPY_analysis_2dfft'

    ef = find_and_delete_superfluous_files(working_path, rename_folder=True)
    print(f'Folders with errors:\n{ef}')

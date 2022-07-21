"""
This file is intend to go into the analysis_2dfft folder, then look
into every folder with the right naming (CrZy4 folders) and first
check if features are already extracted, and otherwise run feature
extraction on these files.

Additionally, this script should allow to go through each folder and
extract new features. First idea is extraction the gradient only after
fitting the function f(x)=a*x

This file needs to be run two times in case there were features newly
created on the first pass - there is either running the postprocessing
pipeline or copying the files happening. Not both in the same pass

"""

import os
from os.path import splitext
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

import pdb

try:
    from create_model.postprocess_2dfft_max_v15 import postprocessing_2dfft
    from create_model.auto_image_extraction import load_json_info_file
except ModuleNotFoundError:
    from postprocess_2dfft_max_v15 import postprocessing_2dfft
    from auto_image_extraction import load_json_info_file


def copy2folder(file, destination_dir) -> None:
    """
    Copies file from parent_dir to parent_dir/folder.
    For this, file needs to be in parent_dir

    args:
        - file (str or Path.object): gull name of file (with path)
            which needs to be copied
        - destination_dir (str or Path.object): directory where file
            should be copied to
    """
    src = file
    dst = destination_dir
    copyfile(src, dst)


def extract_and_move_features(
        feature_name: str = '_lin_simple.',
        analysis_path: Path = Path.cwd(),
):
    """
    Function searches for file with feature_name in files in each folder
    of analysis_2dfft. If this file (which contains the feature(s)) then
    it is copied to analysis_2dfft, otherwise postprocessing_2dfft is
    used to extract the features.
    """

    if cluster:
        folders = [elem for elem in os.listdir(analysis_path) if splitext(elem)[1] == '.']
    else:
        folders = [elem for elem in os.listdir(analysis_path) if elem.find('old') == -1]

    # # select only folders in certain range:
    # folders = [elem for elem in folders if 200 <= int(elem[0:3]) <= 300]

    # select only folders with no gap
    uniform_folders = []
    for idx, folder in enumerate(folders):
        old_model = False
        sim_info = load_json_info_file(analysis_path / folder)
        c_thick = sim_info['c_height']
        try:
            gap_depth = sim_info['geometric_properties']['cg_gap_depth']
        except KeyError:
            gap_depth = 0.
            old_model = True

        if gap_depth == 0 and not old_model:
            uniform_folders.append(folder)

    folders = uniform_folders
    print(f'>>>>> There are {len(folders)} folder to analyze <<<<<')

    output_folder = 'old_' + feature_name[1::].rstrip('.') + '_thesis'

    print(f'There are {len(folders)} folders to analyse')

    copied_folders = []
    error_files = []
    for folder in tqdm(folders):
        # print(f'folder = {folder}')
        if folder.find('old') == -1:
            exists = False

            for file in os.listdir(analysis_path / folder):
                if file.find(f'features{feature_name}') != -1:
                    save_file = file
                    my_file = analysis_path / output_folder / file
                    if my_file.is_file():
                        print(f'File -- {file} -- already existed')
                        save_file = f'{int(file[0]) + 1}' + file[1::]

                    copy2folder(analysis_path / folder / file,
                                analysis_path / output_folder / save_file)

                    sim_file_name = file[0:file.find('features')] + 'info.json'
                    save_sim_file_name = save_file[0:save_file.find('features')] + 'info.json'
                    copy2folder(analysis_path / folder / sim_file_name,
                                analysis_path / output_folder / save_sim_file_name)
                    exists = True
                    copied_folders.append(folder)

            if not exists:
                # extract features and then copy to analysis_path
                print(f'Extraction takes place in folder = {folder}')

                try:
                    postprocessing_2dfft(
                        analysis_path / folder,
                        plot=False,
                        show=False,
                        save=True,
                        add_analytical=False,
                        add_scatter=False,
                        add_fit=False,
                        fitting_style='lin_local',
                        clip_threshold=0.0001,  # 0.01,  # 0.001,
                        m_axis=[0, 17500, 0, 2.5E7],
                        plt_res=300,
                        plt_type='contf',
                        save_cnn_flag=False,
                        cluster=False,
                    )
                except TypeError:
                    print(f'data not found for folder: {folder}')
                    error_files.append(folder)

            # Rename folder
            # os.rename(
            #     analysis_path / folder,
            #     analysis_path / folder[folder.find('coatingheight_') + len('coatingheight_')::]
            # )

    copied_set = set(copied_folders)
    folders_set = set(folders)

    error_set = folders_set.difference(copied_set)
    print(f'error_set is\n{error_set}')

    print('Extracting data completed!')
    print(f'Error files were:\n{error_files}')


if __name__ == '__main__':
    cluster = False

    if not cluster:
        # path = Path.cwd().parent.resolve() / 'analysis_2dfft'
        path = Path('C:\\Users\\Max\\Documents') / 'analysis_2dfft'
    else:
        path = Path.cwd().resolve()

    extract_and_move_features(analysis_path=path,
                              feature_name='_lin_local.')

"""
This file is intend to go into folder above the analysis_2dfft simulations, then look
into every folder with the right naming (CrZy4 folders) and then first
check if .png image for neural net (no legend, title, ...) already exists,
in this case copy it to CNN extraction folder. Otherwise, create neural
net .png file first and copy it then.

"""

import os
from os.path import isfile, join, splitext
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
from typing import Union, Tuple
import json
import numpy as np

from sklearn.model_selection import train_test_split

import pdb

try:
    from create_model.postprocess_2dfft_max_v15 import postprocessing_2dfft
    from create_model.utils import get_newest_file_name
    from create_model.visualize_dataset import visualize_features, get_thickness_info_uniform
except ModuleNotFoundError:
    from postprocess_2dfft_max_v15 import postprocessing_2dfft
    from utils import get_newest_file_name
    from visualize_dataset import visualize_features, get_thickness_info_uniform


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


def check_and_create_dirs(path: Path, f_name: str = 'temp') -> None:
    """
    Checks if folder f_name is already existing in path, otherwise the folders
        - path / f_name / '1'
        - path / f_name / '-1'
    are created. Input for f_name is either 'temp', 'test' or 'train'.

    args:
        - path: Path-object - path where extraction is started from and where all
            simulation data lies
        - f_name: str - either 'temp', 'test' or 'train'
    """
    if not os.path.exists(path / f'old_{f_name}'):
        os.makedirs(path / f'old_{f_name}' / '1')
        os.makedirs(path / f'old_{f_name}' / '-1')


def load_json_info_file(data_path: Path) -> dict:
    """
    Function loads the newest simulation info file and converts it into a dict

    args:
        - data_path: path to folder of a single simulation which contains the
            json simulation info file
    """
    try:
        json_info_file, _ = get_newest_file_name(data_path, extension='.json')
        with open(data_path / json_info_file) as info_file:
            sim_info = json.load(info_file)
    except NameError:
        print('No .json info file with appropriate naming in directory!')
    return sim_info


def extract_folders_via_thickness_threshold(
        folders: list = None,
        thick_threshold: int = 204,
        cutting_operator: str = 'leq'
) -> list:
    """
    Extracts all folders from folders which have a overall thickness with gap
    smaller (leq) or greater (geq) as the thickness threshold

    args:
        - folders: list of folders in directory with simulation data and CNN .png images
        - thick_threshold: threshold where data should be cut
        - cutting_operator: specify if all folder less or greater than threshold should be returned

    """
    new_folders = []
    for folder in folders:
        if folder.count('_') == 0:
            assert len(folder) == 3
            real_thick = int(folder)
        elif folder.count('_') == 1:
            assert len(folder) == 7
            real_thick = int(folder[0:folder.find('_')])
        else:
            c_thick = int(folder[0:folder.find('_') - 1])
            gap_depth = int(folder[folder.rfind('_') + 1:-1])
            real_thick = c_thick - gap_depth

        if real_thick <= thick_threshold and cutting_operator == 'leq':
            new_folders.append(folder)
        if real_thick >= thick_threshold and cutting_operator == 'geq':
            new_folders.append(folder)

    return new_folders


def store_features_in_ndarray(feats: list = None, sim_info: dict = None):
    h = sim_info['c_height']
    try:
        gd = sim_info['geometric_properties']['cg_gap_depth']
        b = sim_info['geometric_properties']['cg_bevel']
    except KeyError:
        gd = 0.
        b = 0.002
    return feats.append([h, gd, b])


def extract_all_information(
        a_path: Path,
        random_suppression: bool = True
) -> list:
    """
    Extracts the features and folder names for all folders in a_path. If random_suppression
        is true it removes randomly some data points in the range between 200 and 300 microns
        coating

    args:
        - a_path: path to analysis data
        - random_suppression: there are a lot of simulations between 200 and 300 microns coating
            thickness. Strictly speaking, there are around 350 within and 250 simulations out
            of this range. If random_suppression is turned on, a certain amount of this the
            simulations between 200 and 300 microns thickness are suppressed and not used for
            CNN training.
    """
    folders = [elem for elem in os.listdir(a_path) if elem.find('old') == -1]

    new_folders = []

    feats = []

    np.random.seed(2718)
    rand_prob = np.random.uniform(0, 1, len(folders))
    # rand_prob = np.ones(len(folders))  # remove random selection

    pop_idx = 0

    for idx, folder in enumerate(folders):
        sim_info = load_json_info_file(a_path / folder)

        c_thick = sim_info['c_height']
        try:
            gap_depth = sim_info['geometric_properties']['cg_gap_depth']
        except KeyError:
            gap_depth = 0.

        new_folders.append(folder)
        store_features_in_ndarray(feats, sim_info)

        # if random_suppression and 200 <= c_thick * 1E6 <= 300:  # and rand_prob <= 0.5:
        # if random_suppression and 200 <= c_thick * 1E6 <= 310 and str(gap_depth*1e6)[-2::]
        # in exclude_digits and not uniform: (rand_prob[idx] < 0.1)
        if random_suppression and (0.00019 < c_thick) and (c_thick < 0.000315) and (gap_depth != 0) \
                and (rand_prob[idx] < 0.7):
            # pdb.set_trace()
            try:
                print(f'pop at {c_thick * 1e6} with gap {gap_depth * 1e6}')
                new_folders.pop()
                feats.pop()
                pop_idx += 1
            except IndexError:
                # pdb.set_trace()
                print(f'new_folders list was empty when you tried to remove last element with thickness {c_thick*1e6}')
    print(f'{pop_idx} elements have been removed')
    # pdb.set_trace()
    return new_folders, np.asarray(feats)


def extract_folders_via_uniformness(
        a_path: Path,
        folders: list = None,
        gap_ratio: int = 5,
        cutting_operator: str = 'uniform',
        random_suppression: bool = False
) -> Tuple[list, np.ndarray]:
    """
    Extracts all folders from folders which are uniform or non-uniform (depending on cutting operator), i.e. for
    which the gap depth is zero
    - The old gap_ratio is deprecated, but still used as an argument for this function to make
        it backwards compatible

    args:
        - a_path: path to analysis data
        - folders: list of folders in directory with simulation data and CNN .png images
        - gap_ratio: ratio (gap_depth/coating_thickness) in percent und which the specimen
            is considered as uniform
        - cutting_operator: specify if folders containing uniform or non-uniform data should
            be returned
        - random_suppression: there are a lot of simulations between 200 and 300 microns coating
            thickness. Strictly speaking, there are around 350 within and 250 simulations out
            of this range. If random_suppression is turned on, a certain amount of this the
            simulations between 200 and 300 microns thickness are suppressed and not used for
            CNN training.
    """
    new_folders = []

    feats = []

    np.random.seed(2718)
    rand_prob = np.random.uniform(0, 1, len(folders))
    # rand_prob = np.ones(len(folders))  # remove random selection

    for idx, folder in enumerate(folders):
        sim_info = load_json_info_file(a_path / folder)

        c_thick = sim_info['c_height']
        try:
            gap_depth = sim_info['geometric_properties']['cg_gap_depth']
        except KeyError:
            gap_depth = 0.

        # if round(100 * gap_depth/c_thick) <= gap_ratio:
        if gap_depth == 0:
            uniform = True
        else:
            uniform = False

        if uniform is True and cutting_operator == 'uniform':
            new_folders.append(folder)
            store_features_in_ndarray(feats, sim_info)
        if uniform is False and cutting_operator == 'non-uniform':
            new_folders.append(folder)
            store_features_in_ndarray(feats, sim_info)

        # if random_suppression and 200 <= c_thick * 1E6 <= 300:  # and rand_prob <= 0.5:
        # pdb.set_trace()
        # exclude_digits = ('15', '25', '50', '60', '70', '80', '90', '100', '110', '120', '130', '150')
        # if random_suppression and 200 <= c_thick * 1E6 <= 310 and str(gap_depth*1e6)[-2::]
        # in exclude_digits and not uniform: (rand_prob[idx] < 0.1)

        if random_suppression and (0.00019 < c_thick) and (c_thick < 0.000315) and (gap_depth != 0) \
                and (cutting_operator == 'non-uniform') and (rand_prob[idx] < 0.7):
            try:
                print(f'pop at {c_thick * 1e6}')
                new_folders.pop()
                feats.pop()
            except IndexError:
                print(f'new_folders list was empty when you tried to remove last element with thickness {c_thick*1e6}')
    return new_folders, np.asarray(feats)


def extract_and_move_nn_image(
        analysis_path: Path,
        copy_files: bool = True,
        thick_threshold: int = 1000,
        gap_ratio: int = 5,
        cutting_operator: str = Union['leq', 'geq', 'uniform', 'non-uniform'],
        feature_name: str = '_lin_simple',
        clf_folder: str = '1',
        cluster: bool = False,
        rs: bool = False
) -> list:
    """
    Function searches for neural net .png file within files in each folder
    containing simulation files. If this .png file (which contains the feature(s))
    exists then it is copied to old_temp.
    Prior, postprocessing_2dfft was called if the corresponding .png file is not
    existing inside to folder to used to extract the features and would be copied then.
    TODO: Since there should be the right .png image file in each folder, track the folders
        where the CNN .png file has not been found and output them to the user to manually
        check what was going wrong in the respective folder.

    Additionally, this script allows to use a thickness threshold.

    This script might needs to be run 2 times. First time create missing cnn
    images - if there are some missing, and second time copy them

    Make sure there are the following folders contained in the working
    directory where this script is getting called from:
        - .. / old_temp / 1
        - .. / old_temp / -1
        - .. / old_test / 1
        - .. / old_test / -1
        - .. / old_train / 1
        - .. / old_train / -1
    This is now handled by this script which creates missing folders
    if they do not exist.

    return:
        - process_errors (list): list of folders where an error occurred during
            extraction

    args:
        - copy files: bool = True: specify if cnn image + sim_info file should
            be copied to higher directory or not. If not, the missing cnn images
            will be created first
        - thick_threshold: int = 216: thickness which should be the threshold if data
            is split according to true thickness
        - gap_ratio: If data is transformed into uniform/non-uniform, gap_ratio is the
            percentage of gap depth per thickness allowed to be considered as uniform
        - cutting_operator: str = 'leq': either 'leg' for <= or 'geq' for >=, or if
            the uniformness mode should be used 'uniform' or 'non-uniform'
        - feature_name: name of lin features which should be copied for standard
            classic machine learning classifiers
        - clf_folder: can be either '1' (thick enough) or '-1' (not thick enough)
        - cluster: bool = False: specify if script runs on cluster or on local
            drive
        - rs: random suppression used in extract_folders_uniform. Look into this function
            for documentation
    """

    check_and_create_dirs(analysis_path, 'temp')
    check_and_create_dirs(analysis_path, 'test')
    check_and_create_dirs(analysis_path, 'val')
    check_and_create_dirs(analysis_path, 'train')

    assert 'old_temp' in os.listdir(analysis_path), '!! old_temp folder is missing in directory!!'
    assert 'old_test' in os.listdir(analysis_path), '!! old_test folder is missing in directory!!'
    assert 'old_train' in os.listdir(analysis_path), '!! old_train folder is missing in directory!!'

    valid_c_operators = ['leq', 'geq', 'uniform', 'non-uniform']
    assert cutting_operator in valid_c_operators, 'Please specify valid cutting_operator'

    if cluster:
        folders = [elem for elem in os.listdir(analysis_path) if splitext(elem)[1] == '.']
    else:
        folders = [elem for elem in os.listdir(analysis_path) if elem.find('old') == -1]

    # extract uniform/non-uniform folders
    new_folders, new_feats = extract_folders_via_uniformness(analysis_path, folders, gap_ratio, cutting_operator, rs)
    save = True
    if cluster:
        save = False
    visualize_features(new_feats, save=save)
    # get_thickness_info_uniform(feats=new_feats, gap_ratio=5, save=save)

    process_errors = []
    print(f'There are {len(new_folders)} folders to analyse')
    for folder in tqdm(new_folders):
        exists = False

        for file in os.listdir(analysis_path / folder):
            if file.find('cnn') != -1:
                if copy_files:
                    copy2folder(analysis_path / folder / file,
                                analysis_path / 'old_temp' / clf_folder / file)
                exists = True

        if not exists:
            process_errors.append(folder)

            # print('Postprocessing started in folder: ', folder)
            # # try:
            # if cluster:
            #     postprocessing_2dfft(
            #         analysis_path / folder,
            #         plot=True,
            #         show=False,
            #         save=True,
            #         add_analytical=False,
            #         add_scatter=False,
            #         add_fit=False,
            #         fitting_style='lin',
            #         clip_threshold=0.0001,  # 0.01,  # 0.001,
            #         m_axis=[0, 8000, 0, 2.5E7],  # was [0, 17500, 0, 2.5E7], now [0, 8000, 0, 2.5E7] for cnn export
            #         plt_res=300,
            #         plt_type='contf',
            #         save_cnn_flag=True,
            #         cluster=cluster,
            #     )
            # else:
            #     process_errors.append(folder)
            # # except (MemoryError, TypeError, UnboundLocalError):
            # #     process_errors.append(folder)
            # #     print(f'Simulation was not completed in {folder}, no file found for postprocessing.'
            # #           f'!!!!! or MemoryError in folder {folder}!!!!!')

    return process_errors


def obtain_train_test_split(path: Path, test_size: float = 0.3, val_and_test_size: float = 0.5, clf_folder: str = '1'):
    """
    Take the data processed and split them randomly into a training, validation and a test set.
    Then copy the data into the appropriate folder

    args:
        - path: Path-object to temp folder where extracted .png data is located
        - val_and_test_size: float which specifies percentage of dataset which should
            be used for validation and testing
        - test_size: float which specifies percentage of validation and test size that should
            be used for the test set
        - clf_folder: can be either '1' (thick enough/uniform) or '-1' (not thick enough/
            non-uniform)
    """
    analysis_path = path.parent.resolve()
    files = [f for f in os.listdir(path / clf_folder) if isfile(join(path / clf_folder, f))]
    files = [elem for elem in files if elem.find('.png') != -1]

    try:
        f_train, f_val_test, *_ = train_test_split(files, test_size=val_and_test_size,
                                               random_state=52)
        f_val, f_test, *_ = train_test_split(f_val_test, test_size=test_size,
                                                   random_state=55)
    except ValueError:
        print('There is an error in the train val test split')
        f_train = files[0:int(len(files)/2)]
        f_test = files[int(len(files)/2)::]

    print(f'size of train set = {len(f_train)}\nsize of val set = {len(f_val)}\nsize of test set = {len(f_test)}\n')
    for file in f_train:
        copy2folder(path / clf_folder / file,
                    analysis_path / 'old_train' / clf_folder / file)
    for file in f_val:
        copy2folder(path / clf_folder / file,
                    analysis_path / 'old_val' / clf_folder / file)
    for file in f_test:
        copy2folder(path / clf_folder / file,
                    analysis_path / 'old_test' / clf_folder / file)


if __name__ == '__main__':
    random_suppression = False
    extract = True
    cluster = True

    # define local path
    path = Path().resolve() / '2dfft_data_selected' / 'cluster_simulations_example'

    if not path.is_dir():
        path = Path.cwd().resolve() / 'simulations'
        cluster = False

    print(f"path = {path}")

    if extract:
        # # thickness classification
        # cut_ops = ['geq', 'leq']
        # threshold = 204

        # # uniformness classification
        cut_ops = ['uniform', 'non-uniform']
        gap_ratio = 5

        labels = ['1', '-1']

        for cut_op, label in zip(cut_ops, labels):
            print(f'--> cut_up = {cut_op}, label = {label} --')
            # not needed to run this twice because the CNN images should exist already
            # pe = extract_and_move_nn_image(path, copy_files=False, gap_ratio=gap_ratio,
            #                                cutting_operator=cut_op, clf_folder=label, cluster=cluster,
            #                                rs=random_suppression)
            # print(f'process errors 1 are: {pe}')

            pe = extract_and_move_nn_image(path, copy_files=True, gap_ratio=gap_ratio,
                                           cutting_operator=cut_op, clf_folder=label, cluster=cluster,
                                           rs=random_suppression)
            print(f'process errors 1 are: {pe}')

            obtain_train_test_split(path / 'old_temp', test_size=0.3, clf_folder=label)

    else:
        folders, features = extract_all_information(a_path=path, random_suppression=True)
        get_thickness_info_uniform(feats=features, save=False)
        visualize_features(f=features, save=True)

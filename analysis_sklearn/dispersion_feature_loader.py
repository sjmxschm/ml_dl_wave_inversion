import pathlib
from os import listdir, path
from os.path import isfile, join

import numpy as np

import json

"""
Script which provides all functions for loading the dispersion data
"""


def get_filenames(data_path, job_name='max_analysis_job', extension='.txt'):
    """ searches the current working directory
        and returns the name of all filenames with
        the according file extension

        Make sure the file is in the appropriate format:
        -> mm-dd_hh-mm-ss_XXX_max_analysis_job_xy.txt
        where XXX is the thickness in microns

        args:
            - data_path (str or Path object): path to data for learning (usually in ../data folder)
            - job_name (str): specifies the name of the job/file for analysis (string before date)
            - extension (str): (optional) specifies file extension to look up - default is '.csv'
    """
    files_in_folder = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    filenames = []
    for file in files_in_folder:
        _, file_extension = path.splitext(file)
        s = file.find('max')
        if file_extension == extension and file[s:s + len(job_name)] == job_name:
            filenames.append(file)
    return filenames


def load_data_from_files(
        filenames,
        data_path,
        num_feats: int = 2
) -> tuple:
    """
    load features from files and obtain layer thickness as label from
    simulation json info file

    Make sure the filenames are in the appropriate format:
    -> mm-dd_hh-mm-ss_max_analysis_job_features.txt
    and respective info file:
    -> mm-dd_hh-mm-ss_max_analysis_job_info.json

    returns:
        - X (numpy array): input data in sklearn shape (n_samples, n_features)
            in case of using the linear function:
            first element in row = a,
            second element in rob = b,
        - y (numpy array): output data in sklearn shape (n_samples,.)

    args:
        - filenames (list): list of files which contain the X data (obtained
            from function get_filenames above)
        - data_path (str or Path object): path to where simulated data is stored
        - num_feats (int): number of features, should be 2 even if just one
            feature is present since this way the jupyter notebook can be run too

    """

    X = np.zeros((len(filenames), num_feats))
    y = np.zeros((len(filenames), 1))

    for idx, file in enumerate(filenames):
        with open(data_path / file, 'r') as data:
            X[idx, :] = np.loadtxt(data, delimiter=',').reshape(1, -1)
            # y[idx] = int(file[15:18])
        json_file = file[0:file.find('features')] + 'info.json'
        with open(data_path / json_file) as info_file:
            sim_info = json.load(info_file)

        try:
            real_thickness = sim_info['c_height'] - sim_info['geometric_properties']['cg_gap_depth']
            y[idx] = real_thickness
        except KeyError:
            y[idx] = sim_info['c_height']

    return X, y

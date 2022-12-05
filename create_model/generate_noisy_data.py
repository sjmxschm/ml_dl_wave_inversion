"""
This file contains the code to create noisy data as follows:
Go through each folder, then for each folder
	• Find the most recent simulation file
	• Load the existing 2D-FFT data
	• Invert the given 2D-FFT data to time-displacement
	• Add noise in time-displacement domain (use a SNR of 40dB for now)
	• Retransform with 2D-FFT from time-displacement to frequency-wavenumber domain
	• Apply non-maximum-suppression to get maxima and save them (find consistent naming scheme)
	• Fit linear functions, extract features and save them (use consistent naming scheme)
	• Create CNN image and store it
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    get_newest_file_name,
    load_2dfft_processed_data,
    invert_2dfft,
    reapply_2dfft,
    add_noise,
    store_fft_data,
    non_maximum_suppression,
    extract_features,
    lin_func
)

# -- Specify folder to work in/where data lays
# TODO: adjust this for the cluster/local machine with try except
data_path = pathlib.Path().resolve() / '2dfft_data_selected' / '200._1._2._1._70'

# -- Specify file names for example file
# TODO: use my own function to do this
fn = '11-09_19-25-13_max_analysis_job_disp_2dfft_fg_0.0002.csv'
# check out if the following works:
fn, is_transformed = get_newest_file_name(
        data_path,
        job_name='max_analysis_job',
        extension='.csv'
)

fg, kg, abs_fft_data, sim_info = load_2dfft_processed_data(fn, data_path)
print('>> frequency-/wavenumber grid, 2D-FFT data, and simulation information file was loaded!')
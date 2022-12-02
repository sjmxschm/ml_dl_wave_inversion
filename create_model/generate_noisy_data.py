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

from utils import load_2dfft_processed_data, invert_2dfft, reapply_2dfft, add_noise, store_fft_data, non_maximum_suppression, extract_features, lin_func

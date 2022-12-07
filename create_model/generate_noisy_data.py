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
import os

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from utils import (
    get_newest_file_name,
    load_2dfft_processed_data,
    invert_2dfft,
    reapply_2dfft,
    add_noise,
    store_fft_data,
    non_maximum_suppression,
    plot_sim_and_analy_data,
    extract_features,
    lin_func,
    get_output_name
)


def create_noisy_files(
        d_path: Path,
        snr: int = 40,
        kernel: int = 15,
        save_features: bool = False,
        save_plot_normal: bool = False,
        save_cnn: bool = False,
        save_data: bool = False
) -> None:
    """
    create_noisy_files loops over all folders in d_path and applies backwards/forward 2D-FFT, adds the noise,
        applies non-maximum-suppression, extracts features, and then saves the features as well as the new
        dispersion map as .png images for processing in a CNN
    :param d_path: Path object that points to the folder in which all simulation folders live
    :param snr: signal-to-noise ratio in dB for noise which should be added
    :param kernel: integer size of kernel for non-maximum-suppression (must be odd!)
    :param save_features: specifies if the features from the conducted fit should be saved
    :param save_plot_normal: specifies if normal plot with title should be saved
    :param save_cnn: specifies if the CNN dispersion map image should be saved
    :param save_data: specifies if the noisy and newly created 2D-FFT data should be saved in folder
    :return:
    """
    assert not save_plot_normal == save_cnn, "Only one save flag can be active at a time!! Neglect if both are False"

    folders = [elem for elem in os.listdir(d_path) if (elem.find('old') == -1 and elem.find('py') == -1)]

    for folder in tqdm(folders):
        print(f'> folder: {folder}')

        fn, is_transformed = get_newest_file_name(
            d_path / folder,
            job_name='max_analysis_job',
            extension='.csv'
        )
        data_file = fn[0:37]
        print(f'>> filename = {fn}')

        fg, kg, abs_fft_data, sim_info = load_2dfft_processed_data(fn, d_path / folder)
        print('>>> frequency-/wavenumber grid, 2D-FFT data, and simulation information file was loaded!')

        displacement_x_time, dt, dx, Nt, Nx = invert_2dfft(fg, kg, abs_fft_data, sim_info)
        print('>>> 2D-FFT data has been inverted to displacement-time data!')

        displacement_x_time_n = add_noise(displacement_x_time, snr_db=snr)
        print(f'>>> noise with SNR = {snr} was added to displacement data!')

        fg_n, kg_n, abs_fft_data_n = reapply_2dfft(displacement_x_time_n, dt, dx, Nt, Nx)
        print('>>> noisy displacement data was transformed back to frequency-wavenumber domain!')

        c_t = 0.0001
        abs_fft_data_n_c, x, y = non_maximum_suppression(
            abs_fft_data_n,
            data_file=f"{data_file}_n_{snr}_k_{kernel}_________",
            sim_path=d_path / folder,
            clip_tr=c_t,
            kernel=kernel,
            save_flag=save_features,
            plot_flag=False
        )
        print(">>> Non-maximum-suppression has finished!")

        feat_lin = extract_features(lin_func, x, y, fg, kg)
        feat_file_name = data_file[0:data_file.find('disp')] + f'features_lin_n_{snr}_k_{kernel}.txt'
        if save_features:
            with open(d_path / folder / feat_file_name, 'w') as f:
                np.savetxt(f, feat_lin, delimiter=',')
        print(">>> Feature-extraction finished!")

        plt_type = 'contf'
        plt_res = 300
        output_file = get_output_name(
            d_path / folder,
            sim_info['job_name'],
            sim_info['c_height'],
            plt_type, plt_res, save_cnn,
            snr=snr, kernel=kernel
        )
        if save_cnn or save_plot_normal:
            plot_sim_and_analy_data(
                fg,
                kg,
                abs_fft_data_n_c,
                sim_info=sim_info,
                output_file=output_file,
                x=x,
                y=y,
                plt_type=plt_type,
                plt_res=plt_res,
                ka_cr=None,
                fa_cr=None,
                mn_cr=None,
                ka_zy4cr=None,
                fa_zy4cr=None,
                mn_zy4cr=None,
                ka_zy=None,
                fa_zy=None,
                mn_zy=None,
                axis=False,
                m_axis=[0, 8000, 0, 2.5E7],
                clip_threshold=c_t,
                add_analytical=False,
                add_fit=True,
                add_scatter=True,
                save_CNN=save_cnn,
                save_flag=save_plot_normal,
                show_plot=False,
            )
            print(">>> Plotting and saving of CNN data finished!")

        if save_data:
            store_fft_data(
                fn[0:-4] + f'_n_{snr}_k_{kernel}.csv',
                d_path / folder,
                abs_fft_data_n_c,
                fg_n,
                kg_n,
                sim_info,
                snr=snr
            )
            print(">>> New, noisy 2D-FFT data was successfully saved!")
        print(">>>> Pipeline for current simulation finished\n_________________")


if __name__ == '__main__':
    # TODO: adjust this for the cluster/local machine with try except
    data_path = Path().resolve() / '2dfft_data_selected' / 'cluster_simulations_example'
    signal_to_noise_ratio_db = 40
    kernel_size_nms = 15

    create_noisy_files(
        data_path,
        snr=signal_to_noise_ratio_db,
        kernel=kernel_size_nms,
        save_features=True,
        save_plot_normal=False,
        save_cnn=True,
        save_data=True
    )

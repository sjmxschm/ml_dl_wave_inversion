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

To generate the noisy files, this file needs to go one level above the simulation folder. An example folder structure
would be:
>> home/simulations_folder/{simulation_1, simulation_2, ..., simulation_n}
then this script needs to be put into
>> home/generate_noisy_data.py

This file is not to be run on the end-node. Therefore, a scheduler script is needed. The scheduler script for this
script is 'generate_noisy_data.pbs' and needs to live in the same folder as this script. The scheduler script is
started via
#################################
>> qsub genenerate_noisy_data.pbs
#################################

It might need to be converted to unix format first which can be done via
#####################################
>> dos2unix genenerate_noisy_data.pbs
#####################################

This file was created by Max Schmitz on 12/06/2022
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from typing import Union

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
    get_output_name,
    send_slack_message
)


def create_noisy_files_in_folder(
        d_path: Path(),
        folder: str,
        incomplete_simulations: list,
        check_for_existing_files: str = None,
        snr: int = 40,
        kernel: int = 15,
        index_thrshld: float = 1.5,
        sup_thrshld: int = 1,
        c_t: float = 0.0001,
        ignore_nms: bool = False,
        save_features: bool = True,
        save_cnn: bool = False,
        save_plot_normal: bool = False,
        save_data: bool = False
) -> bool:
    """
    function is doing the complete noisy data creation pipeline inside a single folder

    :param d_path: path to the folder where all simulation folders are included
    :param folder: folder in which the noisy data should be generated
    :param incomplete_simulations: a list (created outside) of folders which had an error in noise generation
    :param check_for_existing_files: tells if and which noisy files it should check if they are already
        existing (like ~overwrite). Can be '.txt' to look for existing features or '.png' to look for png files
    :param snr: Signal-to-Noise ratio in dB
    :param kernel: kernel size for NMS (should not be even)
    :param index_thrshld: threshold for index slicing inside of NMS
    :param sup_thrshld: threshold for suppression inside of NMS
    :param c_t: clipping threshold for NMS
    :param ignore_nms: flag if the NMS and feature extraction should be excluded (makes sense if only CNN images
        should be created)
    :param save_features: flag if features from linear fit should be saved
    :param save_cnn: flag if image for CNN should be saved
    :param save_plot_normal: flag if a normal plot with title should be created (most often not on cluster)
    :param save_data: flag if new noisy dispersion map data should be saved as a .csv file
    :return:
    """
    print(f'\n> folder: {folder}')
    send_slack_message(f'\n## Noise Generation has started in folder: {folder}')

    fn, is_transformed = get_newest_file_name(
        d_path / folder,
        job_name='max_analysis_job',
        extension='.csv'
    )
    try:
        data_file = fn[0:37]
    except TypeError:
        incomplete_simulations.append(folder)
        print(f'########> No corresponding displacement .csv files found in folder {folder}!!!')
        return True
    print(f'>> filename = {fn}')

    elems_in_folder = [elem for elem in os.listdir(d_path / folder)]
    skip_sim_folder = False
    if check_for_existing_files is not None:
        for elem in elems_in_folder:
            if not elem.find(f'n_{snr}_k_{kernel}_cnn{check_for_existing_files}') == -1:
                skip_sim_folder = True
                print(f'Noisy {check_for_existing_files} file {elem} exists already, move on!')
                send_slack_message(f'\n>># Noisy files exist already in: {folder}\nMove to next folder!')
                break
        if skip_sim_folder:
            return True
    else:
        print('>>> It was not checked for existing files')

    try:
        fg, kg, abs_fft_data, sim_info = load_2dfft_processed_data(fn, d_path / folder)
        print('\n>>> frequency-/wavenumber grid, 2D-FFT data, and simulation information file was loaded!')
    except FileNotFoundError:
        print(f'>>> There was a problem with the file {fn}\n'
              f'>>> in folder {folder}. Move on and ignore this one!')
        return True

    displacement_x_time, dt, dx, Nt, Nx = invert_2dfft(fg, kg, abs_fft_data, sim_info)
    print('>>> 2D-FFT data has been inverted to displacement-time data!')

    displacement_x_time_n = add_noise(displacement_x_time, snr_db=snr)
    print(f'>>> noise with SNR = {snr} was added to displacement data!')

    fg_n, kg_n, abs_fft_data_n = reapply_2dfft(displacement_x_time_n, dt, dx, Nt, Nx)
    print('>>> noisy displacement data was transformed back to frequency-wavenumber domain!')

    if ignore_nms:
        abs_fft_data_n_c, x, y = non_maximum_suppression(
            abs_fft_data_n,
            data_file=f"{data_file}_n_{snr}_k_{kernel}_________",
            sim_path=d_path / folder,
            clip_tr=c_t,
            kernel=kernel,
            suppression_threshold=sup_thrshld,
            idx_threshold=index_thrshld,
            save_flag=save_features,
            plot_flag=False
        )
        print(">>> Non-maximum-suppression has finished!")

        feat_lin = extract_features(lin_func, x, y, fg, kg)
        # feat_file_name = data_file[0:data_file.find('disp')] + f'features_lin_n_{snr}_k_{kernel}.txt'
        feat_file_name = data_file[0:data_file.find('disp')] + \
                         f'features_lin_n_{snr}_k_{kernel}_it_{index_thrshld}_st_{sup_thrshld}.txt'
        if save_features:
            with open(d_path / folder / feat_file_name, 'w') as f:
                np.savetxt(f, feat_lin, delimiter=',')
        print(">>> Feature-extraction finished!")
    else:
        # -- clip values of 2dfft
        fft_data = np.clip(abs_fft_data_n, 0, c_t)
        abs_fft_data_n_c = np.multiply(fft_data.copy(), fft_data > np.median(fft_data))
        x, y = None, None

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
    return False

def create_noisy_files(
        d_path: Path,
        snr: int = 40,
        kernel: int = 15,
        save_features: bool = False,
        save_plot_normal: bool = False,
        save_cnn: bool = False,
        save_data: bool = False,
        check_for_existing_files: bool = True,
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
    :param check_for_existing_files: specifies if script should look for existing 2D-FFT files. If true,
        it will skip folders where noisy data is already existing
    :return:
    """
    if save_plot_normal or save_cnn:
        assert not save_plot_normal == save_cnn, "Only one save flag can be active at a time!! Neglect if both are False"

    folders = [elem for elem in os.listdir(d_path) if (elem.find('old') == -1 and elem.find('py') == -1)
               and not Path(d_path / elem).is_file() and elem.find('export') == -1]

    c_t = 0.0001
    sup_thrshld = 1  # 2
    index_thrshld = 1.5  # 0.4
    incomplete_simulations = []
    for folder in tqdm(folders):
        continue_loop = create_noisy_files_in_folder(
            d_path=d_path,
            folder=folder,
            incomplete_simulations=incomplete_simulations,
            check_for_existing_files=check_for_existing_files,
            snr=snr,
            kernel=kernel,
            index_thrshld=index_thrshld,
            sup_thrshld=sup_thrshld,
            c_t=c_t,
            save_features=save_features,
            save_cnn=save_cnn,
            save_plot_normal=save_plot_normal,
            save_data=save_data,
        )
        if continue_loop:
            continue
    print(f'There are some incomplete files in folders:{incomplete_simulations}')
    send_slack_message(f'\n## Noise Generation has finished!')


if __name__ == '__main__':
    data_path = Path().resolve() / '2dfft_data_selected' / 'cluster_simulations_example'
    # data_path = Path().resolve() / '2dfft_data_selected' / 'cluster_simulations_example_single'
    # data_path = Path().resolve() / '2dfft_data_selected' / 'cluster_simulations_example_duplicate'

    if not data_path.is_dir():
        # data_path = Path(__file__).parent.resolve() / 'simulations'  # in case of cluster
        # data_path = Path(__file__).parent.resolve() / 'batch_1'
        # data_path = Path(__file__).parent.resolve() / 'batch_2'
        # data_path = Path(__file__).parent.resolve() / 'batch_3'
        # data_path = Path(__file__).parent.resolve() / 'batch_4'
        # data_path = Path(__file__).parent.resolve() / 'batch_5'
        # data_path = Path(__file__).parent.resolve() / 'batch_6'
        # data_path = Path(__file__).parent.resolve() / 'batch_7'
        # data_path = Path(__file__).parent.resolve() / 'batch_8'
        data_path = Path(__file__).parent.resolve() / 'batch_9'
        print(f"data_path = {data_path}")

    # used first:
    signal_to_noise_ratio_db = 40
    kernel_size_nms = 15

    # second try (not applied yet):
    # signal_to_noise_ratio_db = 40
    # kernel_size_nms = 61

    create_noisy_files(
        data_path,
        snr=signal_to_noise_ratio_db,
        kernel=kernel_size_nms,
        save_features=False,
        save_plot_normal=False,
        save_cnn=True,
        save_data=False,
        check_for_existing_files='.png'
    )

import pathlib
from os import listdir, path, remove
from os.path import isfile, join
from typing import Tuple, Union
import time

import numpy as np
import matplotlib

# # comment this in if you want to export to latex
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

import matplotlib.pyplot as plt
## do not use this code:
# plt.rcParams.update({
#     "font.family": "serif",  # use serif/main font for text elements
#     "text.usetex": True,     # use inline math for ticks
#     "pgf.rcfonts": False,    # don't setup fonts from rc parameters
#     "pgf.preamble": "\n".join([
#          r"\usepackage{url}",            # load additional packages
#          r"\usepackage{unicode-math}",   # unicode math setup
#          r"\setmainfont{DejaVu Serif}",  # serif font via preamble
#     ])
# })

from datetime import date, datetime

import torch
import torch.nn as nn

import json

# from pushbullet import Pushbullet

import requests

from scipy.optimize import curve_fit

from slack_url import slackurl

"""
This file contains all helper functions which are needed for the work.
Either single functions or the whole script can be imported

created by: Max Schmitz on 08/25/2021

"""


def load_analytical_data(path, material) -> tuple:
    """
    load analytical obtained dispersion data from .npy files as well as
    mode names from .csv files (material) in given directory (path)

    Returns: wave number array, frequency array and mode names for given filename

    args:
        - path - string or path object to the directory
                        which contains analytical dispersion curves
        - material - string with name of stored dispersion data in path without
                        _dispersion_data_analytically_k_.npy,
                        _dispersion_data_analytically_f_.npy, and
                        _dispersion_data_analytically_mode_names_.csv suffix
                        example: 'Chrome_{thick}_mm_dispersion_data_analytically'
    """

    with open(path / f'{material}_dispersion_data_analytically_k_.npy', 'rb') as fa:
        k = np.load(fa)
    with open(path / f'{material}_dispersion_data_analytically_f_.npy', 'rb') as fa:
        f = np.load(fa)
    with open(path / f'{material}_dispersion_data_analytically_mode_names_.csv', 'r') as fa:
        mn = fa.readlines()

    if mn[0].find('\r') == -1:
        mn = [i.rstrip('\n') for i in mn]
    else:
        mn = [str(i)[2:4] for i in mn]

    return k, f, mn


def get_newest_file_name(
        data_path,
        job_name='max_analysis_job',
        extension='.csv'
) -> Tuple:
    """
    searches the current working directory
    and returns the name of the newest csv file

    Make sure the file is in the appropriate format:
        -> mm-dd_hh-mm-ss_max_analysis_job_disp.csv

    returns:
        - newest_file_name - file name of newest .extension file in directory
        - is_transformed - returns True if 2dfft processed file exists in directory

    args:
        - data_path: path to folder with simulation data
        - job_name: specifies the name of the job/file for analysis (string before date)
        - extension: (optional) specifies file extension to look up - default is '.csv'
        - is_transformed: boolean - returns if 2dfft processed data exists already
    """
    newest_file_name = None
    is_transformed = False

    cur_path = data_path  # pathlib.Path().absolute()
    files_in_folder = [f for f in listdir(cur_path) if isfile(join(cur_path, f))]
    newest_creation_date = np.zeros([10, 1], dtype=np.int32).ravel()

    for file in files_in_folder:
        _, file_extension = path.splitext(file)
        if file_extension == extension:
            if file[15:15 + len(job_name)] == job_name and not file.find('2dfft') == -1:
                time_idxs = []
                for idx, c in enumerate(file):
                    if c.isdigit() and idx <= 15:
                        time_idxs.append(int(idx))
                creation_date = np.array([int(file[i]) for i in time_idxs])
                for elem in zip(creation_date, newest_creation_date):
                    if elem[0] > elem[1]:
                        newest_creation_date = creation_date
                        break
                newest_file_name_2dfft = file
                is_transformed = True
            elif file[15:15 + len(job_name)] == job_name and file.find('2dfft') == -1:
                time_idxs = []
                for idx, c in enumerate(file):
                    if c.isdigit() and idx <= 15:
                        time_idxs.append(int(idx))
                creation_date = np.array([int(file[i]) for i in time_idxs])
                for elem in zip(creation_date, newest_creation_date):
                    if elem[0] > elem[1]:
                        newest_creation_date = creation_date
                        break
                newest_file_name = file

    if is_transformed:
        newest_file_name = newest_file_name_2dfft

    return newest_file_name, is_transformed


def print_variables(
        Nt,
        Nx,
        dispXTimeArray,
        tMax,
        xMax,
        dt,
        dx,
        Ny_f,
        Ny_k,
):
    """
    function prints out important analysis variables
    """

    print(f'\nnumber of time increments Nt = {Nt}')
    print(f'number of spacial increments Nx = {Nx}\n')

    print(f'min(inputData) = {np.min(dispXTimeArray)}\n'
          f'max(inputData) = {np.max(dispXTimeArray)}\n')

    print(f'tMax = {tMax}')
    print(f'k resolution: delta k = {1 / xMax}\n'
          f'omega resolution: delta f = {1 / tMax}\n')

    print(f'dt = {dt}')
    print(f'dx = {dx}')

    print(f'Ny_f = {Ny_f} Hz')
    print(f'Ny_k = {Ny_k}')


def get_plotting_layout(lo):
    """
    Specify which plotting layout should be used. Plotting layouts
    to choose from are:
        - koreck
        - analysis_sharp
        - analysis_thick
        - ...

    returns:
        - d (dict): dictionary with entries for color, linestyle, linewidth,
            alpha for the top layer, the bottom layer and the
            layered specimen

    args:
        - lo (str): specify which layout should be used. Must be
            within the set {'koreck', 'analysis_sharp', 'analysis_thick', ...}
    """

    if lo == 'koreck' or lo is None:
        t = {
            'c': 'm',
            'ls': '--',
            'd': (25, 10),
            'lw': .7,
            'a': .8,
        }
        b = {
            'c': 'k',
            'ls': '',
            'd': (None, None),
            'lw': .7,
            'a': .6,
        }
        tb = {
            'c': 'b',
            'ls': 'dashed',
            'd': (5, 5),
            'lw': .7,
            'a': .7,
        }
        dpi = 300
    elif lo == 'analysis_sharp':
        t = {
            'c': 'm',
            'ls': '-',
            'd': (None, None),
            'lw': .5,
            'a': .8,
        }
        b = {
            'c': 'k',
            'ls': '-',
            'd': (None, None),
            'lw': .5,
            'a': .6,
        }
        tb = {
            'c': 'b',
            'ls': 'solid',
            'd': (None, None),
            'lw': .5,
            'a': .7,
        }
        dpi = 600
    elif lo == 'analysis_thick':
        t = {
            'c': 'm',
            'ls': '-',
            'd': (None, None),
            'lw': .9,
            'a': .8,
        }
        b = {
            'c': 'k',
            'ls': '-',
            'd': (None, None),
            'lw': .9,
            'a': .6,
        }
        tb = {
            'c': 'b',
            'ls': 'solid',
            'd': (None, None),
            'lw': .9,
            'a': .7,
        }
        dpi = 600
    elif lo == 'analysis_max_sharp':
        t = {
            'c': 'r',
            'ls': '--',
            'd': (5, 3),
            'lw': .5,
            'a': .8,
        }
        b = {
            'c': 'y',
            'ls': '--',
            'd': (2, 1),
            'lw': .6,
            'a': .7,
        }
        tb = {
            'c': 'c',
            'ls': 'solid',
            'd': (None, None),
            'lw': .5,
            'a': .7,
        }
        dpi = 600
    elif lo == 'max_with_sim_v1':
        t = {
            'c': 'r',
            'ls': '--',
            'd': (5, 3),
            'lw': .7,
            'a': .8,
        }
        b = {
            'c': 'y',
            'ls': '--',
            'd': (2, 1),
            'lw': .7,
            'a': .6,
        }
        tb = {
            'c': 'c',
            'ls': 'solid',
            'd': (None, None),
            'lw': .7,
            'a': .7,
        }
        dpi = 600
    elif lo == 'max_with_sim_v2':
        t = {
            'c': 'r',
            'ls': '--',
            'd': (5, 3),
            'lw': .7,
            'a': .8,
        }
        b = {
            'c': 'y',
            'ls': '--',
            'd': (2, 1),
            'lw': .7,
            'a': .8,
        }
        tb = {
            'c': 'c',
            'ls': 'solid',
            'd': (None, None),
            'lw': .7,
            'a': .7,
        }
        dpi = 600
    else:
        raise (Exception('Specified plotting layout not valid!'
                         'Check for typos'))

    d = {
        't': t,
        'b': b,
        'tb': tb,
        'dpi': dpi
    }

    return d


def lin_func_simple(x, a):
    """
    define basic linear function for optimization in scipy.optimize later.
    Linear function is of the form: a * x + b
    """
    return a * x


def lin_func(x, a, b):
    """
    define basic linear function for optimization in scipy.optimize later.
    Linear function is of the form: a * x + b
    """
    return a * x + b


def poly2_func(x, a, b, c):
    """
    define basic linear function for optimization in scipy.optimize later.
    Linear function is of the form: a * x + b
    """
    return a * x ** 2 + b * x + c


def extract_features(func, x, y, fg, kg):
    """
    Fit curve to NMS data points

    returns:
        - popt (tuple): tuple with coefficients for fitted function

    args:
        - func (function): function which should be fitted
        - fg (float): frequency grid
        - kg (float): wave number grid

    """
    try:
        popt, pcov = curve_fit(func, fg[0, x], kg[y, 0])
    except IndexError:
        popt = (0, 0)
    return popt


def get_values_from_sim_info(
    sim_info: dict,
    attribute: str = 'cg_bevel',
    factor: float = 1E6,
    formatting: str = '.5f'
) -> str:
    """
    Extract attribute from sim_info\geometric_properties dict and reformat it to nice number
    """
    try:
        value = float(sim_info['geometric_properties'][attribute]) * factor
        value = format(value, formatting).rstrip('0')
    except KeyError:
        value = None
        print('Specified attribute not part of ''sim_info[geometric_properties]''! ')
    return value


def plot_sim_and_analy_data(
        fg: np.ndarray,
        kg: np.ndarray,
        fft_data: np.ndarray,
        plt_res: int,
        ka_cr: Union[np.ndarray, None],
        fa_cr: Union[np.ndarray, None],
        mn_cr: Union[np.ndarray, None],
        ka_zy4cr: Union[np.ndarray, None],
        fa_zy4cr: Union[np.ndarray, None],
        mn_zy4cr: Union[np.ndarray, None],
        ka_zy: Union[np.ndarray, None],
        fa_zy: Union[np.ndarray, None],
        mn_zy: Union[np.ndarray, None],
        sim_info: str,
        output_file: Union[str, pathlib.Path],
        plt_type: str = 'contf',
        x: np.ndarray = None,
        y: np.ndarray = None,
        axis: bool = True,
        m_axis: list = None,
        layout: str = 'max_with_sim_v2',
        clip_threshold: float = 1.0,
        func=lin_func,
        fitting_type: str = 'lin',
        add_fit: bool = True,
        add_analytical: bool = True,
        add_scatter: bool = True,
        save_CNN: bool = True,
        save_flag: bool = False,
        show_plot: bool = True,
        save_publication: bool = False
) -> None:
    """
    Plot simulated, non-maximum-suppression and analytical
    dispersion data into one plot

    args:
        - fg - frequency grid from simulation
        - kg - wave number grid from simulation
        - fft_data - spectrum data grid from simulation
        - plt_res - plotting resolution (number of levels)
        - ka_cr - analytically obtained wave numbers for Chrome
        - fa_cr - analytically obtained frequencies for Chrome
        - mn_cr - mode names for Chrome
        - ka_zy4cr - analytically obtained wave numbers for Zirconium-4 with Chrome coating
        - fa_zy4cr - analytically obtained frequencies for Zirconium-4 with Chrome coating
        - mn_zy4cr - mode names for Zirconium-4 with Chrome coating
        - ka_zy - analytically obtained wave numbers for Zirconium-4
        - fa_zy - analytically obtained frequencies for Zirconium-4
        - mn_zy - mode names for Zirconium-4
        - sim_info - simulation information dictionary
        - output_file - name of output file
        - plt_type - specify type of plot for simulated data (either 'cont' or 'contf')
        - x - x-coords of maxima after non-maximum-suppression
        - y - y-coords of maxima after non-maximum-suppression
        - axis (bool = True): specify if default axis scaling should be used or not
        - m_axis (list = None): plug in user axis scaling
        - layout (str): specify which layout/plotting settings should be used
        - clip_threshold - threshold over which elements are clipped
        - func (function object): function which is used for fittung
        - fitting_type (str): type of function which is fitted
        - add_fit (bool): specify if linear and poly2 fit should be conducted and plotted
        - add_analytical (bool): define if analytical curves should be plotted additionally
        - add_scatter (bool): define if scatter plot with nms data should be added to plot
        - save_CNN: bool = True - specify if plot should be saved for use in CNN later. This
            implies that only the dispersion data is ploted, so no axes, no title or further
            additional stuff added, no legend, contf is selected for plotting
        - save_flag: bool = False - flag of plot should be saved
        - show_plot: bool = True - flag determining if plot should be shown
        - save_publication: bool = False : specifies if plot should be exported as pgf for
            publication in latex and if long title with meta data or cleaned title for publication
            should be included. If this is selected, make sure to comment in the rc commands in
            on top after matplotlib import!
    """
    # if save_publication:
    #     plt.rcParams.update({
    #         "text.usetex": True,
    #         "font.family": "sans-serif",
    #         "font.sans-serif": ["Helvetica"]})

    d = get_plotting_layout(layout)

    if not save_CNN:
        fig = plt.figure(1, dpi=d['dpi'])#
        if save_publication:
            fig.set_size_inches(w=6, h=4)
        ax = plt.gca()
    else:
        fig = plt.figure(frameon=False, dpi=600)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

    # -- plot simulation data
    if plt_type == 'contf' or save_CNN:
        # ax.contourf(fg, kg, fft_data, plt_res, cmap='seismic')
        ax.contourf(fg, kg, fft_data, plt_res, cmap='Spectral', zorder=-40)
        # ax.contourf(fg, kg, fft_data, plt_res, cmap='Greys', zorder=-40)
        if save_publication:
            ax.set_rasterization_zorder(-5)
    elif plt_type == 'cont' and not save_CNN:
        # ax.contour(fg, kg, fft_data, plt_res, cmap='seismic')
        ax.contour(fg, kg, fft_data, plt_res, cmap='Spectral')
    else:
        print('Specify plotting type')

    # -- plot nms maxima
    if add_scatter and not save_CNN:
        ax.scatter(fg[0, x], kg[y, 0], color='lime',
                   marker='2', alpha=0.5, label='NMS maxima', zorder=2)  # alpha was 0.7

    if add_fit and not save_CNN:
        coeffs = extract_features(func, x, y, fg, kg)
        x_plot = np.linspace(0, 7000)
        ax.plot(x_plot, func(x_plot, *coeffs), color='b')

    # -- plot analytical data
    if add_analytical and not save_CNN:
        if ka_cr is not None:
            for row in range(ka_cr.shape[1]):  # [0, 1, 5, 6]:
                ax.plot(
                    ka_cr[:, row], fa_cr[:, row],
                    color=d['t']['c'],
                    linestyle=d['t']['ls'],
                    dashes=d['t']['d'],
                    linewidth=d['t']['lw'],
                    alpha=d['t']['a'],
                    # label=f'Cr-{mn_cr[row]}'
                    label=f'Coating-{mn_cr[row]}'
                )
        if ka_zy is not None:
            for row in range(ka_zy.shape[1]):
                ax.plot(
                    ka_zy[:, row],
                    fa_zy[:, row],
                    color=d['b']['c'],
                    linestyle=d['b']['ls'],
                    dashes=d['b']['d'],
                    linewidth=d['b']['lw'],
                    alpha=d['b']['a'],
                    # label=f'Zy4-{mn_zy[row]}'
                    label=f'Plate-{mn_zy[row]}'
                )
        if ka_zy4cr is not None:
            for row in range(ka_zy4cr.shape[1]):
                ax.plot(
                    ka_zy4cr[:, row],
                    fa_zy4cr[:, row],
                    color=d['tb']['c'],
                    linestyle=d['tb']['ls'],
                    dashes=d['tb']['d'],
                    linewidth=d['tb']['lw'],
                    alpha=d['tb']['a'],
                    # label=f'Zy4Cr-{mn_zy4cr[row]}'
                    label=f'Plate + Coating-{mn_zy4cr[row]}'
                )

    if axis:
        plt.axis([0, 17500, 0, 2.5E7])  # [0, 900, 0, 2.5e6]
    if m_axis is not None:
        plt.axis(m_axis)
    if save_CNN:
        plt.axis([0, 8000, 0, 2.5E7])

    # if save_publication:
    # # add this only to include some labeling in plot for some exports
    #     plt.text(7500, 1.4E7, r' $A_{0,plate}$ \& $S_{0,plate}$' + '\n' + r'$+ B_{0,combined}$')
    #     # plt.text(7500, 1.4E7, r' $A_{0,plate}$ \& $S_{0,plate}$ \\ $+ B_{0,combined}$')
    #     plt.text(2300, 0.6E7, r' $A_{0,coating}$')
    #     plt.text(3600, 2E7, r' $S_{0,coating}$')

    if not save_CNN:
        plt.xlabel(r'Wave number $k$ in $1/m$' + '\n')
        plt.ylabel(r'Frequency $f$ in $Hz$')

        cg_top_left = get_values_from_sim_info(sim_info, attribute='cg_top_left',
                                            factor=1E3, formatting='.1f')

        cg_top_right = get_values_from_sim_info(sim_info, attribute='cg_top_right',
                                                factor=1E3, formatting='.1f')

        cg_bevel = get_values_from_sim_info(sim_info, attribute='cg_bevel',
                                            factor=1E3, formatting='.1f')

        cg_width = get_values_from_sim_info(sim_info, attribute='cg_width',
                                            factor=1E2, formatting='.1f')

        cg_gap_depth = get_values_from_sim_info(sim_info, attribute='cg_gap_depth',
                                                factor=1E6, formatting='.5f')

        try:
            if save_publication == True:
                # title = r'Dispersion Graph for uniform coating thickness ' \
                #         r'of ' + str(int(sim_info['c_height']*1E6)) \
                #         + r'$\mu$m' + '\n'
                pass
            else:
                title = 'res=' + str(plt_res) \
                        + ', trsld=' + str(clip_threshold) \
                        + ', c-height=' + str(int(sim_info['c_height']*1E6)) \
                        + ', l=' + str(sim_info['p_width']) + '\n' \
                        + 'cg_tl=' + cg_top_left \
                        + ', cg_bevel=' + cg_bevel \
                        + ', cg_width=' + cg_width \
                        + ', cg_tr=' + cg_top_right \
                        + ', cg_depth=' + cg_gap_depth + '\n' \
                        + sim_info['job_name'] + ' ' + fitting_type

        except TypeError:
            # # old title:
            title = 'res=' + str(plt_res) \
                    + ', trsld=' + str(clip_threshold) \
                    + ', c-height=' + str(int(sim_info['c_height'] * 1E6)) + 'microns' \
                    + ' l=' + str(sim_info['p_width']) + '\n' \
                    + 't_s=' + str(sim_info['t_sampling']) \
                    + ', t_period=' + str(sim_info['t_period']) \
                    + ', t_max=' + str(sim_info['t_max']) + '\n' \
                    + sim_info['job_name'] + ' ' + fitting_type

        if not save_publication:
            plt.title(title)

        h, l = ax.get_legend_handles_labels()
        unique_labels = [lbl for i, lbl in enumerate(l) if i == 0
                         or (lbl[0:4] != l[i - 1][0:4] and lbl[3] != 'S')]
        unique_idx = [l.index(lbl) for lbl in unique_labels]
        unique_labels = [lbl[0:lbl.find('-')] for lbl in unique_labels]

        # unique_idx = [0, 11, 31]  # hardcoding sucks, but for for 200_1_1_1_000
        # unique_labels = ['Coating', 'Plate', 'Plate + Coating']  # hardcoding sucks, but for for 200_1_1_1_000

        plt.legend([h[idx] for idx in unique_idx], unique_labels,
                   title='Analytical Dispersion Curves')

        if save_flag:
            if save_publication:

                # # this can be used as long as there is no contourf plot included,
                # #   otherwise use the .pdf option
                # # pub_out_name = str(output_file.parent) + '/' + \
                # #                str(int(sim_info['c_height'] * 1E6)) \
                # #                + '_sim_disp.pgf'
                # # pub_out_name = f'200_sim_disp.pgf'
                # pub_out_name = str(output_file.parent) + '/' + \
                #                'analytical_curves_200_microns_coating.pgf'
                # plt.savefig(pub_out_name, backend='pgf', format='pgf', dpi=50)
                # plt.savefig(pub_out_name[0:-3] + 'png', backend='pgf', format='png', dpi=300)

                # # use this option for contourf plots, then import .pdf into Inkscape
                # #  with the Poppler/Cairo option to keep the Latex CMU fonts and
                # #  then save with 'Convert texts into paths'
                # pub_out_name = str(output_file.parent) + '/' + \
                #                str(int(sim_info['c_height'] * 1E6)) \
                #                + '_sim_disp.pdf'
                #                # + '_sim_and_analytical_disp.pdf'
                # plt.savefig(pub_out_name, backend='pgf', format='pdf', dpi=300)

                # # use this to plot analytical and simulated curves, then import .pdf into Inkscape
                # #  with the Poppler/Cairo option to keep the Latex CMU fonts and
                # #  then save with 'Convert texts into paths'
                pub_out_name = str(output_file.parent) + '/' + \
                               str(int(sim_info['c_height'] * 1E6)) \
                               + '_sim_and_analytical_disp.pdf'
                plt.savefig(pub_out_name, backend='pgf', format='pdf', dpi=300)
                plt.savefig(pub_out_name[0:-3] + 'png', backend='pgf', format='png', dpi=300)
            else:
                plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)

    else:
        if not save_publication:
            plt.savefig(output_file, dpi=600)
        else:
            pub_out_name = str(output_file.parent) + '/' + \
                           str(int(sim_info['c_height'] * 1E6)) \
                           + '_grays_cnn_disp.pdf'
            plt.savefig(pub_out_name, backend='pgf', format='pdf', dpi=300)

    if show_plot and not save_publication:
        plt.show()

    plt.close(1)


def store_fft_data(
        data_file,
        data_path,
        abs_fft_data,
        fg,
        kg,
        sim_info,
        snr: int = None
) -> None:
    """
    Save 2D-FFT data to files, as well es corresponding frequency and wavenumber grid

    Output: three .csv files containing the obtained data (only if not already existing)

    args:
        - data_file - name of the disp input file
        - abs_fft_data - numpy array containing the 2D-FFT data
        - fg - frequency grid
        - kg - wavenumber grid
        - sim_info - dictionary with simulation information
        - snr - signal-to-noise ratio used in the data (might be none for no added noise)
    """
    file_name, _ = path.splitext(data_file)

    if snr is not None:
        output_name_spectrum = file_name + '.csv'
        output_name_fg = file_name + '.csv'
        output_name_kg = file_name + '.csv'
    else:
        output_name_spectrum = file_name + '_2dfft_sp_' + str(sim_info['c_height']) + '.csv'
        output_name_fg = file_name + '_2dfft_fg_' + str(sim_info['c_height']) + '.csv'
        output_name_kg = file_name + '_2dfft_kg_' + str(sim_info['c_height']) + '.csv'

    if path.exists(data_path / output_name_spectrum):
        pass
    else:
        if snr is not None:
            # save only new spectrum for new noisy data
            np.savetxt(data_path / output_name_spectrum, abs_fft_data, delimiter=',')
        else:
            np.savetxt(data_path / output_name_spectrum, abs_fft_data, delimiter=',')
            np.savetxt(data_path / output_name_fg, fg, delimiter=',')
            np.savetxt(data_path / output_name_kg, kg, delimiter=',')
        print(f'Output file of 2D-FFT {data_file} has been created.')


def get_output_name(data_path, file_name, c_thick, plt_type,
                    plt_res, save_cnn, snr: int = None, kernel: int = None) -> pathlib.Path:
    """
    create the name of the plot output file. Name will be the same as
    the .csv data input file but with time stamp of creation added

    example output:
    -> 09-20_19-23-59_max_analysis_job_04-23_13-35-39_contf_800.png

    (old/deprecated) -> dispersion_graph_2021-04-23_13-35-39_contf_800.png

    args:
        - data_path: path to simulation data (.csv data)
        - file_name: name of simulation file which is analyzed
        - c_thick: thickness of coating
        - plt_type: which plotting type is used (cont or contf)
        - plt_res: which plotting resolution is used
        - snr: signal-to-noise ratio in dB if given for noisy sample
        - kernel: kernel of NMS for noisy sample if given
    """
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    today = date.today()

    if snr is None:
        noise_tail = ''
    else:
        noise_tail = f'_n_{snr}_k_{kernel}'

    output_file = file_name + '_' \
                  + str(today.strftime("%m-%d")) \
                  + '_' \
                  + str(current_time) \
                  + '_' \
                  + plt_type \
                  + '_' \
                  + str(plt_res) \
                  + '_' \
                  + str(int(c_thick*1E6)) \
                  + f'{noise_tail}.png'
    if save_cnn:
        output_file = file_name + '_' \
                      + str(today.strftime("%m-%d")) \
                      + '_' \
                      + str(current_time) \
                      + '_' \
                      + plt_type \
                      + '_' \
                      + str(plt_res) \
                      + '_' \
                      + str(int(c_thick * 1E6)) \
                      + f'{noise_tail}_cnn' \
                      + '.png'
    return data_path / output_file


def my_unsqueeze_2_torch(image) -> torch.Tensor:
    """
    converts np image array of size (n,m) into torch.Tensor of size (1,1,n,m)
    """
    return torch.Tensor(np.expand_dims(np.expand_dims(image, axis=0), axis=0))


def my_squeeze_2_np(image) -> np.array:
    """
    converts image tensor of size (1,1,n,m) into np.array of size (n,m)
    """
    return np.squeeze(np.squeeze(np.array(image)))


def non_maximum_suppression(
        fft_data: np.ndarray,
        data_file: str,
        sim_path: Union[str, pathlib.Path],
        kernel: int = 21,
        gradient: list = None,
        x_lim: list = None,
        y_lim: list = None,
        clip_tr: float = 1.0,
        plot_flag: bool = False,
        save_flag: bool = False,
        dpi: int = 600,
) -> tuple:
    """
        do non-maximum-suppression using Pytorch

        returns:
            - fft_data - clipped and removed all data below median of
                input fft_data,
            - x - x-coords of NMS maxima (in pixel space),
            - y - y-coords of NMW maxima (in pixel space)

        args:
            - fft_data - 2D-FFT data matrix
            - data_file - name of the data file for saving the x-y
                coordinates after non-maximum suppression
            - json_info_file - simulation information dictionary
            - sim_path - path to simulation files
            - kernel=21 - size of kernel for max pooling, needs to be odd,
                61 and 11 look good too
            - gradient = [gl, gu] (list): specify gradient boundaries (lower
                and upper gradient) between which the NMS data should be.
                Ignore data outside of it.
            - x_lim (list): lower and upper limit for extracted max pool
                coordinates on x axis
            - y_lim (list): lower and upper limit for extracted max pool
                coordinates on y axis
            - clip_tr=1 - threshold for clipping, a high clipping value
                (e.g. 1 = clipping off)
            - plot_flag=False - specify if the extration cones should be
                visualized. Watch out, here the pixel values are plotted
            - save_flag=False - specify if NMS coords should be stored
    """
    if x_lim is None:
        x_lim = [50, 300]
    if y_lim is None:
        y_lim = [250, 1800]
    if gradient is None:
        gradient = [2.8, 9]


    # remove all entries smaller than median
    fft_fltr = np.multiply(fft_data.copy(), fft_data > np.median(fft_data))
    fft_fltr = my_unsqueeze_2_torch(fft_fltr)

    # print(fft_fltr.shape)

    mp = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=int(kernel // 2))

    fft_mp = mp(fft_fltr)
    fft_bin = torch.eq(fft_mp, fft_fltr)  # check elementwise if elements equal
    fft_out = np.multiply(my_squeeze_2_np(fft_bin), fft_data)  # extract only MAXs from fft_data

    fft_p = fft_out

    # print(f"fft_p.shape = {fft_p.shape}")

    # -- get coordinates of maxima
    b = np.where(fft_p != 0)

    # print(f"len(b[0]) = {len(b[0])}")
    # print(f"b = {b}")

    c = np.zeros((len(b[0])))  # intensity values at b locations
    # k = 26000 #250
    for i, (x, y) in enumerate(zip(b[0], b[1])):
        c[i] = fft_p[x, y]
    idx = np.argsort(c)
    # 0.2 looks like it gets the job done, but maybe needs to be tuned later on
    idx_reduced = [i for i in idx if c[i] > 0.2*np.mean(c)]

    # import pdb; pdb.set_trace()

    # -- define outputs - remember x and y are swapped for images in python
    x = np.flip(b[1][idx_reduced])
    y = np.flip(b[0][idx_reduced])

    # remove too low and too high values first
    x_new, y_new = [], []
    for (x_elem, y_elem) in zip(x, y):
        if x_lim[0] < x_elem < x_lim[1] and y_lim[0] < y_elem < y_lim[1]:
            x_new.append(x_elem)
            y_new.append(y_elem)
    x = x_new
    y = y_new

    # remove values which are outside a cone
    x_new, y_new = [], []
    gl, gu = gradient[0], gradient[1]
    for (x_elem, y_elem) in zip(x, y):
        if gl * x_elem < y_elem < gu * x_elem:
            x_new.append(x_elem)
            y_new.append(y_elem)
    x = x_new
    y = y_new

    print(f'> number of maxima detected in nms: {len(x_new)}')

    x, y = np.array(x), np.array(y)

    if plot_flag:
        plt.figure(1, dpi=dpi)
        plt.contourf(fft_out, 200, cmap='Spectral')
        plt.scatter(x, y, color='lime', marker='2', alpha=0.7, s=0.5)
        xl_plot = np.linspace(0, 700)
        xu_plot = np.linspace(0, 300)
        plt.plot(xl_plot, gl * xl_plot)
        plt.plot(xu_plot, gu * xu_plot)
        plt.show()

    out = np.concatenate((x, y))
    if save_flag:
        xy_file_name = data_file[0:-9] + 'xy.txt'
        try:
            with open(sim_path / xy_file_name, 'w') as f:
                np.savetxt(f, out, delimiter=',')
        except Exception:
            with open(xy_file_name, 'w') as f:
                np.savetxt(f, out, delimiter=',')

    # -- clip values of 2dfft
    fft_data = np.clip(fft_data, 0, clip_tr)
    fft_data = np.multiply(fft_data.copy(), fft_data > np.median(fft_data))

    return fft_data, x, y


def cut_quadrants(fg, kg, abs_fft_data) -> tuple:
    """
        cuts the absFftArray and remove 2nd, 3rd and 4th quadrant from
        2D-FFT array (only keep 1st quadrant) before plotting it
        - this helps to get better resolution (maxima on spots which make
          physically no sense or contain phase information are ignored)
          and increases computation speed since less data needs to be
          processed

        args:
            - fg
            - kg
            - abs_fft_data
    """
    quarter_y = fg.shape[0] // 2
    quarter_x = fg.shape[1] // 2

    fg = fg[quarter_y:,
         quarter_x:int(1.5 * quarter_x)]
    kg = kg[quarter_y:,
         quarter_x:int(1.5 * quarter_x)]

    abs_fft_data = abs_fft_data[quarter_y:,
                   quarter_x:int(1.5 * quarter_x)]
    return fg, kg, abs_fft_data


# def load_analytical_data(cur_path, coat_thick: float = 0.6):
#     """
#     load analytical obtained dispersion data from .npy files as well as
#     mode names from .csv files
#
#     Returns: wave number array, frequency array and mode names for Cr, Zy4-Cr and Zy4:
#         ka_cr, fa_cr, mn_cr, ka_zy4cr, fa_zy4cr, mn_zy4cr, ka_zy, fa_zy, mn_zy
#
#     args:
#         - cur_path - string or path object of current working directory
#                         with 'analytical_disp_curves' as a subdirectory
#         - coat_thick - coating thickness in mm - specifies which analytical
#                         curves should be loaded. Default = 0.6.
#                         Needs to be in [0.1, 0.3, 0.5, 0.6] for now,
#                         add more layer thicknesses later
#     """
#     analytical_path = cur_path / 'analytical_disp_curves'
#     # MX EDIT: filenames need to be changed back
#     with open(analytical_path / 'Cr_dispersion_curves' / f'Chrome_{coat_thick}_mm_dispersion_curves'
#               / f'Chrome_{coat_thick}_mm_dispersion_data_analytically_k_.npy', 'rb') as f:
#         ka_cr = np.load(f)
#     with open(analytical_path / 'Cr_dispersion_curves' / f'Chrome_{coat_thick}_mm_dispersion_curves'
#               / f'Chrome_{coat_thick}_mm_dispersion_data_analytically_f_.npy', 'rb') as f:
#         fa_cr = np.load(f)
#     with open(analytical_path / 'Cr_dispersion_curves' / f'Chrome_{coat_thick}_mm_dispersion_curves'
#               / f'Chrome_{coat_thick}_mm_dispersion_data_analytically_mode_names_.csv', 'r') as f:
#         mn_cr = f.readlines()
#
#     with open(analytical_path / 'Zy4Cr_dispersion_curves' / f'{coat_thick}_Chrome'
#               / f'Zy4_3_Cr_{coat_thick}_dispersion_data_analytically_k_.npy', 'rb') as f:
#         ka_zy4cr = np.load(f)
#     with open(analytical_path / 'Zy4Cr_dispersion_curves' / f'{coat_thick}_Chrome'
#               / f'Zy4_3_Cr_{coat_thick}_dispersion_data_analytically_f_.npy', 'rb') as f:
#         fa_zy4cr = np.load(f)
#     with open(analytical_path / 'Zy4Cr_dispersion_curves' / f'{coat_thick}_Chrome'
#               / f'Zy4_3_Cr_{coat_thick}_dispersion_data_analytically_mode_names_.csv', 'rb') as f:
#         mn_zy4cr = f.readlines()
#
#     with open(analytical_path / 'Zy4_dispersion_curves_DC'
#               / 'Zirc_dispersion_data_analytically_k_.npy', 'rb') as f:
#         ka_zy = np.load(f)
#     with open(analytical_path / 'Zy4_dispersion_curves_DC'
#               / 'Zirc_dispersion_data_analytically_f_.npy', 'rb') as f:
#         fa_zy = np.load(f)
#     with open(analytical_path / 'Zy4_dispersion_curves_DC'
#               / 'Zirc_dispersion_data_analytically_mode_names_.csv', 'r') as f:
#         mn_zy = f.readlines()
#
#     mn_cr = [i.rstrip('\n') for i in mn_cr]
#     mn_zy4cr = [i.rstrip('\n') for i in mn_zy]
#     mn_zy = [i.rstrip('\n') for i in mn_zy]
#
#     return ka_cr, fa_cr, mn_cr, ka_zy4cr, fa_zy4cr, mn_zy4cr, ka_zy, fa_zy, mn_zy


def load_input_data(data_path):
    """
    load simulation data for 2D-FFT into numpy array and simulation information
    into dictionary
    """
    # -- load data to process in 2dfft
    # print(data_path)
    # print(get_newest_file_name(data_path, extension='.csv'))
    # print(get_newest_file_name(data_path, extension='.json'))
    try:
        data_file, _ = get_newest_file_name(data_path, extension='.csv')
        inputData = np.genfromtxt(open(data_path / data_file), delimiter=',')
    except (NameError, TypeError):
        print('No .csv file with appropriate naming in directory!')

    # -- load json information file
    try:
        json_info_file, _ = get_newest_file_name(data_path, extension='.json')
        with open(data_path / json_info_file) as info_file:
            sim_info = json.load(info_file)
    except NameError:
        print('No .json info file with appropriate naming in directory!')

    return inputData, data_file, sim_info


def load_2dfft_processed_data(fn, data_path):
    """
    Loads 2dfft processed data in case it exists, so Fourier transformation
    does not need to be applied every single time again (slow for large files)

    returns:

    args:
        - fn (string): file name of one of the 2dfft files.
            Example: 09-23_00-54-33_max_analysis_job_disp_2dfft_sp_0.0001
            Hint: the .._sp_.. can be either ..__kg__.. or ..__fg__..
        - data_path (string or Path object): path where to data can be found
    """
    try:
        json_info_file, _ = get_newest_file_name(data_path, extension='.json')
        assert fn[0:fn.find('_disp')] + '_info' == json_info_file[0:-5], \
            'JSON info file and data do not match!'
        with open(data_path / json_info_file) as info_file:
            sim_info = json.load(info_file)
    except NameError:
        print('No .json info file with appropriate naming in directory!')

    # -- load 2DFFT data
    fn = fn[0:43] + 'fg' + fn[45::]
    with open(data_path / fn) as f:
        fg = np.genfromtxt(f, delimiter=',')
    fn = fn[0:43] + 'kg' + fn[45::]
    with open(data_path / fn) as f:
        kg = np.genfromtxt(f, delimiter=',')
    fn = fn[0:43] + 'sp' + fn[45::]
    with open(data_path / fn) as f:
        abs_fft_data = np.genfromtxt(f, delimiter=',')

    return fg, kg, abs_fft_data, sim_info


def delete_unwanted_files(white_list, cur_path=None, del_disp: bool = True) -> None:
    """
    delete all files which's file extension is not given in the white list

    args:
        - file extensions which should not be deleted
            (usually .py, .csv, .json, ...)
        - cur_path: if provided, path to location where files should be deleted.
            If no path is passed, the current working directory is used
        - del_disp: bool = True: specify if the _disp.csv file should be deleted too
    """
    if cur_path is None:
        cur_path = pathlib.Path().absolute()

    files_in_folder = [f for f in listdir(cur_path) if isfile(join(cur_path, f))]
    for file in files_in_folder:
        _, file_extension = path.splitext(file)
        if file_extension not in white_list:
            remove(join(cur_path, file))
        if del_disp and file.find('_disp.csv') != -1:
            remove(cur_path / file)


def parse_input_variables(input_list):
    """
    COPY FROM RUN_SIMULATION.PY
    extracts the input variables for the Abaqus model from the command line input string and outputs them in
      an appropriate format. If no input is given, the respective default value will be used

    output: dictionary with all variables needed for simulation

    args:
        - input_list - list of strings with input arguments. Strings are in the form of 'variable_name=variable_value'

    WATCH OUT: THERE IS A DUPLICATE OF THIS FUNCTION IN RUN_SIMULATION.PY!!
    (duplicate needs to be there because needs to passed directly into Abaqus)
    """
    sd = {
        'coating_height': 600E-6,
        'plate_width': 0.09,
        'base_plate_height': 0.003,
        'coating_density': 7190.0,
        'coating_youngs_mod': 279E9,
        'coating_pois_rat': 0.21,
        'base_plate_density': 6560.0,
        'base_plate_youngs_mod': 99.3E9,
        'base_plate_pois_rat': 0.37,
        'cb_width': 0.005,
        'cg_width': 0.02,
        'cs_width': 0.055,
        'cg_top_left': 0.001,
        'cg_top_right': 0.001,
        'cg_bevel': 0.001,
        'cg_gap_depth': 0.00005,
        'ex_amp': 2e-06,
        'num_mesh': 1,
        't_max': 2E-8,
        't_sampling': 2E-8,  # 5E-8,  # 2E-7,
        't_period': 9.2E-5,
        'run_it': True,
        'save_inp': False
        # 'cores': None
    }
    for elem in input_list:
        # print(elem)
        if elem == 'run_it' and input_list[elem] == False:
            sd['run_it'] = False
        elif elem == 'save_inp':
            sd['save_inp'] = True
        elif '=' in elem:
            eq_idx = elem.find('=')
            if elem[0:eq_idx] in sd:
                elem.strip('\\')
                sd[elem[0:eq_idx]] = float(elem[eq_idx+1:])
    return sd


def send_slack_message(
        message: str = "Abaqus pipeline completed"
):
    """
    Send push notifications to Slack via Slack bot/API

    The URL is unique to the slack bot created and this means tied to a
    certain workspace. In case this should work on a new workspace,
    make sure there is an app created so that the Webhook URL can be accessed:
    Workspace>Settings>Manage Apps>Add new>...

    How to create a Webhook URL is explained here:
    https://api.slack.com/messaging/webhooks

    If you are using this and do not know what this means, I (Max) would be happy when
    you would comment out this function/url/the request.post() command so I do not
    receive all the simulation messages from you to my personal Slack workspace
    -> this is not needed since I am importing my personal url from a file which
        I will not share with you :)

    args:
        - message - string - message which should be send to smartphone/Slack desktop
    """
    payload = '{"text":"%s"}' % message
    url = slackurl  # example: 'https://hooks.slack.com/services/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'
    response = requests.post(url, data=payload)


def invert_2dfft(fg, kg, fft_abs, sim_info) -> Tuple[np.ndarray, float, float, int, int]:
    """
    Inverts the 2D-FFT data, such that the representation in the frequency-wavenumber is
    converted into the time-displacement domain

    :arg:
        - fg - ndarray - frequency grid where unique frequencies are along axis 0
        - kg - ndarray - wavenumber grid where unique wavenumbers are along axis 1
        - fft_abs - ndarray - 2D-FFT transformed data from simulation data
        - sim_info - dict - simulation information file with meta-information

    :return:
        - displacement_x_time - ndarray - array containing the time x displacement values reconstructed from 2D-FFt
        - dt - float - sampling time of simulation
        - dx - float - spatial sampling location difference
        - Nt - int - number of samples in time
        - Nx - int - number of sampling locations in space
    """
    fft_original = np.fft.fftshift(fft_abs)

    ## WATCH OUT I MIXED K AND F IN THE MESHGRID FUNCTION IN THE 2D-FFT FUNCTION! MIX IT BACK HERE
    k = fg[0, :]  # first row contains unique frequencies
    f = kg[:, 0]  # first columns contains unique wavenumbers

    ny_f, ny_k = f[-1], k[-1]

    dt = 1.0 / (2 * ny_f)
    dx = 1.0 / (2 * ny_k)

    Nt = f.shape[0] * 2  # quadrant 2,3,4 were cut, so Nt is actually twice as big
    Nx = k.shape[0] * 2  # quadrant 2,3,4 were cut, so Nx is actually twice as big

    tMax = Nt * dt
    xMax = Nx * dx

    display_2dfft_values = False
    if display_2dfft_values:
        print(f"Nt = {Nt}, Nx = {Nx}")
        print(f"f[-3::] = {f[-3::]}")
        print(f"k[-3::] = {k[-3::]}")
        print(f"f[0:3] = {f[0:3]}")
        print(f"k[0:3] = {k[0:3]}")
        print(f"ny_f = {ny_f}")
        print(f"ny_k = {ny_k}")
        print(f"dt = {dt}")
        print(f"dx = {dx}")
        print(f"tMax = {tMax}")
        print(f"xMax = {xMax}")

    assert sim_info['msmt_len'] == round(xMax, 2), \
        f"There is a missmatch in the measurement length!" \
        f"\n> sim_info['msmt_len'] = {sim_info['msmt_len']} " \
        f"not equal to xMax = {xMax}, with dx = {dx}!"
    assert sim_info['t_sampling'] == round(dt, 8), \
        f"Sampling times do not match" \
        f"\n> sim_info['sampling_time'] = {sim_info['sampling_time']} " \
        f"not equal to xMax = {dt}!"

    # apply ifft2 to get back the time displacement data (invert 2D-FFT)
    displacement_x_time = np.fft.ifft2(fft_original)

    return displacement_x_time, dt, dx, Nt, Nx


def reapply_2dfft(displacement_x_time, dt, dx, Nt, Nx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-apply 2D-FFT transform to the noisy displacement data in the time-displacement domain which was
    originally transformed back from the frequency-wavenumber space

    :arg:
        - time_x_displacement - ndarray - array containing the time x displacement values reconstructed from 2D-FFT
        - dt - float - sampling time of simulation
        - dx - float - spatial sampling location difference
        - Nt - int - number of samples in time
        - Nx - int - number of sampling locations in space

    :returns:
        fg: mesh array with frequencies
        kg: mesh array with wave numbers
        abs_fft_data: absolute values after 2D-FFT transform
    """

    # Get k, f intervals
    ny_f = 1.0 / (2 * dt)
    ny_k = 1.0 / (2 * dx)

    k = np.linspace(-ny_k, ny_k, num=Nx)
    f = np.linspace(-ny_f, ny_f, num=Nt)

    # apply 2D FFT
    start_time = time.time()
    fft_data = np.fft.fftshift(np.fft.fft2(displacement_x_time))  # *dx*dt
    print("--- np 2dfft time: %s seconds ---" % (time.time() - start_time))

    fg, kg = np.meshgrid(k, f)  # f and k are mixed up here - leave mistake to be consistent with older data

    abs_fft_data = np.absolute(fft_data)  # for amplitude spectrum

    return fg, kg, abs_fft_data


def add_noise(d_x_t, snr_db: int = 80) -> np.ndarray:
    """
    Adds random noise to the displacement x time data to simulate real noisy measurements
    and accounts for imperfectness in simulation model

    :param
        - d_x_t: np.array - displacement x time array with respective sampling nodes on y-axis and time on x-axis
        - snr_db: int - signal-to-noise ratio in dB of signal and newly created noise
    :return:
        - d_x_t_noisy: np.array - same dimensions as d_x_t but with added random noise
    """
    noise = np.zeros(d_x_t.shape)
    mean = np.mean(d_x_t)
    max_value = np.amax(d_x_t)

    snr = np.power(10, snr_db / 10.0)  # convert snr from dB to fraction

    # sd = mean/snr
    sd = max_value.real/snr

    noise = np.random.normal(noise, sd, d_x_t.shape)

    d_x_t_noisy = d_x_t + noise

    print_values = False
    if print_values:
        print(d_x_t.shape)
        print(F"mean = {mean} -> {mean.real}")
        print(F"mean = {max_value} -> {max_value.real}")
        print(f"sd = {sd}")
        print(f"d_x_t_noisy.shape = {d_x_t_noisy.shape}")
        print(f"noise = \n{noise}")
        print(f"d_x_t_noisy =\n{d_x_t_noisy}")

    return d_x_t_noisy

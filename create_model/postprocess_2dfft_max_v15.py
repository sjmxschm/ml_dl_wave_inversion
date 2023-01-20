import pathlib

import numpy as np

import time

from os import listdir
from os.path import isfile, join

import json

try:
    from create_model.utils import (
        get_newest_file_name,
        plot_sim_and_analy_data,
        store_fft_data,
        get_output_name,
        non_maximum_suppression,
        cut_quadrants,
        load_analytical_data,
        load_input_data,
        load_2dfft_processed_data,
        lin_func_simple,
        lin_func,
        poly2_func,
        extract_features
    )
except ModuleNotFoundError:
    print('Make sure ''utils.py'' is in the right directory.\n'
          'Ignore this message when running this file on the cluster!')
    from utils import (
        get_newest_file_name,
        plot_sim_and_analy_data,
        store_fft_data,
        get_output_name,
        non_maximum_suppression,
        cut_quadrants,
        load_analytical_data,
        load_input_data,
        load_2dfft_processed_data,
        lin_func_simple,
        lin_func,
        poly2_func,
        extract_features
    )

''' 
This script extract the displacement information from the input textfile and generates the 2d-FFT graph

Execute this script from the command line by cd'ing into the current directory and typing 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
python postprocess_2dfft_max_v15.py
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
into the command line

In input dataFileName has the shape:
_______________________________________________________________________
|time 1  -- disp node 1 - disp node 2 - disp node 3 - ... disp node Nx|
|time 2  -- disp node 1 - disp node 2 - disp node 3 - ... disp node Nx|
|time 3  -- disp node 1 - disp node 2 - disp node 3 - ... disp node Nx|
|  .            .              .            .                 .       |
|  .            .              .            .                 .       |
|  .            .              .            .                 .       |
|time Nt -- disp node 1 - disp node 2 - disp node 3 - ... disp node Nx|
_______________________________________________________________________

make sure that there is no number in the name of the .csv file besides date
and time of creation and the date has the structure:
job_name_mm-dd_hh-mm-ss - Not anymore, now it is:
-> mm-dd_hh-mm-ss_max_analysis_job_disp

Script by Max Schmitz since 04/10/2021

Change log:
v5:
    - function added which automatically searches for the newest .csv file in working directory
v7:
    - print statement removed from script and moved into separate printing function
v8:
    - plotting parameters (contf or cont and resolution) are now saved into the plot's name 
    - simulation information data is loaded into file and displayed in heading of plot
v9:
    - saving the useful part of the 2D-FFT array to a .csv file
v10:
    - create function script to be imported by pipeline
    - add if __name__ = '__main__': function
    - msmt_len is not hardcoded anymore but read from the simulation info file directly
v12:
    - add clipping of 2dfft data to make plot clearer. Threshold for clipping needs to be tuned
    - now previously generated 2dfft.csv files can be in folder bc they are ignored now
    - file can be kept in analysis_2dfft folder and does not need to be copied to every sim folder
v13:
    - now analytical curves of pure Chrome and pure Zirconium-4 can be added to plot too
v14:
    - moved a bunch of code into separate functions to keep code cleaner
    - add maxpooling to get maxima via non maximum suppression (NMS)
    - clip values which are too high if necessary
v15:
    - add analytical curves of coated plate (Zy-4 with Cr-coating) to plots
    - move all helper functions to untils.py
'''


def apply_2dfft(input_data, sim_info):
    """
    TODO: move this function to separate file or to utils
    Apply 2D-FFT transform to the input data which has the shape as shown above.

    args:
        - input_data: raw displacement data from .csv file
        - sim_info: simulation input dict with meta infos

    returns:
        fg: mesh array with frequencies
        kg: mesh array with wave numbers
        abs_fft_data: absolute values after 2D-FFT transform
    """
    # Read data file
    Nt, Nx = input_data.shape
    Nx -= 1  # First column is step time - original -1
    Nt -= 1  # First is step 0

    disp_x_time = np.zeros((Nt, Nx))
    disp_x_time[:, :] = input_data[1:, 1:]

    # remove last column since it is not necessary and provides only NaN values
    disp_x_time = disp_x_time[:, 0:-1]

    # xMax is the horizontal distance from the first measurement node to the last one
    # xMax = 120.0e-3
    xMax = sim_info['msmt_len']
    tMax = input_data[-1, 0]

    # xx = np.linspace(11.5e-3, xMax, num=Nx)
    xx = np.linspace(0.0, xMax, num=Nx)  # specify locations of sampling nodes
    yt = np.linspace(0.0, tMax, num=Nt)  # better: take time from first row?!

    dx = xx[1] - xx[0]
    dt = yt[1] - yt[0]

    # Get k, f intervals
    ny_f = 1.0 / (2 * dt)
    ny_k = 1.0 / (2 * dx)

    k = np.linspace(-ny_k, ny_k, num=Nx)
    f = np.linspace(-ny_f, ny_f, num=Nt)

    # apply 2D FFT
    start_time = time.time()
    fft_data = np.fft.fftshift(np.fft.fft2(disp_x_time))  # *dx*dt
    print("--- np 2dfft time: %s seconds ---" % (time.time() - start_time))

    fg, kg = np.meshgrid(k, f)  # f and k are mixed up here - leave mistake to be consistant with older data

    abs_fft_data = np.absolute(fft_data)  # for amplitude spectrum
    # abs_fft_data = 20 * np.log10(np.absolute(fft_data))       # for amplitude spectrum in dB
    # absFftArray = np.absolute(fftArray)**2                    # for power spectrum

    return fg, kg, abs_fft_data


def postprocessing_2dfft(
        sim_path: str = None,
        plot: bool = False,
        save: bool = False,
        show: bool = False,
        clip_threshold: float = 1,
        fitting_style: str = 'lin_simple',
        add_analytical: bool = True,
        add_scatter: bool = True,
        add_fit: bool = True,
        plt_res: int = 50,
        plt_type: str = 'cont',
        m_axis: list = None,
        save_cnn_flag: bool = False,
        save_pub: bool = False,
        cluster: bool = True
):
    """
    function loads displacement data from .csv file and creates 2D-FFT transformation
     data from it. The 2D-FFT data (3 .csv files) is stored, plots can be created.

    make sure that the displacement .csv file and the simulation information .json
     file are inside the folder where this script was called

    args:
        - sim_path - string - name of folder where the data is stored
        - plot - boolean - specify if plot should be created
        - save - boolean - specify if plot and 2DFFT array should be saved
        - show - boolean - specify if plot should be shown
        (deprecated) - c_thickness - float - thickness of coating in mm to load corresponding analytical
            curve. Currently available are: [0.6,]
        - clip_threshold - float - values which are bigger than clip_threshold are reduced
            to clip_threshold. Big threshold (e.g. =1) means there is no threshold
        - fitting_style - str - specify which kind of funtion should be plotted
            options are: 'lin' (a*x+b), 'lin_simple' (a*x), 'poly2' (a*x**2+b*x+c)
        - add_analytical - boolean - specify if analytical curves should be added to plot
        - add_scatter - boolean - specify if nms data should be added to plot
        - add_fit - boolean - specify if fitted function should be added to plot
        - plt_res - int - number of contour levels for contour plot can be between 50 and 800,
            usually from [100,300,500,800]
        - plt_type - 'cont' or 'contf' - specify which plot should be created
        - m_axis - list - list with 4 numbers specifying the axis limits of the output plot
        - save_CNN_flag - bool - influences how data is plotted and stored for use in CNN later
        - save_pub - bool - specifies if plots should be stored as .pgf for publication in latex
            and which title format (meta information or cleaned) should be used for the dispersion
            plot
        - cluster - bool - specifies  script runs on the cluster or no (influence from where
            data is loaded)
    """

    # -- Specify folder to work in/where data lays
    cur_path = pathlib.Path(__file__).parent.resolve()
    # handle that there is no sim_path on cluster
    if cluster and sim_path is None:
        # data_path = cur_path.parent.resolve()  # was it before
        data_path = cur_path
    elif cluster and sim_path is not None:
        data_path = sim_path
    else:
        # data_path = pathlib.Path(__file__).parents[1].resolve() / 'analysis_2dfft'
        #data_path = pathlib.Path('C:\\Users\\Max\\Documents') / 'analysis_2dfft'
        data_path = pathlib.Path(r"C:\Users\Max\OneDrive\Documents\Uni Gatech MSC\A Lab Research Wave CEE\A Journal Paper\ml_dl_wave_inversion\create_model") / "2dfft_data_selected"
        data_path = data_path / sim_path

    # -- Specify output plot
    plot_flag = plot
    save_flag = save
    show_plot = show
    c_t = clip_threshold
    plt_res = plt_res
    plt_type = plt_type

    # check if there is 2D-FFT processed data already existing
    fn, is_transformed = get_newest_file_name(
        data_path,
        job_name='max_analysis_job',
        extension='.csv'
    )
    print('fn, data_path')
    print(fn, data_path)
    if is_transformed:
        print('-> Found transformed 2D-FFT data. Will load this data!')
        fg, kg, abs_fft_data, sim_info = load_2dfft_processed_data(fn, data_path)
        data_file = fn[0:37]

    else:
        print('-> 2D-FFT processing on raw input data started.')
        input_data, data_file, sim_info = load_input_data(data_path)
        fg, kg, abs_fft_data = apply_2dfft(input_data, sim_info)
        fg, kg, abs_fft_data = cut_quadrants(fg, kg, abs_fft_data)

    output_file = get_output_name(
        data_path,
        sim_info['job_name'],
        sim_info['c_height'],
        plt_type, plt_res, save_cnn_flag
    )

    if format(sim_info['c_height'] * 1E3, ".2f")[2] == '0':
        thick = format(sim_info['c_height'] * 1E3, ".2f")  # convert from m to mm eg. 0.0006m->0.6mm
    else:
        thick = format(sim_info['c_height'] * 1E3, ".1f")

    if cluster:
        a_path = pathlib.Path('/storage/home/hcoda1/8/mschmitz7/scratch/analytical_disp_curves/').resolve()
        print('cluster: a_path = ' + str(a_path))
    else:
        a_path = pathlib.Path(__file__).parents[1].resolve() / 'analytical_disp_curves'
        print('a_path = ' + str(a_path))

    try:
        cr_path = a_path / 'Cr_dispersion_curves' / f'Chrome_{thick}_mm_dispersion_curves'
        ka_cr, fa_cr, mn_cr = load_analytical_data(cr_path, f'Chrome_{thick}_mm')
    except FileNotFoundError:
        ka_cr, fa_cr, mn_cr = None, None, None

    try:
        zy_path = a_path / 'Zy4_dispersion_curves_DC'
        ka_zy, fa_zy, mn_zy = load_analytical_data(zy_path, 'Zirc')
    except FileNotFoundError:
        ka_zy, fa_zy, mn_zy = None, None, None

    try:
        zy4cr_path = a_path / 'Zy4Cr_dispersion_curves' / f'{thick}_Chrome'
        ka_zy4cr, fa_zy4cr, mn_zy4cr = load_analytical_data(zy4cr_path, f'Zy4_3_Cr_{thick}')
    except FileNotFoundError:
        ka_zy4cr, fa_zy4cr, mn_zy4cr = None, None, None

    abs_fft_data, x, y = non_maximum_suppression(
        abs_fft_data,
        data_file,
        data_path,
        clip_tr=c_t,  # 1, 0.0002, 0.0001,
        kernel=15,
        save_flag=save_flag,
        plot_flag=False,
    )

    if fitting_style == 'lin':
        func = lin_func
    elif fitting_style == 'lin_simple':
        func = lin_func_simple
    elif fitting_style == 'poly2':
        func = poly2_func
    elif fitting_style == 'lin_local':
        func = lin_func
    else:
        print('Fitting style not valid. Please specify valid fitting style!')

    feat_lin = extract_features(func, x, y, fg, kg)
    feat_file_name = data_file[0:data_file.find('disp')] + f'features_{fitting_style}.txt'
    if save_flag:
        with open(data_path / feat_file_name, 'w') as f:
            np.savetxt(f, feat_lin, delimiter=',')

    if plot_flag or save_cnn_flag:
        plot_sim_and_analy_data(
            fg,
            kg,
            abs_fft_data,
            plt_res,
            ka_cr,
            fa_cr,
            mn_cr,
            ka_zy4cr,
            fa_zy4cr,
            mn_zy4cr,
            ka_zy,
            fa_zy,
            mn_zy,
            sim_info,
            output_file,
            plt_type,
            x,
            y,
            axis=False,
            m_axis=m_axis,
            clip_threshold=c_t,
            add_analytical=add_analytical,
            add_fit=add_fit,
            add_scatter=add_scatter,
            func=func,
            fitting_type=fitting_style,
            save_CNN=save_cnn_flag,
            save_flag=save_flag,
            show_plot=show_plot,
            save_publication=save_pub
        )

    if save_flag and not is_transformed:
        store_fft_data(
            data_file,
            data_path,
            abs_fft_data,
            fg,
            kg,
            sim_info
        )


if __name__ == "__main__":
    '''
    Call postprocessing_2dfft from above as usually
    '''
    start_time = time.time()

    postprocessing_2dfft(
        # '030_1_1_1_000',
        # '40._1._4._1._0',
        # '170._1._1._1._0',
        '600._1._1._1._0',
        # '200._1._2._1._70',
        #'010_1_1_1_000',
        # '200_1_3_1_100',
        # '200_1_1_1_000',
        # '300_1_2_1_000',
        # '300_1_3_1_200',
        plot=True,
        show=True,
        save=True,
        add_analytical=False,
        add_scatter=False,
        add_fit=False,
        fitting_style='lin_local',
        clip_threshold=0.0001,  # 0.01,  # 0.001,
        m_axis=[0, 10500, 0, 2.5E7],  # [0, 17500, 0, 2.5E7],
        plt_res=300,  # 300, # res for cnns and most stuff was 300, 40 for export
        plt_type='contf',  # 'analytical',  # 'contf',
        save_cnn_flag=False,
        save_pub=True,
        cluster=False,
    )

    print("--- running time: %s seconds ---" % (time.time() - start_time))

    # WG0892


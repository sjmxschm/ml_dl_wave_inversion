import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime

try:
    from create_model.utils import get_newest_file_name
except ModuleNotFoundError:
    from utils import get_newest_file_name
"""
This file helps visualizing different structures in the input simulation
dataset. The first thing is generating a scatter plot visualizing which
thicknesses and gap depths are in dataset

Additionally, basic statistics about the simulation distribution can be
printed into command line and stored in .txt file

created by: Max Schmitz on 11/01/2021
"""


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


def extract_sim_information_from_folders(path: Path) -> np.array:
    """
    function extracts the fundamental simulation information
    for each simulation and puts them into a vector with dimension
    (#sims x #features)

    args:
        - path: Path object with path to working directory. Most probably
        this will be 'analysis_2dfft'

    """

    folders = [elem for elem in os.listdir(path) if elem.find('old') == -1]

    num_feats = 3
    feats = np.zeros((len(folders), num_feats))

    for idx, folder in enumerate(folders):
        sim_info = load_json_info_file(path / folder)

        feats[idx, 0] = sim_info['c_height']
        try:
            feats[idx, 1] = sim_info['geometric_properties']['cg_gap_depth']
            feats[idx, 2] = sim_info['geometric_properties']['cg_bevel']
        except KeyError:
            feats[idx, 1] = 0.
            feats[idx, 2] = 0.002

    return feats


def get_creation_date() -> str:
    """
    return creation date in format:
    month-day_hour-minute-second
    """
    current_time = datetime.now().strftime("%H-%M-%S")
    today = date.today().strftime("%m-%d")
    creation_date = str(today) \
                    + '_' \
                    + str(current_time)
    return creation_date


def visualize_features(
        f: np.array,
        save: bool = False,
        gap_ratio: int = 5,
        thick_threshold: int = 204
) -> None:
    """
    Plot the features from the simulation in an appealing way. Keep in mind
    that the thin values are getting converted to integer values for plotting

    args:
        - f: vector of features with dimension #sims x #features
        - gap_ratio: ratio (gap_depth/coating_thickness) in percent und which the specimen
            is considered as uniform
    """

    fig = plt.figure(1, dpi=200)
    fig.set_size_inches(w=6, h=4)
    ax = plt.gca()

    # f_color = ['g' if elem[0] - elem[1] > thick_threshold * 1E-6 else 'r' for elem in f]
    # f_color = ['g' if round(100 * elem[1]/elem[0]) <= gap_ratio else 'r' for elem in f]
    f_color = ['g' if elem[1] == 0 else 'r' for elem in f]
    f_markers = ["D" if elem[1] == 0 else "o" for elem in f]
    f_size = [6 if elem[1] == 0 else 10 for elem in f]
    # ax.scatter(f[:, 0] * 1E6, f[:, 1] * 1E6, s=f[:, 2] * 8E3, c=f_color, marker=f_markers, alpha=0.8)

    for idx in range(len(f_color)):
        ax.scatter(f[idx, 0] * 1E6, f[idx, 1] * 1E6, s=f_size[idx], c=f_color[idx], marker=f_markers[idx], alpha=0.8)

    x = np.linspace(0, 600, 100)
    # y = x - thick_threshold
    # y = 0.05 * x  # np.ones((x.shape[0]))
    y = np.ones((x.shape[0]))
    ax.plot(x, y, c='k', alpha=0.6)

    # import pdb; pdb.set_trace()

    plt.axis([-10, 610, -10, 210])

    creation_date = get_creation_date()

    # plt.title(f'Simulations Parameters Space for {f.shape[0]} simulations\n'
    #           f'{creation_date}')
    plt.xlabel(r'Coating thickness in $\mu$m (1E-6m)')
    plt.ylabel(r'Gap depth in $\mu$m (1E-6m)')
    if save:
        plt.savefig(Path.cwd() / 'figures_param_space' /
                    f'{creation_date}_sim_params_space.png', dpi=200)
        out_name = Path.cwd() / 'figures_param_space' / \
                   f'{creation_date}_sim_params_space.pgf'
        plt.savefig(out_name, backend='pgf', format='pgf', dpi=200)
    plt.show()


def get_thickness_info_gap(feats: np.array, thick_trshd: int = 204, save: bool = False) -> None:
    """
    Output various basic dataset statistics to terminal and text file

    args:
        - feats: np.array with features. feats[:,1] are thicknesses,
            feats[:,2] are gap depths
        - thick_trshd: integer specifying the thickness threshold for
            thick/not thick enough in microns
        - save: specify if text file with statistics should be saved
    """
    # feats = np.array([
    #     [100E-6, 50E-6, 0.01],
    #     [280E-6, 50E-6, 0.01],
    #     # [280E-6, 100E-6, 0.01],
    #     [400E-6, 100E-6, 0.01]
    # ])

    # statistics - sir = sims in range, soor = sims out of range
    n = [1 if (thick_trshd * 1E-6) <= elem[0] < 300E-6 else 0 for elem in feats]
    sir = sum(n)
    sir_rel = 100 * sum(n) / feats.shape[0]
    soor = feats.shape[0] - sum(n)
    soor_rel = (1 - sum(n) / feats.shape[0]) * 100
    s_total = feats.shape[0]

    sir_te = sum([1 if (thick_trshd * 1E-6) <= elem[0] < 300E-6 and
                       elem[0] - elem[1] > thick_trshd * 1E-6
                  else 0 for elem in feats])
    sir_te_rel = sir_te / sir * 100
    sir_te_rel_tot = sir_te / s_total * 100

    sir_nt = sir - sir_te
    sir_nt_rel = sir_nt / sir * 100
    sir_nt_rel_tot = sir_nt / s_total * 100

    soor_te = sum([1 if ((thick_trshd * 1E-6) > elem[0] or elem[0] >= 300E-6) and
                        elem[0] - elem[1] > thick_trshd * 1E-6
                   else 0 for elem in feats])
    try:
        soor_te_rel = soor_te / soor * 100
        soor_te_rel_tot = soor_te / s_total * 100
    except ZeroDivisionError:
        soor_te_rel = 0
        soor_te_rel_tot = 0

    try:
        soor_nt = soor - soor_te
        soor_nt_rel = soor_nt / soor * 100
        soor_nt_rel_tot = soor_nt / s_total * 100
    except ZeroDivisionError:
        soor_nt = 0
        soor_nt_rel = 0
        soor_nt_rel_tot = 0

    te_tot = sir_te + soor_te
    te_tot_rel = sir_te_rel_tot + soor_te_rel_tot

    nt_tot = sir_nt + soor_nt
    nt_tot_rel = sir_nt_rel_tot + soor_nt_rel_tot

    sims_stats = (
        f'\n---> General Simulation Data Statistics <---\n'
        f'____________________________________________\n'
        # # f'Range: {thick_trshd} <-> 300 microns\n'
        # f'.  .  .  .  .  .  .  .  .  .  .  .  .  .  .\n' 
        # # f'|--> {sir} (= {"{:.2f}".format(sir_rel)}%) sims in range\n'
        # # f'|--> {soor} (= {"{:.2f}".format(soor_rel)}%) sims out of range\n'
        f'|--> {s_total} (= 100%) sims total\n'
        f'.  .  .  .  .  .  .  .  .  .  .  .  .  .  .\n'
        f'|--> {te_tot} (= {"{:.2f}".format(te_tot_rel)}%) sims thick enough\n'
        f'|--> {nt_tot} (= {"{:.2f}".format(nt_tot_rel)}%) sims not thick enough \n'
        f'.  .  .  .  .  .  .  .  .  .  .  .  .  .  .\n'
        f'Within the range there are:\n'
        f'-> {sir_te} = ({"{:.2f}".format(sir_te_rel)}% rel/ '
        f'{"{:.2f}".format(sir_te_rel_tot)}% abs) thick enough\n'
        f'-> {sir_nt} = ({"{:.2f}".format(sir_nt_rel)}% rel/ '
        f'{"{:.2f}".format(sir_nt_rel_tot)}% abs) not thick enough\n'
        f'.  .  .  .  .  .  .  .  .  .  .  .  .  .  .\n'
        f'Out of range there are:\n'
        f'-> {soor_te} = ({"{:.2f}".format(soor_te_rel)}% rel/ '
        f'{"{:.2f}".format(soor_te_rel_tot)}% abs) thick enough\n'
        f'-> {soor_nt} = ({"{:.2f}".format(soor_nt_rel)}% rel/ '
        f'{"{:.2f}".format(soor_nt_rel_tot)}% abs) not thick enough\n'
        f'____________________________________________\n')

    print(sims_stats)

    if save:
        with open(Path.cwd() / 'figures_param_space'
                  / f'{get_creation_date()}_sims_stats.txt', "w") as text_file:
            text_file.write(sims_stats)


def get_thickness_info_uniform(
        feats: np.array,
        gap_ratio: int = 5,
        save: bool = False
) -> None:
    """
    Output various basic dataset statistics to terminal and text file

    args:
        - feats: np.array with features. feats[:,0] are thicknesses,
            feats[:,1] are gap depths
        - gap_ratio: ratio (gap_depth/coating_thickness) in percent und which the specimen
            is considered as uniform
        - save: specify if text file with statistics should be saved
    """
    # feats = np.array([
    #     [100E-6, 50E-6, 0.01],
    #     [280E-6, 50E-6, 0.01],
    #     # [280E-6, 100E-6, 0.01],
    #     [400E-6, 100E-6, 0.01]
    # ])

    # statistics - sir = sims in range, soor = sims out of range
    # n = [1 if elem[1] == 0 else 0 for elem in feats]
    # n = [1 if round(100 * elem[1] / elem[0]) <= gap_ratio else 0 for elem in feats]
    n = [1 if elem[1] == 0 else 0 for elem in feats]
    uf = sum(n)
    uf_rel = 100 * sum(n) / feats.shape[0]
    nuf = feats.shape[0] - sum(n)
    nuf_rel = (1 - sum(n) / feats.shape[0]) * 100
    s_total = feats.shape[0]

    sims_stats = (
        f'\n---> General Simulation Data Statistics <---\n'
        f'____________________________________________\n'
        f'|--> {s_total} (= 100%) sims total\n'
        f'.  .  .  .  .  .  .  .  .  .  .  .  .  .  .\n'
        f'|--> {uf} (= {"{:.2f}".format(uf_rel)}%) sims uniform\n'
        f'|--> {nuf} (= {"{:.2f}".format(nuf_rel)}%) sims not uniform \n'
        f'.  .  .  .  .  .  .  .  .  .  .  .  .  .  .\n'
        f'____________________________________________\n')

    print(sims_stats)

    if save:
        with open(Path.cwd() / 'figures_param_space'
                  / f'{get_creation_date()}_sims_stats.txt', "w") as text_file:
            text_file.write(sims_stats)


if __name__ == '__main__':
    save = True
    gap_ratio = 5  # deprecated - only used for statistics
    thick_trshd = 200
    working_path = Path('C:\\Users\\Max\\Documents') / 'analysis_2dfft'

    feats = extract_sim_information_from_folders(working_path)

    get_thickness_info_uniform(feats, gap_ratio, save=save)

    visualize_features(feats, save=save, thick_threshold=thick_trshd)

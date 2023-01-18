"""
analyze_existing_simulations looks into the existing simulations and casts the simulation information
into a dataframe for further analysis. The idea is to get a list of existing simulations with
given parameters to double-check if the parameter combination is already existing when new simulations
are running.

This file needs to be put at user/scratch/ which is the same level as the 'simulations' folder.
There is no extra scheduler script because this script will run on the end node. To run this file, write

First, activate the right environment
##########################################
>>  module load anaconda3/2020.11
>> conda activate wave_env_cl
##########################################

And then run the script
##########################################
>> python3 analze_existing_simulations.py
##########################################

created by Max Schmitz on 12/03/2022
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib
# # comment this in if you want to export to latex
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from pathlib import Path

from visualize_dataset import load_json_info_file, get_creation_date


def extract_sim_info_to_df(path: Path, save: bool = False) -> pd.DataFrame:
    """
    function extracts the fundamental simulation information
    for each simulation and puts them into a dataframe with dimension
    (#simulations x 5)

    Remarks:
    - This function has been adapted from "extract_sim_information_from_folders" from "visualize_dataset.py"
    - Make sure that there are only the needed files (this one, visualize_dataset, utils and the __pycache__)
        and folders in the directory

    :arg:
        - path: Path object with path to working directory
        - save: Flag if extracted dataframe should be saved to .csv in current folder

    :returns:
        - pd.DataFrame with most important geometric information from all simulations

    """

    folders = [elem for elem in os.listdir(path) if (elem.find('old') == -1 and elem.find('py') == -1
                                                     and elem.find('ext') == -1 and elem.find('.csv') == -1)]

    sim_infos = pd.DataFrame(columns=[
        'c_height', 'cg_top_left', 'cg_bevel', 'cg_top_right', 'cg_gap_depth'])

    for idx, folder in enumerate(tqdm(folders)):
        # print(f'folder>> {folder}')
        sim_info = load_json_info_file(path / folder)

        c_height = sim_info['c_height']
        try:
            cg_gap_depth = sim_info['geometric_properties']['cg_gap_depth']
            cg_bevel = sim_info['geometric_properties']['cg_bevel']
            cg_top_left = sim_info['geometric_properties']['cg_top_left']
            cg_top_right = sim_info['geometric_properties']['cg_top_right']
        except KeyError:
            cg_gap_depth = 0.
            cg_bevel = 0.002
            cg_top_left = 0.
            cg_top_right = 0.

        new_info = pd.DataFrame({
            'c_height': [c_height],
            'cg_top_left': [cg_top_left],
            'cg_bevel': [cg_bevel],
            'cg_top_right': [cg_top_right],
            'cg_gap_depth': [cg_gap_depth]
        })

        sim_infos = pd.concat([sim_infos, new_info])

    print(f"There are {len(sim_infos.index)} simulations which can be used")

    if save:
        fn_path = path / 'sim_infos_py' / Path(get_creation_date() + 'param_infos.csv')
        sim_infos.to_csv(fn_path, index=False)
        print('>> Parameter Infos have been saved!')

    return sim_infos


def visualize_feats(
        sim_df: pd.DataFrame,
        save: bool = False,
        save_path: Path = Path.cwd(),
) -> None:
    """
    Plot the features from the simulation in an appealing way. Keep in mind
    that the thin values are getting converted to integer values for plotting

    This function has been adapted from "visualize_features" from "visualize_dataset.py"

    args:
        - sim_df: dataframe of features with dimension #sims x #5
        - gap_ratio: ratio (gap_depth/coating_thickness) in percent und which the specimen
            is considered as uniform
        - save_path: Path to the directory where the features space visualization should be saved to
    """

    fig = plt.figure(1, dpi=200)
    fig.set_size_inches(w=6, h=4)
    ax = plt.gca()

    sim_color = ['g' if elem == 0.0 else 'r' for elem in sim_df['cg_gap_depth'].tolist()]
    sim_markers = ["D" if elem == 0.0 else "o" for elem in sim_df['cg_gap_depth'].tolist()]
    sim_size = [6 if elem == 0.0 else 10 for elem in sim_df['cg_gap_depth'].tolist()]

    uniform_sims = []
    for thickness in sim_df['cg_gap_depth'].tolist():
        if thickness == 0.0:
            uniform_sims.append(thickness)
    n_uniform_sims = len(uniform_sims)
    n_non_uniform_sims = 1018 - n_uniform_sims

    print(f'There are {n_uniform_sims} simulations with uniform coating and\n'
          f'there are {n_non_uniform_sims} simulations with non-uniform coating')

    for idx in range(len(sim_color)):
        ax.scatter(sim_df['c_height'].iloc[idx] * 1E6, sim_df['cg_gap_depth'].iloc[idx] * 1E6, s=sim_size[idx],
                   c=sim_color[idx], marker=sim_markers[idx], alpha=0.8)

    x = np.linspace(0, 600, 100)
    y = np.ones((x.shape[0]))
    ax.plot(x, y, c='k', alpha=0.6)

    plt.axis([-10, 610, -10, 210])

    creation_date = get_creation_date()

    plt.xlabel(r'Coating thickness in $\mu$m (1E-6m)')
    plt.ylabel(r'Gap depth in $\mu$m (1E-6m)')
    if save:
        try:
            plt.savefig(save_path / 'figures_param_space_py' /
                        f"{creation_date}_sim_params_space.png", dpi=200)
        except FileNotFoundError:
            plt.savefig(save_path / f"{creation_date}_sim_params_space.png", dpi=200)
            out_name = Path.cwd() / f"{creation_date}_sim_params_space.pgf"
            plt.savefig(out_name, backend='pgf', format='pgf', dpi=200)
    plt.show()


if __name__ == '__main__':
    save_param_infos = True
    save_visualization = True

    # working_path = Path(
    #     'C:\\Users\\Max\\OneDrive\\Documents\\Uni Gatech MSC\\A Lab Research Wave CEE\\'
    #     'A Journal Paper\\ml_dl_wave_inversion\\create_model\\2dfft_data_selected\\'
    #     'cluster_simulations_example'
    # )
    working_path = Path(
        'C:\\Users\\Max\\OneDrive\\Documents\\Uni Gatech MSC\\A Lab Research Wave CEE\\'
        'A Journal Paper\\ml_dl_wave_inversion\\create_model\\figures_param_space\\'
    )
    if not working_path.is_dir():
        working_path = Path(__file__).parent.resolve() / 'simulations'  # in case of cluster
        print(f"working path = {working_path}")

    # param_infos = extract_sim_info_to_df(working_path, save=save_param_infos)

    # load the param infos from the .csv file
    parameter_info_path = Path(
        'C:\\Users\\Max\\OneDrive\\Documents\\Uni Gatech MSC\\A Lab Research Wave CEE\\'
        'A Journal Paper\\ml_dl_wave_inversion\\create_model\\figures_param_space\\'
        '01-18_09-53-37param_infos.csv'
    )
    param_infos = pd.read_csv(parameter_info_path)

    visualize_feats(param_infos, save=save_visualization, save_path=working_path)

    #print(param_infos)

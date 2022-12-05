"""
analyze_existing_simulations looks into the existing simulations and casts the simulation information
into a dataframe for further analysis. The idea is to get a list of existing simulations with
given parameters to double-check if the parameter combination is already existing when new simulations
are running.

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from visualize_dataset import load_json_info_file, get_creation_date


def extract_sim_info_to_df(path: Path, save: bool = False) -> pd.DataFrame:
    """
    function extracts the fundamental simulation information
    for each simulation and puts them into a dataframe with dimension
    (#simulations x 5)

    This function has been adapted from "extract_sim_information_from_folders" from "visualize_dataset.py"

    :arg:
        - path: Path object with path to working directory
        - save: Flag if extracted dataframe should be saved to .csv in current folder

    :returns:
        - pd.DataFrame with most important geometric information from all simulations

    """

    folders = [elem for elem in os.listdir(path) if elem.find('old') == -1]
    print(f"folders: {folders}")

    sim_infos = pd.DataFrame(columns=[
        'c_height', 'cg_top_left', 'cg_bevel', 'cg_top_right', 'cg_gap_depth'])

    for idx, folder in enumerate(folders):
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

    if save:
        fn = get_creation_date() + 'param_infos.csv'
        sim_infos.to_csv(fn, index=False)
        print('>> Parameter Infos have been saved!')

    return sim_infos


def visualize_feats(
        sim_df: pd.DataFrame,
        save: bool = False,
) -> None:
    """
    Plot the features from the simulation in an appealing way. Keep in mind
    that the thin values are getting converted to integer values for plotting

    This function has been adapted from "visualize_features" from "visualize_dataset.py"

    args:
        - sim_df: dataframe of features with dimension #sims x #5
        - gap_ratio: ratio (gap_depth/coating_thickness) in percent und which the specimen
            is considered as uniform
    """

    fig = plt.figure(1, dpi=200)
    fig.set_size_inches(w=6, h=4)
    ax = plt.gca()

    sim_color = ['g' if elem == 0 else 'r' for elem in sim_df['cg_gap_depth'].tolist()]
    sim_markers = ["D" if elem == 0 else "o" for elem in sim_df['cg_gap_depth'].tolist()]
    sim_size = [6 if elem == 0 else 10 for elem in sim_df['cg_gap_depth'].tolist()]

    for idx in range(len(sim_color)):
        ax.scatter(sim_df['c_height'] * 1E6, sim_df['cg_gap_depth'] * 1E6, s=sim_size[idx],
                   c=sim_color[idx], marker=sim_markers[idx], alpha=0.8)

    x = np.linspace(0, 600, 100)
    y = np.ones((x.shape[0]))
    ax.plot(x, y, c='k', alpha=0.6)

    plt.axis([-10, 610, -10, 210])

    creation_date = get_creation_date()

    plt.xlabel(r'Coating thickness in $\mu$m (1E-6m)')
    plt.ylabel(r'Gap depth in $\mu$m (1E-6m)')
    if save:
        plt.savefig(Path.cwd() / 'figures_param_space' /
                    f"{creation_date}_sim_params_space.png", dpi=200)
        out_name = Path.cwd() / 'figures_param_space' / \
                   f"{creation_date}_sim_params_space.pgf"
        plt.savefig(out_name, backend='pgf', format='pgf', dpi=200)
    plt.show()


if __name__ == '__main__':
    save_param_infos = True
    save_visualization = False

    working_path = Path(
        'C:\\Users\\Max\\OneDrive\\Documents\\Uni Gatech MSC\\A Lab Research Wave CEE\\'
        'A Journal Paper\\ml_dl_wave_inversion\\create_model\\2dfft_data_selected\\'
        'cluster_simulations_example'
    )

    param_infos = extract_sim_info_to_df(working_path, save=save_param_infos)

    visualize_feats(param_infos, save=save_visualization)

    print(param_infos)

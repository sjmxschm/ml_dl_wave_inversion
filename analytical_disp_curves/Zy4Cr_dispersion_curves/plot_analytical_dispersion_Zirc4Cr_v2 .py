from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analytical_disp_curves.analytical_curves import load_analytical_data, \
    plot_analytical_dispersion_curves_layered

'''

This script plots analytical data from the dispersion curves of
Aluminum, Tape and the AluTape layered system

'''

if __name__ == '__main__':

    # thicknesses = [0.06, 0.1, 0.3, 0.4, 0.5, 0.6]
    thicknesses = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
    # thicknesses = ['0.3']

    layers_combo_path = Path().absolute()
    single_layers_path = layers_combo_path.parent.absolute()

    for th in thicknesses:

        tb_path = layers_combo_path / f'{th}_Chrome'
        f = f'Zy4_3_Cr_{th}'
        k_tb, f_tb, mn_tb = load_analytical_data(f, tb_path)

        t_path = single_layers_path / 'Cr_dispersion_curves' / f'Chrome_{th}_mm_dispersion_curves'
        f = f'Chrome_{th}_mm'
        # import pdb;
        # pdb.set_trace()
        k_t, f_t, mn_t = load_analytical_data(f, t_path)

        b_path = single_layers_path / f'Zy4_dispersion_curves_DC'
        f = f'Zirc'
        k_b, f_b, mn_b = load_analytical_data(f, b_path)

        plot_analytical_dispersion_curves_layered(
            k_t,
            f_t,
            mn_t,
            k_b,
            f_b,
            mn_b,
            k_tb,
            f_tb,
            mn_tb,
            axis=False,
            # m_axis=[0, 4000, 0.5e7, 1.5e7],
            # m_axis=[0, 5000, 0.5e7, 2.5e7],
            # m_axis=[0, 10000, 0.5e7, 4e7],
            m_axis=[0, 10000, 0e7, 2.5e7],
            output_name=f'Zy4-Cr_{th}_dc_extracted_3.png',
            layout='analysis_max_sharp',
            save=False,
            show=True
        )

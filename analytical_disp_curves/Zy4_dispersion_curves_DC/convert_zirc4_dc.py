import numpy as np
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
from pathlib import Path

from analytical_disp_curves.analytical_curves import load_data_from_exel_layered, convert_fc2kf, \
    plot_analytical_dispersion_curves_single, save_dispersion_np_array, save_mode_names, load_analytical_data

if __name__ == '__main__':
    # f_a, v_a, mn, output_name = load_data_from_exel_layered('Zirc', file_name='A_Lamb')
    # k_a = convert_fc2kf(f_a, v_a, output_name)
    # f_s, v_s, mn_s, output_name = load_data_from_exel_layered('Zirc', file_name='S_Lamb')
    # k_s = convert_fc2kf(f_s, v_s, output_name)

    # f = np.hstack((f_a, f_s))
    # k = np.hstack((k_a, k_s))
    # v = np.hstack((v_a, v_s))
    # mn.extend(mn_s)

    # save_dispersion_np_array(output_name, f, 'f')
    # save_dispersion_np_array(output_name, k, 'k')
    # save_mode_names(output_name, mn)
    # save_dispersion_np_array(output_name, v, 'v')

    k, f, mn = load_analytical_data('Zirc', Path.cwd())
    with open(Path.cwd() / Path('Zirc_dispersion_data_analytically_v_.npy'), 'rb') as v_in:
        v = np.load(v_in)

    save_publication = True

    if True:
        fig = plt.figure(1, dpi=300)
        fig.set_size_inches(w=6, h=4)

        for row in range(f.shape[1]):
            if mn[row][0] == 'S':
                plt.plot(f[0:np.where(f == 0)[0][0], row], v[0:np.where(f == 0)[0][0], row],
                         label=mn[row], linewidth=.7)
            else:
                plt.plot(f[0:np.where(f == 0)[0][0], row], v[0:np.where(f == 0)[0][0], row],
                         linestyle='--', label=mn[row], linewidth=.7)

        if not save_publication:
            plt.title(f'Analytically obtained dispersion curves of Zirc-4 1mm')
        plt.xlabel(r'Frequency $f$ in MHz')
        plt.ylabel(r'Phase speed in $\frac{m}{s}$')
        plt.legend(loc='lower right', ncol=4)

        plt.axis([0, 3E7, 0, 6000])
        if save_publication:
            pub_out_name = f'Zirc-4_c_f_dispersion_plot.pgf'
            plt.savefig(pub_out_name[0:-3] + 'png', backend='pgf', format='png', dpi=200)
            plt.savefig(pub_out_name, backend='pgf', format='pgf', dpi=200)
        else:
            # plt.savefig(Path('Zirc-4_c_f_dispersion_plot' + '.eps'), format='eps', bbox_inches='tight', dpi=300)
            plt.show()

    if save_publication:
        plot_analytical_dispersion_curves_single(
            'Zirc-4 1mm',
            k, f, mn,
            axis=False,
            # m_axis=[0, 4000, 0, 9e6],
            m_axis=[0, 5000, 0, 2E7],
            save=True,
            save_publication=save_publication
        )

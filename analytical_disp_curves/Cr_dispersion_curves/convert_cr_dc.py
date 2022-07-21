import numpy as np

from pathlib import Path

from analytical_disp_curves.analytical_curves import load_data_from_exel_layered, convert_fc2kf, \
    plot_analytical_dispersion_curves_single, save_dispersion_np_array, save_mode_names

if __name__ == '__main__':

    # thicknesses = [0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4]
    # thicknesses = ['0.4', '0.5', '0.6']
    # thicknesses = ['0.02', '0.04', '0.06', '0.1', '0.2', '0.3']

    thicknesses = ['0.03']
    for th in thicknesses:
        p = Path(__file__).parent.resolve()
        # import pdb; pdb.set_trace()
        m = f'Chrome_{th}_mm'
        f_a, v_a, mn, output_name = load_data_from_exel_layered(
            m, thickness=float(th), file_name=f'{p}/{m}_dispersion_curves/A_Lamb')
        k_a = convert_fc2kf(f_a, v_a, output_name)
        f_s, v_s, mn_s, output_name = load_data_from_exel_layered(
            m, thickness=float(th), file_name=f'{p}/{m}_dispersion_curves/S_Lamb')
        k_s = convert_fc2kf(f_s, v_s, output_name)

        f = np.hstack((f_a, f_s))
        k = np.hstack((k_a, k_s))
        mn.extend(mn_s)

        save_dispersion_np_array(output_name, f, 'f', path=Path(p / f'{m}_dispersion_curves'))
        save_dispersion_np_array(output_name, k, 'k', path=Path(p / f'{m}_dispersion_curves'))
        save_mode_names(output_name, mn, path=Path(p / f'{m}_dispersion_curves'))

        plot_analytical_dispersion_curves_single(
            m,
            k,
            f,
            mn,
            axis=False,
            # m_axis=[0, 4000, 0, 1.5e7],
            save=True,
            path=Path(p / f'{m}_dispersion_curves')
        )

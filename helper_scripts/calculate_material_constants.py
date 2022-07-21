import numpy as np


def calculate_wave_speeds(
        material='Chrome',
        rho=7190,
        nu=0.21,
        E=279E9
):
    """
      calculate dilational and transverse wave velocity from material properties
      
      args:
            - rho = 7190      density - kg/m^3
            - nu = 0.21       Poisson's ratio
            - E = 279E9       Young's modulus - Pa
      """

    # Lame parameters:
    lmd = nu / ((1 - 2 * nu) * (1 + nu)) * E
    mu = 1 / (2 + 2 * nu) * E

    c_d = np.sqrt((lmd + 2 * mu) / rho)
    c_s = np.sqrt(mu / rho)

    print(f'\n'
          f'for {material}:\n'
          f'dilatational wave speed: c_d = {c_d} m/s\n'
          f'shear wave speed: c_s = {c_s} m/s'
          f'\n')


def calculate_max_sim_time(
        c_g=6370,
        d_plate=0.09,
):
    """
      plug the wave speeds into the dispersion software from Vallen
       and obtain the group velocity at omega=0.
      
      Calculate the maximum simulation time
      
      args:
            - c_g = 6370      # m/s
            - d_plate = 0.09  # m
      """
    t_sim_max = d_plate / c_g
    print(f'The maximum simulation time is: t_sim,max = {t_sim_max} s')


def calculate_shear_velocity(v_long, pois_rat, density):
    """
    calculate the missing variables
    """
    v_l = v_long
    nu = pois_rat
    rho = density

    # calculate material constants
    lmd = rho * nu/(- nu + 1) * v_l**2
    mu = lmd * (1-2*nu)/(2*nu)

    v_s = np.sqrt(mu/rho)

    print(f'\n'
          f'for the dilatational wave speed {v_l} m/s the material constants are\n'
          f'lambda = {lmd}\n'
          f'mu = {mu}\n'
          f'shear wave speed: v_s = {v_s} m/s' # - 2526.121856704588 m/s
          f'\n')

def calculate_mesh_elements_per_height(
        material,
        height,
        c_pois_rat,
        c_youngs_mod,
        c_density,
        num_mesh: int = 1,
        t_max: float = 5e-8
):

    f_max = 4 * np.pi / t_max
    c_eta = (0.87 + 1.12 * c_pois_rat) / (1 + c_pois_rat)
    c_mu = c_youngs_mod / (2 + 2 * c_pois_rat)
    c_r_chrome = c_eta * np.sqrt(c_mu / c_density)

    n_mesh = int(num_mesh)  # 2  #int(num_mesh) # was 1
    c_elem_size = c_r_chrome / (n_mesh * f_max)

    print(f'For {material} the number of mesh elements per height are\n'
          f'{height//c_elem_size}')


if __name__ == '__main__':

    calculate_mesh_elements_per_height(
        material='Chrome',
        height=0.12,
        c_pois_rat=0.21,
        c_youngs_mod=279E9,
        c_density=7190.0,
        num_mesh=1,
        t_max=5e-8
    )

    calculate_mesh_elements_per_height(
        material='Zirconium',
        height=0.0036, #0.0036
        c_pois_rat=0.37,
        c_youngs_mod=99.3E9,
        c_density=6560.0,
        num_mesh=1,
        t_max=5e-8
    )

    # # for Chrome:
    # calculate_wave_speeds(
    #     material='Chrome',
    #     rho=7190,
    #     nu=0.21,
    #     E=279E9
    # )
    #
    # # for Zirconium-4:
    # calculate_wave_speeds(
    #     material='Zirconium-4',
    #     rho=6560,
    #     nu=0.37,
    #     E=99.3E9
    # )
    #
    # calculate_shear_velocity(
    #     v_long=5561,
    #     pois_rat=0.37,
    #     density=6560
    # )

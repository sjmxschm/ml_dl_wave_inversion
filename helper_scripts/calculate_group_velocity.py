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

if __name__ == '__main__':
    # for Chrome:
    calculate_wave_speeds(
        material='Chrome',
        rho=7190,
        nu=0.21,
        E=279E9
    )

    # for Zirconium-4:
    calculate_wave_speeds(
        material='Zirconium-4',
        rho=6560,
        nu=0.37,
        E=99.3E9
    )

    calculate_shear_velocity(
        v_long=5561,
        pois_rat=0.37,
        density=6560
    )

    # for Aluminum:
    calculate_wave_speeds(
        material='Alu',
        rho=2700,
        nu=0.3375,
        E=70.758E9
    )
    # for Tape:
    calculate_wave_speeds(
        material='Tape',
        rho=1106,
        nu=0.35,
        E=1E9
    )

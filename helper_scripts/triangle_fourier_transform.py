import numpy as np
import matplotlib.pyplot as plt


def plot_sinc_fct():
    r = 10E7
    f = np.arange(-r, r, 100)

    # hg = np.sinc(t)

    t_max = 5E-8 #2E-8  # 5E-8 is currently used in model
    hg = np.square(np.sinc(t_max * f)) # * np.exp(2*np.pi*1j*f*t_max)
    # hg = np.sinc(1 * f) * np.exp(2 * np.pi * 1j * f * t_max)

    plt.plot(f, hg)
    plt.grid(True)
    plt.axis([-.5E8, .5E8, -0.1, 1.1])
    # plt.axis([-1E3, 1E3, -1, 1])
    plt.xticks([-4E7, -3E7, -2E7, -1E7, 0, 1E7, 2E7, 3E7, 4E7])

    plt.xlabel('Frequency in Hz')
    plt.ylabel('Intensity')
    plt.title('Fourier transform of triangular function (sinc**2 function)'
              f'\n for an excitation duration of 2*{t_max}')
    plt.savefig('ft_triangular_func', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_sinc_fct()

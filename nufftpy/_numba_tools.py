from __future__ import division, print_function
import numpy as np
import numba


@numba.jit(nopython=True)
def slow_grid(x, c, tau, Msp, ftau):
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        for mm in range(-Msp, Msp):
            spread = np.exp(-0.25 * (xi - hx * (m + mm)) ** 2 / tau)
            ftau[(m + mm) % Mr] += c[i] * spread


@numba.jit(nopython=True)
def fast_grid(x, c, tau, Msp, ftau, E3):
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr

    # precompute some exponents
    for j in range(Msp + 1):
        E3[j] = np.exp(-(np.pi * j / Mr) ** 2 / tau)

    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        xi = (xi - hx * m)
        E1 = np.exp(-0.25 * xi ** 2 / tau)
        E2 = np.exp((xi * np.pi) / (Mr * tau))
        E2mm = 1
        for mm in range(Msp):
            ftau[(m + mm) % Mr] += c[i] * E1 * E2mm * E3[mm]
            E2mm *= E2
            ftau[(m - mm - 1) % Mr] += c[i] * E1 / E2mm * E3[mm + 1]


def _gaussian_grid_numba_1D(x, c, Mr, Msp, tau, fast_gridding):
    """Compute the 1D Gaussian gridding with numba"""
    ftau = np.zeros(Mr, dtype=c.dtype)
    if fast_gridding:
        E3 = np.zeros(Msp + 1, dtype=x.dtype)
        fast_grid(x, c, tau, Msp, ftau, E3)
    else:
        slow_grid(x, c, tau, Msp, ftau)
    return ftau

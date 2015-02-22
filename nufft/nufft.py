from __future__ import division, print_function
import numpy as np


def compute_k(M):
    return np.arange(-(M // 2), M - (M // 2))


def dirft1d(x, c, M, iflag=-1):
    """Direct Non-Uniform Fourier Transform in 1 Dimension"""
    sign = -1 if iflag < 0 else 1
    M = int(M)
    x, c = map(np.ravel, np.broadcast_arrays(x, c))
    return np.dot(c, np.exp(sign * 1j * compute_k(M) * x[:, None]))


def nufft1d(x, c, M, iflag=-1, eps=1E-8, use_fft=True):
    """Fast Non-Uniform Fourier Transform in 1 Dimension"""
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy "
                         "1e-33 < eps < 1e-1.".format(eps))
    sign = -1 if iflag < 0 else 1
    M = int(M)
    x, c = map(np.ravel, np.broadcast_arrays(x, c))
    N = len(x)

    # Choose ratio, Msp, lambda, and tau following Dutt & Rokhlin (1993)
    ratio = 2 if eps > 1E-11 else 3
    Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
    Mr = max(ratio * M, 2 * Msp)
    lambda_ = Msp / (ratio * (ratio - 0.5))
    tau = np.pi * lambda_ / M ** 2

    # Construct the convolved grid
    hx = 2 * np.pi / Mr
    xmod = x % (2 * np.pi)
    m = (xmod // hx).astype(int)
    mspread = np.arange(-Msp, Msp)
    mm = (m + mspread[:, None])

    ftau = np.zeros(Mr, dtype=c.dtype)
    np.add.at(ftau, mm % Mr, c * np.exp(-(xmod - hx * mm) ** 2 / (4 * tau)))
            
    # Compute the DFT on the convolved grid
    k = compute_k(M)
    if use_fft:
        # TODO: multiply ftau by phase to replace concatenation step
        if sign < 0:
            Ftau = (1 / Mr) * np.fft.fft(ftau)
        else:
            Ftau = np.fft.ifft(ftau)
        Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])
    else:
        m = hx * np.arange(Mr)
        Ftau = (1 / Mr) * np.dot(ftau, np.exp(1j * sign * k * m[:, None]))
        
    # Deconvolve the grid
    return np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau

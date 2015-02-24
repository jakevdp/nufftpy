from __future__ import division, print_function
import warnings
import numpy as np

try:
    import numba
except ImportError:
    numba = None
else:
    from ._numba_tools import _gaussian_grid_numba_1D


def nufftfreqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins"""
    return df * np.arange(-(M // 2), M - (M // 2))


def _check_inputs(x, c, df):
    x = df * np.asarray(x)
    c = np.asarray(c)
    if x.ndim != 1:
        raise ValueError("Expected one-dimensional input arrays")
    if x.shape != c.shape:
        raise ValueError("Array shapes must match")
    return x, c


def _compute_grid_params(M, eps):
    if eps <= 1E-33 or eps >= 1E-1:
        raise ValueError("eps = {0:.0e}; must satisfy "
                         "1e-33 < eps < 1e-1.".format(eps))

    # Choose Msp & tau from eps following Dutt & Rokhlin (1993)
    ratio = 2 if eps > 1E-11 else 3
    Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
    Mr = max(ratio * M, 2 * Msp)
    lambda_ = Msp / (ratio * (ratio - 0.5))
    tau = np.pi * lambda_ / M ** 2
    return Mr, Msp, tau


def _gaussian_grid_1D(x, c, Mr, Msp, tau, fast_gridding):
    """Compute the 1D gaussian gridding with Numpy"""
    N = len(x)
    ftau = np.zeros(Mr, dtype=c.dtype)
    hx = 2 * np.pi / Mr
    xmod = x % (2 * np.pi)

    m = 1 + (xmod // hx).astype(int)
    msp = np.arange(-Msp, Msp)[:, np.newaxis]
    mm = m + msp

    if fast_gridding:
        # Greengard & Lee (2004) approach
        E1 = np.exp(-0.25 * (xmod - hx * m) ** 2 / tau)

        # The following lines basically compute this:
        # E2 = np.exp(msp * (xmod - hx * m) * np.pi / (Mr * tau))
        E2 = np.empty((2 * Msp, N), dtype=xmod.dtype)
        E2[Msp] = 1
        E2[Msp + 1:] = np.exp((xmod - hx * m) * np.pi / (Mr * tau))
        E2[Msp + 1:].cumprod(0, out=E2[Msp + 1:])
        E2[Msp - 1::-1] = 1. / (E2[Msp + 1] * E2[Msp:])

        E3 = np.exp(-(np.pi * msp / Mr) ** 2 / tau)
        spread = (c * E1) * E2 * E3
    else:
        spread = c * np.exp(-0.25 * (xmod - hx * mm) ** 2 / tau)
    np.add.at(ftau, mm % Mr, spread)

    return ftau


def nufft1(x, c, M, df=1.0, eps=1E-15, iflag=1,
           direct=False, fast_gridding=True, use_numba=True):
    """Fast Non-Uniform Fourier Transform (Type 1: uniform frequency grid)

    Compute the non-uniform FFT of one-dimensional points x with complex
    values c. Result is computed at frequencies (df * m)
    for integer m in the range -M/2 < m < M/2.

    Parameters
    ----------
    x, c : array_like
        real locations x and complex values c of the points to be transformed.
    M, df : int & float
        Parameters specifying the desired frequency grid. Transform will be
        computed at frequencies df * (-(M//2) + arange(M))
    eps : float
        The desired approximate error for the FFT result. Must be in range
        1E-33 < eps < 1E-1, though be aware that the errors are only well
        calibrated near the range 1E-12 ~ 1E-6. eps is not referenced if
        direct = True.
    iflag : float
        if iflag<0, compute the transform with a negative exponent.
        if iflag>=0, compute the transform with a positive exponent.
    direct : bool (default = False)
        If True, then use the slower (but more straightforward)
        direct Fourier transform to compute the result.
    fast_gridding : bool (default = True)
        If True, use the fast Gaussian grid algorithm of Greengard & Lee (2004)
        Otherwise, use a more naive gridding approach
    use_numba : bool (default = True)
        If True, use numba to compute the result. If False or if numba is not
        installed, default to the numpy version, which is ~5x slower.

    Returns
    -------
    Fk : ndarray
        The complex discrete Fourier transform

    See Also
    --------
    nufftfreqs : compute the frequencies of the nufft results
    """
    x, c = _check_inputs(x, c, df)
    M = int(M)
    N = len(x)
    k = nufftfreqs(M)

    if direct:
        # Direct Fourier Transform: this is easy (but slow)
        sign = -1 if iflag < 0 else 1
        return (1 / N) * np.dot(c, np.exp(sign * 1j * k * x[:, None]))
    else:
        # FFT version: a bit more involved
        Mr, Msp, tau = _compute_grid_params(M, eps)

        # Construct the convolved grid
        if use_numba and (numba is None):
            warnings.warn('numba is not installed. Using slower version.')
            use_numba = False
        if use_numba:
            ftau = _gaussian_grid_numba_1D(x, c, Mr, Msp, tau, fast_gridding)
        else:
            ftau = _gaussian_grid_1D(x, c, Mr, Msp, tau, fast_gridding)

        # Compute the FFT on the convolved grid
        if iflag < 0:
            Ftau = (1 / Mr) * np.fft.fft(ftau)
        else:
            Ftau = np.fft.ifft(ftau)
        Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])

        # Deconvolve the grid using convolution theorem
        return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau

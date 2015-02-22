from __future__ import division, print_function
import numpy as np


def nufft_freqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins"""
    return df * np.arange(-(M // 2), M - (M // 2))


def nufft1d(x, c, M, df=1.0, iflag=-1, eps=1E-8, direct=False):
    """Fast Non-Uniform Fourier Transform in 1 Dimension

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
    iflag : float
        if iflag<0, compute the transform with a negative exponent.
        if iflag>=0, compute the transform with a positive exponent.
    eps : float
        the desired approximate error for the FFT result. Must be in range
        1E-33 < eps < 1E-1, though be aware that the errors are only well
        calibrated near the range 1E-12 ~ 1E-6.
    direct : bool
        If True, then use the slower (but more straightforward) direct Fourier
        transform to compute the result. If direct method is used, then eps
        is not referenced.

    Returns
    -------
    Fk : ndarray
        The complex discrete Fourier transform 

    See Also
    --------
    nufft_freqs : compute the frequencies of the nufft results
    """
    # Check inputs
    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=complex)
    if x.ndim != 1:
        raise ValueError("Expected one-dimensional input arrays")
    if x.shape != c.shape:
        raise ValueError("Array shapes must match")
    sign = -1 if iflag < 0 else 1
    M = int(M)

    # For direct, compute the easy direct DFT, otherwise compute the NUFFT
    if direct:
        return np.dot(c, np.exp(sign * 1j * nufft_freqs(M, df) * x[:, None]))
    else:
        # Validate the remaining inputs
        N = len(x)
        if eps <= 1E-33 or eps >= 1E-1:
            raise ValueError("eps = {0:.0e}; must satisfy "
                             "1e-33 < eps < 1e-1.".format(eps))

        # Choose ratio, Msp, lambda, and tau following Dutt & Rokhlin (1993)
        ratio = 2 if eps > 1E-11 else 3
        Msp = int(-np.log(eps) / (np.pi * (ratio - 1) / (ratio - 0.5)) + 0.5)
        Mr = max(ratio * M, 2 * Msp)
        lambda_ = Msp / (ratio * (ratio - 0.5))
        tau = np.pi * lambda_ / M ** 2

        # Construct the convolved grid
        # TODO: break-up exponential as in Greengard & Lee (2004)
        ftau = np.zeros(Mr, dtype=c.dtype)
        hx = 2 * np.pi / Mr
        xmod = (df * x) % (2 * np.pi)
        m = (xmod // hx).astype(int) + 1 + np.arange(-Msp, Msp)[:, np.newaxis]
        np.add.at(ftau, m % Mr, c * np.exp(-(xmod - hx * m) ** 2 / (4 * tau)))

        # Compute the FFT on the convolved grid
        # TODO: multiply ftau by phase to replace concatenation step
        if sign < 0:
            Ftau = (1 / Mr) * np.fft.fft(ftau)
        else:
            Ftau = np.fft.ifft(ftau)
        Ftau = np.concatenate([Ftau[-(M//2):], Ftau[:M//2 + M % 2]])

        # Deconvolve the grid using convolution theorem
        k = nufft_freqs(M)
        return np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau

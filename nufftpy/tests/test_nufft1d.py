from .. import nufft1

import numpy as np
from numpy.testing import assert_allclose

from nose import SkipTest


def test_nufft1_vs_direct_1D():
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(100)
    c = np.exp(1j * x)

    def check_results(df, iflag, M, eps, use_numba, fast_gridding):
        dft = nufft1(x, c, M, iflag=iflag, direct=True)
        fft = nufft1(x, c, M, iflag=iflag, eps=eps,
                     use_numba=use_numba, fast_gridding=fast_gridding)
        assert_allclose(dft, fft, rtol=eps ** 0.95)

    for df in [0.5, 1.0, 2.0]:
        for iflag in [1, -1]:
            for M in [51, 100, 111]:
                for eps in [1E-8, 1E-12]:
                    for use_numba in [True, False]:
                        for fast_gridding in [True, False]:
                            yield (check_results, df, iflag, M, eps,
                                   use_numba, fast_gridding)


def test_nufft1_vs_fortran_1D():
    """Test against the Fortran implementation"""
    try:
        from nufft import nufft1 as nufft1_fortran
    except ImportError:
        raise SkipTest("python-nufft package is not installed: "
                       "see http://github.com/dfm/python-nufft")
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(100)
    c = np.exp(1j * x)

    def check_results(df, iflag, M, eps):
        fft1 = nufft1(x, c, M, iflag=iflag, eps=eps)
        fft2 = nufft1_fortran(x, c, M, iflag=iflag, eps=eps)
        assert_allclose(fft1, fft2)

    for df in [0.5, 1.0, 2.0]:
        for iflag in [1, -1]:
            for M in [51, 100, 111]:
                for eps in [1E-8, 1E-12]:
                    yield check_results, df, iflag, M, eps


def test_nufft1_fastgridding_1D():
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(100)
    c = np.exp(1j * x)

    def check_results(df, iflag, M, eps, use_numba):
        fft1 = nufft1(x, c, M, iflag=iflag, eps=eps,
                      use_numba=use_numba, fast_gridding=True)
        fft2 = nufft1(x, c, M, iflag=iflag, eps=eps,
                      use_numba=use_numba, fast_gridding=False)
        assert_allclose(fft1, fft2)

    for df in [0.5, 1.0, 2.0]:
        for iflag in [1, -1]:
            for M in [51, 100, 111]:
                for eps in [1E-8, 1E-12]:
                    for use_numba in [True, False]:
                        yield check_results, df, iflag, M, eps, use_numba

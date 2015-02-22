from .. import nufft1d

import numpy as np
from numpy.testing import assert_allclose


def test_dirft_nufft_1d():
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(100)
    c = np.exp(1j * x)

    def check_results(df, iflag, M, eps):
        dft = nufft1d(x, c, M, iflag=iflag, direct=True)
        fft = nufft1d(x, c, M, iflag=iflag, eps=eps)
        assert_allclose(dft, fft, rtol=eps ** 0.95)

    for df in [0.5, 1.0, 2.0]:
        for iflag in [1, -1]:
            for M in [100, 101]:
                for eps in [1E-8, 1E-12]:
                    yield check_results, df, iflag, M, eps

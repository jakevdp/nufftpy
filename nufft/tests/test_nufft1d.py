from .. import dirft1d, nufft1d

import numpy as np
from numpy.testing import assert_allclose


def test_dirft_nufft_1d():
    rng = np.random.RandomState(0)
    x = 100 * rng.rand(100)
    c = np.exp(1j * x) 

    def check_results(iflag, use_fft, M, eps):
        dft = dirft1d(x, c, M, iflag=iflag)
        fft = nufft1d(x, c, M, iflag=iflag, eps=eps, use_fft=use_fft)
        assert_allclose(dft, fft, rtol=eps ** 0.8)

    for use_fft in [True, False]:
        for iflag in [1, -1]:
            for M in [100, 200]:
                for eps in [1E-8, 1E-12]:
                    yield check_results, iflag, use_fft, M, eps

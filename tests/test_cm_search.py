import numpy
import itertools

from witnessmess import (
    cm_optimal_witness, 
    cm_optimal_gaussian,
    cm_is_physical,
    cm_is_pairwise_ppt, 
    cm_build_random
)

def test_cm_search ():
    mode_count_list = [ 3, 4, 5 ]
    spectral_factor_list = numpy.linspace(1, 3, 31)
    test_cases = itertools.product(mode_count_list, spectral_factor_list)

    for mode_count, spectral_factor in test_cases:
        G = cm_build_random(mode_count, spectral_factor)
        w, W = cm_optimal_witness(G, mode_count)

        assert cm_is_physical(G, mode_count)

        for index in range(5):
            h, G = cm_optimal_gaussian(W, mode_count, min_ppt = 1e-3, min_phy = 1e-3)
            w, W = cm_optimal_witness(G, mode_count)

            assert cm_is_physical(G, mode_count)
            assert cm_is_pairwise_ppt(G, mode_count)
            assert w < 0


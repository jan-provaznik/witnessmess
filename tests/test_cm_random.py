import numpy
import itertools

from witnessmess import cm_build_random, cm_is_physical

def test_cm_random ():
    '''
    Tests whether the random covariance matrices are actually physical.
    '''

    for _ in range(50):
        mode_count_list = numpy.arange(1, 10)
        spectral_factor_list = numpy.linspace(1, 10, 201)
        test_cases = itertools.product(mode_count_list, spectral_factor_list)
        
        for mode_count, spectral_factor in test_cases:
            G = cm_build_random(mode_count, spectral_factor)
            assert cm_is_physical(G, mode_count)


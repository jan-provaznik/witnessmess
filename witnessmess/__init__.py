#!/bin/env python
#
# 2019 - 2023 Jan Provaznik (jan@provaznik.pro)

'''
Witnessing genuine multi-partite entanglement from a subset of marginalia both
in multi-partite qubit systems and multi-mode continuous variable Gaussian
systems.
'''

version = '0.4.0'

# Genuine multi-partite entanglement witness for discrete variable multi-qudit
# systems inferred from their density matrics.

from ._dm import (
    dm_optimal_witness, 
    dm_is_physical
)    

# Genuine multi-partite entanglement witness for continuous variable multi-mode
# systems inferred from their covariance matrics.

from ._cm import (
    cm_optimal_witness, 
    cm_optimal_gaussian,
    cm_build_random, 
    cm_build_sigma,
    cm_build_ptranspose,
    cm_build_quadrature_reordering,
    cm_pairwise_pt, 
    cm_is_physical, 
    cm_is_pairwise_ppt,
)

# Module exports (for documentation).

__all__ = [
    'dm_optimal_witness', 
    'dm_is_physical',
    'cm_optimal_witness', 
    'cm_optimal_gaussian',
    'cm_build_random', 
    'cm_build_sigma',
    'cm_build_ptranspose',
    'cm_build_quadrature_reordering',
    'cm_pairwise_pt', 
    'cm_is_physical', 
    'cm_is_pairwise_ppt',
]


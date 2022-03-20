#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Witnessing genuine multi-partite entanglement from a subset of marginalia
# both in multi-partite qubit systems and multi-mode continuous variable
# Gaussian systems.

# (1.0) Discrete variable (qubit) version

from ._discrete import (
    optimalGenuineMultipartiteEntanglementWitness 
        as optimal_discrete_witness
)

# (2.0) Continuous variable (Gaussian) version

from ._gaussian import (
    optimalGenuineMultipartiteEntanglementWitness 
        as optimal_gaussian_witness,
    optimalGenuineMultipartiteEntanglementWitnessLinearGraph
        as optimal_gaussian_witness_blind_linear
)

# (2.1) Continuous variable (Gaussian) utilities

from ._gaussian import (
    optimalVarianceMatrixFromWitness 
        as optimal_gaussian_state,
    randomVarianceMatrix 
        as random_gaussian_state,
    computePairwisePPTcondition
        as gaussian_pairwise_ppt
)


#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Witnessing genuine multi-partite entanglement from a subset of marginalia
# both in multi-partite qubit systems and multi-mode continuous variable
# Gaussian systems.

version = '0.2.2'

# (1.0) Discrete variable (qubit) version

from ._discrete import (
    optimalGenuineMultipartiteEntanglementWitness 
        as _optimal_discrete_witness
)

def optimal_discrete_witness (rho, num, use_system_list = []):
    '''
    Find an optimal partially blind witness of genuine multi-partite
    entanglement respective to a particular multi-qubit quantum system
    comprising at least three qubits.

    Parameters
    ----------
    rho : numpy.ndarray
        Matrix of the multi-qubit system, in Kronecker form.
    num : int
        Number of qubits making up the system.
    use_system_list : iterable over pairs
        Which two-body correlations to consider. If empty, the original
        configuration (see Miklin, 10.1103/PhysRevA.93.020104) is used. 
        In particular the three qubit case uses [ (0, 1), (0, 2), (1, 2) ].

    Returns
    -------
    float64
        Witness value.
    numpy.ndarray
        Witness matrix.
    '''

    return _optimal_discrete_witness(rho, num, list(use_system_list))

# (2.0) Continuous variable (Gaussian) version

from ._gaussian import (
    optimalGenuineMultipartiteEntanglementWitness 
        as optimal_gaussian_witness,
    optimalGenuineMultipartiteEntanglementWitnessLinearGraph
        as optimal_gaussian_witness_blind_linear
)

# (2.1) Continuous variable (Gaussian) utilities

from ._gaussian import (
    randomVarianceMatrix 
        as random_gaussian_state,
    computePairwisePPTcondition
        as gaussian_pairwise_ppt,
    optimalVarianceMatrixFromWitness 
        as optimal_gaussian_state
)


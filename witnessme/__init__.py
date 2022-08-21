#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Witnessing genuine multi-partite entanglement from a subset of marginalia
# both in multi-partite qubit systems and multi-mode continuous variable
# Gaussian systems.

version = '0.3.2'

# Discrete variable multi-qudit 
# genuine multi-partite entanglement witness

from ._discrete import (
    optimalGenuineMultipartiteEntanglementWitness 
        as _discrete_optimal_witness
)

def discrete_optimal_witness (rho, dim, use_pairs_list = []):
    '''
    Finds an optimal partially blind witness of genuine multi-partite
    entanglement respective to a particular multi-partite quantum system
    comprising at least three parts.

    Parameters
    ----------
    rho : numpy.ndarray
        Matrix of the multi-qudit system, in Kronecker form.
    dim : list of integers 
        Dimensions of individual parts making up the multi-partite system.
    use_pairs_list : iterable over pairs
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

    return _discrete_optimal_witness(rho, dim, use_pairs_list)

# Continuous variable Gaussian
# genuine multi-partite entanglement witness

from ._gaussian import (
    optimalGenuineMultipartiteEntanglementWitness 
        as _gaussian_optimal_witness,
    optimalGenuineMultipartiteEntanglementWitnessLinearGraph
        as _gaussian_optimal_witness_linear,
    optimalVarianceMatrixFromWitness 
        as _gaussian_optimal_covmat,
    randomVarianceMatrix 
        as _gaussian_random_covmat,
    computePairwisePPTcondition
        as _gaussian_pairwise_ppt
)

def gaussian_optimal_witness (cov, num, use_pairs_list = []):
    '''
    Finds an optimal partially blind witness of genuine multi-partite
    entanglement respective to a particular multi-mode continous variable 
    Gaussian quantum state characterized by its covariance matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix, interleaved (X, P) form.
    num : int
        Number of modes.
    use_pairs_list : iterable over pairs
        Which two-body correlations to consider. 
        If empty, the full covariance matrix is used.

    Returns
    -------
    float64
        Witness value.
    numpy.ndarray
        Witness matrix.
    '''

    return _gaussian_optimal_witness(cov, num, use_pairs_list)

def gaussian_optimal_witness_linear (cov, num):
    '''
    Finds an optimal partially blind witness of genuine multi-partite
    entanglement respective to a particular multi-mode continous variable 
    Gaussian quantum state characterized by its covariance matrix.

    The witness is partially blind and relies only on the nearest-neighbor
    two-body correlations, that is, on [ (0, 1), (1, 2), ... (num - 1, num) ].

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix, interleaved (X, P) form.
    num : int
        Number of modes.

    Returns
    -------
    float64
        Witness value.
    numpy.ndarray
        Witness matrix.
    '''

    return _gaussian_optimal_witness_linear(cov, num)

def gaussian_optimal_covmat (wit, num, min_eig = 0.2, min_diag = 1, max_diag = 10):
    '''
    Finds covariance matrix of Gaussian state minimizing the provided
    entanglement witness.

    Elements of the covariance matrix can be constrained using the optional
    parameters min_eig, min_diag and max_diag. For example, setting a lower
    bound on eigenvalues limits squeezing.

    Parameters
    ----------
    wit : numpy.ndarray
        Witness matrix.
    num : int
        Number of modes.
    min_eig : float64
        Lower bound on eigenvalues.
    min_diag : float64
        Lower bound on diagonal elements.
    max_diag : float64
        Upper bound on diagonal elements.

    Returns
    -------
    float64
        Witness value.
    numpy.ndarray
        Covariance matrix.
    '''

    return _gaussian_optimal_covmat(wit, num, min_eig, min_diag, max_diag)

def gaussian_random_covmat (num):
    '''
    Constructs a random Gaussian covariance matrix.

    Parameters
    ----------
    num : int
        Number of modes.

    Returns
    -------
    numpy.ndarray
        Covariance matrix.
    '''

    return _gaussian_random_covmat(num)

def gaussian_pairwise_ppt (cov, num):
    '''
    Computes Gaussian PPT criterion for all two-mode marginals of Gaussian
    state characterized by its covariance matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix, interleaved (X, P) form.
    num : int
        Number of modes.

    Returns
    -------
    float64 generator
        Least eigenvalues of the PPT condition applied to the marginals.
        Negative value implies the two-mode marginal is entangled.
    '''

    return _gaussian_pairwise_ppt(cov, num)


#!/bin/env python

# 2019 - 2023 Jan Provaznik (jan@provaznik.pro)
#
# Witnessing genuine multipartite entanglement [1] presents the semi-definite
# program (44 - 45) which can be used to detect genuinely multi-partite
# entangled quantum systems from their $X, P$ variance matrices $\gamma$.
# 
# Our paper builds on the original concept by introducing additional
# constraints, effectively restricting the Witness to a subset of marginalia,
# similar to the discrete variable approach described by Paraschiv [3].
#
# [1] https://doi.org/10.1088/1367-2630/8/4/051         (Hyllus, Eisert)
# [2] https://doi.org/10.1103/PhysRevA.106.062410       (Nordgren, Mista)
# [3] https://doi.org/10.1103/PhysRevA.98.062102        (Paraschiv)
# [4] https://doi.org/10.1142/s1230161214400010         (Adesso)
# [5] https://doi.org/10.1103/PhysRevA.49.1567          (Simon)

import picos
import numpy
import scipy.linalg

from ._pi import picos_solve_problem
from ._pi import picos_debug_constraints
from ._pi import bipartition_count, bipartition_index_list

# Implementation of the optimal genuine multipartite entanglement witness as
# per equations (44 - 45). Including our modifications.

def _picos_block_diagonal (u, v):
    return picos.block([ [u, 0], [0, v] ])

def _picos_sigma_matrix (mode_count):
    matrix = picos.Constant([ [ 0, 1 ], [ -1, 0 ]], shape = (2, 2))
    result = matrix
    for index in range(mode_count - 1):
        result = _picos_block_diagonal(result, matrix)
    return result

def _picos_partial_transpose_matrix (mode_count, mode_target):
    d = cm_build_ptranspose(mode_count, mode_target)
    return picos.Constant(d)

# Map system component index to its respective matrix indices.

def _system_index_to_matrix_indices (system_index):
    index = 2 * system_index
    return index, index + 1

# Constraints (1, k) as per (H.44)
    
def _set_partition_constraints (P, partition_index_list, X_1, X_k_1):
    length = len(partition_index_list)

    for i in range(length):
        system_index_i = partition_index_list[i]
        _set_partition_constraints_ij(P, system_index_i, system_index_i, X_1, X_k_1)

        for j in range(i + 1, length):
            system_index_j = partition_index_list[j]
            _set_partition_constraints_ij(P, system_index_i, system_index_j, X_1, X_k_1)

def _set_partition_constraints_ij (P, system_index_i, system_index_j, X_1, X_k_1):
    I = _system_index_to_matrix_indices(system_index_i)
    J = _system_index_to_matrix_indices(system_index_j)

    F = (X_1[I, J].real == X_k_1[I, J].real)
    P.add_constraint(F)

# Essentially the adjacency matrix complement, 
# more or less I suppose?

def _build_blind_list (use_system_list, N):
    use_system_list = set(use_system_list)
    all_system_list = set((i, j) for i in range(N) for j in range(i + 1, N))

    return list(all_system_list - use_system_list)

# Prohibit correlations between the (i, j) subsystems in the witness matrix.

def _set_blindness_constraints_ij (P, system_index_i, system_index_j, X_1):
    I = _system_index_to_matrix_indices(system_index_i)
    J = _system_index_to_matrix_indices(system_index_j)
        
    F = (X_1[I, J] == 0)
    P.add_constraint(F)

def cm_optimal_witness (covariance_matrix, mode_count, use_pairs_list = []):
    '''
    Finds an optimal (partially blind) witness of genuine multi-partite
    entanglement respective to a particular multi-mode continuous variable 
    Gaussian quantum state characterized by its covariance matrix.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Covariance matrix in X-P interleaved form.
    mode_count : int
        Number of modes.
    use_pairs_list : list of pairs
        Which two-body correlations to consider. If empty or unset, the entire
        covariance matrix is used.

    Returns
    -------
    float64
        Witness value.
    numpy.ndarray
        Witness matrix.
    '''

    G = covariance_matrix
    N = mode_count

    D = 2 * N
    K = bipartition_count(N)
    S = _picos_sigma_matrix(N)

    X_1 = picos.SymmetricVariable('X_1', shape = (D, D))
    X_K_2 = picos.RealVariable('X_{0}'.format(K + 2))
    X_K_3 = picos.RealVariable('X_{0}'.format(K + 3))
    
    P = picos.Problem()
    
    # Original (Eisert, Hyllus) semi-definite constraints for genuine
    # multipartite (GME) witnesses with separable bipartite marginalia.

    for k in range(1, K + 1):
        X_k_1   = picos.HermitianVariable('X_{0}'.format(k + 1), shape = (D, D))
        X_K_3_k = picos.RealVariable('X_{0}'.format(K + 3 + k))
        
        # Constraint (1, k) as per (H.44)
        for partition_index_list in bipartition_index_list(k, N):
            _set_partition_constraints(P, partition_index_list, X_1, X_k_1)
        
        # Constraint (2, k) as per (H.44)
        F = (picos.trace(1j * S * X_k_1) + X_K_2 - X_K_3 + X_K_3_k == 0)
        P.add_constraint(F)
        
        # Constraints on matrix positivity
        P.add_constraint(X_k_1   >> 0)
        P.add_constraint(X_K_3_k >> 0)
    
    # Constraint (3) as per (H.44)
    P.add_constraint(X_K_2 - X_K_3 == 1)
    
    # Constraints on matrix positivity
    P.add_constraint(X_1   >> 0)
    P.add_constraint(X_K_2 >> 0)
    P.add_constraint(X_K_3 >> 0)
    
    # Additional constraints on partial-blindness, permitting only certain 
    # correlations between the (i, j) subsystems in the witness matrix.
    #
    # Apply only if we specified the use_pairs_list.

    if use_pairs_list:
        nil_system_list = _build_blind_list(use_pairs_list, N)
        for (system_index_i, system_index_j) in nil_system_list:
            _set_blindness_constraints_ij(P, system_index_i, system_index_j, X_1)
    
    # Objective function as per equation (H.45)
    F = (picos.trace(G * X_1.real) - 1)
    P.set_objective('min', F)

    # Fire in the hole!
    S = picos_solve_problem(P)

    # Return values
    return S.value, numpy.array(X_1.real.value)

# Implementation of search for optimal state $\gamma$ respective to particular
# witness $W$ with additional constraints on diagonal elements and minimal
# eigenvalues.

def cm_optimal_gaussian (witness_matrix, mode_count, min_phy = 1e-6, min_ppt = 1e-6, min_eig = 0.2, min_diag = 1, max_diag = 10):
    '''
    Finds a covariance matrix of some Gaussian state minimizing the provided
    entanglement witness. 

    The optimal state is constrained to have separable two-body marginals.
    Elements of its covariance matrix can be further constrained using the
    optional parameters min_eig, min_diag and max_diag. For example, setting a
    lower bound on eigenvalues through min_eig limits amount of squeezing.

    Parameters
    ----------
    witness_matrix : numpy.ndarray
        Witness matrix.
    mode_count : int
        Number of modes.
    min_phy: float64
        Lower bound on physicality criterion.
    min_ppt: float64
        Lower bound on the pairwise Peres-Horodecki criterion.
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

    W = witness_matrix
    N = mode_count

    S = _picos_sigma_matrix(N)
    s = _picos_sigma_matrix(2)
    T = _picos_partial_transpose_matrix(2, 1)

    D = 2 * N
    G = picos.SymmetricVariable('G', shape = (D, D))
    P = picos.Problem()
    
    # Objective function as per equation (H.45)
    
    F = picos.trace(G * W)
    P.set_objective('min', F)

    # Constraint (1) on uncertainty relations (cm_is_physical)
    P.add_constraint(min_phy <= picos.lambda_min(G + 1j * S))
    
    # Constraint (2) on pairwise separability (cm_is_pairwise_ppt)
    for i in range(N):
        for j in range(i + 1, N):
            I = _system_index_to_matrix_indices(i)
            J = _system_index_to_matrix_indices(j)
            K = (* I, * J)

            # The (i, j) marginal.
            M = G[K, K]

            # To ensure pairwise separability, the respective 
            # partially transposed (i, j) marginal must be physical, that is,
            # min eig (T * M * T + 1j * sigma) > 0.
            Q = T * M * T + 1j * s

            # To ensure it is sufficiently physical, we further require that
            # min eig (T * M * T + 1j * sigma) > min_ppt.
            P.add_constraint(min_ppt <= picos.lambda_min(Q))
            
    # Constraints (3) (remove X, P correlations)
    for i in range(N):
        for j in range(N):
            P.add_constraint(G[2 * i, 2 * j + 1] == 0)
            P.add_constraint(G[2 * i + 1, 2 * j] == 0)
            
    # Constraints (A) (impose limits on diagonals)
    for i in range(D):
        P.add_constraint(min_diag <= G[i, i])
        P.add_constraint(G[i, i] <= max_diag)

    # Constraints (B) (impose limits on squeezing)
    P.add_constraint(min_eig <= picos.lambda_min(G))
    
    # Fire in the hole!
    S = picos_solve_problem(P)
    
    # Return values
    return S.value, numpy.array(G.value)

def cm_build_ptranspose (mode_count, mode_target):
    '''
    Constructs a transformation matrix facilitating partial transposition in
    phase space. This corresponds to mirror reflection with respect to the
    position quadrature (https://doi.org/10.1103/PhysRevLett.84.2726).

    Parameters
    ----------
    mode_count : int
        Number of modes.
    mode_target : int
        Target mode to be partially transposed.

    Returns
    -------
    numpy.ndarray
        Transformation matrix in X-P interleaved form.
    '''

    T = numpy.ones(2 * mode_count)
    T[2 * mode_target + 1] = - 1
    return numpy.diag(T)


def cm_build_quadrature_reordering (mode_count):
    '''
    Constructs transformation matrix facilitating transition from
    X-P interleaved to X-P grouped ordering of quadratures.

    Parameters
    ----------
    mode_count : int
        Number of modes.

    Returns
    -------
    numpy.ndarray
        A transformation matrix.
    '''

    D = 2 * mode_count
    T = numpy.zeros((D, D))
    
    for j in range(0, mode_count):
        T[j, 2 * j] = 1
        T[mode_count + j, 2 * j + 1] = 1
        
    return T

def cm_build_random (mode_count, spectral_factor = 2.0):
    '''
    Constructs a bona-fide random (multi-mode) Gaussian covariance matrix.

    The construction process is loosely based on the derivation of Theorem 3
    from (https://doi.org/10.1103/PhysRevA.49.1567). 

    - Per Equation (2.28) the symplectic eigenvalues in the canonical form of
      bona-fide covariance matrix are lower bound. The lower bound, adjusted
      for the alternative definition of covariance matrix, equals to 1. 

    The algorithm works as follows.

    - Construct a random vector of physical symplectic eigenvalues
      from the interval (1, 1 + spectral_factor).

    - Construct a random invertible matrix. This can be done by sampling some
      random distribution until the determinant is sufficiently non-zero.

    - Construct a random symplectic matrix from the orthogonal matrix.
      We employ the generator expression (ii) for symplectic rotations from
      Lemma 1 of (https://doi.org/10.2307/1993590). 

    - Construct a random covariance matrix in X-P grouped format. Transform it
      to the expected X-P interleaved form.

    Parameters
    ----------
    mode_count : int
        Number of modes.
    spectral_factor : float64
        Factor (positive) for scaling of symplectic eigenvalues.

    Returns
    -------
    numpy.ndarray
        Covariance matrix.
    '''

    assert mode_count > 0
    assert spectral_factor > 0

    # (1) Construct a random vector of valid symplectic eigenvalues
    V = numpy.diag(1 + spectral_factor * numpy.random.rand(mode_count))
    
    # (2) Construct a random invertible matrix
    M = numpy.random.rand(mode_count, mode_count)
    while numpy.abs(scipy.linalg.det(M)) < 1e-1:
        M = numpy.random.rand(mode_count, mode_count)

    # (3) Construct a random symplectic matrix
    X = M
    P = numpy.linalg.inv(M).T
    
    # (3) Construct a random covariance matrix without X-P correlations 
    # in X-P grouped matrix format
    G = scipy.linalg.block_diag(X @ V @ X.T, P @ V @ P.T)

    # (4) Transformation matrix from X-P interleaved to X-P grouped format
    T = cm_build_quadrature_reordering(mode_count)

    return T.T @ G @ T

def cm_is_physical (covariance_matrix, mode_count):
    '''
    Every physical covariance matrix G must satisfy Heisenberg uncertainty
    relations. This can be determined by checking if (G + 1j * sigma) is a
    positive semi-definite matrix.

    See Theorem 3 in (https://doi.org/10.1103/PhysRevA.49.1567) for the
    original derivation of the condition. 

    See Equation 5 in (https://doi.org/10.1088/1367-2630/8/4/051) for the
    condition adapted to the alternative definition of variance matrix used
    throughout the paper and consequently this library.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Covariance matrix (multi-mode).
    mode_count : int
        Number of modes in G.

    Returns
    -------
    float64
        The lowest eigenvalue of (G + 1j * sigma) matrix.

    References
    ----------
    '''

    value_sigma = cm_build_sigma(mode_count)
    value_simon = numpy.linalg.eigvalsh(covariance_matrix + 1j * value_sigma).min() 

    condition_simon = (value_simon >= 0)
    condition_issym = scipy.linalg.issymmetric(covariance_matrix, rtol = 1e-8)

    return (condition_simon and condition_issym)

def cm_build_sigma (mode_count):
    '''
    Constructs a (multi-mode) symplectic form (sigma).

    Parameters
    ----------
    mode_count : int
        Number of modes.

    Returns
    -------
    numpy.ndarray
        Resulting (multi-mode) symplectic form.
    '''

    sigma = numpy.array([ [ 0, 1 ], [ -1, 0 ]], dtype = numpy.float64)
    array = [ sigma ] * mode_count
    return scipy.linalg.block_diag(* array)

def cm_pairwise_pt (covariance_matrix, mode_count):
    '''
    The continuous variable version of the Peres-Horodecki separability
    criterion (positive partial transpose, PPT) uses covariance matrices. 

    If the matrix represents a separable state, its partially transposed
    covariance matrix necessarily represents a physical state. For certain
    special classes of continuous variable states, such as those with Gaussian
    Wigner functions, it actually is a sufficient condition of separability.

    Given a covariance matrix of some multi-mode state this function produces
    its two-mode marginals. It partially partially transposes their covariance
    matrices and computes the value of the physicality condition. It then
    returns a list of these values. 

    These values can be used to indicate that the two-body marginals are
    separable. This depends on the state, namely whether the separability
    criterion is sufficient. Should it be sufficient and should all the values
    be positive, we say the state is pairwise separable.

    See Simon (https://doi.org/10.1103/PhysRevLett.84.2726) for details on the
    continuous variable version of Peres-Horodecki criterion.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Covariance matrix in X-P interleaved form.
    mode_count: int
        Number of modes.

    Returns
    -------
    numpy.ndarray 
        A vector of values. See above for their interpretation.
    '''

    generator = _pairwise_PT_impl(covariance_matrix, mode_count)
    return numpy.array(list(generator))

def cm_is_pairwise_ppt (covariance_matrix, mode_count):
    '''
    Checks the continuous variable version of the Peres-Horodecki separability
    criterion (positive partial transpose, PPT) for all two-mode marginals of 
    continuous variable state characterized by its covariance matrix.

    See Simon (https://doi.org/10.1103/PhysRevLett.84.2726) for details on the
    continuous variable version of Peres-Horodecki criterion.

    Parameters
    ----------
    covariance_matrix : numpy.ndarray
        Covariance matrix in X-P interleaved form.
    mode_count: int
        Number of modes.

    Returns
    -------
    bool
        True if the partial transpose is positive for every two-mode marginal.
    '''

    return cm_pairwise_pt(covariance_matrix, mode_count).min() > 0

def _pairwise_PT_impl (G, mode_count):
    for mode_index_i in range(mode_count):
        for mode_index_j in range(mode_index_i + 1, mode_count):
            yield _pairwise_PT_impl_ij(G, mode_count, mode_index_i, mode_index_j)

def _pairwise_PT_impl_ij (G, mode_count, mode_index_i, mode_index_j):
    I = _system_index_to_matrix_indices(mode_index_i)
    J = _system_index_to_matrix_indices(mode_index_j)
    K = * I, * J

    M = G[numpy.ix_(K, K)]

    S = cm_build_sigma(2)
    T = cm_build_ptranspose(2, 1)
    Q = T @ M @ T + 1j * S

    return numpy.linalg.eigvalsh(Q).min()


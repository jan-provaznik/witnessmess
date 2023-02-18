#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
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

import picos
import numpy
import scipy.linalg

def picos_block_diagonal (u, v):
    return picos.block([ [u, 0], [0, v] ])

# The symplectic $X, P$ commutator matrix $\sigma$ as per the equation (H.3).

def picos_sigma_matrix (N):
    matrix = picos.Constant([ [ 0, 1 ], [ -1, 0 ]], shape = (2, 2))
    result = matrix
    for index in range(N - 1):
        result = picos_block_diagonal(result, matrix)
    return result

# Constructs a CV-partial-transpose transformation matrix.
#
# Suppose there are $N$ modes total and the mode $j$ is to be partially
# transposed. The diagonal matrix $\Lambda$ facilitates this transformation
# as $\gamma^{T_{j}} \equiv \Lambda \gamma \Lambda$.

def build_partial_transpose_matrix (N, j):
    T = numpy.ones(2 * N, dtype = numpy.int64)
    T[2 * j + 1] = - 1
    return numpy.diag(T)

def picos_partial_transpose_matrix (N, j):
    d = build_partial_transpose_matrix(N, j)
    return picos.Constant(d)

# The number $K$ of unique unordered bi-partitions of $N$ elements is given by
# the Stirling number of the second kind [4] as $K := 2^{N - 1} - 1$.
#
# [4] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind

def bipartition_count (N):
    return 2 ** (N - 1) - 1

def bipartition_index_list (k, N):
    partition_p_index_list = [ j for j in range(N) if     (k & (1 << j)) ]
    partition_q_index_list = [ j for j in range(N) if not (k & (1 << j)) ]
    
    return partition_p_index_list, partition_q_index_list

# Implementation of the optimal genuine multipartite entanglement witness as
# per equations (44 - 45). Including our modifications.

def _genuineMultipartiteEntanglementWitness (X_1, G, N, use_system_list = []):

    D = 2 * N
    K = bipartition_count(N)
    S = picos_sigma_matrix(N)

    # Previously the X_1 block would be set up here, however, 
    # this is deferred to the caller now.
    #
    # X_1 = picos.SymmetricVariable('X_1', shape = (D, D))

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
            set_partition_constraints(P, partition_index_list, X_1, X_k_1)
        
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
    # Apply only if we specified the use_system_list.

    if use_system_list:
        nil_system_list = build_blind_list(use_system_list, N)
        for (system_index_i, system_index_j) in nil_system_list:
            set_blindness_constraints_ij(P, system_index_i, system_index_j, X_1)
    
    # Return the problem (without objective) 

    return P

def optimalGenuineMultipartiteEntanglementWitness (G, N, use_system_list = []):

    D = 2 * N
    W = picos.SymmetricVariable('X_1', shape = (D, D))
    P = _genuineMultipartiteEntanglementWitness(W, G, N, use_system_list)
    
    # Objective function as per equation (H.45)
    F = (picos.trace(G * W.real) - 1)
    P.set_objective('min', F)

    # Fire in the hole!
    S = P.solve(solver = 'mosek', mosek_params = { 'MSK_IPAR_NUM_THREADS' : 0 })

    # ...
    ret = S.value, numpy.array(W.value)

    # We have a memory leak somewhere. 
    # One of the suspects is MOSEK and the way it is used by PICOS.
    #
    # Per its documentation (section 10.1)
    # https://docs.mosek.com/latest/pythonapi/guidelines-optimizer.html
    # we try to manually release the internal allocations.
    #
    # P.strategy.solver.int.__del__()
    # P.strategy.solver.env.__del__()

    # We must also reset some of the MOSEKSolver internals of PICOS sice the
    # manual release of the mosek.Env is not expected to happen.
    #
    # delattr(picos.solvers.MOSEKSolver, 'mosekEnvironment')

    # ...
    # And the memory keeps increasing anyway. 
    # MOSEK and PICOS are not the culprits I suppose.
    
    # Return values
    return ret

def confirmGenuineMultipartiteEntanglementWitness (W, G, N, use_system_list = []):

    D = 2 * N
    W = picos.Constant('X_1', W, shape = (D, D))
    P = _genuineMultipartiteEntanglementWitness(W, G, N, use_system_list)
    
    # Objective function NOT set as per documentation
    P.set_objective('find')
    
    # Fire in the hole!
    S = P.solve(solver = 'mosek', mosek_params = { 'MSK_IPAR_NUM_THREADS' : 0 })
    
    # Return values
    return S

# Map system component index to its respective matrix indices.

def system_index_to_matrix_indices (system_index):
    index = 2 * system_index
    return index, index + 1

# Constraints (1, k) as per (H.44)
    
def set_partition_constraints (P, partition_index_list, X_1, X_k_1):
    length = len(partition_index_list)

    for i in range(length):
        system_index_i = partition_index_list[i]
        set_partition_constraints_ij(P, system_index_i, system_index_i, X_1, X_k_1)

        for j in range(i + 1, length):
            system_index_j = partition_index_list[j]
            set_partition_constraints_ij(P, system_index_i, system_index_j, X_1, X_k_1)

def set_partition_constraints_ij (P, system_index_i, system_index_j, X_1, X_k_1):
    I = system_index_to_matrix_indices(system_index_i)
    J = system_index_to_matrix_indices(system_index_j)

    F = (X_1[I, J].real == X_k_1[I, J].real)
    P.add_constraint(F)

# Essentially the adjacency matrix complement, 
# more or less I suppose?

def build_blind_list (use_system_list, N):
    use_system_list = set(use_system_list)
    all_system_list = set((i, j) for i in range(N) for j in range(i + 1, N))

    return list(all_system_list - use_system_list)

# Prohibit correlations between the (i, j) subsystems in the witness matrix.

def set_blindness_constraints_ij (P, system_index_i, system_index_j, X_1):
    I = system_index_to_matrix_indices(system_index_i)
    J = system_index_to_matrix_indices(system_index_j)
        
    F = (X_1[I, J] == 0)
    P.add_constraint(F)

# That is all, folks!

def optimalGenuineMultipartiteEntanglementWitnessLinearGraph (G, N):
    use_system_list = [ (i, i + 1) for i in range(N - 1) ]
    return optimalGenuineMultipartiteEntanglementWitness(G, N, use_system_list)

def confirmGenuineMultipartiteEntanglementWitnessLinearGraph (W, G, N):
    use_system_list = [ (i, i + 1) for i in range(N - 1) ]
    return confirmGenuineMultipartiteEntanglementWitness(W, G, N, use_system_list)

# Implementation of search for optimal state $\gamma$ respective to particular
# witness $W$ with additional constraints on diagonal elements and minimal
# eigenvalues.

def optimalVarianceMatrixFromWitness (W, N, mineig = 0.2, mindiag = 1, maxdiag = 10):
    S = picos_sigma_matrix(N)
    s = picos_sigma_matrix(2)
    T = picos_partial_transpose_matrix(2, 1)

    D = 2 * N
    G = picos.SymmetricVariable('G', shape = (D, D))
    P = picos.Problem()
    
    # Objective function as per equation (H.45)
    
    F = picos.trace(G * W)
    P.set_objective('min', F)

    # Constraint (1)
    
    P.add_constraint(G + 1j * S >> 0)
    
    # Constraints (2) (partial transpose)
    
    for i in range(N):
        for j in range(i + 1, N):
            I = system_index_to_matrix_indices(i)
            J = system_index_to_matrix_indices(j)
            K = (* I, * J)
            
            P.add_constraint(T * G[K, K] * T + 1j * s >> 0)
            
    # Constraints (3) (remove X, P correlations)
    
    for i in range(N):
        for j in range(N):
            P.add_constraint(G[2 * i, 2 * j + 1] == 0)
            P.add_constraint(G[2 * i + 1, 2 * j] == 0)
            
    # Constraints (A) (impose limits on diagonals and squeezing)
    
    for i in range(D):
        P.add_constraint(mindiag <= G[i, i])
        P.add_constraint(G[i, i] <= maxdiag)
    
    P.add_constraint(mineig <= picos.lambda_min(G))
    
    # Fire in the hole!
    
    S = P.solve(solver = 'mosek', mosek_params = { 'MSK_IPAR_NUM_THREADS' : 0 })
    
    # Return values
    
    return S.value, numpy.array(G.value)

# Constructs a reordering transformation matrix $T$, 
# used in construction of random states with uncorrelated $X, P$ quadratures.

def build_reordering_matrix (N):
    D = 2 * N
    T = numpy.zeros([ D, D ])
    
    for j in range(0, N):
        T[j, 2 * j] = 1
        T[N + j, 2 * j + 1] = 1
        
    return T

# Constructs a random variance matrix $\gamma$ without $X, P$ correlations.

def randomVarianceMatrix (N):
    R = numpy.random.rand(N, N)
    S = numpy.linalg.inv(R)
    T = build_reordering_matrix(N)
    
    v = numpy.random.rand(N)
    V = numpy.diag(v)
    
    X = R   @ V @ R.T
    P = S.T @ V @ S
    
    s = scipy.linalg.block_diag(X, P)
    G = T.T @ s @ T
    
    return G

# Pairwise entanglement computation.

def build_sigma_matrix (N):
    matrix = numpy.array([ [ 0, 1 ], [ -1, 0 ]])
    result = matrix
    for index in range(N - 1):
        result = scipy.linalg.block_diag(result, matrix)
    return result

def pairwise_PPT_impl (G, N):
    S = build_sigma_matrix(2)
    T = build_partial_transpose_matrix(2, 1)

    for system_index_i in range(N):
        for system_index_j in range(system_index_i + 1, N):

            I = system_index_to_matrix_indices(system_index_i)
            J = system_index_to_matrix_indices(system_index_j)
            K = * I, * J

            r = numpy.array(K)[:, numpy.newaxis]
            c = numpy.array(K)[numpy.newaxis, :]
            M = G[r, c]

            m = T @ M @ T + 1j * S
            e = numpy.real(numpy.linalg.eigvals(m)).min()

            yield e

def computePairwisePPTcondition (G, N):
    return list(pairwise_PPT_impl(G, N))

# Module exports

__all__ = [
    'optimalGenuineMultipartiteEntanglementWitness',
    'optimalGenuineMultipartiteEntanglementWitnessLinearGraph',
    'confirmGenuineMultipartiteEntanglementWitness',
    'confirmGenuineMultipartiteEntanglementWitnessLinearGraph',
    'optimalVarianceMatrixFromWitness',
    'computePairwisePPTcondition',
    'randomVarianceMatrix',
]


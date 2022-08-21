#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Witnessing genuine multipartite entanglement from a subset of marginalia
# using a semi-definite program (9) introduced in [1], discussed in [2]
# and experimentally demonstrated in [3].
#
# [1] https://dx.doi.org/10.1103/PhysRevA.98.062102 (Paraschiv)
# [2] https://dx.doi.org/10.1103/PhysRevA.93.020104 (Miklin)
# [3] https://dx.doi.org/10.1364/QIM.2019.F5A.45    (Micuda)

import picos
import numpy
import functools

def bipartition_count (N):
    return 2 ** (N - 1) - 1

def bipartition_index_list (k, N):
    partition_p_index_list = [ j for j in range(N) if     (k & (1 << j)) ]
    partition_q_index_list = [ j for j in range(N) if not (k & (1 << j)) ]
    
    return partition_p_index_list, partition_q_index_list

def picos_pauli_matrix (j):
    if (j == 0):
        return picos.Constant([ [1, 0], [0, 1] ], shape = (2, 2))
    if (j == 1):
        return picos.Constant([ [0, 1], [1, 0] ], shape = (2, 2))
    if (j == 2):
        return picos.Constant([ [0, -1j], [1j, 0] ], shape = (2, 2))
    if (j == 3):
        return picos.Constant([ [1, 0], [0, -1] ], shape = (2, 2))
    raise ValueError

# Wrapper for functional programming (since picos.kron is deprecated) 

def picos_kron_wrapper (p, q):
    return p @ q

# The program (9) from [1] constrains the witness structure to only the
# bipartite subsystems. The witness is defined as weighted sum of Kronecker
# products of Pauli matrices and consequently the optimal 
# weights $w_{i, j}^{p, q}$ are found.

def optimalGenuineMultipartiteEntanglementWitness (rho, N, use_system_list = []):
    D = 2 ** N
    K = bipartition_count(N)
    
    W = 0
    P = picos.Problem()
    
    # Unless the caller specifies the marginals, we presume the original
    # genuine multipartite entanglement presented by Miklin [2] in (3)
    
    if not use_system_list:
        use_system_list = []
        for system_index_p in range(0, N):
            for system_index_q in range(system_index_p + 1, N):
                use_system_list.append((system_index_p, system_index_q))
    
    # Construct the Witness matrix with the desired structure
    
    for (system_index_p, system_index_q) in use_system_list:
        for sigma_index_i in range(4):
            for sigma_index_j in range(4):
                wij = picos.RealVariable('w_({0},{1})^({2},{3})'.format(sigma_index_i, sigma_index_j, system_index_p, system_index_q))
                mij = picos_witness_matrix_pq_ij(system_index_p, system_index_q, sigma_index_i, sigma_index_j, N)

                W = W + mij * wij
    
    # Constrain its normalisation
    
    P.add_constraint(picos.trace(W) == 1)
    
    # Constrain its marginals, prohibiting bipartite entanglement
    
    for k in range(1, K + 1):
        index_list, _ = bipartition_index_list(k, N)

        Pk = picos.HermitianVariable('P_{0}'.format(k), shape = (D, D))
        Qk = picos.partial_transpose(W - Pk, index_list, 2)

        P.add_constraint(Qk >> 0)
        P.add_constraint(Pk >> 0)
        
    # Construct the objective

    F = picos.trace(W * rho)
    F = F.real
    
    P.set_objective('min', F)
    
    # Fire in the hole!
    
    S = P.solve(solver = 'mosek', mosek_params = { 'MSK_IPAR_NUM_THREADS' : 0 })
    
    # Return values
    
    return S.value, numpy.array(W.value)

# Assuming $p < q$, we construct the large Kronecker product of Pauli matrices
# respective to the $w_{i, j}^{p, q}$ coefficient.

def picos_witness_matrix_pq_ij (system_index_p, system_index_q, sigma_index_i, sigma_index_j, N):
    vector = build_witness_matrix_pq_ij_vector(system_index_p, system_index_q, sigma_index_i, sigma_index_j, N)
    vector = map(picos_pauli_matrix, vector)
    result = functools.reduce(picos_kron_wrapper, vector)
    return result

# We first determine the sequence of Pauli matrices in the large matrix,
# we then map this sequence of indices into a sequence of Pauli matrices,
# and then the caller produces the large matrix by reducing the matrix sequence
# with the Kronecker product into a single matrix.

def build_witness_matrix_pq_ij_vector (system_index_p, system_index_q, sigma_index_i, sigma_index_j, N):
    assert(system_index_p < system_index_q)
    
    result = []
    result = result + [ 0 ] * system_index_p
    result = result + [ sigma_index_i ]
    result = result + [ 0 ] * (system_index_q - system_index_p - 1)
    result = result + [ sigma_index_j ]
    result = result + [ 0 ] * (N - system_index_q - 1)
    return result

# Yes, it really is this short.

__all__ = [
    'optimalGenuineMultipartiteEntanglementWitness'
]


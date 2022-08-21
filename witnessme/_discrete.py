#!/bin/env python

# 2019 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Witnessing genuine multipartite entanglement from a subset of marginalia
# using a semi-definite program (9) introduced in [1], discussed in [2]
# and experimentally demonstrated in [3].
#
# A general multi-qudit states are supported by using generalized Gell-Mann
# matrices [4, (3-5)], a generalization of Pauli matrices to higher dimensions.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.98.062102     (Paraschiv)
# [2] https://dx.doi.org/10.1103/PhysRevA.93.020104     (Miklin)
# [3] https://dx.doi.org/10.1364/QIM.2019.F5A.45        (Micuda)
# [4] http://dx.doi.org/10.1088/1751-8113/41/23/235303  (Bertlmann)

import picos
import numpy
import operator
import functools

def bipartition_count (N):
    return 2 ** (N - 1) - 1

def bipartition_index_list (k, N):
    partition_p_index_list = [ j for j in range(N) if     (k & (1 << j)) ]
    partition_q_index_list = [ j for j in range(N) if not (k & (1 << j)) ]
    
    return partition_p_index_list, partition_q_index_list

# Wrapper for functional programming (since picos.kron is deprecated) 

def picos_kron_wrapper (p, q):
    return p @ q

# The program (9) from [1] constrains the witness structure to only the
# bipartite subsystems. The witness is defined as weighted sum of Kronecker
# products of Pauli matrices and consequently the optimal 
# weights $w_{i, j}^{p, q}$ are found.
#
# The density matrix (rho) is given in Kronecker form and its components may be
# differently dimensional, their dimensions are given by the (dims) list.

def optimalGenuineMultipartiteEntanglementWitness (rho, dims, use_pairs_list = []):
    N = len(dims)
    D = numpy.prod(dims)
    K = bipartition_count(N)
    
    W = 0
    P = picos.Problem()

    # Construct the GGM matrices beforehand.

    M = [ build_ggm_matrices(dim) for dim in dims ]
    
    # Unless the caller specifies the marginals, we presume the original
    # genuine multipartite entanglement presented by Miklin [2] in (3), 
    # that is ALL the marginals.
    
    if not use_pairs_list:
        use_pairs_list = []
        for part_index_p in range(0, N):
            for part_index_q in range(part_index_p + 1, N):
                use_pairs_list.append((part_index_p, part_index_q))
    
    # Construct the Witness matrix with the desired structure
    
    for (part_index_p, part_index_q) in use_pairs_list:
        part_count_p = dims[part_index_p] ** 2
        part_count_q = dims[part_index_q] ** 2

        for sigma_index_i in range(part_count_p):
            for sigma_index_j in range(part_count_q):

                wij_label = 'w_({0},{1})^({2},{3})'.format(
                    sigma_index_i, 
                    sigma_index_j, 
                    part_index_p, 
                    part_index_q)

                wij = picos.RealVariable(wij_label)
                mij = picos_witness_matrix_pq_ij(
                    part_index_p, 
                    part_index_q, 
                    sigma_index_i, 
                    sigma_index_j, 
                    M, N)

                W = W + mij * wij
    
    # Constrain its normalisation
    
    P.add_constraint(picos.trace(W) == 1)
    
    # Constrain its marginals, prohibiting bipartite entanglement
    
    for k in range(1, K + 1):
        index_list, _ = bipartition_index_list(k, N)

        Pk = picos.HermitianVariable('P_{0}'.format(k), shape = (D, D))
        Qk = picos.partial_transpose(W - Pk, index_list, dims)

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

# Assuming $p < q$, we construct the large Kronecker product of GGM matrices
# respective to the $w_{i, j}^{p, q}$ coefficient.

def picos_witness_matrix_pq_ij (system_index_p, system_index_q, sigma_index_i, sigma_index_j, M, N):
    vector = build_witness_matrix_pq_ij_vector(system_index_p, system_index_q, sigma_index_i, sigma_index_j, N)
    vector = map(operator.getitem, M, vector)
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


# Generalized Gell-Mann matrices [4, (3-5)] are a generalization of Pauli 
# matrices to higher dimensions. The referenced paper contains explicit
# examples of three-, four- and five-level GGM matrices.

def _build_ejk (j, k, d):
    E = numpy.zeros([ d, d ])
    E[j, k] = 1
    return picos.Constant(E)

def _build_ggm_sjk (j, k, d):
    return (_build_ejk(j, k, d) + _build_ejk(k, j, d))

def _build_ggm_ajk (j, k, d):
    return (_build_ejk(k, j, d) - _build_ejk(j, k, d)) * 1j

def _build_ggm_dl (l, d):
    m = l + 1
    n = l + 2

    u = numpy.sqrt(2 / (m * n))
    v = sum(_build_ejk(j, j, d) for j in range(m))
    w = m * _build_ejk(m, m, d)

    return u * (v - w)

# Including the identity matrix, which is always placed at position 0, 
# build_ggm_matrices should return a list with d ** 2 matrices.

def build_ggm_matrices (d):
    matrices = [ picos.I(d) ]
    for k in range (d):
        for j in range(k):
            matrices.append(_build_ggm_sjk(j, k, d))
            matrices.append(_build_ggm_ajk(j, k, d))
    for l in range(d - 1):
        matrices.append(_build_ggm_dl(l, d))
    return matrices

# Yes, it really is this short.

__all__ = [
    'optimalGenuineMultipartiteEntanglementWitness'
]


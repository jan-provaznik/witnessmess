#!/bin/env python

# 2019 - 2023 Jan Provaznik (jan@provaznik.pro)
#
# Shared components.

import picos

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

def picos_solve_problem (problem):
    '''
    Executes problem.solve(...) with a solver we prefer.

    While picos.Problem.solve can select a compatible solver on its own, it
    only checks whether their respective modules exist. Unfortunately this 
    does not determine whether MOSEK (the preferred solver) can be actually
    used as it requires a valid license.

    Parameters
    ----------
    problem : picos.Problem
        The instance of picos.Problem this functions wraps.

    Returns
    -------
    picos.Solution
        Solution that would be returned by picos.Problem.solve call.
    '''

    solvers = picos.available_solvers()

    # We prefer MOSEK for its unmatched speed.
    if 'mosek' in solvers:
        if _mosek_check_license():
            return problem.solve(solver = 'mosek', 
                mosek_params = { 'MSK_IPAR_NUM_THREADS' : 0 })

    # We can always fall back onto CVXOPT. It is considerably slower.
    if 'cvxopt' in solvers:
        return problem.solve(solver = 'cvxopt')

    # Fallback to whatever PICOS deems good enough. Might not work.
    return problem.solve()

def _mosek_check_license ():
    '''
    Checks whether MOSEK recognizes its license and deems it good enough.

    Returns
    -------
    bool
        True if license was deemed good enough.
    '''

    try:
        import mosek
    except ImportError as error:
        return False

    try:
        mosek_env = picos.solvers.get_solver('mosek')._get_environment()
        mosek_env.checkoutlicense(mosek.feature.pton)
        mosek_env.checkinall()
    except mosek.Error as error:
        return False

    return True

def picos_debug_constraints (problem):
    '''
    Use on solved problem to debug constraint values.

    Parameters
    ----------
    problem : picos.Problem
        The problem you are debugging.
    '''

    problem_constraints = problem.constraints.items()
    print(f'A problem with {len(problem_constraints)} constraints')
    for con_index, con_object in problem_constraints:
        con_expressions = list(con_object.expressions)
        print(f'  Constraint {con_index}')
        print(f'  {picos.value(con_object)}')
        print(f'    With {len(con_expressions)} components below')
        for exp_index, exp_object in enumerate(con_expressions):
            print(f'    {exp_index} => {picos.value(exp_object)}')


import numpy
from braket import ket2dm, eyemat, qubit_from_mask
from witnessme import optimal_discrete_witness

def paraschiv4balt (q):
    phi1 = 1 / numpy.sqrt(2) * (
        qubit_from_mask([ 1, 1, 0, 0 ]) * numpy.exp(3j / 7 * numpy.pi) -
        qubit_from_mask([ 0, 0, 0, 0 ])
    )
    phi2 = 1 / numpy.sqrt(5) * (
        qubit_from_mask([ 0, 1, 0, 1 ]) * numpy.exp(1j / 4 * numpy.pi) -
        qubit_from_mask([ 0, 1, 1, 1 ]) -
        qubit_from_mask([ 1, 0, 0, 0 ]) -
        qubit_from_mask([ 1, 0, 0, 1 ]) -
        qubit_from_mask([ 1, 1, 1, 1 ])
    )
    phi3 = 1 / numpy.sqrt(6) * (
        qubit_from_mask([ 0, 0, 1, 0 ]) * numpy.exp(- 2j / 3 * numpy.pi) +
        qubit_from_mask([ 0, 0, 1, 1 ]) +
        qubit_from_mask([ 0, 1, 0, 0 ]) +
        qubit_from_mask([ 1, 0, 1, 0 ]) +
        qubit_from_mask([ 1, 1, 0, 1 ]) +
        qubit_from_mask([ 1, 1, 1, 0 ])
    )
    phi4 = 1 / numpy.sqrt(2) * (
        qubit_from_mask([ 0, 1, 1, 0 ]) -
        qubit_from_mask([ 1, 0, 1, 1 ])
    )
    
    X = 1 / numpy.sqrt(87) * (
        phi1 * 5 +
        phi2 * numpy.sqrt(10) +
        phi3 * numpy.sqrt(3) +
        phi4 * 7
    )
    
    R = ket2dm(X)
    E = eyemat(4)

    return (1 - q) * R + q * E

# According to [1] the noise tolerance for the paraschiv4balt state is ~ 3% when
# inferring from the { (0, 1), (1, 2), (1, 3) } graph.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.98.062102

for q in numpy.linspace(0, 0.05, 11):
    rho = paraschiv4balt(q)
    w, W = optimal_discrete_witness(rho, 4, [ (0, 1), (1, 2), (1, 3) ])
    print('{:8f} {:+8f}'.format(q, w))  


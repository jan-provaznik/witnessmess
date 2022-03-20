import numpy
from braket import ket2dm, eyemat, qubit_from_mask
from witnessme import optimal_discrete_witness

def miklin4a (q):
    X = (
        qubit_from_mask([ 0, 0, 1, 1 ]) +
        qubit_from_mask([ 0, 1, 0, 1 ]) +
        qubit_from_mask([ 0, 1, 1, 0 ]) + 
        qubit_from_mask([ 1, 0, 0, 1 ]) -
        qubit_from_mask([ 1, 0, 1, 0 ])
    ) * 1 / numpy.sqrt(5)

    R = ket2dm(X)
    E = eyemat(4)

    return (1 - q) * R + q * E
    
# According to [1] the noise tolerance for the miklin3b state is ~ 21% when
# inferring from the { (0, 1), (1, 2), (2, 3) } graph.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.93.020104

for q in numpy.linspace(0, 0.3, 31):
    rho = miklin4a(q)
    w, W = optimal_discrete_witness(rho, 4, [ (0, 1), (1, 2), (2, 3) ])
    print('{:8f} {:+8f}'.format(q, w))


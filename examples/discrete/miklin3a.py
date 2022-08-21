import numpy
from braket import ket2dm, eyemat, qubit_from_mask
from witnessme import discrete_optimal_witness

def miklin3a (q):
    X001 = 1 / 3 * numpy.exp(+ 1j * numpy.pi / 3)
    X010 = 1 / 3 * numpy.exp(- 1j * numpy.pi / 3)
    X100 = 1 / 3 * (-1)
    X111 = numpy.sqrt(2 / 3)

    W011 = 1 / numpy.sqrt(3)
    W101 = W011
    W110 = W011

    X = (
        X001 * qubit_from_mask([ 0, 0, 1 ]) +
        X010 * qubit_from_mask([ 0, 1, 0 ]) +
        X100 * qubit_from_mask([ 1, 0, 0 ]) +
        X111 * qubit_from_mask([ 1, 1, 1 ])
    )
    W = (
        W011 * qubit_from_mask([ 0, 1, 1 ]) +
        W101 * qubit_from_mask([ 1, 0, 1 ]) +
        W110 * qubit_from_mask([ 1, 1, 0 ])
    )

    R = ket2dm(X) * 2 / 3 + ket2dm(W) / 3
    E = eyemat(3)

    return (1 - q) * R + q * E

# According to [1] the noise tolerance for the miklin3a state is ~ 13%.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.93.020104

for q in numpy.linspace(0, 0.2, 21):
    rho = miklin3a(q)
    w, W = discrete_optimal_witness(rho, [ 2, 2, 2 ])
    print('{:8f} {:+8f}'.format(q, w))


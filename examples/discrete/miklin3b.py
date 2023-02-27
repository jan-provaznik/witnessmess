import numpy
from braket import ket2dm, eyemat, qubit_from_mask
from witnessmess import dm_optimal_witness

def miklin3b (q):
    S000 = qubit_from_mask([ 0, 0, 0 ])
    S001 = qubit_from_mask([ 0, 0, 1 ])
    S010 = qubit_from_mask([ 0, 1, 0 ])
    S100 = qubit_from_mask([ 1, 0, 0 ])
    S011 = qubit_from_mask([ 0, 1, 1 ])
    S101 = qubit_from_mask([ 1, 0, 1 ])
    S111 = qubit_from_mask([ 1, 1, 1 ])

    X1 = 1 / numpy.sqrt(10) * (
        S000 * numpy.sqrt(5) +
        S011 * numpy.sqrt(4) * numpy.exp(- 1j * 3 / 4 * numpy.pi) +
        S101 * numpy.exp(- 1j * 3 / 4 * numpy.pi)
    )
    X2 = 1 / numpy.sqrt(10) * (
        numpy.sqrt(3) * (
            S001 +
            S010 * numpy.exp(1j * 2 / 3 * numpy.pi) +
            S100 * numpy.exp(- 1j * 1 / 3 * numpy.pi)
        ) + S111
    )

    R = ket2dm(X1) / 2 + ket2dm(X2) / 2
    E = eyemat(3)

    return (1 - q) * R + q * E

# According to [1] the noise tolerance for the miklin3b state is ~ 5% when
# inferring from the partial { (0, 2), (1, 2) } graph.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.93.020104

for q in numpy.linspace(0, 0.1, 21):
    rho = miklin3b(q)
    w, W = dm_optimal_witness(rho, [ 2, 2, 2 ], [ (0, 2), (1, 2) ])
    print('{:8f} {:+8f}'.format(q, w))


import numpy
import functools
from witnessmess import dm_optimal_witness 

def kron (* args):
    return functools.reduce(numpy.kron, args, 1)

def dag (mat):
    return numpy.conjugate(mat.T)

def qudit (i, d):
    ket = numpy.zeros([ d, 1 ])
    ket[i] = 1
    return ket

def qutrits (* args):
    return kron(* [ qudit(val, 3) for val in args ])

# Supplementary material of [1], equation (13) gives tripartite qutrit system 
# with noise tolerance 29.5%.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.93.020104

def miklin13 (q):
    q000 = qutrits(0, 0, 0)
    q012 = qutrits(0, 1, 2)
    q021 = qutrits(0, 2, 1)

    q102 = qutrits(1, 0, 2)
    q111 = qutrits(1, 1, 1)
    q120 = qutrits(1, 2, 0)

    q201 = qutrits(2, 0, 1)
    q210 = qutrits(2, 1, 0)
    q222 = qutrits(2, 2, 2)

    c1 = 1 / numpy.sqrt(12)
    c2 = numpy.sqrt(5) / 6
    c3 = 1 / numpy.sqrt(6)

    eta1 = c1 * (q000 - q222) - 1j * c2 * (q012 + q021 - q102 + q120 + q201 + q210)
    eta2 = c3 * q111 - c2 * (q012 - q021 + q102 + q120 + q201 - q210)

    rho = 0.5 * (eta1 @ dag(eta1) + eta2 @ dag(eta2))
    eye = numpy.eye(27) / 27

    return (1 - q) * rho + q * eye

# According to [1] the noise tolerance should be ~ 29.5%

for q in numpy.linspace(0, 0.35, 36):
    rho = miklin13(q)
    wit, mat = dm_optimal_witness(rho, [ 3, 3, 3 ])
    print('{:.4f} {:.6f}'.format(q, wit))


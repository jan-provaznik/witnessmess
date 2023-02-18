import numpy
from braket import ket2dm, eyemat, ketnorm
from witnessmess import dm_optimal_witness 

def paraschiv4b (q):
    X = numpy.array([
        -2 / 33  +  5j / 38,
         1 / 21  +  2j / 13,
        50 / 149 +  3j / 35,
        -1 / 17  +  3j / 25,
         7 / 26  -  5j / 28,
        -3 / 32  + 10j / 61,
         7 / 62  +  6j / 17,
        -2 / 17  +  1j / 111,
         5 / 23  +  8j / 29,
         3 / 31  +  5j / 41,
         2 / 41  -  5j / 34,
        -1 / 13  + 11j / 30,
        -1 / 289 +  1j / 51,
        -1 / 270 +  1j / 24,
        -1 / 58  +  7j / 24,
        11 / 31
    ]).reshape(16, 1)
    X = X / ketnorm(X)

    R = ket2dm(X)
    E = eyemat(4)

    return (1 - q) * R + q * E
    
# According to [1] the noise tolerance for the paraschiv4b state is ~ 5% when
# inferring from the { (0, 1), (1, 2), (1, 3) } graph.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.98.062102

for q in numpy.linspace(0, 0.10, 21):
    rho = paraschiv4b(q)
    w, W = dm_optimal_witness(rho, [ 2, 2, 2, 2 ], [ (0, 1), (1, 2), (1, 3) ])
    print('{:8f} {:+8f}'.format(q, w))  


import numpy
from braket import ket2dm, eyemat, ketnorm
from witnessmess import dm_optimal_witness 

def paraschiv5a (q):
    X = numpy.array([
         -3 / 35  -  1j / 22,
          4 / 31  +  1j / 40,
          4 / 29  -  1j / 22,
         -1 / 28  +  4j / 29,
          4 / 35  -  2j / 25,
         -1 / 24  +  1j / 19,
         -6 / 35  -  5j / 28,
          2 / 33  -  4j / 45,
          1 / 32  -  1j / 3,
          3 / 35  - 19j / 94,
         -5 / 24  -  3j / 16,
          1 / 74  -  2j / 33,
         -4 / 27  +  1j / 207,
         -1 / 186 -  2j / 39,
          5 / 41  -  2j / 13,
          2 / 19  +  5j / 34,
         -2 / 27  +  1j / 6,
        - 3 / 29  -  8j / 33,
        - 1 / 8   -  5j / 36,
          7 / 30  -  3j / 40,
        - 4 / 31  -  5j / 28,
          1 / 7   -  3j / 35,
        -11 / 36  +  1j / 83,
          1 / 50  -  2j / 35,
          1 / 10  -  8j / 41,
          1 / 26  -  1j / 50,
         -4 / 39  -  2j / 29,
         -2 / 29  -  2j / 19,
         -1 / 18  -  1j / 295,
         -2 / 27  -  2j / 23,
         -1 / 18  -  4j / 33,
          1 / 10
    ]).reshape(32, 1)
    X = X / ketnorm(X)
    
    R = ket2dm(X)
    E = eyemat(5)
    
    return (1 - q) * R + q * E
    
# According to [1] the noise tolerance for the paraschiv4b state is ~ 0.11% when
# inferring from the { (0, 1), (1, 2), (2, 3), (3, 4) } graph.
#
# [1] https://dx.doi.org/10.1103/PhysRevA.98.062102

for q in numpy.linspace(0, 0.02, 11):
    rho = paraschiv5a(q)
    w, W = dm_optimal_witness(rho, [ 2, 2, 2, 2, 2 ], [ (0, 1), (1, 2), (2, 3), (3, 4) ])
    print('{:8f} {:+8f}'.format(q, w))  

# Note to self: testing this on i5-7200 was not the brightest idea.


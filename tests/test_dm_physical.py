import numpy
import scipy.linalg

from witnessmess import dm_is_physical


def test_dm_physical ():
    for loop_index in range(1000):

        # Construct a random hermitian matrix
        H = numpy.random.rand(16, 16) + 1j * numpy.random.rand(16, 16)
        H = H + numpy.conjugate(H.T)

        # Construct a random unitary operator
        U = scipy.linalg.expm(-1j * H)

        # Construct a random normalized positive diagonal matrix
        D = numpy.diag(1 + numpy.random.rand(16))
        D = D / D.trace()

        # Construct a random normalized positive-definite matrix
        R = U @ D @ numpy.conjugate(U.T)

        assert dm_is_physical(R)


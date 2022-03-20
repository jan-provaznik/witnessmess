import numpy
import functools

def dag (q):
    return numpy.conjugate(q.T)

def ket2dm (ket):
    return ket @ dag(ket)

def eyemat (nsys):
    dim = 2 ** nsys
    return numpy.eye(dim) / dim

def build_qubit (q):
    if (q == 0):
        return numpy.array([ 1, 0 ]).reshape(2, 1)
    if (q == 1):
        return numpy.array([ 0, 1 ]).reshape(2, 1)
    raise ValueError

def qubit_from_mask (mask):
    return functools.reduce(numpy.kron, map(build_qubit, mask))

def ketnorm (ket):
    return numpy.sqrt(numpy.sum(numpy.abs(ket) ** 2))


import numpy
import functools
import witnessmess

# Utilities

def dag (mat):
    return numpy.conjugate(mat.T)

def ket2dm (ket):
    return ket @ dag(ket)

def kron (* args):
    return functools.reduce(numpy.kron, args, 1)

def ketnorm (ket):
    return numpy.sqrt(numpy.sum(numpy.abs(ket) ** 2))

# Creation of fully mixed density matrix

def build_mixed (count, qdim):
    dim = qdim ** count
    return numpy.eye(dim) / dim

# Creation of general qudits

def build_qudit (val, qdim):
    ket = numpy.zeros([ qdim, 1 ])
    ket[val] = 1
    return ket

def build_qudit_from_mask (mask, qdim):
    return kron(* [ build_qudit(val, qdim) for val in mask ])

# Specialization for qubits

def build_qubit (val):
    return build_qudit(val, 2)

def build_qubit_from_mask (mask):
    return build_qudit_from_mask(mask, 2)

# Specialization for qutrits

def build_qutrit (val):
    return build_qudit(val, 3)

def build_qutrit_from_mask (mask):
    return build_qudit_from_mask(mask, 3)

# Test: Miklin (3a) from (https://dx.doi.org/10.1103/PhysRevA.93.020104)
#
# Noise tolerance for the state is ~ 13%.

def make_dm_miklin3a (noise_level):
    X001 = 1 / 3 * numpy.exp(+ 1j * numpy.pi / 3)
    X010 = 1 / 3 * numpy.exp(- 1j * numpy.pi / 3)
    X100 = 1 / 3 * (-1)
    X111 = numpy.sqrt(2 / 3)

    W011 = 1 / numpy.sqrt(3)
    W101 = W011
    W110 = W011

    X = (
        X001 * build_qubit_from_mask([ 0, 0, 1 ]) +
        X010 * build_qubit_from_mask([ 0, 1, 0 ]) +
        X100 * build_qubit_from_mask([ 1, 0, 0 ]) +
        X111 * build_qubit_from_mask([ 1, 1, 1 ])
    )
    W = (
        W011 * build_qubit_from_mask([ 0, 1, 1 ]) +
        W101 * build_qubit_from_mask([ 1, 0, 1 ]) +
        W110 * build_qubit_from_mask([ 1, 1, 0 ])
    )

    R = ket2dm(X) * 2 / 3 + ket2dm(W) / 3
    E = build_mixed(3, 2)

    return (1 - noise_level) * R + noise_level * E

def test_dm_miklin3a ():
    test_source = [ 
        0.00, 0.01, 0.02,
        0.03, 0.04, 0.05,
        0.06, 0.07, 0.08,
        0.09, 0.10, 0.11,
        0.12, 0.13, 0.14,
        0.15, 0.16, 0.17,
        0.18, 0.19, 0.20
    ]
    test_target = [
        -0.019959, -0.018509, -0.017060,
        -0.015610, -0.014161, -0.012711,
        -0.011261, -0.009812, -0.008362,
        -0.006913, -0.005463, -0.004014,
        -0.002564, -0.001114, +0.000335,
        +0.001785, +0.003234, +0.004684,
        +0.006134, +0.007583, +0.009033,
    ]

    state_dims = [ 2, 2, 2 ]
    state_tree = [ (0, 1), (0, 2), (1, 2) ]

    for (source_value, target_value) in zip(test_source, test_target):
        r = make_dm_miklin3a(source_value)
        w, _ = witnessmess.dm_optimal_witness(r, state_dims, state_tree)
        assert numpy.abs(w - target_value) < 1e-6

# Test: Miklin (3b) from (https://dx.doi.org/10.1103/PhysRevA.93.020104)
#
# Noise tolerance for the state is ~ 5% when
# inferring from the partial { (0, 2), (1, 2) } graph.

def make_dm_miklin3b (noise_level):
    S000 = build_qubit_from_mask([ 0, 0, 0 ])
    S001 = build_qubit_from_mask([ 0, 0, 1 ])
    S010 = build_qubit_from_mask([ 0, 1, 0 ])
    S100 = build_qubit_from_mask([ 1, 0, 0 ])
    S011 = build_qubit_from_mask([ 0, 1, 1 ])
    S101 = build_qubit_from_mask([ 1, 0, 1 ])
    S111 = build_qubit_from_mask([ 1, 1, 1 ])

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
    E = build_mixed(3, 2)

    return (1 - noise_level) * R + noise_level * E

def test_dm_miklin3b ():
    test_source = [ 
        0.000, 0.005, 0.010, 
        0.015, 0.020, 0.025, 
        0.030, 0.035, 0.040, 
        0.045, 0.050, 0.055, 
        0.060, 0.065, 0.070, 
        0.075, 0.080, 0.085, 
        0.090, 0.095, 0.100 
    ]
    test_target = [
        -0.006546, -0.005888, -0.005231,
        -0.004573, -0.003915, -0.003257, 
        -0.002600, -0.001942, -0.001284, 
        -0.000627, +0.000031, +0.000689, 
        +0.001347, +0.002004, +0.002662, 
        +0.003320, +0.003978, +0.004635, 
        +0.005293, +0.005951, +0.006608
    ]

    state_dims = [ 2, 2, 2 ]
    state_tree = [ (0, 2), (1, 2) ]

    for (source_value, target_value) in zip(test_source, test_target):
        r = make_dm_miklin3b(source_value)
        w, _ = witnessmess.dm_optimal_witness(r, state_dims, state_tree)
        assert numpy.abs(w - target_value) < 1e-6

# Test: Miklin (13) from (https://dx.doi.org/10.1103/PhysRevA.93.020104)
# supplementary material.
#
# Noise tolerance for the state is ~ 29.5%.

def make_dm_miklin13 (noise_level):
    q000 = build_qutrit_from_mask([ 0, 0, 0 ])
    q012 = build_qutrit_from_mask([ 0, 1, 2 ])
    q021 = build_qutrit_from_mask([ 0, 2, 1 ])

    q102 = build_qutrit_from_mask([ 1, 0, 2 ])
    q111 = build_qutrit_from_mask([ 1, 1, 1 ])
    q120 = build_qutrit_from_mask([ 1, 2, 0 ])

    q201 = build_qutrit_from_mask([ 2, 0, 1 ])
    q210 = build_qutrit_from_mask([ 2, 1, 0 ])
    q222 = build_qutrit_from_mask([ 2, 2, 2 ])

    c1 = 1 / numpy.sqrt(12)
    c2 = numpy.sqrt(5) / 6
    c3 = 1 / numpy.sqrt(6)

    eta1 = c1 * (q000 - q222) - 1j * c2 * (q012 + q021 - q102 + q120 + q201 + q210)
    eta2 = c3 * q111 - c2 * (q012 - q021 + q102 + q120 + q201 - q210)

    rho = 0.5 * (eta1 @ dag(eta1) + eta2 @ dag(eta2))
    eye = build_mixed(3, 3)

    return (1 - noise_level) * rho + noise_level * eye

def test_dm_miklin13 ():
    test_source = [
        0.00, 0.01, 0.02, 0.03,
        0.04, 0.05, 0.06, 0.07,
        0.08, 0.09, 0.10, 0.11,
        0.12, 0.13, 0.14, 0.15,
        0.16, 0.17, 0.18, 0.19,
        0.20, 0.21, 0.22, 0.23,
        0.24, 0.25, 0.26, 0.27,
        0.28, 0.29, 0.30, 0.31,
        0.32, 0.33, 0.34, 0.35
    ]
    test_target = [
        -0.015525, -0.014999, -0.014473, -0.013948,
        -0.013422, -0.012897, -0.012371, -0.011845,
        -0.011320, -0.010794, -0.010268, -0.009743,
        -0.009217, -0.008692, -0.008166, -0.007640,
        -0.007115, -0.006589, -0.006064, -0.005538,
        -0.005012, -0.004487, -0.003961, -0.003435,
        -0.002910, -0.002384, -0.001859, -0.001333,
        -0.000807, -0.000282, +0.000244, +0.000769,
        +0.001295, +0.001821, +0.002346, +0.002872
    ]

    state_dims = [ 3, 3, 3 ]
    state_tree = [ (0, 1), (0, 2), (1, 2) ]

    for (source_value, target_value) in zip(test_source, test_target):
        r = make_dm_miklin13(source_value)
        w, _ = witnessmess.dm_optimal_witness(r, state_dims, state_tree)
        assert numpy.abs(w - target_value) < 1e-6

# Test: Miklin (4a) from (https://dx.doi.org/10.1103/PhysRevA.93.020104)
#
# Noise tolerance for the state is ~ 21%.

def make_dm_miklin4a (noise_level):
    X = (
        build_qubit_from_mask([ 0, 0, 1, 1 ]) +
        build_qubit_from_mask([ 0, 1, 0, 1 ]) +
        build_qubit_from_mask([ 0, 1, 1, 0 ]) + 
        build_qubit_from_mask([ 1, 0, 0, 1 ]) -
        build_qubit_from_mask([ 1, 0, 1, 0 ])
    ) * 1 / numpy.sqrt(5)

    R = ket2dm(X)
    E = build_mixed(4, 2)

    return (1 - noise_level) * R + noise_level * E
    
def test_dm_miklin4a ():
    test_source = [
        0.00, 0.01, 0.02,
        0.03, 0.04, 0.05,
        0.06, 0.07, 0.08,
        0.09, 0.10, 0.11,
        0.12, 0.13, 0.14,
        0.15, 0.16, 0.17,
        0.18, 0.19, 0.20,
        0.21, 0.22, 0.23,
        0.24, 0.25, 0.26,
        0.27, 0.28, 0.29,
        0.30
    ]
    test_target = [
        -0.016865, -0.016071, -0.015277,
        -0.014484, -0.013690, -0.012896,
        -0.012103, -0.011309, -0.010515,
        -0.009722, -0.008928, -0.008134,
        -0.007341, -0.006547, -0.005754,
        -0.004960, -0.004166, -0.003373,
        -0.002579, -0.001785, -0.000992,
        -0.000198, +0.000596, +0.001389,
        +0.002183, +0.002977, +0.003770,
        +0.004564, +0.005358, +0.006151,
        +0.006945
    ]

    state_dims = [ 2, 2, 2, 2 ]
    state_tree = [ (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3) ]

    for (source_value, target_value) in zip(test_source, test_target):
        r = make_dm_miklin4a(source_value)
        w, _ = witnessmess.dm_optimal_witness(r, state_dims, state_tree)
        assert numpy.abs(w - target_value) < 1e-6

# Test: Paraschiv (4b) from (https://dx.doi.org/10.1103/PhysRevA.98.062102)
#
# Noise tolerance for the state is ~ 5% when
# inferring from the partial { (0, 1), (1, 2), (1, 3) } graph.

def make_dm_paraschiv4b (noise_level):
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
    E = build_mixed(4, 2)

    return (1 - noise_level) * R + noise_level * E

def test_dm_paraschiv4b ():
    test_source = [
        0.000, 0.005, 0.010, 
        0.015, 0.020, 0.025, 
        0.030, 0.035, 0.040, 
        0.045, 0.050, 0.055, 
        0.060, 0.065, 0.070, 
        0.075, 0.080, 0.085, 
        0.090, 0.095, 0.100
    ]
    test_target = [
        -0.003571, -0.003241, -0.002911,
        -0.002580, -0.002250, -0.001920,
        -0.001589, -0.001259, -0.000929,
        -0.000598, -0.000268, +0.000062,
        +0.000393, +0.000723, +0.001054,
        +0.001384, +0.001714, +0.002045,
        +0.002375, +0.002705, +0.003036
    ]

    state_dims = [ 2, 2, 2, 2]
    state_tree = [ (0, 1), (1, 2), (1, 3) ]

    for (source_value, target_value) in zip(test_source, test_target):
        r = make_dm_paraschiv4b(source_value)
        w, _ = witnessmess.dm_optimal_witness(r, state_dims, state_tree)
        assert numpy.abs(w - target_value) < 1e-6

# Test: Paraschiv (5a) from (https://dx.doi.org/10.1103/PhysRevA.98.062102)
#
# Noise tolerance for the state is ~ 0.11% when
# inferring from the partial { (0, 1), (1, 2), (2, 3), (3, 4) } graph.

def make_dm_paraschiv5a (noise_level):
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
    E = build_mixed(5, 2)
    
    return (1 - noise_level) * R + noise_level * E

def test_dm_paraschiv5a ():
    test_source = [
        0.000, 0.002, 0.004,
        0.006, 0.008, 0.010,
        0.012, 0.014, 0.016,
        0.018, 0.020
    ]
    test_target = [
        -0.001139, -0.001074, -0.001009,
        -0.000944, -0.000880, -0.000815,
        -0.000750, -0.000685, -0.000620,
        -0.000556, -0.000491
    ]

    state_dims = [ 2, 2, 2, 2, 2 ]
    state_tree = [ (0, 1), (1, 2), (2, 3), (3, 4) ]

    for (source_value, target_value) in zip(test_source, test_target):
        r = make_dm_paraschiv5a(source_value)
        w, _ = witnessmess.dm_optimal_witness(r, state_dims, state_tree)
        assert numpy.abs(w - target_value) < 1e-6


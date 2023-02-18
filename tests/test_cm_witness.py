import numpy
from witnessmess import cm_optimal_witness, cm_is_pairwise_ppt

# Hyllus (97) and (99) examples were taken 
# from (https://doi.org/10.1088/1367-2630/8/4/051)

def test_cm_witness_hyllus97 ():
    G = numpy.array([
        [ 2,  0, 1,  0, 1,  0 ],
        [ 0,  3, 0, -1, 0, -1 ],
        [ 1,  0, 2,  0, 1,  0 ],
        [ 0, -1, 0,  3, 0, -1 ],
        [ 1,  0, 1,  0, 2,  0 ],
        [ 0, -1, 0, -1, 0,  3 ]
    ]) * 0.5
    w, _ = cm_optimal_witness(G, 3)

    assert numpy.abs(w + 0.3056) < 1e-4

def test_cm_witness_hyllus99 ():
    x =  6.4980
    y = -4.3142

    G = numpy.array([
        [ x,  0, y,  0,  y,  0,  0,  0 ],
        [ 0,  x, 0, -y,  0, -y,  0,  0 ],
        [ y,  0, x,  0,  0,  0,  y,  0 ],
        [ 0, -y, 0,  x,  0,  0,  0, -y ],
        [ y,  0, 0,  0,  x,  0, -y,  0 ],
        [ 0, -y, 0,  0,  0,  x,  0,  y ],
        [ 0,  0, y,  0, -y,  0,  x,  0 ],
        [ 0,  0, 0, -y,  0,  y,  0,  x ]
    ])

    w, _ = cm_optimal_witness(G, 4)
    assert numpy.abs(w + 0.230) < 1e-3

# Nordgren examples (with partially blind witnesses) were taken 
# from (https://doi.org/10.1103/PhysRevA.106.062410)

def test_cm_witness_nordgren3 ():
    G = numpy.array([
        [  1.34,    0, -0.35,    0, -0.82,    0 ],
        [     0, 10.0,     0, 8.45,     0, 1.87 ],
        [ -0.35,    0,  7.80,    0, -8.05,    0 ],
        [     0, 8.45,     0, 7.92,     0, 2.09 ],
        [ -0.82,    0, -8.05,    0,  10.0,    0 ],
        [     0, 1.87,     0, 2.09,     0, 1.62 ]
    ])

    assert cm_is_pairwise_ppt(G, 3)

    w, _ = cm_optimal_witness(G, 3, [ (0, 1), (1, 2) ])

    assert w < 0
    assert numpy.abs(w + 0.143) < 1e-3

def test_cm_witness_nordgren4a ():
    G = numpy.array([
        [  2.83,     0, -0.02,     0, -1.38,     0,  2.83,     0 ],
        [     0,  7.18,     0,  8.06,     0,  7.09,     0, -4.12 ],
        [ -0.02,     0,  3.91,     0, -2.46,     0,  4.73,     0 ],
        [     0,  8.06,     0,  9.79,     0,  8.47,     0, -4.81 ],
        [ -1.38,     0, -2.46,     0,  2.58,     0, -4.68,     0 ],
        [     0,  7.09,     0,  8.47,     0,  10.0,     0, -3.08 ],
        [  2.83,     0,  4.73,     0, -4.68,     0,  10.0,     0 ],
        [     0, -4.12,     0, -4.81,     0, -3.08,     0,  3.22 ]
    ])

    assert cm_is_pairwise_ppt(G, 4)

    w, _ = cm_optimal_witness(G, 4, [ (0, 1), (1, 2), (2, 3) ])

    assert w < 0
    assert numpy.abs(w + 0.0693) < 1e-4

def test_cm_witness_nordgren4b ():
    G = numpy.array([
        [  5.23,    0,  0.45,    0, -0.02,    0, -2.43,    0 ],
        [     0, 1.16,     0,  3.0,     0, 1.15,     0, 0.51 ],
        [  0.45,    0,  3.35,    0,  0.91,    0, -5.20,    0 ],
        [     0,    3,     0,   10,     0, 3.52,     0, 2.06 ],
        [ -0.02,    0,  0.91,    0,  4.09,    0, -2.97,    0 ],
        [     0, 1.15,     0, 3.52,     0, 1.62,     0, 0.62 ],
        [ -2.43,    0, -5.20,    0, -2.97,    0,    10,    0 ],
        [     0, 0.51,     0, 2.06,     0, 0.62,     0, 1.49 ]
    ])

    assert cm_is_pairwise_ppt(G, 4)

    w, _ = cm_optimal_witness(G, 4, [ (0, 1), (1, 2), (1, 3) ])

    assert w < 0
    assert numpy.abs(w + 0.0681) < 1e-4


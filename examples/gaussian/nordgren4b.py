import numpy
from witnessmess import cm_optimal_witness, cm_pairwise_pt

# The matrices were introduced in section 5 of [1]

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

# Per table 3 of [1] the 2-mode marginalia are separable

e = cm_pairwise_pt(G, 4)
print(e)

# Per table 1 of [1] the value should be ~ -0.0681

w, W = cm_optimal_witness(G, 4, [ (0, 1), (1, 2), (1, 3) ])
print(w)

# Per text the witness matrix should be blind in the (0, 2) block.
print(W.round(4))

# References
#   [1] https://doi.org/10.1103/PhysRevA.106.062410


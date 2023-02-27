import numpy
from witnessmess import cm_optimal_witness, cm_pairwise_pt

# The matrices were introduced in section 5 of [1]

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

# Per table 3 of [1] the 2-mode marginalia are separable

e = cm_pairwise_pt(G, 4)
print(e)

# Per table 1 of [1] the value should be ~ -0.0693

w, W = cm_optimal_witness(G, 4, [ (0, 1), (1, 2), (2, 3) ])
print(w)

# Per text the witness matrix should be blind in the (0, 3) block.
print(W.round(4))

# References
#   [1] https://doi.org/10.1103/PhysRevA.106.062410


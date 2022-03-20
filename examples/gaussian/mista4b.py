import numpy
from witnessme import optimal_gaussian_witness, gaussian_pairwise_ppt

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

print(G.shape)

# Per table 3 of [1] the 2-mode marginalia separable

e = gaussian_pairwise_ppt(G, 4)
print(e)

# Per table 1 of [1] the value should be ~ -0.0681

w, W = optimal_gaussian_witness(G, 4, [ (0, 1), (1, 2), (1, 3) ])
print(w)

# Per text the witness matrix should be blind in the (0, 2) block.
print(W.round(4))

# References
#   [1] https://arxiv.org/abs/2103.07327


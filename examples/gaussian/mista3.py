import numpy
from witnessme import optimal_gaussian_witness, gaussian_pairwise_ppt

# The matrices were introduced in section 5 of [1]

G = numpy.array([
    [  1.34,    0, -0.35,    0, -0.82,    0 ],
    [     0, 10.0,     0, 8.45,     0, 1.87 ],
    [ -0.35,    0,  7.80,    0, -8.05,    0 ],
    [     0, 8.45,     0, 7.92,     0, 2.09 ],
    [ -0.82,    0, -8.05,    0,  10.0,    0 ],
    [     0, 1.87,     0, 2.09,     0, 1.62 ]
])

# Per table 2 of [1 ] the 2-mode marginalia separable

e = gaussian_pairwise_ppt(G, 3)
print(e)

# Per table 1 of [1] the value should be ~ -0.143

w, W = optimal_gaussian_witness(G, 3, [ (0, 1), (1, 2) ])
print(w)

# Per text the witness matrix should be blind in the (0, 2) block.
print(W.round(4))

# References
#   [1] https://arxiv.org/abs/2103.07327

import numpy
from witnessme import optimal_gaussian_witness, gaussian_pairwise_ppt

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

# Per table 3 of [1] the 2-mode marginalia separable

e = gaussian_pairwise_ppt(G, 4)
print(e)

# Per table 1 of [1] the value should be ~ -0.0693

w, W = optimal_gaussian_witness(G, 4, [ (0, 1), (1, 2), (2, 3) ])
print(w)

# Per text the witness matrix should be blind in the (0, 3) block.
print(W.round(4))

# References
#   [1] https://arxiv.org/abs/2103.07327


import numpy
from witnessme import optimal_gaussian_witness

# Covariance matrix (92) of [1].

G = numpy.array([
    [ 2,  0, 1,  0, 1,  0 ],
    [ 0,  3, 0, -1, 0, -1 ],
    [ 1,  0, 2,  0, 1,  0 ],
    [ 0, -1, 0,  3, 0, -1 ],
    [ 1,  0, 1,  0, 2,  0 ],
    [ 0, -1, 0, -1, 0,  3 ]
]) * 0.5

w, W = optimal_gaussian_witness(G, 3)

# According to [1] we should get -0.3056

print(w)

# References
# [1] https://doi.org/10.1088/1367-2630/8/4/051

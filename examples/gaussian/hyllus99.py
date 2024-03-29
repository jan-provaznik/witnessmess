import numpy
from witnessmess import cm_optimal_witness

# Covariance matrix (99) of [1].

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

w, W = cm_optimal_witness(G, 4, [ (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3) ])

# According to [1] we should get -0.2305

print(w)

# References
#   [1] https://doi.org/10.1088/1367-2630/8/4/051


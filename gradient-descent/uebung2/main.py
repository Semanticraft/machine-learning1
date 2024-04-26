import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import uebung2.logic as logic

sample_size = 100
n_features = 2
x_min = [1.5, -0.5]
x_max = [11., 5.0]

X = logic.create_feature_matrix(sample_size, n_features, x_min, x_max)
print(X)

assert len(X[:, 0]) == sample_size
assert len(X[0, :]) == n_features
for i in range(n_features):
    assert np.max(X[:, i]) <= x_max[i]
    assert np.min(X[:, i]) >= x_min[i]

assert len(logic.linear_hypothesis([.1, .2, .3])(X)) == sample_size

theta = (2., 3., -4.)
sigma = 3.
y = logic.generate_targets(X, theta, sigma)

assert len(y) == sample_size

logic.plot_data_scatter(X, y)
plt.clf()

J = logic.mse_cost_function(X, y)
print(J(theta))

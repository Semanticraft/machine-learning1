import matplotlib.pyplot as plt

import regression
import numpy as np

# generation of artificial design matrix
X = np.random.randn(30, 3)
X[:, 0] = np.ones(30)
theta = np.array([1.1, 2.0, -.9])
h = regression.get_linear_hypothesis(theta)
y_noise_sigma = 4.
m = 30
y = h(X) + np.random.randn(m).reshape(-1, 1) * y_noise_sigma

# plotting generated data distribution for multivariate linear regression
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], y, c='r', marker='o', edgecolor='black')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Distribution of Data')
fig.savefig('distribution-of-data.png')
fig.clf()

# plotting the hypothesis with the least cost after all iterations of the gradient descent
steps = 400
thetas = regression.gradient_descent(0.01, np.ones(3), X, y, steps)
regression.plot_cost_over_iterations(X, y, thetas, "cost-over-time-1.png")
regression.plot_best_hypothesis(X, y, thetas, "best-hypothesis-1.png")


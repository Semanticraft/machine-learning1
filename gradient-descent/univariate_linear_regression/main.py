import matplotlib.pyplot as plt

import regression
import numpy as np

# generation of synthetic training data
x_min = -10.
x_max = 10.
m = 10

x = np.random.uniform(x_min, x_max, m)
a = 10.
b = 5.
y_noise_sigma = 4.
y = a + b * x + np.random.randn(m) * y_noise_sigma

# plotting generated data distribution for univariate linear regression
plt.plot(x, y, "bo")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Distribution of Data")
plt.savefig("distribution-of-data.png")
plt.clf()

# generating a linear hypothesis based on the gradient descent procedure for univariate linear regression
steps = 15000
thetas_0, thetas_1 = regression.gradient_descent(x, y, 1, 1, 0.0001, steps)
regression.plot_cost_over_iterations(x, y, thetas_0, thetas_1, "cost-over-time-1.png")
regression.plot_best_hypothesis(x, y, thetas_0, thetas_1, "best-hypothesis-1.png")

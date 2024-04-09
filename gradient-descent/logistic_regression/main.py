import numpy as np
import matplotlib.pyplot as plt
import logistic_regression.logic as lr

# class 0:
# covariance matrix and mean
cov0 = np.array([[5, -4], [-4, 4]])
mean0 = np.array([2., 3])
# number of data points
m0 = 1000

# class 1
# covariance matrix
cov1 = np.array([[5, -3], [-3, 3]])
mean1 = np.array([1., 1])
# number of data points
m1 = 1000

# generate m gaussian distributed data points with
# mean and cov.
r0 = np.random.multivariate_normal(mean0, cov0, m0)
r1 = np.random.multivariate_normal(mean1, cov1, m1)

lr.plot_data_distribution(r0, r1, 'logistic-data-distribution.png')
plt.clf()

X = np.concatenate((r0, r1))
y = np.ones(len(r0) + len(r1))
y[:len(r0), ] = 0

lr.plot_logistic_function(np.arange(-10, 10, 0.001))
plt.clf()

theta = np.array([1.1, 2.0, -.9])
h = lr.logistic_hypothesis(theta)
print(h(X))

J = lr.cross_entropy_costs(lr.logistic_hypothesis, X, y)
print(J(theta))
assert len(J(theta)) == len(X)

J = lr.mean_cross_entropy_costs(X, y, lr.logistic_hypothesis, lr.cross_entropy_costs, 0.1)
print(J(theta))

theta = lr.compute_new_theta(X, y, theta, .1, lr.logistic_hypothesis, .1)
print(theta)

alpha = 0.16
num_iters = 100
history_cost, history_theta = lr.gradient_descent(X, y, theta, alpha, num_iters, .1)

lr.plot_progress(history_cost)
plt.clf()
print("costs before the training:\t ", history_cost[0])
print("costs after the training:\t ", history_cost[-1])

best_theta = history_theta[-1]
decision_boundary = lr.get_decision_boundary(best_theta)
x1 = np.arange(-10, 10, 0.001)
plt.plot(x1, decision_boundary(x1))
lr.plot_data_distribution(r0, r1, 'best-hypothesis.png')
plt.clf()

print('accuracy: ' + str(np.mean((decision_boundary(X[:, 0]) <= X[:, 0]).astype(float) +
                                 (decision_boundary(X[:, 1]) >= X[:, 1]).astype(float))))

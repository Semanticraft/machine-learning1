import numpy as np
import matplotlib.pyplot as plt


def logistic_function(x):
    """
    Applies the logistic function to x, element-wise.

    Parameters:
    x (numpy.ndarray): Intervall in the set of floats.
    .
    Returns:
    numpy.ndarray: Intervall with application of logistic function.
    """
    return 1. / (1. + np.exp(-x))


def plot_logistic_function(x):
    """
    Maps all x to y based on the logistic function and saves the plot.

    Parameters:
    x (numpy.ndarray): Intervall that is to be plotted.
    """
    plt.plot(x, logistic_function(x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('logistic-function.png')


def logistic_hypothesis(theta):
    """
    Combines given list argument in a logistic equation and returns it as a function.

    Parameters:
    theta (numpy.ndarray): List of coefficients.

    Returns:
    function: Lambda that models a logistc function based on thetas and X.
    """
    return lambda X: logistic_function(transform_design_matrix(X).dot(theta.T))


def transform_design_matrix(X):
    """
    Transforms X to X', where X' is a new matrix that appends a column of ones to the left side of X.

    Parameters:
    X (numpy.ndarray): The matrix that is to be transformed.

    Returns:
    numpy.ndarray: X'.
    """
    new_design_matrix = np.ones((len(X), len(X.T) + 1))
    new_design_matrix[:, 1:] = X
    return new_design_matrix


def cross_entropy_costs(h, X, y):
    """
    Implements cross-entropy as a function costs(theta) on given training data.

    Parameters:
    h (function): the hypothesis as function.
    X (numpy.ndarray): features as 2D array with shape (m_examples, n_features).
    y (numpy.ndarray): ground truth labels for given features with shape (m_examples).

    Returns:
    function: Lambda costs(theta) that models the cross-entropy for each x^i.
    """
    epsilon = 1e-10
    return lambda theta: -y * np.log(h(theta)(X) + epsilon) - (1 - y) * np.log(1 - h(theta)(X) + epsilon)


def mean_cross_entropy_costs(X, y, hypothesis, cost_func, lambda_reg=0.1):
    """
    Implements mean cross-entropy as a function J(theta) on given training data .

    Parameters:
    X (numpy.ndarray): Features as 2D array with shape (m_examples, n_features).
    y (numpy.ndarray): Ground truth labels for given features with shape (m_examples).
    hypothesis (function): The hypothesis as function.
    cost_func (function): The cost function.

    Returns:
    function: Lambda J(theta) that models the mean cross-entropy.
    """
    m = len(X)
    return lambda theta: (1 / m) * cost_func(hypothesis, X, y)(theta)


def compute_new_theta(X, y, theta, learning_rate, hypothesis, lambda_reg=0.1):
    """
    Updates learnable parameters theta.

    The update is done by calculating the partial derivatives of
    the cost function including the linear hypothesis. The
    gradients scaled by a scalar are subtracted from the given
    theta values.

    Parameters:
    X (numpy.ndarray): 2D numpy array of x values.
    y (numpy.ndarray): Array of y values corresponding to x.
    theta (numpy.ndarray): Current theta values.
    learning_rate (float): Value to scale the negative gradient.
    hypothesis (function): The hypothesis as function.

    Returns:
    numpy.ndarray: Updated theta
    """
    m = len(X)
    return theta - learning_rate * (1 / m) * transform_design_matrix(X).T.dot((hypothesis(theta)(X) - y))


def gradient_descent(X, y, theta, learning_rate, num_iters, lambda_reg=0.1):
    """
    Minimize theta values of a logistic model based on cross-entropy cost function.

    Parameters:
    X (numpy.ndarray): 2D numpy array of x values.
    y (numpy.ndarray): Array of y values corresponding to x.
    theta (numpy.ndarray): Current theta values.
    learning_rate (float): Value to scale the negative gradient.
    num_iters (int): Number of iterations updating thetas.
    lambda_reg (float): Regularization strength.
    cost_function (function): Python function for computing the cost.

    Returns:
    history_cost (numpy.ndarray): Cost after each iteration.
    history_theta (numpy.ndarray): Updated theta values after each iteration.
    """
    history_cost = np.zeros(num_iters + 1)
    history_theta = np.zeros((num_iters + 1, len(theta)))
    history_cost[0] = mean_cross_entropy_costs(X, y, logistic_hypothesis, cross_entropy_costs, 0.1)(theta).sum()
    history_theta[0] = theta
    for i in range(num_iters):
        history_theta[i + 1] = compute_new_theta(X, y, history_theta[i], learning_rate, logistic_hypothesis, lambda_reg)
        history_cost[i + 1] = (mean_cross_entropy_costs(X, y, logistic_hypothesis, cross_entropy_costs, 0.1)
                               (history_theta[i + 1])).sum()
    return history_cost, history_theta


def plot_progress(costs):
    """
    Plots the costs over the iterations.

    Parameters:
    costs (numpy.ndarray): History of costs.
    """
    plt.plot(np.arange(0, len(costs), 1), costs)
    plt.ylabel('costs')
    plt.xlabel('iteration')
    plt.savefig('cost-over-iterations.png')


def get_decision_boundary(theta):
    """
    Get the decision boundary based on a parameter set theta and an intervall in x1.

    Parameters:
    theta (numpy.ndarray): The parameter set.

    Returns:
    function: The decision boundary function based on a parameter set theta and an intervall in x1.
    """
    return lambda x1: (theta[0] + x1 * theta[1]) / -theta[2]


def plot_data_distribution(r0, r1, filename):
    """
    Plot the 2D data distribution and saves it into a file 'filename'.

    Parameters:
    r0 (numpy.ndarray): Data points with class 0.
    r1 (numpy.ndarray): Data points with class 1.
    """
    plt.scatter(r0[..., 0], r0[..., 1], c='b', marker='*', label="class 0")
    plt.scatter(r1[..., 0], r1[..., 1], c='r', marker='.', label="class 1")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.legend()
    plt.savefig(filename)

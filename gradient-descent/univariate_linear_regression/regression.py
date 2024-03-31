import numpy as np
import matplotlib.pyplot as plt


def get_linear_hypothesis(theta_0, theta_1):
    """
    Get linear hypothesis based on the given parameters.

    Parameters:
    theta_0 (float): The intercept parameter.
    theta_1 (float): The parameter that is to be multiplied by x.

    Returns:
    function: Linear hypothesis based on the given parameters and x.
    """
    return lambda x: theta_0 + theta_1 * x


def get_squared_error_cost_function(x, y, get_hypothesis):
    """
    Get squared error cost function.

    Parameters:
    x (numpy.ndarray): The x values of the observed data points.
    y (numpy.ndarray): The y values of the observed data points.
    get_hypothesis (function): Linear hypothesis based on x and y.

    Returns:
    function: The squared error cost function.
    """
    assert (len(x) == len(y))
    m = len(x)
    return lambda theta_0, theta_1: 1. / (2. * m) * ((get_hypothesis(theta_0, theta_1)(x) - y) ** 2).sum()


def compute_new_thetas(x, y, theta_0, theta_1, alpha):
    """
    Computes new parameters based on the standard gradient descent procedure.

    Parameters:
    x (numpy.ndarray): The x values of the observed data points.
    y (numpy.ndarray): The y values of the observed data points.
    theta_0 (float): The intercept parameter.
    theta_1 (float): The parameter that is to be multiplied by x.
    alpha (float): Learning rate.

    Returns:
    new_theta_0 (float): The new intercept parameter.
    new_theta_1 (float): The new parameter that is to be multiplied by x.
    """
    assert (len(x) == len(y))

    # partial derivative for theta_0
    temp_sum = 0
    for i in range(len(x)):
        temp_sum = temp_sum + theta_0 + theta_1 * x[i] - y[i]
    new_theta_0 = theta_0 - (alpha / len(x)) * temp_sum

    # partial derivative for theta_1
    temp_sum = 0
    for i in range(len(x)):
        temp_sum = temp_sum + (theta_0 + theta_1 * x[i] - y[i]) * x[i]
    new_theta_1 = theta_1 - (alpha / len(x)) * temp_sum

    return new_theta_0, new_theta_1


def gradient_descent(x, y, theta_0, theta_1, alpha, steps):
    """
    Iteratively applies the standard gradient descent procedure with "steps" amount of steps.

    Parameters:
    x (numpy.ndarray): The x values of the observed data points.
    y (numpy.ndarray): The y values of the observed data points.
    theta_0 (float): The intercept parameter.
    theta_1 (float): The parameter that is to be multiplied by x.
    alpha (float): Learning rate.
    steps (int): Amount of iterations.

    Returns:
    new_thetas_0 (numpy.ndarray): The new intercept parameters over all iterations.
    new_thetas_1 (numpy.ndarray): The new parameters that are to be multiplied by x over each iteration.
    """
    new_thetas_0 = np.zeros(steps + 1)
    new_thetas_1 = np.zeros(steps + 1)
    new_thetas_0[0] = theta_0
    new_thetas_1[0] = theta_1
    for i in range(steps):
        new_thetas_0[i+1], new_thetas_1[i+1] = compute_new_thetas(x, y, new_thetas_0[i], new_thetas_1[i], alpha)
    return new_thetas_0, new_thetas_1


def get_costs(x, y, thetas_0, thetas_1):
    """
    Computes squared error costs for given data entries and parameter combinations for a linear hypothesis.

    Parameters:
    x (numpy.ndarray): The x values of the observed data points.
    y (numpy.ndarray): The y values of the observed data points.
    thetas_0 (numpy.ndarray): The intercept parameters.
    thetas_1 (numpy.ndarray): The parameters that are to be multiplied by x.

    Returns:
    (numpy.ndarray): The squared error costs for given data entries and parameter combinations for a linear hypothesis.
    """
    assert len(thetas_0) == len(thetas_1)
    cost_function = get_squared_error_cost_function(x, y, get_linear_hypothesis)
    costs = np.zeros(len(thetas_0))
    for i in range(len(thetas_0)):
        costs[i] = cost_function(thetas_0[i], thetas_1[i])
    return costs


def plot_cost_over_iterations(x, y, thetas_0, thetas_1, filename):
    """
    Plots the cost over all iterations of a gradient descent.

    Parameters:
    x (numpy.ndarray): The x values of the observed data points.
    y (numpy.ndarray): The y values of the observed data points.
    thetas_0 (numpy.ndarray): The intercept parameters.
    thetas_1 (numpy.ndarray): The parameters that are to be multiplied by x.
    filename (str): Filename under which the generated plot will be saved.
    """
    costs = get_costs(x, y, thetas_0, thetas_1)
    plt.plot(np.arange(0, len(thetas_0), 1), costs)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("Cost Over Iterations")
    plt.savefig(filename)
    plt.clf()


def plot_best_hypothesis(x, y, thetas_0, thetas_1, filename):
    """
    Plots the hypothesis that minimizes the squared error cost.

    Parameters:
    x (numpy.ndarray): The x values of the observed data points.
    y (numpy.ndarray): The y values of the observed data points.
    thetas_0 (numpy.ndarray): The intercept parameters.
    thetas_1 (numpy.ndarray): The parameters that are to be multiplied by x.
    filename (str): Filename under which the generated plot will be saved.
    """
    costs = get_costs(x, y, thetas_0, thetas_1)
    best_theta_combination = [thetas_0[np.argmin(costs)], thetas_1[np.argmin(costs)]]
    plt.scatter(x, y)
    plt.plot(x, get_linear_hypothesis(best_theta_combination[0], best_theta_combination[1])(x))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Best Hypothesis")
    plt.savefig(filename)
    plt.clf()

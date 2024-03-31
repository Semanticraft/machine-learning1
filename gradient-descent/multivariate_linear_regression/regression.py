import numpy
import numpy as np
import matplotlib.pyplot as plt


def get_linear_hypothesis(theta):
    """
    Get linear hypothesis based on the given parameters.

    Parameters:
    theta (numpy.ndarray): A ndarray containing the parameters for the linear hypothesis.

    Returns:
    function: Linear hypothesis based on the given parameters and design matrix.
    """
    theta = np.array(theta)
    return lambda X: np.dot(X, theta.reshape(-1, 1))


def get_cost_function(X, y):
    """
    Get squared error cost function based on the design matrix and y.

    Parameters:
    X (numpy.ndarray): The design matrix.
    y (numpy.ndarray): The output vector.

    Returns:
    function: The squared error cost function based on the design matrix and y.
    """
    assert (len(X) == len(y))
    m = len(X)
    return lambda theta: 1. / (2. * m) * ((get_linear_hypothesis(theta)(X) - y) ** 2).sum()


def compute_new_theta(X, y, theta, alpha):
    """
    Compute new vector of theta parameters based on the standard gradient descent procedure.

    Parameters:
    X (numpy.ndarray): The design matrix.
    y (numpy.ndarray): The output vector.
    theta (numpy.ndarray): The parameter vector.
    alpha (float): The learning rate.

    Returns:
    numpy.ndarray: The new vector of theta parameters.
    """
    assert (len(X) == len(y))
    m = len(X)
    return theta.reshape(-1, 1) - alpha * (1.0 / m) * X.T.dot(get_linear_hypothesis(theta)(X) - y)


def gradient_descent(alpha, theta, X, y, steps):
    """
    Iteratively applies the standard gradient descent procedure with "steps" amount of steps.

    Parameters:
    alpha (float): The learning rate.
    theta (numpy.ndarray): The parameter vector.
    X (numpy.ndarray): The design matrix.
    y (numpy.ndarray): The output vector.
    steps (int): The amount of steps.

    Returns:
    numpy.ndarray: A matrix containing the computed parameter vectors over each iteration as rows.
    """
    computed_matrix = numpy.zeros((steps, len(theta)))
    computed_theta = theta
    for i in range(steps):
        computed_theta = compute_new_theta(X, y, computed_theta, alpha)
        computed_matrix[i] = computed_theta.flatten()
    return computed_matrix


def get_costs(X, y, thetas):
    """
    Computes squared error costs for given data entries and parameter combinations for a linear hypothesis.

    Parameters:
    X (numpy.ndarray): The design matrix.
    y (numpy.ndarray): The output vector.
    theta (numpy.ndarray): Matrix containing parameter vectors as rows.

    Returns:
    (numpy.ndarray): The squared error costs for given data entries and parameter combinations for a linear hypothesis.
    """
    cost_function = get_cost_function(X, y)
    costs = np.zeros(len(thetas))
    for i in range(len(thetas)):
        costs[i] = cost_function(thetas[i])
    return costs


def plot_cost_over_iterations(X, y, thetas, filename):
    """
    Plots the cost over all iterations of a gradient descent.

    Parameters:
    X (numpy.ndarray): The design matrix.
    y (numpy.ndarray): The output vector.
    thetas (numpy.ndarray): Matrix containing all parameter vectors as rows.
    filename (str): Filename under which the generated plot will be saved.
    """
    costs = get_costs(X, y, thetas)
    plt.plot(np.arange(0, len(thetas), 1), costs)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("Cost Over Iterations")
    plt.savefig(filename)
    plt.clf()


def plot_best_hypothesis(X, y, thetas, filename):
    """
    Plots the hypothesis that minimizes the squared error cost (only for 3D planes).

    Parameters:
    X (numpy.ndarray): The design matrix.
    y (numpy.ndarray): The output vector.
    thetas (numpy.ndarray): Matrix containing all parameter vectors as rows.
    filename (str): Filename under which the generated plot will be saved.
    """
    assert(len(X[0]) == 3)
    costs = get_costs(X, y, thetas)
    best_theta_combination = np.array([thetas[np.argmin(costs)]])
    X_1_mesh, X_2_mesh = np.meshgrid(np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.01),
                                     np.arange(np.min(X[:, 2]), np.max(X[:, 2]), 0.01))
    Z = best_theta_combination[:, 0] + best_theta_combination[:, 1] * X_1_mesh + best_theta_combination[:, 2] * X_2_mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_1_mesh, X_2_mesh, Z, cmap='jet', cstride=5, rstride=5, alpha=0.25)
    ax.scatter(X[:, 1], X[:, 2], y, c='r', marker='o', edgecolor='black')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.set_title('Best Hypothesis')
    fig.savefig(filename)
    plt.clf()

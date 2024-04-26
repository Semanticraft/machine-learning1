import numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


def create_feature_matrix(sample_size, n_features, x_min, x_max):
    """
    creates random feature vectors based on a linear function in a given interval

    Args:
        sample_size: number feature vectors
        n_features: number of features for each vector
        x_min: lower bound value ranges
        x_max: upper bound value ranges

    Returns:
        x: 2D array containing feature vectors with shape (sample_size, n_features)
    """
    x = np.zeros((sample_size, n_features))
    for i in range(n_features):
        x[:, i] = np.random.uniform(x_min[i], x_max[i], sample_size)
    return x


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


def linear_hypothesis(thetas):
    """
    Combines given list argument in a linear equation and returns it as a function

    Args:
        thetas: list of coefficients

    Returns:
        lambda that models a linear function based on thetas and x
    """
    return lambda X: np.dot(transform_design_matrix(X), thetas)


def generate_targets(X, theta, sigma):
    """
    Combines given arguments in a linear equation with X,
    adds some Gaussian noise and returns the result

    Args:
        X: 2D numpy feature matrix
        theta: list of coefficients
        sigma: standard deviation of the gaussian noise

    Returns:
        target values for X
    """
    return np.dot(transform_design_matrix(X), theta) + sigma * np.random.randn(len(X))


def mse_cost_function(x, y):
    """
    Implements MSE cost function as a function J(theta) on given training data

    Args:
        x: vector of x values
        y: vector of ground truth values y

    Returns:
        lambda J(theta) that models the cost function
    """
    return lambda theta: (1 / 2 * len(x)) * ((linear_hypothesis(theta)(x) - y)**2).sum()


def plot_data_scatter(features, targets):
    """
    Plots the features and the targets in a 3D scatter plot

    Args:
        features: 2D numpy-array features
        targets: targets
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], targets, c='r', marker='o', edgecolor='black')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('Distribution of Data')
    fig.savefig('distribution-of-data.png')


def update_theta(x, y, theta, learning_rate):
    """
    Updates learnable parameters theta

    The update is done by calculating the partial derivatives of
    the cost function including the linear hypothesis. The
    gradients scaled by a scalar are subtracted from the given
    theta values.

    Args:
        x: 2D numpy array of x values
        y: array of y values corresponding to x
        theta: current theta values
        learning_rate: value to scale the negative gradient

    Returns:
        theta: Updated theta vector
    """
    return theta - learning_rate * (1 / len(x)) * transform_design_matrix(x).T.dot(linear_hypothesis(theta)(x) - y)


def gradient_descent(learning_rate, theta, iterations, x, y, cost_function):
    """
    Minimize theta values of a linear model based on MSE cost function

    Args:
        learning_rate: scalar, scales the negative gradient
        theta: initial theta values
        x: vector, x values from the data set
        y: vector, y values from the data set
        iterations: scalar, number of theta updates
        cost_function: python function for computing the cost

    Returns:
        history_cost: cost after each iteration
        history_theta: Updated theta values after each iteration
    """
    

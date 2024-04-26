import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


def linear_random_data(sample_size: int, a: float, b: float, x_min: float, x_max: float, noise_factor: float) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a random data set based on a linear function in a given interval.

    Args:
        sample_size (int): Number of data points.
        a (float): Coefficient of x^0
        b (float): Coefficient of x^1
        x_min (float): Lower bound value range.
        x_max (float): Upper bound value range.
        noise_factor (float): Strength of noise added to y.

    Returns:
        x (numpy.ndarray): Array of x values | len(x)==len(y).
        y (numpy.ndarray): Array of y values corresponding to x | len(x)==len(y).
    """
    x = np.random.uniform(x_min, x_max, sample_size)
    y = a + b * x + np.random.randn(sample_size) * noise_factor
    return x, y


def linear_hypothesis(theta_0: float, theta_1: float) -> Callable[[np.ndarray], float]:
    """
    Combines given arguments in a linear equation and returns it as a function.

    Args:
        theta_0 (float): First coefficient.
        theta_1 (float): Second coefficient.

    Returns:
        function: Lambda that models a linear function based on theta_0, theta_1 and x.
    """
    return lambda x: theta_0 + theta_1 * x


def mse_cost_function(x: np.ndarray, y: np.ndarray) -> Callable[[float, float], float]:
    """
    Implements MSE cost function as a function J(theta_0, theta_1) on given training data.

    Args:
        x (numpy.ndarray): Vector of x values.
        y (numpy.ndarray): Vector of ground truth values y.

    Returns:
        function: Lambda J(theta_0, theta_1) that models the cost function.
    """
    assert (x.size == y.size)
    return lambda theta_0, theta_1: (1 / (2 * x.size)) * (linear_hypothesis(theta_0, theta_1)(x) - y).sum()


def plot_data_with_hypothesis(x: np.ndarray, y: np.ndarray, theta_0: float, theta_1: float):
    """
    Plots the data (x, y) together with a hypothesis given theta0 and theta1.

    Args:
        x (numpy.ndarray): Vector of x values.
        y (numpy.ndarray): Vector of ground truth values y.
        theta_0 (float): First hypothesis parameter.
        theta_1 (float): Second hypothesis parameter.
    """
    plt.plot(x, y, 'xr', label='data set')
    plt.plot(x, linear_hypothesis(theta_0, theta_1)(x), label='hypothesis')
    plt.legend()
    plt.text(1, -40, s='θ₀: ' + str(theta_0) + '\nθ₁: ' + str(theta_1) + '\nCost: ' +
                       str(mse_cost_function(x, y)(theta_0, theta_1)), ha='left', va='bottom')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Hypothesis')
    plt.savefig('hypothesis.png')


def create_cost_plt_grid(cost: Callable[[float, float], float], interval: float, num_samples: int, theta_0_offset=0.,
                         theta_1_offset=0.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates mesh points for a 3D plot based on a given interval and a cost function.
    The function creates a numpy meshgrid for plotting a 3D-plot of the cost function.
    Additionally, for the mesh grid points cost values are calculated and returned.

    Args:
        cost (function): A function that is used to calculate costs. The function "cost" was typically e.g.
              created by calling "cost = mse_cost_function(x, y)". So, the data x,y and the model
              are used internally in cost. The arguments of the function cost are
              theta_0 und theta_1, i.e. cost(theta_0, theta_0).
        interval (float): A scalar that defines the range [-interval, interval] of the mesh grid.
        num_samples (int): The total number of points in the mesh grid is num_samples * num_samples (equally distributed
        ).
        theta_0_offset (float): Shifts the plotted interval for theta_0 by a scalar.
        theta_1_offset (float): Shifts the plotted interval for theta_1 by a scalar.

    Returns:
        T0 (numpy.ndarray): A matrix representing a meshgrid for the values of the first plot dimension (Theta 0).
        T1 (numpy.ndarray): A matrix representing a meshgrid for the values of the second plot dimesion (Theta 1).
        C (numpy.ndarray): A matrix representing cost values (third plot dimension).
    """
    x, y = np.meshgrid(theta)


def create_cost_plt(T0: np.ndarray, T1: np.ndarray, Costs: np.ndarray):
    """
    Creates a contour and a surface plot based on the given data.

    Args:
        T0 (numpy.ndarray): A matrix representing a meshgrid for X values (Theta 0).
        T1 (numpy.ndarray): A matrix representing a meshgrid for Y values (Theta 1).
        Costs (numpy.ndarray): A matrix representing cost values .
    """
    raise NotImplementedError("You should implement this!")
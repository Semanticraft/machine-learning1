import numpy as np
import matplotlib.pyplot as plt

import hashlib


def round_and_hash(value, precision=4, dtype=np.float32):
    """
    Function to round and hash a scalar or numpy array of scalars.
    Used to compare results with true solutions without spoiling the solution.
    """
    rounded = np.array([value], dtype=dtype).round(decimals=precision)
    hashed = hashlib.md5(rounded).hexdigest()
    return hashed


def train_data():
    x_values = np.random.uniform(0, 2 * np.pi, 2)
    y_values = np.sin(x_values)
    return x_values, y_values


def get_thetas(x, y):
    theta1 = (y[1] - y[0]) / (x[1] - x[0])
    theta0 = y[0] - theta1 * x[0]
    return np.array([theta0, theta1])


def get_w(x, y):
    return np.array((y[0] + y[1]) / 2)


def get_hypothesis_1(thetas):
    return lambda x: thetas[0] + x * thetas[1]


def get_hypothesis_2(w):
    return lambda x: np.full(len(x), w)


def plot_true_target_function_x_y_h1_h2(x, y, hypothesis1, hypothesis2, filename):
    x_grid = np.linspace(0, 2 * np.pi, 1000)
    y_grid = np.sin(x_grid)
    plt.plot(x_grid, y_grid, label='true target')
    plt.plot(x_grid, hypothesis1(x_grid), label='linear hypothesis')
    plt.plot(x_grid, hypothesis2(x_grid), label='constant hypothesis')
    plt.scatter([x[0], x[1]], [y[0], y[1]])
    plt.xlim(0, 2 * np.pi)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def out_of_sample_error(y_preds, y):
    return np.mean((y_preds - y) ** 2)


def run_experiment(m):
    xs = np.ndarray((m, 2))
    ys = np.ndarray((m, 2))
    t0s = np.ndarray(m)
    t1s = np.ndarray(m)
    ws = np.ndarray(m)
    e_out_h1s = np.ndarray(m)
    e_out_h2s = np.ndarray(m)
    x_grid = np.linspace(0, 2 * np.pi, 1000)
    y_grid = np.sin(x_grid)

    for i in range(m):
        xs[i], ys[i] = train_data()
        t0s[i], t1s[i] = get_thetas(xs[i], ys[i])
        ws[i] = get_w(xs[i], ys[i])
        e_out_h1s[i] = out_of_sample_error(get_hypothesis_1([t0s[i], t1s[i]])(x_grid), y_grid)
        e_out_h2s[i] = out_of_sample_error(get_hypothesis_2(ws[i])(x_grid), y_grid)

    return xs, ys, t0s, t1s, ws, e_out_h1s, e_out_h2s


def bias_square(y_true, y_avg):
    """
     Returns the bias^2 of a hypothesis set for the sin-example.

            Parameters:
                    y_true(np.array): The y-values of the target function
                                      at each position on the x_grid
                    y_avg(np.array): The y-values of the avg hypothesis
                                     at each position on the x_grid

            Returns:
                    variance (double):  Bias^2 of the hypothesis set
    """
    return np.mean((y_avg - y_true) ** 2)


def variances(hypothesis_func, param_func, xs, ys, x_grid, y_avg):
    """
    Returns the variance of a hypothesis set for the sin-example.

            Parameters:
                    hypothesis_func (function): The hypothesis function 1 or 2
                    param_func (function): the function to calculate the parameters
                            from the training data, i.e., get_theta or get_w
                    xs(np.array): 2D-Array with different training data values for x
                                first dimension: different training data sets
                                second dimension: data points in a data set
                    ys(np.array): 2D-Array with different training data values for y
                                first dimension: different training data sets
                                second dimension: data points in a data set
                    x_grid(np.array): The x-values for calculating the expectation E_x
                    y_avg(np.array): The y-values of the average hypothesis at the
                                     positions of x_grid

            Returns:
                    variance (double):  Variance of the hypothesis set for
                                        a type for training data
                                        (here two examples per training data set)
    """
    num_datasets = xs.shape[0]
    predictions = []

    for i in range(num_datasets):
        x_train = xs[i]
        y_train = ys[i]
        params = param_func(x_train, y_train)
        y_pred = hypothesis_func(params)(x_grid)
        predictions.append(y_pred)

    predictions = np.array(predictions)
    squared_diffs = (predictions - y_avg) ** 2
    variance = np.mean(squared_diffs)

    return variance

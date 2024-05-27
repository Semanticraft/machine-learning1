import numpy as np

import logic

x_train, y_train = logic.train_data()
print(x_train, y_train)

assert len(x_train) == 2
assert len(y_train) == 2
np.testing.assert_array_equal(np.sin(x_train), y_train)
for i in range(1000):
    x_tmp, _ = logic.train_data()
    assert x_tmp.min() >= 0.0
    assert x_tmp.max() <= 2 * np.pi

thetas = logic.get_thetas(x_train, y_train)
w = logic.get_w(x_train, y_train)
print(thetas[0], thetas[1])
print(w)

x_train_temp = np.array([0, 1])
y_train_temp = np.array([np.sin(x_i) for x_i in x_train_temp])
thetas_test = logic.get_thetas(x_train_temp, y_train_temp)
w_test = logic.get_w(x_train_temp, y_train_temp)

np.testing.assert_almost_equal(thetas_test[0], 0.0)
np.testing.assert_almost_equal(thetas_test[1], 0.8414709848078965)
np.testing.assert_almost_equal(w_test, 0.42073549240394825)
# we want to compute numerically the expectation w.r.t. x
# p(x) is const. in the intervall [0, 2pi]
x_grid = np.linspace(0, 2 * np.pi, 100)
y_grid = np.sin(x_grid)

print(w_test)
h1_test = logic.get_hypothesis_1(thetas_test)
h2_test = logic.get_hypothesis_2(w_test)
np.testing.assert_almost_equal(h1_test(x_grid)[10], 0.5340523361780719)
np.testing.assert_almost_equal(h2_test(x_grid)[10], 0.42073549240394825)

x_train, y_train = logic.train_data()
thetas = logic.get_thetas(x_train, y_train)
w = logic.get_w(x_train, y_train)
logic.plot_true_target_function_x_y_h1_h2(x_train, y_train, logic.get_hypothesis_1(thetas), logic.get_hypothesis_2(w),
                                          'plot.png')

e_out_h1_test = logic.out_of_sample_error(h1_test(x_grid), y_grid)
np.testing.assert_almost_equal(e_out_h1_test, 11.52548591)

num_training_data = 10000
xs, ys, t0s, t1s, ws, e_out_h1s, e_out_h2s = logic.run_experiment(num_training_data)

t0_avg = t0s.mean()
t1_avg = t1s.mean()
thetas_avg = [t0_avg, t1_avg]
w_avg = ws.mean()
h1_avg = logic.get_hypothesis_1(thetas_avg)
h2_avg = logic.get_hypothesis_2(w_avg)
print(thetas_avg)

logic.plot_true_target_function_x_y_h1_h2(x_grid, y_grid, h1_avg, h2_avg, 'average-models.png')

expectation_Eout_1 = e_out_h1s.mean()
print("expectation of E_out of model 1:", expectation_Eout_1)

expectation_Eout_2 = e_out_h2s.mean()
print("expectation of E_out of model 2:", expectation_Eout_2)

bias_1 = logic.bias_square(y_grid, h1_avg(x_grid))
print("Bias of model 1:", bias_1)

bias_2 = logic.bias_square(y_grid, h2_avg(x_grid))
print("Bias of model 2:", bias_2)

var_hypothesis_set_1 = logic.variances(logic.get_hypothesis_1,
                                       logic.get_thetas,
                                       xs, ys,
                                       x_grid,
                                       h1_avg(x_grid))
print(var_hypothesis_set_1)

var_hypothesis_set_2 = logic.variances(logic.get_hypothesis_2,
                                       logic.get_w,
                                       xs, ys,
                                       x_grid,
                                       h2_avg(x_grid))
print(var_hypothesis_set_2)

print("model 1: E_out ≈ bias^2 + variance:  %f ≈ %f + %f" % (expectation_Eout_1, bias_1, var_hypothesis_set_1))
print("model 2: E_out ≈ bias^2 + variance:  %f ≈ %f + %f" % (expectation_Eout_2, bias_2, var_hypothesis_set_2))

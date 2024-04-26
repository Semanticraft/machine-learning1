import uebung1.logic as logic
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

x, y = logic.linear_random_data(sample_size=50, a=0., b=5., x_min=-10, x_max=10, noise_factor=5)
plt.plot(x, y, "rx")
plt.xlabel('x')
plt.ylabel('y')
plt.title('linear random data')
plt.savefig('linear-random-data.png')
plt.clf()

j = logic.mse_cost_function(x, y)
print(j(2.1, 2.9))
print(j(2.3, 4.9))

t0 = 0
t1 = 1

logic.plot_data_with_hypothesis(x, y, theta_0=t0, theta_1=t1)
plt.clf()

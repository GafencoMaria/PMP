import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# ex1
def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    # prior = np.repeat(1 / grid_points, grid_points)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


# aruncam de 28 ori o moneda si observam 8 steme
data = np.repeat([0, 1], (20, 8))
points = 10
h = data.sum()
t = len(data) - h
grid_, posterior_ = posterior_grid(points, h, t)
plt.plot(grid_, posterior_, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')


# ex2
def find_pi(n):
    x, y = np.random.uniform(-1, 1, size=(2, n))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / n
    error = abs((pi - np.pi) / pi) * 100
    return error


error_list = [find_pi(100), find_pi(1000), find_pi(10000)]
error_mean = np.mean(error_list)
error_deviation = np.std(error_list)
plt.errorbar(error_mean)
plt.errorbar(error_deviation)


# ex3
def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


# folosim parametrii distributiei a priori din cursul 2
func_ = stats.beta[(1, 1), (20, 20), (1, 4)]
trace_ = metropolis(func=func_)
x_ = np.linspace(0.01, .99, 100)
y_ = func_.pdf(x_)
plt.xlim(0, 1)
plt.plot(x_, y_, 'C1-', lw=3, label='True distribution')
plt.hist(trace_[trace_ > 0], bins=25, density=True, label='Estimated distribution')
plt.xlabel('x')
plt.ylabel('pdf(x)')
plt.yticks([])
plt.legend()

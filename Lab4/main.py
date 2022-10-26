import numpy as np
import math

lam = 20
traffic = np.random.poisson(lam, 1000)
time = np.random.normal(1, 0.5, 1000)
alpha = 1
preparation_time = np.random.exponential(1/alpha, 1000)


def ex2():
    alpha_max = -(15 * math.log(2.71828) / math.log(0.05))
    return alpha_max


def ex3():
    avg_time = (math.log(1.6) / 7.5 * math.log(2.71828)) * 60
    return avg_time

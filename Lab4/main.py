import numpy as np

lam = 20
traffic = np.random.poisson(lam, 1000)
time = np.random.normal(1, 0.5, 1000)
alpha = 1
preparation_time = np.random.exponential(1/alpha, 1000)

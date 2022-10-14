import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

m1 = stats.expon.rvs(0.1/4)
m2 = stats.expon.rvs(0.1/6)
x = stats.uniform.rvs(0, (m1+m2)/2, size=10000)


az.plot_posterior({'x':x}) 
plt.show() 

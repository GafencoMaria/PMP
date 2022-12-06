import numpy as np
import pymc3 as pm


min_stations = 1
min_checkout_point = 1
min_tables = 1
for i in range(10):
    model = pm.Model()
    with model:
        clients_nr = pm.Poisson('N', mu=20)
        checkout_point_time = pm.Normal('T_c', mu=1, sd=0.5, shape=50)
        cooking_time = pm.Exponential('T_g', lam=2, shape=50)
        table_time = pm.TruncatedNormal('T_m', mu=10, sd=2, lower=0)
        idx = np.arange(50)
        time = pm.math.switch(clients_nr > idx, checkout_point_time[idx] + cooking_time[idx], 0)
        success1 = pm.Deterministic('S', pm.math.prod(pm.math.switch(time < 15, 1, 0)))
        success2 = pm.Deterministic('S', pm.math.prod(pm.math.switch(time > 0, 1, 0)))
        trace = pm.sample(10000)
    prob1 = len(success1[(success1 == 1)]) / len(success1)
    prob2 = len(success2[(success2 == 1)]) / len(success2)
    if prob1 < 0.95:
        min_checkout_point = min_checkout_point + 1
        min_stations += 1
    if prob2 < 0.90:
        min_tables += 1

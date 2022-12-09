import random

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')


dummy_data = np.loadtxt('date.csv')

x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)


# 1.a.
def print_graf():
    α_p_post_a = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post_a = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx_a = np.argsort(x_1s[0])
    y_p_post_a = α_p_post_a + np.dot(β_p_post_a, x_1s)

    plt.plot(x_1s[0][idx_a], y_p_post_a[idx_a], 'C1', label=f'model order {order}')

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()


# 1.b.
# sd=100
with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

print_graf()

# sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

print_graf()


# 2.
for i in range(len(x_1) + 1, 501):
    x_1 = np.append(x_1, round(random.uniform(-1.081, 3.970), 5))
    y_1 = np.append(y_1, round(random.uniform(9.357, -6.080), 5))
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

print_graf()

# sd=100
with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

print_graf()

# sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

print_graf()


# 3.
# model cubic
order = 3
x_1p_c = np.vstack([x_1**i for i in range(1, order+1)])
x_1s_c = (x_1p_c - x_1p_c.mean(axis=1, keepdims=True)) / x_1p_c.std(axis=1, keepdims=True)
y_1s_c = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s_c[0], y_1s_c)
plt.xlabel('x_c')
plt.ylabel('y_c')

with pm.Model() as model_c:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_c = pm.sample(2000, return_inferencedata=True)

waic = az.waic(dummy_data, scale="deviance")
loo = az.loo(dummy_data, scale="deviance")

# model liniar si patratic
order = 2
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

with pm.Model() as model_l:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10)
    ε = pm.HalfNormal('ε', 5)
    μ = α + β * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

# cubic
α_c_post = idata_c.posterior['α'].mean(("chain", "draw")).values
β_c_post = idata_c.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s_c[0])
y_c_post = α_c_post + np.dot(β_c_post, x_1s_c)

plt.plot(x_1s_c[0][idx], y_c_post[idx], 'C1', label=f'model order {order}')

# liniar
x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
y_l_post = α_l_post + β_l_post * x_new

plt.plot(x_new, y_l_post, 'C2', label='linear model')

# patratic
α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()

# comparare
cmp_df_waic = az.compare({'model_c': idata_c, 'model_l': idata_l, 'model_p': idata_p}, method='BB-pseudo-BMA', ic="waic"
                         , scale="deviance")
cmp_df_loo = az.compare({'model_c': idata_c, 'model_l': idata_l, 'model_p': idata_p}, method='BB-pseudo-BMA', ic="loo"
                        , scale="deviance")

import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
# tt.config.blas__ldflags = ''

if __name__ == "__main__":
    data = pd.read_csv('Admission.csv')
    data.head()
    admission = data['Admission'].values
    df = data.query("Admission == ('0', '1')")
    y_1 = pd.Categorical(df(['Admission'])).codes
    x_n = ['GRE', 'GPA']
    x_1 = df[x_n].values

    with pm.Model as model:
        beta0 = pm.Normal('beta0', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

        miu = beta0 + pm.math.dot(x_1, beta)
        teta = pm.Deterministic('teta', 1 / (1 + pm.math.exp(-miu)))
        bd0 = pm.Deterministic('bd0', -beta0/beta[1] - beta[0]/beta[1] * x_1[:, 0])

        y1 = pm.Bernoulli('y1', p=teta, observed=y_1)

        idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    idx = np.argsort(x_1[:, 0])
    bd = idata_1.posterior['bd'].mean(("chain", "draw"))[idx]
    plt.scatter(x_1[:, 0], x_1[:, 1], c=[f'C{x}' for x in y_1])
    plt.plot(x_1[:, 0][idx], bd, color='k');
    az.plot_hdi(x_1[:, 0], idata_1.posterior['bd'], color='k')
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])
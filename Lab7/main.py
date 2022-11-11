import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0, 0].scatter(speed, price, alpha=0.6)
    axes[0, 1].scatter(hardDrive, price, alpha=0.6)
    axes[1, 0].scatter(ram, price, alpha=0.6)
    axes[1, 1].scatter(premium, price, alpha=0.6)
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].set_xlabel("Speed")
    axes[0, 1].set_xlabel("HardDrive")
    axes[1, 0].set_xlabel("Ram")
    axes[1, 1].set_xlabel("Premium")

    with pm.Model() as model_md:
        alpha = pm.Normal('alpha', sd=10)
        beta1 = pm.Normal('beta1', sd=10)
        beta2 = pm.Normal('beta2', sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        x1 = speed
        x2 = np.log(hardDrive)
        mu = pm.Deterministic('mu', alpha + beta1*x1 + beta2*x2)
        price_prediction = pm.Normal('price_prediction', mu=mu, sd=sigma)
        trace_md = pm.sample(1000, tune=1000)

    ppc = pm.sample_posterior_predictive(trace_md, samples=100, model=model_md)
    plt.plot(speed, hardDrive, 'C0.', alpha=0.1)
    az.plot_posterior(model_md, combined=False, colors='cycle', figsize=(8, 3))

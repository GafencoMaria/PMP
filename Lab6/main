import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm


if __name__ == "__main__":
# 1.
        data = pd.read_csv("data.csv")
        print(data)

        plt.scatter(data['ppvt'], data['momage'])
        plt.show()

# 2.
        with pm.Model() as model_g:
                alpha = pm.Normal('alpha', mu=0, sd=10)
                beta = pm.Normal('beta', mu=0, sd=1)
                epsilon = pm.HalfCauchy('epsilon', 5)
                mu - pm.Deterministic('mu', alpha + beta * data['ppvt'])
                y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
                idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)

                az.plot_pair(idata_g, var_names=['alpha', 'beta'], scatter_kwargs={'alpha': 0.1})

                plt.plot(x, y, 'C0.')
                posterior_g = idata_g.posterior.stack(samples={"chain", "draw"})
                alpha_m = posterior_g['alpha'].mean().item()
                beta_m = posterior_g['beta'].mean().item()
                draws = range(0, posterior_g.samples.size, 10)
                plt.plot(x, posterior_g['alpha'][draws].values

                         + posterior_g['beta'][draws].values * x[:, None],
                         c='gray', alpha=0.5)
                plt.plot(x, alpha_m + beta_m * x, c='k',
                         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
                plt.xlabel('x')
                plt.ylabel('y', rotation=0)
                plt.legend()

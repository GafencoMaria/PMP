import numpy as np
import arviz as az
import pymc3 as pm

# 1
clusters = 3
with pm.Model() as model_3:
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 3, 0]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))


# 2
clusters = 2
with pm.Model() as model_2:
    n_cluster = [350, 150]
    n_total = sum(n_cluster)
    means = [5, 0]
    std_devs = [2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    data_2 = pm.sample(500, return_inferencedata=True)

clusters = 3
with pm.Model() as model_3:
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 3, 0]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    data_3 = pm.sample(500, return_inferencedata=True)

clusters = 4
with pm.Model() as model_4:
    n_cluster = [200, 150, 100, 50]
    n_total = sum(n_cluster)
    means = [5, 4, 3, 0]
    std_devs = [2, 2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    data_4 = pm.sample(500, return_inferencedata=True)


# 3
waic = az.waic(500, scale="deviance")
loo = az.loo(500, scale="deviance")

cmp_df_waic = az.compare({'model_2': data_2, 'model_3': data_3, 'model_4': data_4}, method='BB-pseudo-BMA', ic="waic"
                         , scale="deviance")
cmp_df_loo = az.compare({'model_2': data_2, 'model_3': data_3, 'model_4': data_4}, method='BB-pseudo-BMA', ic="loo"
                        , scale="deviance")

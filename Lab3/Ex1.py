import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

if __name__ == '__main__':
    model = pm.Model()

    # cutremur si incendiu 98%
    # cutremur fara incendiu 2%
    # fara cutremur dar incendiu 95%
    # fara cutremur si incendiu 0.01%

    with model:
        cutremur = pm.Bernoulli('C', 0.0005)
        incendiu = pm.Bernoulli('I', pm.math.switch(cutremur, 0.03, 0.01))
        alarma_p = pm.Deterministic('A_p', pm.math.switch(cutremur, pm.math.switch(incendiu, 0.98, 0.02), pm.math.switch(incendiu, 0.95, 0.0001)))
        alarma_da = pm.Bernoulli('A1', p=alarma_p, observed=1)
        # alarma_nu = pm.Bernoulli('A2', p=alarma_p, observed=0)
        trace = pm.sample(50000)

    dictionary = {
        'cutremur': trace['C'].tolist(),
        'incendiu': trace['I'].tolist()
    }
    df = pd.DataFrame(dictionary)

    # prob sa fi avut loc un cutremur stiind ca alarma a fost declansata
    p_cutremur = df[(df['cutremur'] == 1)].shape[0]/df.shape[0]
    print(p_cutremur)

    # prob sa fi avut loc un incendiu stiind ca alarma nu a fost declansata
    # p_incendiu = df[(df['incendiu'] == 1)].shape[0]/df.shape[0]
    # print(p_incendiu)



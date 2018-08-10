import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tabulate import tabulate

import util
from problem.branin import Branin
from fidelity_strat import FidelityPropToBudget

num_configs = 1000
budgets = [9, 27, 81, 243]
problems = [
    ('Zero', Branin(deviation_kwargs={'bz': 0, 'cz': 0, 'tz': 0}), FidelityPropToBudget([True]*3)),
    ('Default', Branin(), FidelityPropToBudget([True]*3)),
    ('1.', Branin(deviation_kwargs={'tz': 1., 'cz': 1., 'bz': 1.}), FidelityPropToBudget([True]*3)),
    ('1.5', Branin(deviation_kwargs={'tz': 1., 'cz': 1., 'bz': 0.05}), FidelityPropToBudget([True]*3)),
    ('1.', Branin(deviation_kwargs={'tz': 0., 'cz': 0., 'bz': 1.}), FidelityPropToBudget([True]*3)),
    ('1.', Branin(deviation_kwargs={'tz': 0., 'cz': 0., 'bz': 0.05}), FidelityPropToBudget([True]*3)),
]


def calc_loss(config, budget, problem, strat):
    norm_budget = util.normalize_budget(budget, budgets[-1])
    z = strat.calc_fidelities(norm_budget)
    return problem.calc_loss(config, z)


# Bookkeeping
losses = np.zeros((len(problems), len(budgets), num_configs))
n = len(budgets) - 1  # Gaußsche Summenformel -> Anzahl der Kombinationen
rank_corr = np.zeros((len(problems), (n**2+n)//2))

# Iterate all problems,strategies
for i, (_, p, s) in enumerate(problems):
    cs = p.build_config_space()
    configs = [cs.sample_configuration() for _ in range(num_configs)]
    # Iterate all budgets
    for j, b in enumerate(budgets):
        # Iterate all configurations
        for k, c in enumerate(configs):
            losses[i, j, k] = calc_loss(c, b, p, s)

combinations = [(j, k) for j in range(len(budgets)) for k in range(j+1, j+1+len(budgets[j+1:]))]
combinations.sort(key=lambda x: np.abs(x[0]-x[1]))
# combinations = combinations[0:3]
fix, axes = plt.subplots(len(problems), 1, figsize=(12, 16))

for i, axis in zip(range(len(problems)), axes):
    axis.set_title(problems[i][0])

    for j, (i1, i2) in enumerate(combinations):
        rank_corr[i, j], _ = scipy.stats.spearmanr(losses[i, i1, :], losses[i, i2, :])
        axis.scatter(losses[i, i1, :], losses[i, i2, :], s=0.5, label="{} -> {}".format(budgets[i1], budgets[i2]))


headers = ['Name']
headers.extend(["{} -> {}".format(budgets[i1], budgets[i2]) for i1, i2 in combinations])

data = []
for i, (n, _, _) in enumerate(problems):
    d = [n]
    d.extend(rank_corr[i, :])
    data.append(d)

print(tabulate(data, headers=headers))
plt.legend()
plt.show()
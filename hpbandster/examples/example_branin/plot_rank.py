import numpy as np
import scipy.stats
from tabulate import tabulate

import util
from branin import Branin
from fidelity_strat import FidelityPropToBudget

num_configs = 100
budgets = [9, 27, 81, 243]
problems = [
    ('Zero', Branin(deviation_kwargs={'bz': 0, 'cz': 0, 'tz': 0}), FidelityPropToBudget([True]*3)),
    ('Default', Branin(), FidelityPropToBudget([True]*3)),
    ('0.05', Branin(deviation_kwargs={'bz': 0.05, 'cz': 0.05, 'tz': 0.05}), FidelityPropToBudget([True]*3)),
]


def calc_loss(config, budget, problem, strat):
    norm_budget = util.normalize_budget(budget, budgets[-1])
    z = strat.calc_fidelities(norm_budget)
    return problem.calc_loss(config, z)


def calc_rank(losses):
    tmp = [(l, i) for i, l in enumerate(losses)]
    return [i for (_, i) in sorted(tmp)]

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
for i in range(len(problems)):
    ranks = []
    for j in range(len(budgets)):
        ranks.append(calc_rank(losses[i, j, :]))

    for j, (i1, i2) in enumerate(combinations):
        rank_corr[i, j], _ = scipy.stats.spearmanr(ranks[i1], ranks[i2])


headers = ['Name']
headers.extend(["{} -> {}".format(budgets[i1], budgets[i2]) for i1, i2 in combinations])

data = []
for i, (n, _, _) in enumerate(problems):
    d = [n]
    d.extend(rank_corr[i, :])
    data.append(d)

print(tabulate(data, headers=headers))

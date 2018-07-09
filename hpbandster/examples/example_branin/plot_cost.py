#!/usr/bin/env python3
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np

from branin import Branin
from fidelity_strat import FidelityPropToCost
import util

resolution = 50
hb_steps = 5
alpha = 1
cost_pow = dict(pow_z1=3, pow_z2=2, pow_z3=1.5)
branin = Branin(**cost_pow)
min_cost = branin.cost(0, 0, 0)
max_cost = branin.cost(1, 1, 1)
extent = (0, 1, 0, 1)

z = np.linspace(0, 1, resolution, endpoint=True)
Z1, Z2, Z3 = np.meshgrid(z, z, z, sparse=True)
C = branin.cost(Z1, Z2, Z3) / max_cost
budgets = np.array([9, 27, 81, 243])
norm_budgets = budgets / budgets.max()
plot_data = [
    (1, 2, C[:, :, -1]),
    (3, 2, C[:, -1, :]),
    (3, 1, C[-1, :, :]),
]


def log3(x):
    return np.log(x) / np.log(3)


def calc_trajectory(xi, yi):
    def cost(*z):
        zz = np.ones(3)
        zz[xi] = z[0]
        zz[yi] = z[1]
        return branin.cost(*zz)

    trajectory = np.empty((len(budgets), 2))
    local_branin = Branin(**cost_pow)
    local_branin.cost = cost
    strategy = FidelityPropToCost(2, local_branin, False, alpha=alpha)
    for i, b in enumerate(norm_budgets):
        z = strategy.calc_fidelities(b)
        trajectory[i] = z
    return trajectory


f, axis = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

for ax, (xl, yl, CC) in zip(axis, plot_data):
    print(CC.min(), CC.max(), norm_budgets)
    ax.set_title('$cost \\propto z_{0}^{{{1}}} \\cdot z_{2}^{{{3}}}$'
                 .format(xl, cost_pow['pow_z' + str(xl)],
                         yl, cost_pow['pow_z' + str(yl)]))
    ax.set_xlabel('$z_{}$'.format(xl))
    ax.set_ylabel('$z_{}$'.format(yl))
    im = ax.imshow(CC, interpolation='bilinear', origin='lower', extent=extent)
    co = ax.contour(CC, levels=norm_budgets, colors='k', origin='lower', extent=extent)
    trajectory = calc_trajectory(xl - 1, yl - 1)
    print(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-')

plt.tight_layout()
plt.show()

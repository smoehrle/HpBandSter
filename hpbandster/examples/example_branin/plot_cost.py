#!/usr/bin/env python3
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

import branin
import util

resolution = 50
hb_steps = 5
alpha = 1
cost_pow = dict(pow_z1=3, pow_z2=2, pow_z3=1.5)
min_cost = branin.cost(0, 0, 0, **cost_pow)
max_cost = branin.cost(1, 1, 1, **cost_pow)
extent = (0, 1, 0, 1)

z = np.linspace(0, 1, resolution, endpoint=True)
Z1, Z2, Z3 = np.meshgrid(z, z, z, sparse=True)
C = branin.cost(Z1, Z2, Z3, **cost_pow)
budgets = min_cost + (max_cost - min_cost) * 3.**np.arange(1 - hb_steps, 1)
plot_data = [
    (1, 2, C[:, :, -1]),
    (3, 2, C[:, -1, :]),
    (3, 1, C[-1, :, :]),
]


def log3(x):
    return np.log(x) / np.log(3)


def calc_trajectory(xi, yi):
    def objective(z, b):
        zz = np.ones(3)
        zz[xi] = z[0]
        zz[yi] = z[1]
        return branin.cost_objective(zz, b, cost_kwargs=cost_pow)

    trajectory = np.empty((len(budgets), 2))
    x0 = np.ones(2)
    for i, b in enumerate(budgets[::-1]):
        # Linesearch fails too often for default solver L-BFGS
        # Investigate why linesearch fails - they are also an issue with TNC
        z = util.fidelity_propto_cost(b, x0, objective, fallback=False)
        trajectory[i] = z
        x0 = z
    return trajectory


f, axis = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

for ax, (xl, yl, CC) in zip(axis, plot_data):
    ax.set_title('$cost \\propto z_{0}^{{{1}}} \\cdot z_{2}^{{{3}}}$'
                 .format(xl, cost_pow['pow_z' + str(xl)],
                         yl, cost_pow['pow_z' + str(yl)]))
    ax.set_xlabel('$z_{}$'.format(xl))
    ax.set_ylabel('$z_{}$'.format(yl))
    im = ax.imshow(CC, interpolation='bilinear', origin='lower', extent=extent)
    co = ax.contour(CC, levels=budgets, colors='k', origin='lower', extent=extent)
    ax.clabel(co, budgets[0:-1:2],  # label every second level
              inline=1, fmt='%1.3f')
    trajectory = calc_trajectory(xl - 1, yl - 1)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-')

plt.tight_layout()
plt.show()

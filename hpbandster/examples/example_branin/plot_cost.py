#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from problem.branin import Branin
from strategy import PropToCost

resolution = 50
hb_steps = 5
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
    fid_filter = np.full((3,), False)
    fid_filter[xi] = True
    fid_filter[yi] = True
    trajectory = np.empty((len(budgets), 2))
    strategy = PropToCost(branin, use_fidelity=fid_filter)
    for i, b in enumerate(norm_budgets):
        z = strategy.calc_fidelities(b)
        trajectory[i, 0] = z[xi]
        trajectory[i, 1] = z[yi]

    return trajectory


f, axis = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

for ax, (xl, yl, CC) in zip(axis, plot_data):
    ax.set_title('$cost \\propto z_{0}^{{{1}}} \\cdot z_{2}^{{{3}}}$'
                 .format(xl, cost_pow['pow_z' + str(xl)],
                         yl, cost_pow['pow_z' + str(yl)]))
    ax.set_xlabel('$z_{}$'.format(xl))
    ax.set_ylabel('$z_{}$'.format(yl))
    im = ax.imshow(CC, interpolation='bilinear', origin='lower', extent=extent)
    co = ax.contour(CC, levels=norm_budgets, colors='k', origin='lower', extent=extent)
    trajectory = calc_trajectory(xl - 1, yl - 1)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-')

plt.tight_layout()
plt.show()

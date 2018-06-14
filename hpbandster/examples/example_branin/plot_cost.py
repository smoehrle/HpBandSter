#!/usr/bin/env python3
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

import branin


resolution = 50
hb_steps = 10
margin = 0.01
cost_pow = dict(pow_z1=1, pow_z2=1, pow_z3=1.5)
min_cost = branin.cost(0, 0, 0, **cost_pow)
max_cost = branin.cost(1, 1, 1, **cost_pow)

z = np.linspace(0, 1 + margin, resolution, endpoint=True)
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
    def traj_func(z, b):
        zz = np.ones(3)
        zz[xi] = z[0]
        zz[yi] = z[1]
        return np.abs(branin.cost(*zz, **cost_pow) - b) - np.linalg.norm(zz, ord=2)
    trajectory = np.empty((len(budgets), 2))
    x0 = np.ones(2)
    for i, b in enumerate(budgets[::-1]):
        z = minimize(traj_func, x0, b, bounds=[(0, 1 + margin), (0, 1 + margin)])['x']
        trajectory[i] = z
        x0 = z
    return trajectory


extent = (0, 1 + margin, 0, 1 + margin)
for i, (xl, yl, CC) in enumerate(plot_data):
    plt.title('$cost \\propto z_{0}^{{{1}}} \\cdot z_{2}^{{{3}}}$'
              .format(xl, cost_pow['pow_z' + str(xl)],
                      yl, cost_pow['pow_z' + str(yl)]))
    plt.xlabel('$z_{}$'.format(xl))
    plt.ylabel('$z_{}$'.format(yl))
    im = plt.imshow(CC, interpolation='bilinear', origin='lower', extent=extent)
    co = plt.contour(CC, levels=budgets, colors='k', origin='lower', extent=extent)
    plt.clabel(co, budgets[0::2],  # label every second level
               inline=1, fmt='%1.3f')
    trajectory = calc_trajectory(xl - 1, yl - 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-')
    plt.colorbar(im)

    plt.show()

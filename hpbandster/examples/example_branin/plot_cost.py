#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import branin


resolution = 50
hb_steps = 5
margin = 0.01
cost_pow = dict(pow_z1=3, pow_z2=2, pow_z3=1.5)
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

    plt.colorbar(im)

    plt.show()

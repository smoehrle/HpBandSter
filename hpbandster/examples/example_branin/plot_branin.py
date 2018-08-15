#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from problem.branin import Branin

num = 50

# Draw branin
x1 = np.linspace(-5, 10, num)
x2 = np.linspace(0, 15, num)
X1, X2 = np.meshgrid(x1, x2, sparse=True)

branin = Branin()
MFB = branin.calc_mf(X1, X2, 0, 0, 0)
NB = branin.calc_noisy(X1, X2, 0, 0, 0)

for i, Y in enumerate([MFB, NB]):
    plt.subplot(2, 2, i + 2)
    im = plt.imshow(Y, interpolation='bilinear')
    plt.colorbar(im)


plt.show()

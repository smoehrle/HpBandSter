#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from example_branin import problem

num = 50

# Draw branin
x1 = np.linspace(-5, 10, num)
x2 = np.linspace(0, 15, num)
X1, X2 = np.meshgrid(x1, x2, sparse=True)
B = branin.branin(X1, X2)
MFB = branin.mf_branin(X1, X2, 0, 0, 0)
NB = branin.noisy_branin(X1, X2, 0, 0, 0)

for i, Y in enumerate([B, MFB, NB]):
    plt.subplot(2, 2, i + 2)
    im = plt.imshow(B, interpolation='bilinear')
    plt.colorbar(im)


plt.show()

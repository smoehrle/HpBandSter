import time
import numpy as np
import random

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.INFO)


class MyWorker(Worker):

    def __init__(self, true_y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_y = true_y

    def compute(self, config, budget, *args, **kwargs):
        """
            Simple example for a compute function

            The loss is just a the config + some noise (that decreases with the budget)
            There is a 10 percent failure probability for any run, just to demonstrate
            the robustness of Hyperband agains these kinds of failures.

            For dramatization, the function sleeps for one second, which emphasizes
            the speed ups achievable with parallel workers.
        """
        time.sleep(budget/100)

        x1, x2 = config['x1'], config['x2']
        y = self.calc_noisy_branin(x1, x2)

        return({
            'loss': np.abs(self.true_y - y),   # this is the a mandatory field to run hyperband
            'info': 'x1: {}, x2: {}, y: {}, y_t: {}'.format(x1, x2, y, self.true_y)     # can be used for any user-defined information - also mandatory
        })

    def calc_noisy_branin(self, x1, x2, noise_variance=0.05):
        return self.calc_branin(x1, x2) + np.random.normal(0, noise_variance)

    @staticmethod
    def calc_branin(x1, x2, z1=1, z2=1, z3=1):
        a = 1
        b = 5.1 / (4 * np.pi**2) - 0.01 * (1 - z1)
        c = 5 / np.pi - 0.1 * (1 - z2)
        r = 6
        s = 10
        t = 1 / (8 * np.pi) + 0.05 * (1 - z3)

        return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1-t) * np.cos(x1) + s
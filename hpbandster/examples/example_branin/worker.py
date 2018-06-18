import numpy as np
import logging
from typing import Callable

from hpbandster.core.worker import Worker

import branin
import util
logging.basicConfig(level=logging.INFO)


class BraninWorker(Worker):

    def __init__(self, true_y: float, cost: Callable[[float, float, float], float],
                 min_budget: float, max_budget: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_y = true_y
        self.cost = cost if cost else self._cost
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.z = None

    def compute(self, config, budget, *args, **kwargs):
        """
            Simple example for a compute function

            The loss is just a the config + some noise (that decreases with the budget)
            There is a 10 percent failure probability for any run, just to demonstrate
            the robustness of Hyperband agains these kinds of failures.

            For dramatization, the function sleeps for one second, which emphasizes
            the speed ups achievable with parallel workers.
        """
        strategy = 'fid_propto_cost'
        norm_budget = util.normalize_budget(budget, self.min_budget, self.max_budget)
        if self.z is None:
            z0 = np.ones(3) * 0.5
        else:
            z0 = self.z
        self.z = util.fidelity_propto_cost(norm_budget, z0, branin.cost_objective)
        cost = self.cost(*self.z)

        x1, x2 = config['x1'], config['x2']
        y = self.calc_noisy_branin(x1, x2, *self.z)

        return({
            'loss': y,  # this is the a mandatory field to run hyperband
            'info': {   # can be used for any user-defined information - also mandatory
                'cost': cost,
                'fidelity': np.array2string(self.z),
                'fidelity_strategy': strategy
            }
        })

    @staticmethod
    def calc_noisy_branin(x1: float, x2: float, z1: float, z2: float, z3: float,
                          noise_variance: float = 0.05):
        return branin.noisy_branin(x1, x2, z1, z2, z3, noise_variance)

    @staticmethod
    def _cost(z1: float, z2: float, z3: float) -> float:
        return branin.cost(z1, z2, z3)

    @staticmethod
    def calc_branin(x1: float, x2: float, z1: float = 1, z2: float = 1, z3: float = 1) -> float:
        return branin.mf_branin(x1, x2, z1, z2, z3)

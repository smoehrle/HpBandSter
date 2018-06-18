import numpy as np
import ConfigSpace as CS
from typing import Callable

from hpbandster.core.worker import Worker

import branin
import fidelity_strat
import util


class BraninWorker(Worker):
    def __init__(
            self, true_y: float,
            cost: Callable[[float, float, float], float],
            strategie: fidelity_strat.FidelityStrat,
            min_budget: float, max_budget: float, *args, **kwargs):
        """
        Initialize the BraninWorker with a groundtruth and a fidelity strategie

        Parameters
        ----------
        true_y :
            Groundtruth
        cost :
            Plugable cost function
        strategie :
            Responsible for the step 'budget to fidelity'
        min_budget :
            The lowest possible budget
        max_budget :
            The highest possible budget
        """
        super().__init__(*args, **kwargs)
        self.true_y = true_y
        self.cost = cost
        self.strategie = strategie
        self.min_budget = min_budget
        self.max_budget = max_budget

    def compute(self, config: CS.ConfigurationSpace, budget: float, *args, **kwargs) -> dict:
        """
        Simple compute function which evaluates a noisy branin function for
        given parameters and budget

        Parameters
        ----------
        config :
            Contains the parameters which should be evaluated in this run
        budget :
            Available budget, which can be used for this run

        Returns
        -------
        dict with the 'loss' and another 'info'-dict.
        """
        norm_budget = util.normalize_budget(budget, self.min_budget, self.max_budget)
        z = self.strategie.calc_fidelities(norm_budget)
        cost = self.cost(*z)

        x1, x2 = config['x1'], config['x2']
        y = branin.noisy_branin(x1, x2, *z)

        return({
            'loss': np.abs(y-self.true_y),  # this is the a mandatory field to run hyperband
            'info': {   # can be used for any user-defined information - also mandatory
                'cost': cost,
                'fidelity': np.array2string(z),
                'fidelity_strategy': self.strategie.name
            }
        })

import time
import numpy as np
import ConfigSpace as CS
from typing import Callable

from hpbandster.core.worker import Worker

import branin
import fidelity_strat
import problem
import util


class SimM2FWorker(Worker):
    """
    Worker which simulates a run with a given problem instance and fidelity strategy
    """
    def __init__(
            self,
            problem: problem.Problem,
            strategy: fidelity_strat.FidelityStrat,
            max_budget: float,
            *args, **kwargs):
        """
        Initialize the SimM2FWorker with a problem instance and a fidelity strategy

        Parameters
        ----------
        problem :
            A problem instance (e.g. branin)
        strategy :
            A strategy which splits the budget
        cost :
            Plugable cost function for time simulation
        max_budget :
            The highest possible budget
        args*, kqargs*:
            Worker parameter
        """

        super().__init__(*args, **kwargs)
        self.problem = problem
        self.strategie = strategy
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
        time.sleep(0.01)

        norm_budget = util.normalize_budget(budget, self.max_budget)
        z = self.strategie.calc_fidelities(norm_budget)

        # Temporary (discuss with David)
        cost = self.problem.cost(*z, config=config)
        if cost is None:
            loss, cost = self.problem.calc_loss(config, z, kwargs['config_id'])
        else:
            loss = self.problem.calc_loss(config, z)

        return({
            'loss': loss,  # this is the a mandatory field to run hyperband
            'info': {   # can be used for any user-defined information - also mandatory
                'cost': cost,
                'fidelity': np.array2string(z),
                'fidelity_strategy': repr(self.strategie),
                'strategy_info': self.strategie.info,
                'problem': repr(self.problem),
            }
        })

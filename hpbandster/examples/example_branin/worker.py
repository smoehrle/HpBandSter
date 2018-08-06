import time
from typing import Tuple
import numpy as np
import ConfigSpace as CS

from hpbandster.core.worker import Worker

import util
from models import Run


class SimM2FWorker(Worker):
    """
    Worker which simulates a run with a given problem instance and fidelity strategy
    """
    def __init__(
            self,
            run: Run,
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
        self.run_config = run
        self.max_budget = max_budget

    def compute(self, config: CS.ConfigurationSpace, budget: float,
                config_id: Tuple[float, float, float], *args, **kwargs) -> dict:
        """
        Simple compute function which evaluates problem and strategy functions for
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
        self.run_config.config_id = config_id
        self.run_config.config = config
        time.sleep(0.01)

        norm_budget = util.normalize_budget(budget, self.max_budget)
        z = self.run_config.strategy.calc_fidelities(norm_budget)
        fid_config = self.run_config.problem.fidelity_config(fidelity_vector=z)
        cost = self.run_config.problem.cost(config, fidelity_config=fid_config)
        loss, info = self.run_config.problem.loss(config, fid_config)

        return({
            'loss': loss,  # this is the a mandatory field to run hyperband
            'info': {   # can be used for any user-defined information - also mandatory
                'cost': cost,
                'fidelity_vector': np.array2string(z),
                'fidelity_config': repr(fid_config),
                'fidelity_strategy': repr(self.run_config.strategy),
                'strategy_info': self.run_config.strategy.info,
                'problem': repr(self.run_config.problem),
                **info
            }
        })

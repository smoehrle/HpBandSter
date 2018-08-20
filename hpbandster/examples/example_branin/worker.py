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
            run_config: Run,
            run_id2: int,
            max_budget: float,
            *args, **kwargs):
        """
        Initialize the SimM2FWorker with a problem instance and a fidelity strategy

        Parameters
        ----------
        run :
            A run combining a problem with a strategy
        run_id2 :
            The actual run_id. run_id is used by the worker which is actual the job_id
            this run_id indicates the repetition of the given run_config
        max_budget :
            The highest possible budget
        args*, kqargs*:
            Worker parameter
        """

        super().__init__(*args, **kwargs)
        self.run_config = run_config
        run_config.problem.run_id = run_id2
        self.max_budget = max_budget

    def compute(self, config: CS.ConfigurationSpace, budget: float,
                config_id: Tuple[int, int, int], *args, **kwargs) -> dict:
        """
        Simple compute function which evaluates problem and strategy functions for
        given parameters and budget

        Parameters
        ----------
        config :
            Contains the parameters which should be evaluated in this run
        budget :
            Available budget, which can be used for this run
        config_id :
            HpBandSter config_id

        Returns
        -------
        dict with the 'loss' and another 'info'-dict.
        """
        time.sleep(0.01)

        norm_budget = util.normalize_budget(budget, self.max_budget)
        z, strat_info = self.run_config.strategy.calc_fidelities(norm_budget, config, config_id)
        fid_config = self.run_config.problem.fidelity_config(config, config_id, fidelity_vector=z)
        cost = self.run_config.problem.cost(config, config_id, fidelity_config=fid_config)
        loss, prob_info = self.run_config.problem.loss(config, config_id, fid_config)

        return({
            'loss': loss,  # this is the a mandatory field to run hyperband
            'info': {   # can be used for any user-defined information - also mandatory
                'cost': cost,
                'fidelity_vector': np.array2string(z),
                'fidelity_config': repr(fid_config),
                'fidelity_strategy': repr(self.run_config.strategy),
                'problem': repr(self.run_config.problem),
                **strat_info,
                **prob_info
            }
        })

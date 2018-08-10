from typing import Dict, Tuple

import ConfigSpace as CS
import numpy as np

from strategy import Strategy


class FullFidelity(Strategy):
    """
    This is a trivial strategie which always returns 1 as fidelity
    """
    def __init__(self):
        """
        Parameters
        ----------
        num_fidelities :
            Number of fidelities
        """
        super().__init__('full_fidelity')

    def calc_fidelities(
            self, norm_budget: float,
            config: CS.Configuration, config_id: Tuple[int, int, int] 
            )-> Tuple[np.ndarray, Dict[str, str]]:
        num_fidelities = len(self.run.problem.fidelity_space(config, config_id)
                             .get_hyperparameter_names())
        return np.array(num_fidelities * [1.]), {}

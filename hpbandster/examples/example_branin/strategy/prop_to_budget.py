from typing import Dict, Tuple

import ConfigSpace as CS
import numpy as np

from strategy import Strategy


class PropToBudget(Strategy):
    """
    This is a trivial strategie which uses the normalised budget directly as fidelity
    """
    def __init__(self, use_fidelity: [bool]):
        """
        Parameters
        ----------
        use_fidelity :
            This spezifies which fidelity should be used
            [True, True, True] -> use all three
            [True, False, False] -> use only z1
        """
        fidelities = ['z{}'.format(i) for i, v in enumerate(use_fidelity) if v]
        super().__init__('propto_budget_{}'.format('_'.join(fidelities)))
        self.use_fidelity = np.array(use_fidelity)

    def calc_fidelities(
            self, norm_budget: float,
            config: CS.Configuration, config_id: Tuple[int, int, int]
            ) -> Tuple[np.ndarray, Dict[str, str]]:
        z = np.array(len(self.use_fidelity) * [1.])
        z[self.use_fidelity] = norm_budget
        return z, {}

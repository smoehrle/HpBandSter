from typing import Dict, Tuple

import ConfigSpace as CS
import numpy as np

from models import Run


class Strategy:
    """
    This is a parent class defining a common interface for arbitrary fidelity
    strategies.
    """
    def __init__(self, name):
        self.name = name
        self.info = dict()
        self._run = None

    def calc_fidelities(
            self, norm_budget: float,
            config: CS.Configuration, config_id: Tuple[int, int, int]
            ) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        This function is the core of the strategie. It receives a
        budget, a config and the HpBandSter config_id and should return
        an ndarray containing the fidelity parameters.

        Parameters
        ----------
        norm_budget :
            Budget between [0,1]
        config :
            config for the current evaluation
        config_id :
            Tuple of iteration, xxx, uid

        Returns
        -------
            The fidelity parameters
        """
        pass

    @property
    def run(self) -> Run:
        return self._run

    @run.setter
    def run(self, value: Run):
        self._run = value

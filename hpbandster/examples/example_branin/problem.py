from typing import Dict, Optional

import numpy as np
import ConfigSpace as CS


class Problem:
    """
    Abstract class defining a problem for the SimM2FWorker
    """
    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError('__repr__ of Problem')

    def cost(self, fidelity_config: Optional[CS.Configuration] = None,
             fidelity_values: Optional[Dict] = None,
             fidelity_vector: Optional[np.ndarray] = None) -> float:
        if fidelity_config is not None:
            if fidelity_vector is not None:
                raise ValueError('Fidelity specified both as config and '
                                 'vector, can only do one.')
            if fidelity_values is not None:
                raise ValueError('Fidelity specified both as config and '
                                 'dictionary, can only do one.')
        else:
            fidelity_config = self.fidelity_config(fidelity_vector=fidelity_vector,
                                                   fidelity_values=fidelity_values)
        return self._cost(fidelity_config)

    def _cost(self, fidelity_config: CS.Configuration) -> float:
        raise NotImplementedError()

    def fidelity_config(self,
                        fidelity_values: Optional[Dict] = None,
                        fidelity_vector: Optional[np.ndarray] = None):
        fid_space = self.build_fidelity_space()
        fid_config = CS.Configuration(fid_space,
                                      vector=fidelity_vector,
                                      values=fidelity_values)
        return fid_config

    def calc_loss(self, config: CS.Configuration, fidelities: CS.Configuration) -> float:
        """
        Calculate the loss for given configuration and fidelities

        Parameters
        ----------
        config :
            ConfigurationSpace for this calculation
        fidelities :
            Three fidelities between [0,1]

        Returns
        -------
            The loss
        """
        raise NotImplementedError()

    @staticmethod
    def build_fidelity_space() -> CS.ConfigurationSpace:
        """
        Returns
        -------
            The fidelity space used by this problem
        """
        raise NotImplementedError()

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        """
        Returns
        -------
            The configspace used by this problem
        """
        raise NotImplementedError()

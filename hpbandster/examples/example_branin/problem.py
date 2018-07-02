import ConfigSpace as CS
import numpy as np


class Problem:
    """
    Abstract class defining a problem for the SimM2FWorker
    """
    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError('__repr__ of Problem')

    def calc_loss(self, config: CS.ConfigurationSpace, fidelities: np.ndarray) -> float:
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
    def build_config_space() -> CS.ConfigurationSpace:
        """
        Returns
        -------
            The configspace used by this problem
        """
        raise NotImplementedError()

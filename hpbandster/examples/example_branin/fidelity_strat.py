import collections
from typing import Callable, Optional, Iterable
import logging

import numpy as np
import scipy.optimize

from problem import Problem


class FidelityStrat:
    """
    This is like an abstract class for arbitrary fidelity strategies.
    The only things a valid strategie class needs is a name and the
    calc_fidelities function below
    """
    def __init__(self, name):
        self.name = name

    def calc_fidelities(self, norm_budget: float) -> np.ndarray:
        """
        This function is the core of the strategie. It receives a given
        budget and should return an ndarray containing the fidelity
        parameters.

        Parameters
        ----------
        norm_budget :
            Budget between [0,1]

        Returns
        -------
            The fidelity parameters
        """
        pass


class FullFidelity(FidelityStrat):
    """
    This is a trivial strategie which always returns 1 as fidelity
    """
    def __init__(self, num_fidelities: int):
        """
        Parameters
        ----------
        num_fidelities :
            Number of fidelities
        """
        super().__init__('full_fidelity')
        self.num_fidelities = num_fidelities

    def calc_fidelities(self, norm_budget: float) -> np.ndarray:
        return np.array(self.num_fidelities * [1.])


class FidelityPropToBudget(FidelityStrat):
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

    def calc_fidelities(self, norm_budget: float) -> np.ndarray:
        z = np.array(len(self.use_fidelity) * [1.])
        z[self.use_fidelity] = norm_budget
        return z


class FidelityPropToCost(FidelityStrat):
    """
    This is a more sophisticated strategie which uses a cost-to-fidelity model.
    """
    def __init__(
            self,
            problem: Problem,
            use_fidelity: Iterable[bool],
            alpha: float = 1.):
        self.use_fidelity = np.array(use_fidelity, dtype=np.bool)
        fidelities = ['z{}'.format(i) for i, v in enumerate(self.use_fidelity) if v]
        super().__init__('fid_propto_cost_{}'.format('_'.join(fidelities)))

        self.cost = problem.cost
        self.max_cost = problem.cost(*np.ones(self._num_fidelities))
        self.alpha = alpha
        self.logger = logging.getLogger()

    @property
    def _num_variable_fidelities(self) -> int:
        return self.use_fidelity.sum()

    @property
    def _num_fidelities(self) -> int:
        return len(self.use_fidelity)

    def calc_fidelities(self, norm_budget: float) -> np.ndarray:
        def cost_objective(z: np.ndarray, b: float):
            Z = np.ones_like(self.use_fidelity, dtype=np.float)
            Z[self.use_fidelity] = z
            return (self.cost(*Z) / self.max_cost - b)**2

        def fidelity_objective(z: np.ndarray):
            Z = np.ones_like(self.use_fidelity, dtype=np.float)
            Z[self.use_fidelity] = z
            return 1 - np.linalg.norm(Z, ord=2)

        constraint = dict(type='eq', fun=cost_objective, args=(norm_budget,))
        options = dict()
        extend = self._num_variable_fidelities * [(0, 1)]
        init_z = self._num_variable_fidelities * [norm_budget]
        result = scipy.optimize.minimize(
            fidelity_objective, init_z,
            method='SLSQP', bounds=extend, constraints=constraint, options=options)

        z = np.ones_like(self.use_fidelity, dtype=np.float)
        if result['success']:
            z[self.use_fidelity] = result['x']
        else:
            self.logger.warning("FAILED NUMERICAL FIDELITY SEARCH".format())
            self.logger.warning("{}, norm budget {:.2f} ".format(self.name, norm_budget))
            self.logger.warning(result)
            z[self.use_fidelity] = init_z
        return z

    @staticmethod
    def _log3(x):
        return np.log(x) / np.log(3)

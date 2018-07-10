import logging
from typing import Callable

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
        # XXX: Do we want the |z| to be propto budget, or each single z dimension
        z = np.array(len(self.use_fidelity) * [1.])
        z[self.use_fidelity] = norm_budget
        return z


class FidelityPropToCost(FidelityStrat):
    """
    This is a more sophisticated strategie which uses a cost-to-fidelity model.
    """
    def __init__(
            self, num_fidelities: float,
            problem: Problem,
            fallback: bool):
        super().__init__('fid_propto_cost')
        self.num_fidelities = num_fidelities
        self.init_z = np.ones(num_fidelities) * 0.5
        self.cost = problem.cost
        self.max_cost = problem.cost(*np.ones(num_fidelities))
        self.fallback = fallback
        self.logger = logging.getLogger()

    def calc_fidelities(self, norm_budget: float) -> np.ndarray:
        options = dict(maxiter=1000)
        extend = self.num_fidelities * [(0, 1)]

        def cost_objective(z: np.ndarray, b: float):
            return (self.cost(*z) / self.max_cost - b)**2

        def fidelity_objective(z: np.ndarray):
            return 1 - np.linalg.norm(z, ord=2)

        constraint = dict(type='eq', fun=cost_objective, args=(norm_budget,))
        result = scipy.optimize.minimize(
            fidelity_objective, [norm_budget] * self.num_fidelities,
            method='SLSQP', bounds=extend, constraints=constraint, options=options)

        if result['success'] or not self.fallback:
            z = result['x']
            self.init_z = z
        else:
            self.logger.info("FAILED NUMERICAL FIDELITY SEARCH")
            self.logger.info(result)
            z = np.array(self.num_fidelities * [norm_budget])
            self.init_z = z
        return z

    @staticmethod
    def _log3(x):
        return np.log(x) / np.log(3)

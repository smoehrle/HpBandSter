from typing import Iterable, Optional, Tuple, Dict
import logging

import numpy as np
import scipy.optimize
import ConfigSpace as CS

from models import Run
from problem import Problem


class FidelityStrat:
    """
    This is like an abstract class for arbitrary fidelity strategies.
    The only things a valid strategie class needs is a name and the
    calc_fidelities function below
    """
    def __init__(self, name):
        self.name = name
        self.info = dict()
        self._run = None

    def calc_fidelities(self, norm_budget: float,
                        config: CS.Configuration, config_id: Tuple[int, int, int])\
                        -> Tuple[np.ndarray, Dict[str, str]]:
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

    @property
    def run(self) -> Run:
        return self._run

    @run.setter
    def run(self, value: Run):
        self._run = value


class FullFidelity(FidelityStrat):
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

    def calc_fidelities(self, norm_budget: float,
                        config: CS.Configuration, config_id: Tuple[int, int, int])\
                        -> Tuple[np.ndarray, Dict[str, str]]:
        num_fidelities = len(self.run.problem.fidelity_space(config, config_id)
                             .get_hyperparameter_names())
        return np.array(num_fidelities * [1.]), {}


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

    def calc_fidelities(self, norm_budget: float,
                        config: CS.Configuration, config_id: Tuple[int, int, int])\
                        -> Tuple[np.ndarray, Dict[str, str]]:
        z = np.array(len(self.use_fidelity) * [1.])
        z[self.use_fidelity] = norm_budget
        return z, {}


class FidelityPropToCost(FidelityStrat):
    """
    This is a more sophisticated strategie which uses a cost-to-fidelity model.
    """
    def __init__(
            self,
            use_fidelity: Iterable[bool],
            alpha: float = 1.):
        self.use_fidelity = np.array(use_fidelity, dtype=np.bool)
        fidelities = ['z{}'.format(i) for i, v in enumerate(self.use_fidelity) if v]
        super().__init__('propto_cost_{}'.format('_'.join(fidelities)))

        self.alpha = alpha
        self.logger = logging.getLogger()

    @property
    def _num_variable_fidelities(self) -> int:
        return self.use_fidelity.sum()

    @property
    def _num_fidelities(self) -> int:
        return len(self.use_fidelity)

    def max_cost(self, config, config_id):
        return self.run.problem.cost(config, config_id,
                                     fidelity_vector=np.ones(self._num_fidelities))

    def calc_fidelities(self, norm_budget: float,
                        config: CS.Configuration, config_id: Tuple[int, int, int])\
                        -> Tuple[np.ndarray, Dict[str, str]]:
        def cost_objective(z: np.ndarray, b: float):
            Z = np.ones_like(self.use_fidelity, dtype=np.float)
            Z[self.use_fidelity] = z
            cost = self.run.problem.cost(config, config_id, fidelity_vector=Z)
            return (b - cost / self.max_cost(config, config_id))**2

        def fidelity_objective(z: np.ndarray):
            Z = np.ones_like(self.use_fidelity, dtype=np.float)
            Z[self.use_fidelity] = z
            return 1 - np.linalg.norm(Z, ord=2)

        constraint = dict(type='eq', fun=cost_objective, args=(norm_budget,))
        options = dict(maxiter=1000)
        extend = self._num_variable_fidelities * [(0, 1)]
        init_z = self._num_variable_fidelities * [norm_budget]
        if self._num_variable_fidelities == 1:
            result = scipy.optimize.minimize(
                cost_objective, init_z, norm_budget, bounds=extend, options=options)
        else:
            result = scipy.optimize.minimize(
                fidelity_objective, init_z,
                method='SLSQP', bounds=extend,
                constraints=constraint, options=options)

        z = np.ones_like(self.use_fidelity, dtype=np.float)
        if result['success']:
            z[self.use_fidelity] = result['x']
        else:
            self.logger.warning("FAILED NUMERICAL FIDELITY SEARCH".format())
            self.logger.warning("{}, norm budget {:.2f} ".format(self.name, norm_budget))
            self.logger.warning(result)
            z[self.use_fidelity] = init_z
        info = dict(minimize_success=bool(result['success']))
        return z, info

    @staticmethod
    def _log3(x):
        return np.log(x) / np.log(3)

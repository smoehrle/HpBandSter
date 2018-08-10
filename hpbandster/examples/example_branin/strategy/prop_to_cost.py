import logging
from typing import Dict, Iterable, Tuple

import ConfigSpace as CS
import numpy as np
import scipy

from strategy import Strategy


class PropToCost(Strategy):
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

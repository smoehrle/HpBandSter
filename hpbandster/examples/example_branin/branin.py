#! /usr/bin/env python3

from problem import Problem
from typing import Callable, Dict
import ConfigSpace as CS
import numpy as np

default_param = dict(
    a=1,
    b=5.1 / (4 * np.pi**2),
    c=5 / np.pi,
    r=6,
    s=10,
    t=1 / (8 * np.pi),
    bz=-0.01,
    cz=-0.1,
    tz=0.05)


class Branin(Problem):
    """
    Implementation of the branin function as a toy black box.

    Read more about branin: https://www.sfu.ca/~ssurjano/branin.html
    Usually choose -5 <= x1 <= 10 and 0 <= x2 <= 15.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
            Change the default behavior of the branin function.
            Possible parameters are: a,b,c,r,s,t,bz,cz,tz,pow_z1,pow_z2,pow_z3
        """
        self.params = {**default_param, **kwargs}
        self.pow_z1 = kwargs['pow_z1'] if 'pow_z1' in kwargs else 3.
        self.pow_z2 = kwargs['pow_z2'] if 'pow_z2' in kwargs else 2.
        self.pow_z3 = kwargs['pow_z3'] if 'pow_z3' in kwargs else 1.5

    def __repr__(self):
        return "Branin({})".format(repr(self.params))

    def calc_loss(self, config: CS.ConfigurationSpace, fidelities: np.ndarray):
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
        y = self.calc_noisy(config['x1'], config['x2'], *fidelities, param=self.params)
        return np.abs(self.min - y)

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        """
        Returns
        -------
            The configspace used by branin
        """
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x1', lower=-5, upper=10))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x2', lower=0, upper=15))
        return config_space

    def cost(self, z1, z2, z3) -> float:
        """
        Cost function which calculates the cost for given fidelity parameters. This cost function is
        based on the BOCA paper

        Parameters
        ----------
        z1, z2, z3 :
            The fidelities between [0,1]

        Returns
        -------
        The cost
        """
        return 0.05 + ((1+z1)**self.pow_z1 * (1+z2)**self.pow_z2 * (1+z3)**self.pow_z3)

    @staticmethod
    def calc_noisy(
            x1: float, x2: float,
            z1: float=1, z2: float=1, z3: float=1,
            noise_std: float=0.05,
            param: Dict[str, float]=default_param) -> float:
        """
        Calculate multi fidelity branin function for given parameters and add some noise

        Parameters
        ----------
        x1 :
            First coordinate. Range: [-5:10]
        x2 :
            Second coordinate. Range: [0:15]
        z1, z2, z3 :
            Fidelity parameter. Range: [0:1]
        noise_std :
            standard deviation of the 0-mean gaussian noise
        param :
            Branin tuning parameter. Default: parameters from BOCA

        Returns
        -------
            The result of the calculation
        """
        return Branin.calc_mf(x1, x2, z1, z2, z3, param) + np.random.normal(0, noise_std)

    @staticmethod
    def calc_mf(
            x1: float, x2: float,
            z1: float=1, z2: float=1, z3: float=1,
            param: Dict[str, float]=default_param) -> float:
        """
        Calculate multi fidelity branin function for given parameters

        Parameters
        ----------
        x1 :
            First coordinate. Range: [-5:10]
        x2 :
            Second coordinate. Range: [0:15]
        z1, z2, z3 :
            Fidelity parameter. Range: [0:1]
        param :
            Branin tuning parameter. Default: parameters from BOCA

        Returns
        -------
            The result of the calculation
        """
        assert z1 >= 0 and z1 <= 1 and z2 >= 0 and z2 <= 1 and z3 >= 0 and z3 <= 1,\
            "Assure fidelity 0 <= z <= 1."
        merged_param = {**default_param, **param}
        merged_param['b'] += merged_param['bz'] * (1 - z1)
        merged_param['c'] += merged_param['cz'] * (1 - z2)
        merged_param['t'] += merged_param['tz'] * (1 - z3)
        del merged_param['bz']
        del merged_param['cz']
        del merged_param['tz']
        return Branin.calc(x1, x2, **merged_param)

    @staticmethod
    def calc(
            x1: float, x2: float,
            a: float = default_param['a'],
            b: float = default_param['b'],
            c: float = default_param['c'],
            r: float = default_param['r'],
            s: float = default_param['s'],
            t: float = default_param['t']) -> float:
        """
        Calculate branin function for given parameters.
        This branin function is based on the function defined in the BOCA paper.

        Parameters
        ----------
        x1 :
            First coordinate. Range: [-5:10]
        x2 :
            Second coordinate. Range: [0:15]
        a, b, c, r, s, t :
            Branin tuning parameter. Default: parameters from BOCA

        Returns
        -------
            The result of the calculation
        """
        return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

    @property
    def min(self) -> float:
        """
        Returns
        -------
        The minimum which the BOCA branin function can reach. This minimum is reachable in three points
        (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
        """
        return 0.397887

#! /usr/bin/env python3

from problem import Problem
import ConfigSpace as CS
import numpy as np


def _boca_deviation(x: np.ndarray, z: np.ndarray,
                    bz: float, cz: float, tz: float, **kwargs) -> np.ndarray:
    """
    Constant error terms (depend on z but not x) according to boca paper.
    """

    zz = 1 - z
    return np.array([zz[0] * bz, zz[1] * cz, zz[2] * tz])


def _linear_deviation(x: np.ndarray, z: np.ndarray,
                      bz: float, cz: float, tz: float, **kwargs) -> np.ndarray:
    zz = 1 - z
    xx = np.linalg.norm(x) / np.linalg.norm([10, 15])
    return np.array([zz[0] * bz * xx,
                     zz[1] * cz * xx,
                     zz[2] * tz * xx])


class Branin(Problem):
    """
    Implementation of the branin function as a toy black box.

    Read more about branin and its parameter on
    https://www.sfu.ca/~ssurjano/branin.html

    Usually choose -5 <= x1 <= 10 and 0 <= x2 <= 15.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
            Change the default behavior of the branin function.
            Possible parameters are: a,b,c,r,s,t,bz,cz,tz, deviation.


        """
        default_kwargs = dict(a=1,
                              b=5.1 / (4 * np.pi**2),
                              c=5 / np.pi,
                              r=6,
                              s=10,
                              t=1 / (8 * np.pi),
                              deviation='boca',
                              deviation_kwargs=dict(
                                  bz=-0.01,
                                  cz=-0.1,
                                  tz=0.05),
                              pow_z1=3,
                              pow_z2=2,
                              pow_z3=1.5)
        kwargs = {
            **default_kwargs,
            **kwargs
        }

        self.a = kwargs['a']  # type: float
        self.b = kwargs['b']  # type: float
        self.c = kwargs['c']  # type: float
        self.r = kwargs['r']  # type: float
        self.s = kwargs['s']  # type: float
        self.t = kwargs['t']  # type: float
        if kwargs['deviation'] == "boca":
            self.deviation = _boca_deviation
        elif kwargs['deviation'] == "linear":
            self.deviation = _linear_deviation
        else:
            raise Exception("Expected 'deviation' parameter to be 'boca' or 'linear'")
        self.deviation_kwargs = kwargs['deviation_kwargs']  # type: Dict[str, float]
        self.pow_z1 = kwargs['pow_z1']  # type: float
        self.pow_z2 = kwargs['pow_z2']  # type: float
        self.pow_z3 = kwargs['pow_z3']  # type: float

    def __repr__(self):
        return "Branin Problem"

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
        y = self.calc_noisy(config['x1'], config['x2'], *fidelities)
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
        Cost function which calculates the cost for given fidelity parameters.
        This cost function is based on the BOCA paper

        Parameters
        ----------
        z1, z2, z3 :
            The fidelities between [0,1]

        Returns
        -------
        The cost
        """
        return 0.05 + ((1 + z1)**self.pow_z1 * (1 + z2)**self.pow_z2 * (1 + z3)**self.pow_z3)

    def calc_noisy(
            self,
            x1: float, x2: float,
            z1: float, z2: float, z3: float,
            noise_std: float = 0.05) -> float:
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
        return self.calc_mf(x1, x2, z1, z2, z3)\
            + np.random.normal(0, noise_std)

    def calc_mf(
            self,
            x1: float, x2: float,
            z1: float, z2: float, z3: float) -> float:
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
        delta = self.deviation(np.array([x1, x2]),
                               np.array([z1, z2, z3]),
                               **self.deviation_kwargs)
        return self.calc(x1, x2, *delta)

    def calc(self,
             x1: float, x2: float,
             delta_a: float = 0.,
             delta_b: float = 0.,
             delta_c: float = 0.) -> float:
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
            Branin tuning parameter.

        Returns
        -------
            The result of the calculation
        """
        a = self.a + delta_a
        b = self.b + delta_b
        c = self.c + delta_c
        return a * (x2 - b * x1**2 + c * x1 - self.r)**2\
            + self.s * (1 - self.t) * np.cos(x1) + self.s

    @property
    def min(self) -> float:
        """
        Returns
        -------
        The minimum which the BOCA branin function can reach.
        This minimum is reachable in three points
            (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
        """
        return 0.397887

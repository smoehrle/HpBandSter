#! /usr/bin/env python3
"""
Implementation of the branin function as a toy black box.

Read more about branin: https://www.sfu.ca/~ssurjano/branin.html
Usually choose -5 <= x1 <= 10 and 0 <= x2 <= 15.
"""

from typing import Callable, Dict
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


def noisy_branin(
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
    return mf_branin(x1, x2, z1, z2, z3, param) + np.random.normal(0, noise_std)


def mf_branin(
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
    return branin(x1, x2, **merged_param)


def branin(
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


def build_cost(pow_z1: float = 3, pow_z2: float = 2, pow_z3: float = 1.5) -> Callable[[float, float, float], float]:
    """
    Generate a cost function which calculates the cost for given fidelity parameters. This cost function is
    based on the BOCA paper

    Parameters
    ----------
    pow_z1, pow_z2, pow_z3 :
        The power each fidelity is raised to. Default: powers from BOCA

    Returns
    -------
    A cost function with the given powers
    """
    def _cost(z1: float, z2: float, z3: float) -> float:
        return 0.05 + (z1**pow_z1 * z2**pow_z2 * z3**pow_z3)
    return _cost


def min() -> float:
    """
    Returns
    -------
    The minimum which the BOCA branin function can reach. This minimum is reachable in three points
    (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
    """
    return 0.397887

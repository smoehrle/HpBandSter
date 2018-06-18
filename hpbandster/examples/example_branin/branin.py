#! /usr/bin/env python3
"""
Implementation of the branin function as a toy black box.

Read more about branin: https://www.sfu.ca/~ssurjano/branin.html
Usually choose -5 <= x1 <= 10 and 0 <= x2 <= 15.
"""

from typing import Dict
import numpy as np

default_param = dict(a=1,
                     b=5.1 / (4 * np.pi**2),
                     c=5 / np.pi,
                     r=6,
                     s=10,
                     t=1 / (8 * np.pi),
                     bz=-0.01,
                     cz=-0.1,
                     tz=0.05)


def noisy_branin(x1: float, x2: float,
                 z1: float = 1, z2: float = 1, z3: float = 1,
                 noise_std: float = 0.05,
                 param: Dict[str, float] = default_param) -> float:
    return mf_branin(x1, x2, z1, z2, z3, param)


def mf_branin(x1: float, x2: float,
              z1: float = 0.5, z2: float = 1, z3: float = 1,
              param: Dict[str, float] = default_param) -> float:
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


def branin(x1: float, x2: float,
           a: float = default_param['a'],
           b: float = default_param['b'],
           c: float = default_param['c'],
           r: float = default_param['r'],
           s: float = default_param['s'],
           t: float = default_param['t']) -> float:
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def cost(z1: float, z2: float, z3: float,
         pow_z1: float = 3, pow_z2: float = 2, pow_z3: float = 1.5) -> float:
    return 0.05 + (z1**pow_z1 * z2**pow_z2 * z3**pow_z3)


def __log3(x):
    return np.log(x) / np.log(3)


def cost_objective(z: np.ndarray, b: float, alpha: float = 1., cost_kwargs: Dict = {}):
    return (np.abs(__log3(cost(*z, **cost_kwargs)) - __log3(b)) - alpha * np.linalg.norm(z, ord=2))

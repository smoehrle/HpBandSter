import numpy as np
import os
import yaml
from typing import NamedTuple

from hpbandster.optimizers import HyperBand, RandomSearch

import branin
import fidelity_strat as strat


class ExperimentConfig(NamedTuple):
    """
    Contains all configuration parameters which are not
    specific for a single run.

    Members
    -------
    num_hb_runs :
        The number of HyperBand runs for a single runs.
        A single HyperBand run has num_bracket iterations
    num_runs :
        The number of repetitions / independent runs
    runs :
        Tuple of run configurations in the format (name, config)
    working_dir :
        Location where the result pickles are saved to
    min_budget :
        Minimum budget given to each configuration (default: 9)
    max_budget :
        Maximum budget given to each configuration (default: 243)
    offset :
        Offset for the result pickle name. If you already have 10
        results set this value to 10. Then the next run will be #11
    """
    # Mandatory config values
    num_hb_runs: int
    num_runs: int
    runs: tuple

    # auto-generated:
    working_dir: str

    # Optional config values
    min_budget: int = 9
    max_budget: int = 243
    offset: int = 0

    @property
    def num_brackets(self) -> int:
        return int(round((np.log(self.max_budget) - np.log(self.min_budget)) / np.log(3)))


class RunConfig():
    """
    Somewhat an abstract class which tells you what a specific RunConfig should implement
    """
    @property
    def display_name(self) -> str:
        pass

    @property
    def constructor(self):
        pass

    @property
    def strategy(self):
        pass

    @property
    def cost(self):
        pass

    @property
    def branin_params(self):
        pass


class RandomSearchConfig(NamedTuple):
    name: str

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def constructor(self):
        return RandomSearch

    @property
    def strategy(self):
        return strat.FidelityPropToBudget([True]*3)
    
    @property
    def cost(self):
        return branin.build_cost()

    @property
    def branin_params(self):
        return {}


class HyperBandConfig(NamedTuple):
    name: str
    z1: bool = False
    z1_pow: float = 3
    z2: bool = False
    z2_pow: float = 2
    z3: bool = False
    z3_pow: float = 1.5
    bz: float = branin.default_param['bz']
    cz: float = branin.default_param['cz']
    tz: float = branin.default_param['tz']

    @property
    def display_name(self) -> str:
        return '{}_{}'.format(self.name, self.strategy.name)

    @property
    def constructor(self):
        return HyperBand

    @property
    def strategy(self):
        return strat.FidelityPropToBudget([self.z1, self.z2, self.z3])

    @property
    def cost(self):
        return branin.build_cost(self.z1_pow, self.z2_pow, self.z3_pow)

    @property
    def branin_params(self):
        return {'bz': self.bz, 'cz': self.cz, 'tz': self.tz}


def load(file_path: str) -> ExperimentConfig:
    """
    Load a config by the given filepath. Working_dir is autmatically set
    to the config location.

    Parameters
    ----------
    file_path :
        Path to a yaml config file

    Returns
    -------
    A complete ExperimentConfig
    """
    with open(file_path, 'r') as f:
        dict_ = yaml.load(f)
    dict_['working_dir'] = os.path.dirname(file_path)
    runs = []
    for run in dict_['runs']:
        if run['name'] == 'RandomSearch':
            runs.append(RandomSearchConfig(**run))
        if run['name'] == 'HyperBand':
            runs.append(HyperBandConfig(**run))
    dict_['runs'] = tuple(runs)
    return ExperimentConfig(**dict_)

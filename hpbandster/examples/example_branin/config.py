import numpy as np
import os
import yaml
from typing import NamedTuple

from hpbandster.optimizers import HyperBand, RandomSearch



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


class RandomSearchConfig(NamedTuple):
    name: str

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def constructor(self):
        return RandomSearch


class HyperBandConfig(NamedTuple):
    name: str

    @property
    def display_name(self) -> str:
        return '{} with {}'.format(self.name, "stratxy")

    @property
    def constructor(self):
        return HyperBand


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

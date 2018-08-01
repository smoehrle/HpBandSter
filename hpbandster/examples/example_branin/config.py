import os
from typing import Dict
from collections import namedtuple

import yaml
import numpy as np
from hpbandster.optimizers import HyperBand, RandomSearch

import algorithm_configuration
import branin
import openml_problem
from problem import Problem
import fidelity_strat as strat


class ExperimentConfig(namedtuple('ExperimentConfig1',
        [ 'num_hb_runs', 'num_runs', 'runs', 'working_dir', 'min_budget', 'max_budget', 'offset'])):
    __slots__ = ()
    def __new__(cls, num_hb_runs, num_runs, runs, working_dir, min_budget=9, max_budget=243, offset=0):
        return super().__new__(cls, num_hb_runs, num_runs, runs, working_dir, min_budget, max_budget, offset)
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
    # NamedTuple with type hints and default values are not supported by Python 3.5
    # # Mandatory config values
    # num_hb_runs: int
    # num_runs: int
    # runs: tuple

    # # auto-generated:
    # working_dir: str

    # # Optional config values
    # min_budget: int = 9
    # max_budget: int = 243
    # offset: int = 0

    @property
    def num_brackets(self) -> int:
        return int(round((np.log(self.max_budget) - np.log(self.min_budget)) / np.log(3)))


class RunConfig():
    """
    Abstract class specifying a run config
    """
    @property
    def display_name(self) -> str:
        raise NotImplementedError('RunConfig.display_name')

    @property
    def constructor(self):
        raise NotImplementedError('RunConfig.constructor')

    @property
    def strategy(self):
        raise NotImplementedError('RunConfig.strategy')

    @property
    def problem(self):
        raise NotImplementedError('RunConfig.problem')


class RandomSearchConfig(RunConfig):
    """
    Run config for a random search run
    """
    def __init__(self, problem, strategy, displayname):
        self._problem = problem
        self._strategy = strategy
        self._displayname = displayname

    @property
    def display_name(self) -> str:
        if self._displayname is not None:
            return self._displayname
        return 'RandomSearch'

    @property
    def constructor(self):
        return RandomSearch

    @property
    def strategy(self):
        return self._strategy

    @property
    def problem(self):
        return self._problem


class HyperBandConfig(RunConfig):
    """
    Run config for a hyperband run
    """
    def __init__(self, problem, strategy, displayname):
        self._problem = problem
        self._strategy = strategy
        self._displayname = displayname

    @property
    def display_name(self) -> str:
        if self._displayname is not None:
            return self._displayname
        return '{}_{}'.format('hyperband', self.strategy.name)

    @property
    def constructor(self):
        return HyperBand


    @property
    def strategy(self):
        return self._strategy

    @property
    def problem(self):
        return self._problem


def load(file_path: str, run_id: str) -> ExperimentConfig:
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

    problems = load_problems(dict_['problems'], run_id)
    del dict_['problems']
    strategies = load_strategies(problems, dict_['strategies'])
    del dict_['strategies']
    runs = []
    for run in dict_['runs']:
        name = run['name']
        p = problems[run['problem']]
        s = strategies[run['strategy']]
        displayname = run['displayname'] if 'displayname' in run else None
        if name == 'RandomSearch':
            runs.append(RandomSearchConfig(p, s, displayname))
        elif name == 'HyperBand':
            runs.append(HyperBandConfig(p, s, displayname))
        else:
            raise NotImplementedError('The run type "{}" is not implemented'.format(name))
    dict_['runs'] = tuple(runs)
    return ExperimentConfig(**dict_)


def load_problems(problems: dict, run_id: str) -> dict:
    if not problems:
        raise LookupError('No problem instances defined!')

    result = {}
    for p in problems:
        name = p['name'].lower()
        del p['name']
        label = p['label'].lower()
        del p['label']

        if name == 'branin':
            obj = branin.Branin(**p)
        elif name == 'openmlrf':
            obj = openml_problem.OpenMLRF(**p)
        elif name == 'openmlgb':
            obj = openml_problem.OpenMLGB(**p)
        elif name == 'ac':
            p['seed'] = run_id
            obj = algorithm_configuration.AlgorithmConfiguration(**p)
        else:
            raise NotImplementedError('The problem type "{}" is not implemented'.format(name))

        result[label] = obj
    return result


def load_strategies(problems: Dict[str, Problem], strategies: dict) -> dict:
    if not strategies:
        raise LookupError('No strategy instance defined!')

    result = {}
    for s in strategies:
        name = s['name'].lower()
        del s['name']
        label = s['label'].lower()
        del s['label']

        if name == 'full':
            obj = strat.FullFidelity(**s)
        elif name == 'proptobudget':
            obj = strat.FidelityPropToBudget(**s)
        elif name == 'proptocost':
            prob = problems[s['problem']]
            del s['problem']
            obj = strat.FidelityPropToCost(problem=prob, **s)
        else:
            raise NotImplementedError('The problem type "{}" is not implemented'.format(name))

        result[label] = obj
    return result

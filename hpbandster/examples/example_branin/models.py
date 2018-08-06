from typing import Type, List
from collections import namedtuple

import numpy as np
from hpbandster.core.master import Master


class Experiment(namedtuple('ExperimentBase',
        ['num_hb_runs', 'num_runs', 'runs', 'working_dir', 'run_id', 'min_budget', 'max_budget', 'offset'])):
    __slots__ = ()

    def __init__(self, *args, runs: List['Run'], **kwargs):
        for r in runs:
            r.experiment = self

    def __new__(cls, num_hb_runs, num_runs, runs, working_dir, run_id, min_budget=9, max_budget=243, offset=0):
        return super().__new__(cls, num_hb_runs, num_runs, runs, working_dir, run_id, min_budget, max_budget, offset)
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


class Run():
    experiment: Experiment = None

    def __init__(self,
                 label: str,
                 optimizer_class: Type[Master],
                 problem: 'Problem',
                 strategy: 'strat.FidelityStrat',
                 ):
        self.optimizer_class = optimizer_class
        self.problem = problem
        self.strategy = strategy
        self.label = label

        self.problem.run = self
        self.strategy.run = self

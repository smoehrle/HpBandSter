import os
import inspect
import types
from typing import Dict, Union, TypeVar, List
import difflib
from collections import Mapping

import yaml
import hpbandster.optimizers

from problem import Problem
import fidelity_strat as strat
from models import Run, Experiment


def load(file_path: str, run_id: str) -> Experiment:
    """
    Load a config by the given filepath. Working_dir is automatically set
    to the config location.

    Parameters
    ----------
    file_path :
        Path to a yaml config file

    Returns
    -------
    A complete ExperimentConfig
    """
    dict_ = load_yaml(file_path)
    dict_['working_dir'] = os.path.dirname(file_path)
    problems = load_problems(dict_.pop('problems'))
    strategies = load_strategies(dict_.pop('strategies'))
    runs = load_runs(dict_.pop('runs'), strategies, problems)
    return Experiment(runs=runs, run_id=run_id, **dict_)


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        dict_ = yaml.load(f)
    if 'extend' in dict_:
        base_file_path = os.path.join(os.path.dirname(file_path), dict_['extend'])
        del dict_['extend']
        base_ = load_yaml(base_file_path)
        dict_ = _merge_dicts(base_, dict_, overwrite=True)
    return dict_


def load_runs(
        runs: List[Dict[str, str]],
        strategies: Dict[str, strat.FidelityStrat],
        problems: Dict[str, Problem]) -> List[Run]:
    result = []
    for kwargs in runs:
        opt_class_name = kwargs.pop('optimizer')
        opt_cls = _load_class(opt_class_name,
                              kwargs.pop('optimizer_module', hpbandster.optimizers))
        opt_label = opt_cls.__name__
        problem_label = kwargs['problem']
        strategy_label = kwargs['strategy']
        run_label = '{}-{}-{}'.format(opt_label, problem_label, strategy_label).lower()

        ctor, kwargs = problems[problem_label]
        problem = ctor(**kwargs)

        ctor, kwargs = strategies[strategy_label]
        strategy = ctor(**kwargs)

        run = Run(
            optimizer_class=opt_cls,
            problem=problem,
            strategy=strategy,
            label=run_label)

        result.append(run)
    return result


def load_problems(problems: Dict[str, Dict[str, str]]) -> dict:
    result = {}
    for label, kwargs in problems.items():
        cls_ = _load_class(kwargs.pop('class'), kwargs.pop('module'))
        result[label] = cls_, kwargs
    return result


def load_strategies(strategies: Dict[str, Dict[str, str]]) -> dict:
    result = {}
    for label, kwargs in strategies.items():
        cls_ = _load_class(kwargs.pop('class'), kwargs.pop('module', strat))
        result[label] = cls_, kwargs
    return result


T = TypeVar('T')


def _load_class(class_name: str, module: Union[str, types.ModuleType]) -> T:
    if isinstance(module, str):
        module = __import__(module)

    class_members = inspect.getmembers(module, inspect.isclass)
    for name, cls_ in class_members:
        if class_name == name:
            return cls_

    err_msg = ('Tried loading class {} of module {}, but failed finding.'
               .format(class_name, repr(module)))
    similar_names = difflib.get_close_matches(
        class_name, [n for n, c in class_members], n=1)
    if len(similar_names) > 0:
        err_msg += ' Did you mean class "{}"?'.format(similar_names[0])
    raise ValueError(err_msg)


def _merge_dicts(base: Dict, ext: Dict, overwrite: bool) -> Dict:
    merge = base.copy()
    for k, ext_v in ext.items():
        if k in base:
            base_v = base[k]
            if isinstance(base_v, Mapping) and isinstance(ext_v, Mapping):
                ext_v = _merge_dicts(base_v, ext_v, overwrite)
            elif not overwrite:
                raise ValueError('Found key {} in base and extension dict, '
                                 'expected values to be dicts to merge, '
                                 'but found {} and {}.'
                                 .format(k, repr(base_v), repr(ext_v)))
        merge[k] = ext_v
    return merge

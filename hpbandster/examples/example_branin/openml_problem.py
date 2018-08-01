from typing import List
import warnings

import ConfigSpace as CS
import numpy as np
import openml
import sklearn.ensemble
import sklearn.model_selection

from problem import Problem


warnings.filterwarnings("ignore", ".*n_jobs=1.*")


def _int_fraction(perc: float, start: int, end: int):
    return int(_float_fraction(perc, start, end))


def _float_fraction(perc: float, start: float, end: float):
    return start + perc * float(end - start)


class OpenMLRF(Problem):
    def __init__(self, task_id: int):
        self.task = openml.tasks.get_task(task_id)
        self.data = self.task.get_X_and_y()

    def __repr__(self):
        return 'Problem RandomForrest on task {}'.format(self.task.task_id)

    def _from_fidelities(self, fidelities: List[float], config: CS.Configuration):
        n_trees = _int_fraction(fidelities[0], 20, 2000)
        max_depth = _int_fraction(fidelities[1], 2, 10)
        return n_trees, max_depth

    def cost(self, *args: float, config=None)-> float:
        n_trees, max_depth = self._from_fidelities(args, config)
        return n_trees * max_depth

    def calc_loss(self, config: CS.Configuration, fidelities: np.ndarray) -> float:
        n_trees, max_depth = self._from_fidelities(fidelities, config)
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees,
                                                        max_depth=max_depth,
                                                        criterion=config['criterion'],
                                                        max_features=config['max_features'],
                                                        min_samples_leaf=config['min_samples_leaf'],
                                                        n_jobs=1,
                                                        bootstrap=True,
                                                        oob_score=True)
        X, y = self.data
        model.fit(X, y)
        return model.oob_score_

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.CategoricalHyperparameter('criterion', choices=['gini', 'entropy']),
            CS.CategoricalHyperparameter('max_features', choices=['sqrt', 'log2', None]),
            CS.UniformIntegerHyperparameter('n_estimators', lower=20, upper=2000, log=True),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=10),
        ])
        return config_space


class OpenMLGB(Problem):
    def __init__(self, task_id: int):
        self.task = openml.tasks.get_task(task_id)
        self.data = self.task.get_X_and_y()

    def __repr__(self):
        return 'Problem Gradient Boosted Trees on task {}'.format(self.task.task_id)

    def cost(self, *args: float, **kwargs) -> float:
        n_stages, subsamples, max_depth = self._from_fidelities(args)
        return n_stages * subsamples * max_depth

    def calc_loss(self, config: CS.Configuration, fidelities: np.ndarray) -> float:
        n_stages, subsamples, max_depth = self._from_fidelities(fidelities)
        model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_stages,
                                                            max_depth=max_depth,
                                                            subsample=subsamples,
                                                            max_features=config['max_features'],
                                                            learning_rate=config['learning_rate'],
                                                            min_samples_leaf=config['min_samples_leaf'])
        X, y = self.data
        X_train, X_test, y_train, y_test =\
            sklearn.model_selection.train_test_split(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        return 1. - acc

    def _from_fidelities(self, fidelities: List[float]):
        n_stages = _int_fraction(fidelities[0], 20, 200)
        subsamples = _float_fraction(fidelities[1], 0.1, 1)
        max_depth = _int_fraction(fidelities[2], 2, 10)
        return n_stages, subsamples, max_depth

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=1., log=True),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=4),
            CS.CategoricalHyperparameter('max_features', choices=['sqrt', 'log2', None]),
        ])
        return config_space

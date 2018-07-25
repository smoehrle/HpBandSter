from typing import List

import ConfigSpace as CS
import numpy as np
import openml
import sklearn

from problem import Problem


def _int_fraction(perc: float, start: int, end: int):
    return int(_float_fraction(perc, start, end))


def _float_fraction(perc: float, start: float, end: float):
    return start + perc * float(end - start)


class OpenMLRF(Problem):
    def __init__(self, task_id: int):
        task = openml.tasks.get_task(task_id)
        self.data = task.get_X_and_y()

    def __repr__(self):
        pass

    def cost(self, *args: float) -> float:
        n_trees, n_samples_per_tree, n_classes = args
        # ??? Is this correct ???
        return n_trees * n_samples_per_tree * n_classes

    def calc_loss(self, config: CS.Configuration, fidelities: np.ndarray) -> float:
        n_trees = _int_fraction(fidelities[0], 5, 20)
        n_samples_per_tree = _int_fraction(fidelities[1], 2, 10)
        n_classes = _int_fraction(fidelities[2], 2, 10)

        model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees,
                                                        max_depth=config['max_depth'],
                                                        min_samples_leaf=config['min_samples_leaf'])
        X = self.data[0]
        y = self.data[1]
        scores = sklearn.model_selection.cross_val_score(model, X=X, y=y,
                                                         scoring='l2', cv=3)
        return np.mean(scores)

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('max_depth', lower=10, upper=100),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=4),
        ])
        return config_space


class OpenMLGB(Problem):
    def __init__(self, task_id: int):
        task = openml.tasks.get_task(task_id)
        self.data = task.get_X_and_y()

    def __repr__(self):
        pass

    def cost(self, *args: float) -> float:
        n_stages, subsamples, max_depth = self._from_fidelities(args)
        # ??? Is this correct ???
        return n_stages * subsamples * np.log(max_depth)

    def calc_loss(self, config: CS.Configuration, fidelities: np.ndarray) -> float:
        n_stages, subsamples, max_depth = self._from_fidelities(fidelities)
        model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_stages,
                                                            max_depth=max_depth,
                                                            subsample=subsamples,
                                                            learning_rate=config['learning_rate'],
                                                            min_samples_leaf=config['min_samples_leaf'])
        X = self.data[0]
        y = self.data[1]
        scores = sklearn.model_selection.cross_val_score(model, X=X, y=y,
                                                         scoring='l2', cv=3)
        return np.mean(scores)

    @staticmethod
    def _from_fidelities(fidelities: List[float]):
        n_stages = _int_fraction(fidelities[0], 5, 20)
        subsamples = _float_fraction(fidelities[1], 0.1, 1)
        max_depth = _int_fraction(fidelities[2], 2, 10)
        return n_stages, subsamples, max_depth

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=1., log=True),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=4),
        ])
        return config_space

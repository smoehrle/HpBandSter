from typing import Tuple, Dict
import warnings

import ConfigSpace as CS
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

    def _cost(self, config: CS.Configuration, config_id: Tuple[int, int, int],
              fidelities: CS.Configuration)-> float:
        return fidelities['n_estimators'] * fidelities['max_depth']

    def loss(self, config: CS.Configuration, config_id: Tuple[int, int, int],
             fidelities: CS.Configuration) -> Tuple[float, Dict[str, str]]:
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=fidelities['n_estimators'],
                                                        max_depth=fidelities['max_depth'],
                                                        criterion=config['criterion'],
                                                        max_features=config['max_features'],
                                                        min_samples_leaf=config['min_samples_leaf'],
                                                        n_jobs=1,
                                                        bootstrap=True,
                                                        oob_score=True)
        X, y = self.data
        model.fit(X, y)
        return model.oob_score_, {}

    @staticmethod
    def build_fidelity_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_estimators', lower=20, upper=2000),
            CS.UniformIntegerHyperparameter('max_depth', lower=2, upper=10),
        ])
        return config_space

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.CategoricalHyperparameter('criterion', choices=['gini', 'entropy']),
            CS.CategoricalHyperparameter('max_features', choices=['sqrt', 'log2', None]),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=10),
        ])
        return config_space


class OpenMLGB(Problem):
    def __init__(self, task_id: int):
        self.task = openml.tasks.get_task(task_id)
        self.data = self.task.get_X_and_y()

    def __repr__(self):
        return 'Problem Gradient Boosted Trees on task {}'.format(self.task.task_id)

    def _cost(self, config: CS.Configuration, config_id: Tuple[int, int, int],
              fidelities: CS.Configuration)-> float:
        return fidelities['n_estimators'] * fidelities['subsample'] * fidelities['max_depth']

    def loss(self, config: CS.Configuration, config_id: Tuple[int, int, int],
             fidelities: CS.Configuration) -> Tuple[float, Dict[str, str]]:
        model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=fidelities['n_estimators'],
                                                            max_depth=fidelities['max_depth'],
                                                            subsample=fidelities['subsample'],
                                                            max_features=config['max_features'],
                                                            learning_rate=config['learning_rate'],
                                                            min_samples_leaf=config['min_samples_leaf'])
        X, y = self.data
        X_train, X_test, y_train, y_test =\
            sklearn.model_selection.train_test_split(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        return 1. - acc, {}

    @staticmethod
    def build_fidelity_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_estimators', lower=20, upper=200),
            CS.UniformFloatHyperparameter('subsample', lower=0.1, upper=1.),
            CS.UniformIntegerHyperparameter('max_depth', lower=2, upper=10),
        ])

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=1.),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=4),
            CS.CategoricalHyperparameter('max_features', choices=['sqrt', 'log2', None]),
        ])
        return config_space

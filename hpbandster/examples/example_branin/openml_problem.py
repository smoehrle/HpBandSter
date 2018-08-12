from typing import Tuple, Dict, List
import warnings

import ConfigSpace as CS
import openml
import sklearn.ensemble
import sklearn.model_selection
import sklearn.linear_model
import numpy as np

from problem import Problem


warnings.filterwarnings("ignore", ".*n_jobs=1.*")


def _inorder_subsample(rs, a, n):
    ix = rs.choice(a, n, replace=False)
    ix.sort()
    return ix


class OpenMLClassification(Problem):
    SUB_CLASS_HP = '_n_classes'
    SUB_FEAT_HP = '_n_features'
    SUB_SAMP_HP = '_n_samples'

    def __init__(self, task_id: int):
        super().__init__()
        self.task = openml.tasks.get_task(task_id)

    def _random_state(self, config_id: Tuple[int, int, int]) -> np.random.RandomState:
        iteration, _, _ = config_id
        intseq_run_id = list(map(ord, self.run.experiment.run_id))
        return np.random.RandomState([iteration] + intseq_run_id)

    def get_X_and_y(self, config_id: Tuple[int, int, int], fidelity_config: CS.Configuration):
        X, y = self.task.get_X_and_y()

        rs = self._random_state(config_id)
        if self.SUB_CLASS_HP in fidelity_config:
            ix = np.isin(y, rs.choice(np.unique(y), fidelity_config[self.SUB_CLASS_HP], replace=False))
            X = X[ix]
            y = y[ix]
        if self.SUB_SAMP_HP in fidelity_config:
            nsamples = min(X.shape[0], fidelity_config[self.SUB_SAMP_HP])
            ix = _inorder_subsample(rs, X.shape[0], nsamples)
            X, y = X[ix], y[ix]
        if self.SUB_FEAT_HP in fidelity_config:
            ix = _inorder_subsample(rs, X.shape[1], fidelity_config[self.SUB_FEAT_HP])
            X = X[:, ix]

        return X, y

    def task_fidelities(self, samples=True, classes=True, features=True)\
        -> List[CS.hyperparameters.Hyperparameter]:
        X, y = self.task.get_X_and_y()
        hp = []
        if samples:
            hp += CS.UniformIntegerHyperparameter(self.SUB_SAMP_HP, lower=1, upper=X.shape[0]),
        if features:
            hp += CS.UniformIntegerHyperparameter(self.SUB_FEAT_HP, lower=1, upper=X.shape[1]),
        nclasses = len(np.unique(y))
        if classes and nclasses > 2:
            hp += CS.UniformIntegerHyperparameter(self.SUB_CLASS_HP, lower=2, upper=nclasses),
        return hp


class LogisticRegressionOpenML(OpenMLClassification):
    def __init__(self, task_id: int):
        super().__init__(task_id)

    def __repr__(self):
        return 'Problem: LogisticRegression on OpenML task {}'.format(self.task.task_id)

    def _cost(self, config: CS.Configuration, config_id: Tuple[int, int, int],
              fidelity_config: CS.Configuration)-> float:
        cost = 1
        if self.SUB_CLASS_HP in fidelity_config:
            cost *= fidelity_config[self.SUB_CLASS_HP]
        cost *= fidelity_config[self.SUB_SAMP_HP]
        cost *= fidelity_config[self.SUB_FEAT_HP]
        # this is not completely corret, because classes subsampling also reduces number of samples
        return cost

    def loss(self, config: CS.Configuration, config_id: Tuple[int, int, int],
             fidelities: CS.Configuration) -> Tuple[float, Dict[str, str]]:
        model = sklearn.linear_model.LogisticRegression(penalty=config['penalty'],
                                                        C=config['C'])
        X, y = self.get_X_and_y(config_id, fidelities)
        X_train, X_test, y_train, y_test =\
            sklearn.model_selection.train_test_split(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        return 1. - acc, {}

    def build_fidelity_space(self, config, config_id) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(self.task_fidelities())
        return config_space

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.CategoricalHyperparameter('penalty', choices=['l1', 'l2']),
            CS.UniformFloatHyperparameter('C', lower=0.1, upper=1.),
        ])
        return config_space


class OpenMLRF(OpenMLClassification):
    def __init__(self, task_id: int):
        super().__init__(task_id)

    def __repr__(self):
        return 'Problem RandomForest on task {}'.format(self.task.task_id)

    def _cost(self, config: CS.Configuration, config_id: Tuple[int, int, int],
              fidelities: CS.Configuration)-> float:
        return fidelities['n_estimators'] * fidelities['max_depth'] * fidelities[self.SUB_SAMP_HP]

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
        X, y = self.get_X_and_y(config_id, fidelities)
        model.fit(X, y)
        return 1 - model.oob_score_, {}

    def build_fidelity_space(self, config, config_id) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_estimators', lower=20, upper=500),
            CS.UniformIntegerHyperparameter('max_depth', lower=2, upper=10),
        ] + self.task_fidelities(classes=False, samples=True, features=False))
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


class OpenMLGB(OpenMLClassification):
    def __init__(self, task_id: int):
        super().__init__(task_id)

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
        X, y = self.get_X_and_y(config_id, fidelities)
        X_train, X_test, y_train, y_test =\
            sklearn.model_selection.train_test_split(X, y, test_size=0.3)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        return 1. - acc, {}

    @staticmethod
    def build_fidelity_space(config, config_id) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_estimators', lower=10, upper=100),
            CS.UniformFloatHyperparameter('subsample', lower=0., upper=1.),
            CS.UniformIntegerHyperparameter('max_depth', lower=2, upper=5),
        ])
        return config_space

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformFloatHyperparameter('learning_rate', lower=0.01, upper=1.),
            CS.UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=4),
            CS.CategoricalHyperparameter('max_features', choices=['sqrt', 'log2', None]),
        ])
        return config_space

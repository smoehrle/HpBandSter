import ConfigSpace as CS
import numpy as np
import openml
import sklearn

from problem import Problem


class OpenMLForrest(Problem):
    def __init__(self, task_id: int):
        task = openml.tasks.get_task(task_id)
        self.data = task.get_X_and_y()

        pass

    def __repr__(self):
        pass

    def cost(self, *args: float) -> float:
        raise NotImplementedError()

    def calc_loss(self, config: CS.ConfigurationSpace, fidelities: np.ndarray) -> float:
        model = sklearn.ensemble.RandomForestClassifier()
        scores = sklearn.model_selection.cross_val_score(model, X=self.data[0], y=self.data[1],
                                                         scoring='l2', cv=3)
        return np.mean(scores)

    @staticmethod
    def build_config_space() -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_trees', lower=5, upper=20),
            CS.UniformIntegerHyperparameter('n_samples_per_tree', lower=2, upper=10),
            CS.UniformIntegerHyperparameter('n_classes', lower=2, upper=10),
        ])
        return config_space

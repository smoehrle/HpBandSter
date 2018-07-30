import tarfile as tar
from collections import namedtuple

import ConfigSpace as CS
import numpy as np

from problem import Problem


Dataset = namedtuple('Dataset', ['filename', 'max_cutoff', 'time_scale_factor'])

datasets = {
    'SPEAR_QCP': Dataset(
        filename='data/AMAI_data/SPEAR/random-SPEAR-QCP-5s-results-1000train-1000test.txt',
        max_cutoff=50, time_scale_factor=10),
}


class AlgorithmConfiguration(Problem):
    """
    """
    def __init__(self, tarfile, dataset_name):
        if dataset_name not in datasets:
            raise Exception("Dataset {} not found.".format(dataset_name))

        self.dataset_name = dataset_name
        self.dataset = datasets[dataset_name]

        with tar.open(tarfile, 'r:gz') as archive:
            with archive.extractfile(self.dataset.filename) as f:
                self.num_configs = len(f.readline().decode().split(',')) - 1
                f.seek(0)
                self.instance_config_result_matix = np.loadtxt(f, delimiter=',', usecols=range(1, self.num_configs+1))
        self.num_instances = self.instance_config_result_matix.shape[0]

        # Fix outlier
        i = np.where(self.instance_config_result_matix > self.dataset.max_cutoff)
        self.instance_config_result_matix[i] = self.dataset.max_cutoff

        # Cutoff time where at least 0.3 instance and algorithm combinations finish with a result
        self.min_cutoff = np.percentile(self.instance_config_result_matix, 30)

    def __repr__(self):
        return "AC_{}".format(self.dataset_name)

    def cost(self, *args: float) -> float:
        return None

    def calc_loss(self, config: CS.ConfigurationSpace, fidelities: np.ndarray, config_id: tuple) -> (float, float):
        """
        Calculate the loss for given configuration and fidelities

        Parameters
        ----------
        config :
            ConfigurationSpace for this calculation
        fidelities :
            Two fidelities between [0,1]
            The first fidelity is the number of problem instances which are evaluated
            The second fidelity is the cutoff time.
        config_id :
            Config_id for the current iteration

        Returns
        -------
            (loss, cost)
            Since the cost is dependent on the configuration, it cannot be calculated independently 
        """
        if fidelities[0] < 1:
            num_instances = int(self.num_instances * fidelities[0])
            # Seed random generator with iteration id. This ensures that the same instances
            # are compared in the same iteration
            # Problem: since the seed is not run but only iteration dependend the same
            # problem instances are evaluated across different runs 
            np.random.seed(config_id[0])
            i = sorted(np.random.choice(self.num_instances, num_instances, replace=False))
        else:
            i = range(self.num_instances)

        if fidelities[1] < 1:
            cutoff_time = fidelities[1] * (self.dataset.max_cutoff - self.min_cutoff) + self.min_cutoff
        else:
            cutoff_time = self.dataset.max_cutoff

        results = self.instance_config_result_matix[i, config['x']]
        cutoff_i = np.where(results > cutoff_time)
        results[cutoff_i] = cutoff_time

        loss = results.sum()
        return loss, loss / self.dataset.time_scale_factor

    def build_config_space(self) -> CS.ConfigurationSpace:
        """
        Returns
        -------
            The configspace used by this problem

            The only hyperparameter for this problem is the index which selects the predefined algorithm configuration
        """
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('x', lower=0, upper=self.num_configs-1))
        return config_space

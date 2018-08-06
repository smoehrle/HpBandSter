import logging
import random
import tarfile as tar
from collections import namedtuple
from typing import Dict

import ConfigSpace as CS
import numpy as np

from problem import Problem


Dataset = namedtuple('Dataset', ['filename', 'max_cutoff', 'time_scale_factor'])

datasets = {
    'SPEAR_QCP': Dataset(
        filename='data/AMAI_data/SPEAR/random-SPEAR-QCP-5s-results-1000train-1000test.txt',
        max_cutoff=50, time_scale_factor=10),
    'SPEAR_SWGCP': Dataset(
        filename='data/AMAI_data/SPEAR/random-SPEAR-SWGCP-5s-results-1000train-1000test.txt',
        max_cutoff=50, time_scale_factor=10),
}


class AlgorithmConfiguration(Problem):
    """
    """
    def __init__(self, tarfile, dataset_name, seed):
        self.logger = logging.getLogger(__name__)

        if dataset_name not in datasets:
            raise Exception("Dataset {} not found.".format(dataset_name))

        self.dataset_name = dataset_name
        self.dataset = datasets[dataset_name]

        if type(seed) is int:
            self.seed = seed

        if type(seed) is str:
            self.seed = int.from_bytes(seed.encode(), byteorder='big')

        if not self.seed:
            raise Exception("Currently only 'str' and 'int' seeds are supported.")

        self.logger.debug("___AC_INIT___")
        self.logger.debug("Seed: {}".format(seed))

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

        self.logger.debug("Num_config: {}, num_instances: {}".format(self.num_configs, self.num_instances))
        self.logger.debug("Outlier: {}, min_cutoff: {}".format(len(i), self.min_cutoff))

    def __repr__(self):
        return "AC_{}".format(self.dataset_name)

    def _cost(self, *args: float, **kwargs) -> float:
        return None

    def loss(self, config: CS.Configuration, config_id: Tuple[int, int, int],
             fidelities: CS.Configuration) -> (float, Dict):
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
        self.logger.debug("___CALC_LOSS___{}".format(config_id))
        self.logger.debug("Config: {}, Fidelity: {}, {}".format(config['x'], fidelities['n_instances'], fidelities['cutoff']))
        if fidelities['n_instances'] < self.num_instances:
            # Seed random generator with iteration id. This ensures that the same instances
            # are compared in the same iteration
            # Problem: since the seed is not run but only iteration dependend the same
            # problem instances are evaluated across different runs 
            seed = self.generate_seed(config_id[0])
            print("ConfigId: {}, Seed: {}".format(config_id[0], seed))
            np.random.seed(seed)
            i = sorted(np.random.choice(self.num_instances,
                                        fidelities['n_instances'],
                                        replace=False))
        else:
            i = range(self.num_instances)
        len_i = len(i)
        max_samples = 10 if len_i > 10 else len_i
        self.logger.debug("Len: {}, samples: {}...".format(len_i, i[0:max_samples]))

        cutoff_time = fidelities['cutoff']
        self.logger.debug("Cutoff time: {}".format(cutoff_time))

        loss = self._calc_loss(self.instance_config_result_matix[i, config['x']], cutoff_time)
        test_loss = self._calc_loss(self.instance_config_result_matix[:, config['x']], self.dataset.max_cutoff)
        self.logger.debug("Loss: {}, test_loss: {}".format(loss, test_loss))
        return loss, {
            'cost': loss / self.dataset.time_scale_factor,
            'test_loss': test_loss
        }

    def _calc_loss(self, results, cutoff_time):
        cutoff_i = np.where(results > cutoff_time)
        results[cutoff_i] = cutoff_time

        return results.sum()

    def build_fidelity_space(self) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_instances', lower=0, upper=self.num_instances),
            CS.UniformFloatHyperparameter('cutoff', lower=self.min_cutoff, upper=self.dataset.max_cutoff),
        ])
        return cs

    def build_config_space(self) -> CS.ConfigurationSpace:
        """
        Returns
        -------
            The configspace used by this problem

            The only hyperparameter for this problem is the index which selects the predefined algorithm configuration
        """
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter('x', lower=0, upper=self.num_configs-1))
        return config_space

    def generate_seed(self, iteration):
        """
        Generate a seed depending on the iteration and the base seed which should be different for each run

        Parameters
        ----------
        iteration :
            The current iteration. This ensures that in each iteration the same instances are sampled

        Returns
        -------
            A number between 0 and 2**32-1
        """

        random.seed(self.seed)
        for _ in range(iteration + 1):
            val = random.getrandbits(32)
        return val

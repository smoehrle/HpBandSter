import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tabulate import tabulate

import config
import util
from models import Experiment
from problem.branin import Branin
from strategy import PropToBudget

num_configs = 1000


def main():
    cli_param = parse_cli()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if cli_param.aggregated_result_file:
        handle_aro_file(cli_param.aggregated_result_file)

    if cli_param.config:
        handle_config_file(cli_param.config)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HpBandSter rank utility functions.')
    parser.add_argument(
        '--aggregated-result-file',
        help='An aggregated result file if you want to analyse the ranks of an actual run',
        type=str,
        default=None)
    parser.add_argument(
        '--config',
        help='A configfile if you want to analyse the ranks over all budgets for the defined problems',
        type=str,
        default=None)
    return parser.parse_args()


def handle_aro_file(filename):
    pass


def handle_config_file(filename):
    config_ = config.load(filename, "plot_rank")
    budgets = generate_budgets(config_)

    # Bookkeeping
    losses = np.zeros((len(config_.runs), len(budgets), num_configs))
    n = len(budgets) - 1  # Gaußsche Summenformel -> Anzahl der Kombinationen
    rank_corr = np.zeros((len(config_.runs), (n**2+n)//2))

    # Iterate all problems,strategies
    for i, r in enumerate(config_.runs):
        cs = r.problem.build_config_space()
        configs = [cs.sample_configuration() for _ in range(num_configs)]
        # Iterate all budgets
        for j, b in enumerate(budgets):
            norm_budget = util.normalize_budget(b, budgets[-1])

            # Iterate all configurations
            for k, c in enumerate(configs):
                losses[i, j, k] = calc_loss(j, c, norm_budget, r.problem, r.strategy)

    combinations = [(j, k) for j in range(len(budgets)) for k in range(j+1, j+1+len(budgets[j+1:]))]
    combinations.sort(key=lambda x: np.abs(x[0]-x[1]))
    # combinations = combinations[0:3]
    fix, axes = plt.subplots(len(config_.runs), 1, figsize=(12, 16))

    for i, axis in zip(range(len(config_.runs)), axes):
        axis.set_title(config_.runs[i].label)

        for j, (i1, i2) in enumerate(combinations):
            rank_corr[i, j], _ = scipy.stats.spearmanr(losses[i, i1, :], losses[i, i2, :])
            axis.scatter(losses[i, i1, :], losses[i, i2, :], s=0.5, label="{} -> {}".format(budgets[i1], budgets[i2]))

    headers = ['Name']
    headers.extend(["{} -> {}".format(budgets[i1], budgets[i2]) for i1, i2 in combinations])

    data = []
    for i, r in enumerate(config_.runs):
        d = [r.label]
        d.extend(rank_corr[i, :])
        data.append(d)

    print(tabulate(data, headers=headers))
    plt.legend()
    plt.show()


def generate_budgets(config: Experiment) -> List[float]:
    result = []
    budget = float(config.max_budget)
    while budget >= config.min_budget:
        result.append(budget)
        budget /= 3

    return list(reversed(result))


def calc_loss(i, config, budget, problem, strat):
    fidelities, _ = strat.calc_fidelities(budget, config, (i, 0, 0))
    fid_config = problem.fidelity_config(config, (i, 0, 0), fidelity_vector=fidelities)
    return problem.loss(config, (i, 0, 0), fid_config)[0]


if __name__ == '__main__':
    main()

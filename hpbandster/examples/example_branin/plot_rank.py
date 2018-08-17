import argparse
import logging
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
from tabulate import tabulate

import config
import util
from result_aggregator import AggregatedResults
from models import Experiment
from problem.branin import Branin
from strategy import PropToBudget
from hpbandster.core.result import Result

num_configs = 1000
logger = logging.getLogger(__name__)


def main():
    cli_param = parse_cli()

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
    ar = AggregatedResults.load(filename)

    for config_id in ar.runs.keys():
        logger.info("Current config_id: {}".format(config_id))
        rank_df_lst = [rank_result(run) for run in ar.runs[config_id]]

        mean_df = pd.concat(rank_df_lst).groupby(level=0).mean()
        std_df = pd.concat(rank_df_lst).groupby(level=0).std()
        logger.info(mean_df)

        plt.title(config_id)
        plot_ranks(mean_df, std_df)
        plt.show()


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


def rank_result(result: Result) -> pd.DataFrame:
    configs_by_budget = defaultdict(set)
    for config_id, config_data in result.data.items():
        for budget, data in config_data.results.items():
            configs_by_budget[budget].add(config_id)
    budgets = sorted(configs_by_budget.keys())

    infos = list()

    for i in range(len(budgets)):
        for j in range(i + 1, len(budgets)):
            left_budget, right_budget = budgets[i], budgets[j]
            configs = configs_by_budget[left_budget] & configs_by_budget[right_budget]
            configs_results = [result.data[config_id].results
                               for config_id in configs]
            left_loss = [results[left_budget]['loss']
                         for results in configs_results]
            right_loss = [results[right_budget]['loss']
                          for results in configs_results]
            scorr, _ = scipy.stats.spearmanr(left_loss, right_loss)
            infos.append(dict(budget_x=left_budget,
                              budget_y=right_budget,
                              spearman_corr=scorr,
                              num_samples=len(configs)))
    return pd.DataFrame(infos)


def plot_ranks(mean_df: pd.DataFrame, std_df: pd.DataFrame) -> None:
    plt.ylim(0, 1)
    plt.xlabel('budget')
    plt.ylabel('spearman rank correlation')
    for (_, mean_row), (_, std_row) in zip(mean_df.iterrows(), std_df.iterrows()):
        x = [mean_row.budget_x, mean_row.budget_y]
        y = 2 * [mean_row.spearman_corr]
        plt.errorbar(x, y, yerr=std_row.spearman_corr, capsize=4, elinewidth=0.5)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    main()

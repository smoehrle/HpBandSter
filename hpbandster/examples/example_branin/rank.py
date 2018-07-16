import pickle
import argparse
import logging
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from hpbandster.core.result import Result


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
            scorr, _ = spearmanr(left_loss, right_loss)
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


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HpBandSter rank utility functions.')
    parser.add_argument('--result-files',
                        nargs='+',
                        help='Result object to load as pickle file.',
                        type=str)
    parser.add_argument('--show-plot', action='store_true',
                        help='Show plot of ranks.')
    parser.add_argument('--save-plot', type=str,
                        help='Save plot of ranks.')
    return parser.parse_args()


def main():
    cli_param = parse_cli()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    rank_df_lst = []
    for file_name in cli_param.result_files:
        with open(file_name, 'rb') as f:
            result = pickle.load(f)

        rank_df_lst.append(rank_result(result))
    mean_df = pd.concat(rank_df_lst).groupby(level=0).mean()
    std_df = pd.concat(rank_df_lst).groupby(level=0).std()
    logger.info(mean_df)

    if cli_param.show_plot or cli_param.save_plot is not None:
        plot_ranks(mean_df, std_df)
    if cli_param.show_plot:
        plt.show()
    if cli_param.save_plot is not None:
        plt.savefig(cli_param.save_plot)

if __name__ == '__main__':
    main()

import pickle
import typing
import argparse
import logging
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from hpbandster.core.result import Result


def rank(lst: typing.Iterable[float]) -> typing.Iterable[int]:
    return np.argsort(lst) + 1


def spearman_corr(x: typing.Iterable[int], y: typing.Iterable[int]) -> float:
    n = len(x)
    d2 = sum((x - y)**2)
    norm = n * (n**2 - 1)
    return 1 - (6 * d2 / norm)


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
            left_ranks = rank([results[left_budget]['loss']
                               for results in configs_results])
            right_ranks = rank([results[right_budget]['loss']
                                for results in configs_results])
            scorr = spearman_corr(left_ranks, right_ranks)
            infos.append(dict(budget_x=left_budget,
                              budget_y=right_budget,
                              spearman_corr=scorr,
                              num_samples=len(configs)))
    return pd.DataFrame(infos)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HpBandSter rank utility functions.')
    parser.add_argument('--result-file', help='Result object to load as pickle file.',
                        type=str)
    return parser.parse_args()


def main():
    cli_param = parse_cli()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    if cli_param.result_file is not None:
        with open(cli_param.result_file, 'rb') as f:
            result = pickle.load(f)
        rank_df = rank_result(result)
        logger.info(rank_df)


if __name__ == '__main__':
    main()

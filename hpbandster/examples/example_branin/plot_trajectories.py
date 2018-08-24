import os
import glob
import pickle
import sys
import argparse
import re
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from result_aggregator import AggregatedResults

logger = logging.getLogger(__name__)


def fill_trajectories(pandas_data_frame):
    # forward fill to make it a propper step function
    df = pandas_data_frame.fillna(method='ffill')

    # backward fill to replace the NaNs for the early times by
    # the performance of a random configuration
    df = df.fillna(method='bfill')
    return(df)


def plot_losses(incumbent_trajectories, title, regret=True, incumbent=None,
                show=True, linewidth=3, marker_size=10,
                xscale='log', xlabel='wall clock time [s]',
                yscale='log', ylabel=None,
                legend_loc='best',
                xlim=None, ylim=None,
                plot_mean=True, labels={}, markers={}, colors={}, figsize=(16, 9)):

    fig, ax = plt.subplots(1, figsize=figsize)

    if regret:
        if ylabel is None:
            ylabel = 'regret'
        # find lowest performance in the data to update incumbent

        if incumbent is None:
            incumbent = np.inf
            for tr in incumbent_trajectories.values():
                incumbent = min(tr['losses'][:, -1].min(), incumbent)
        print('incumbent value: ', incumbent)
    if ylabel is None:
        ylabel = 'loss'

    for m, tr in incumbent_trajectories.items():

        trajectory = np.copy(tr['losses'])
        if (trajectory.shape[0] == 0):
            continue
        if regret:
            trajectory -= incumbent

        sem = np.sqrt(trajectory.var(axis=0, ddof=1) / tr['losses'].shape[0])
        if plot_mean:
            mean = trajectory.mean(axis=0)

        else:
            mean = np.median(trajectory, axis=0)
            sem *= 1.253

        ax.fill_between(tr['time_stamps'], mean - 2 * sem, mean + 2 * sem,
                        color=colors.get(m, 'black'), alpha=0.3)
        ax.plot(tr['time_stamps'], mean,
                label=labels.get(m, m), color=colors.get(m, None), linewidth=linewidth,
                marker=markers.get(m, None), markersize=marker_size, markevery=(0.1, 0.1))

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(which='both', alpha=0.3, linewidth=2)

    if legend_loc is not None:
        ax.legend(loc=legend_loc, framealpha=1)

    if show:
        plt.show()

    return (fig, ax)


def load_trajectories(runs, time_col: str, value_col: str, bigger_is_better: bool):
    df = pd.DataFrame()
    for i, datum in enumerate(runs):
        times = np.array(datum[time_col])
        tmp = pd.DataFrame({str(i): datum[value_col]}, index=times)
        df = df.join(tmp, how='outer')
    df = fill_trajectories(df)

    return {
        'time_stamps': np.array(df.index),
        'losses': np.array(df.T)
    }


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HpBandSter plot trajectories utility functions.')
    parser.add_argument(
        '--file',
        help='Aggregated result file',
        type=str,
        required=True)
    parser.add_argument(
        '--time-column',
        help='Data column shown as x-axis. Usually "cummulative_cost" or "times_finished".',
        type=str,
        default=None)
    parser.add_argument(
        '--value-column',
        help='Data column shown as y-axis. Usually "losses" or "test_losses".',
        type=str,
        default=None)
    parser.add_argument(
        '--bigger-is-better',
        help='Set this flag if a bigger budget should be always better',
        action='store_true')

    return parser.parse_args()


def main():
    args = parse_cli()

    ar = AggregatedResults.load(args.file)

    all_losses = dict()
    time_col = args.time_column if args.time_column else ar.config.plot.time_column
    value_col = args.value_column if args.value_column else ar.config.plot.value_column
    bigger_is_better = args.bigger_is_better if args.bigger_is_better else ar.config.plot.bigger_is_better

    for config_id in sorted(ar.runs.keys(), reverse=True):
        logger.info("Loading trajectories for {}".format(config_id))
        all_losses[config_id] = load_trajectories(
            ar.runs[config_id],
            time_col,
            value_col,
            bigger_is_better
        )

    plot_losses(
        all_losses,
        ar.config.plot.title,
        xlabel=time_col,
        ylabel=value_col,
        show=True)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()

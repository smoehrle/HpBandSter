import os
import glob
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_result(results_object):
    """
        Returns the best configurations over time, but also returns the cummulative budget

        Parameters:
        -----------
            all_budgets: bool
                If set to true all runs (even those not with the largest budget) can be the incumbent.
                Otherwise, only full budget runs are considered

        Returns:
        --------
            dict:
                dictionary with all the config IDs, the times the runs
                finished, their respective budgets, and corresponding losses
    """
    all_runs = results_object.get_all_runs(only_largest_budget=False)
    all_runs.sort(key=lambda r: r.time_stamps['finished'])

    return_dict = {
        'config_ids': [],
        'times_finished': [],
        'budgets': [],
        'losses': [],
        'info': [],
        'test_losses': [],
        'cummulative_budget': [],
        'cummulative_cost': []
    }

    cummulative_budget = 0
    cummulative_cost = 0
    current_incumbent = float('inf')
    incumbent_budget = -float('inf')

    for r in all_runs:

        cummulative_budget += r.budget
        try:
            cummulative_cost += r.info['cost']
        except: pass

        if r.loss is None:
            continue

        if (r.budget >= incumbent_budget and r.loss < current_incumbent):
        # if ((r.budget == incumbent_budget and r.loss < current_incumbent) or\
        #      r.budget > incumbent_budget):
            current_incumbent = r.loss
            incumbent_budget = r.budget

            return_dict['config_ids'].append(r.config_id)
            return_dict['times_finished'].append(r.time_stamps['finished'])
            return_dict['budgets'].append(r.budget)
            return_dict['losses'].append(r.loss)
            return_dict['cummulative_budget'].append(cummulative_budget)
            return_dict['cummulative_cost'].append(cummulative_cost)
            try:
                return_dict['test_losses'].append(r.info['test_loss'])
            except:
                pass

    if current_incumbent != r.loss:
        r = all_runs[-1]

        return_dict['config_ids'].append(return_dict['config_ids'][-1])
        return_dict['times_finished'].append(r.time_stamps['finished'])
        return_dict['budgets'].append(return_dict['budgets'][-1])
        return_dict['losses'].append(return_dict['losses'][-1])
        return_dict['cummulative_budget'].append(cummulative_budget)
        return_dict['cummulative_cost'].append(cummulative_cost)
        try:
            return_dict['test_losses'].append(return_dict['test_losses'][-1])
        except:
            pass

    return_dict['configs'] = {}

    id2conf = results_object.get_id2config_mapping()

    for c in return_dict['config_ids']:
        return_dict['configs'][c] = id2conf[c]

    return_dict['HB_config'] = results_object.HB_config

    return (return_dict)


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


def load_trajectories(config_id, working_dir: str = '.'):
    df = pd.DataFrame()
    for fn in glob.glob(os.path.join(working_dir, 'results.{}*.pkl'.format(config_id))):
        with open(fn, 'rb') as fh:
            result = pickle.load(fh)
        datum = extract_result(result)
        times = np.array(datum['cummulative_cost'])
        print(fn, datum['test_losses'], times)
        tmp = pd.DataFrame({fn: datum['losses']}, index=times)
        df = df.join(tmp, how='outer')
    df = fill_trajectories(df)

    return {
        'time_stamps': np.array(df.index),
        'losses': np.array(df.T)
    }


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else '001'
    all_losses = {config_id: load_trajectories('{}-{}'.format(run_id, config_id))
                  for config_id in [
                      'randomsearch-',
                      'hyperband_propto_budget_z0-',
                      'hyperband_propto_budget_z1-',
                      'hyperband_propto_budget_z2-',
                      'hyperband_propto_budget_z0_z1_z2-',
                      'hyperband_fid_propto_cost-',
                      ]}

    plot_losses(all_losses, 'Branin', show=True)


if __name__ == "__main__":
    main()

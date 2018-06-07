import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_results_to_pickle(results_object):
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

        if (r.budget == incumbent_budget and r.loss < current_incumbent) or \
           r.budget > incumbent_budget:
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


def main():
    df = pd.DataFrame()
    directory, m = '.', 'results'
    for fn in glob.glob(directory + '/' + m + '*.pkl'):
        with open(fn, 'rb') as fh:
            datum = pickle.load(fh)
            print(datum)
            times = np.array(datum['cummulative_budget']) / datum['budgets'][-1]
            tmp = pd.DataFrame({fn: datum['test_losses']}, index=times)

            df = df.join(tmp, how='ouer')

    df = fill_trajectories(df)

    all_trajectories = {
        'time_stamps': np.array(df.index),
        'losses': np.array(df.T)
    }

    plot_losses(all_trajectories, show=True)


if __name__ == "__main__":
    main()

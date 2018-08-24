import itertools

from hpbandster.core.result import Result
import numpy as np
import scipy
import ConfigSpace as CS
import argparse


def extract_result(results_object, bigger_is_better):
    """
        Returns the best configurations over time, but also returns the cummulative budget

        Parameters:
        -----------
            result_object:
                HyperBand result object

            bigger_is_better: bool
                If set to true then a run with a bigger budget is always considered better than
                the current best run with a lower budget

                If set to false a run is considered better only if the budget is equal or greater
                and the loss is smaller

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
        except:
            pass

        if r.loss is None:
            continue

        if bigger_is_better:
            is_better = r.budget > incumbent_budget or (r.budget == incumbent_budget and r.loss < current_incumbent)
        else:
            is_better = r.budget >= incumbent_budget and r.loss < current_incumbent

        if is_better:
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


def start_worker(num_worker: int, worker_opts: dict, worker_class) -> None:
    if worker_opts.get('cost') is None:
        worker_opts['cost'] = None
    for i in range(num_worker):
        w = worker_class(id=i, **worker_opts)
        w.run(background=True)


def run_hb(constructor, constructor_opts: dict, num_worker: int, iterations: int = 10) -> Result:
    print('Start {} run'.format(constructor))
    HB = constructor(**constructor_opts)
    res = HB.run(iterations, min_n_workers=num_worker)
    HB.shutdown(shutdown_workers=True)
    return res


def log_results(res: Result, *, simulate_time: bool) -> dict:
    id2config = res.get_id2config_mapping()

    res.data
    # Log number of configurations
    print('A total of {} unique configurations where sampled.'.format(
        len(id2config.keys())))
    runs = sorted(res.get_all_runs(), key=lambda x: x.budget)

    # Log number of runs
    print('A total of {} runs where executed.'.format(len(runs)))

    # Log runs per budget
    for b, r in itertools.groupby(runs, key=lambda x: x.budget):
        print('Budget: {:3}, #Runs: {:3}'.format(b, sum([1 for _ in r])))

    # Log total budget
    sum_ = sum([x.budget for x in runs])
    print('Total budget: {}'.format(sum_))

    # Log configurations
    for k in id2config.items():
        print("Key: {}".format(k))

    if simulate_time:
        incumbent_trajectory = extract_result(res, False)
    else:
        incumbent_trajectory = res.get_incumbent_trajectory(all_budgets=True)

    ids = incumbent_trajectory['config_ids']
    times = incumbent_trajectory['times_finished']
    budgets = incumbent_trajectory['budgets']
    losses = incumbent_trajectory['losses']

    for i in range(len(ids)):
        print("Id: ({0[0]}, {0[1]}, {0[2]:>2}), time: {1:>5.2f}, budget: {2:>5}, loss: {3:>5.2f}"
              .format(ids[i], times[i], budgets[i], losses[i]))

    return incumbent_trajectory


def normalize_budget(budget: float, min_budget: float, max_budget: float) -> float:
    return (budget - min_budget) / (max_budget - min_budget)

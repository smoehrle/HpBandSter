import itertools

from hpbandster.core.result import Result

from worker import BraninWorker


def get_simulated_incumbent_trajectory(res: Result, all_budgets: bool=True):
    """
        Returns the best configurations over time simulated by adding up the used budget


        Parameters:
        -----------
            res: Result
                Result object returnd by a HB run
            all_budgets: bool
                If set to true all runs (even those not with the largest budget) can be the incumbent.
                Otherwise, only full budget runs are considered

        Returns:
        --------
            dict:
                dictionary with all the config IDs, the times the runs
                finished, their respective budgets, and corresponding losses
    """
    all_runs = res.get_all_runs(only_largest_budget=not all_budgets)

    if not all_budgets:
        all_runs = list(filter(lambda r: r.budget == res.HB_config['max_budget'], all_runs))

    all_runs.sort(key=lambda r: r.time_stamps['finished'])

    return_dict = {
        'config_ids': [],
        'times_finished': [],
        'budgets': [],
        'losses': [],
    }

    current_incumbent = float('inf')
    incumbent_budget = -float('inf')
    total_budget = .0

    for r in all_runs:
        if r.loss is None:
            continue

        total_budget += int(r.info)
        if ((r.budget == incumbent_budget and r.loss < current_incumbent) or
           (r.budget > incumbent_budget)):
            current_incumbent = r.loss
            incumbent_budget = r.budget

            return_dict['config_ids'].append(r.config_id)
            return_dict['times_finished'].append(total_budget)
            return_dict['budgets'].append(r.budget)
            return_dict['losses'].append(r.loss)

    if current_incumbent != r.loss:
        return_dict['config_ids'].append(return_dict['config_ids'][-1])
        return_dict['times_finished'].append(total_budget)
        return_dict['budgets'].append(return_dict['budgets'][-1])
        return_dict['losses'].append(return_dict['losses'][-1])

    return return_dict


def start_worker(num_worker: int, worker_opts: dict) -> None:
    if worker_opts.get('cost') is None:
        worker_opts['cost'] = None
    for i in range(num_worker):
        w = BraninWorker(id=i, **worker_opts)
        w.run(background=True)


def run_hb(constructor, constructor_opts: dict, num_worker: int, iterations: int=10) -> Result:
    print('Start {} run'.format(constructor))
    HB = constructor(**constructor_opts)
    res = HB.run(iterations, min_n_workers=num_worker)
    HB.shutdown(shutdown_workers=True)
    return res


def log_results(res: Result, *, simulate_time: bool) -> dict:
    id2config = res.get_id2config_mapping()

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
        incumbent_trajectory = get_simulated_incumbent_trajectory(res, all_budgets=True)
    else:
        incumbent_trajectory = res.get_incumbent_trajectory(all_budgets=True)

    ids = incumbent_trajectory['config_ids']
    times = incumbent_trajectory['times_finished']
    budgets = incumbent_trajectory['budgets']
    losses = incumbent_trajectory['losses']

    for i in range(len(ids)):
        print("Id: ({0[0]}, {0[1]}, {0[2]:>2}), time: {1:>5.2f}, budget: {2:>5}, loss: {3:>5.2f}".format(ids[i], times[i], budgets[i], losses[i]))

    return incumbent_trajectory

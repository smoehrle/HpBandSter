from hpbandster.core.result import Result


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
    total_budget = 0

    for r in all_runs:
        if r.loss is None:
            continue

        total_budget += r.budget
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

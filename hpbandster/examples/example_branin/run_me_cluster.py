import os
import argparse
import pickle
import logging
import random
import numpy as np
from typing import Optional, NamedTuple

from hpbandster.optimizers import HyperBand, BOHB, RandomSearch
import hpbandster.core.nameserver as hpns
import ConfigSpace as CS

import branin
import util
from worker import BraninWorker
from fidelity_strat import FidelityPropToBudget, FidelityPropToCost

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -------------------------------- Run config ---------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Each strategie is run 'num_runs' times
# Each run consists of 'num_hb_runs' * 'num_brakets' iterations

min_budget = 9
max_budget = 243
num_brackets = int(round((np.log(max_budget) - np.log(min_budget)) / np.log(3)))
num_hb_runs = 2000

num_runs = 3
strategies = [
    #FidelityPropToBudget([False, False, False]),
    #FidelityPropToBudget([True, False, False]),
    FidelityPropToBudget([False, True, False]),
    FidelityPropToBudget([False, False, True]),
    # FidelityPropToBudget([True, True, False]),
    # FidelityPropToBudget([True, False, True]),
    # FidelityPropToBudget([False, True, True]),
    # FidelityPropToBudget([True, True, True]),
]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ------------------------------ Run config end -------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HpBandSter branin toy function example.')
    parser.add_argument('--run-id', help='unique id to identify the HPB run.',
                        default='HPB_branin', type=str)
    parser.add_argument('--working-dir', help='working directory to store live data.',
                        default='.', type=str)
    parser.add_argument('--nic-name', help='name of the Network Interface Card.',
                        default='lo', type=str)
    parser.add_argument('--master', help='start master.',
                        action='store_true')
    parser.add_argument('--worker', help='start worker.',
                        action='store_true')
    parser.add_argument('--num-worker', help='number of worker to start.',
                        default=1, type=int)
    return parser.parse_args()


def build_config_space() -> CS.ConfigurationSpace:
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x1', lower=-5, upper=10))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x2', lower=0, upper=15))
    return config_space


def start_worker(
        run_id: str,
        strategie,
        working_dir: str,
        host: Optional[str] = None,
        background: bool = False) -> None:
    true_y = branin.min()
    cost = branin.build_cost()
    w = BraninWorker(
        true_y,
        cost,
        strategie,
        min_budget, max_budget,
        run_id=run_id, host=host)

    assert working_dir is not None, "Need working_dir to load nameserver credentials."
    w.load_nameserver_credentials(working_dir)

    w.run(background)


def run_master(run_id: str, pickle_name: str, ns: hpns.NameServer, working_dir: str):
    config_space = build_config_space()
    hb = HyperBand(
        configspace=config_space,
        run_id=run_id,
        min_budget=min_budget,
        max_budget=max_budget,
        eta=3,
        host=ns.host,
        nameserver=ns.host,
        nameserver_port=ns.port,
        ping_interval=3600
    )

    res = hb.run(n_iterations=num_brackets*num_hb_runs, min_n_workers=1)

    # pickle result here for later analysis
    with open(os.path.join(working_dir, 'results.{}.pkl'.format(pickle_name)), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown all workers and namespace
    hb.shutdown(shutdown_workers=True)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    args = parse_cli()

    if not args.master and not args.worker:
        logger.warning("Nothing to do. Please specify --master and/or --worker.")

    # start name server
    if args.master:
        ns = hpns.NameServer(
            run_id=args.run_id, nic_name=args.nic_name,
            working_directory=args.working_dir)
        ns.start()

    runs = [(strat, i) for strat in strategies for i in range(num_runs)]
    for strat, i in runs:
        print("Start run {} with strat {}".format(i, strat.name))
        if args.worker:
            host = hpns.nic_name_to_host(args.nic_name)
            for j in range(args.num_worker):
                start_worker(
                    args.run_id,
                    strat,
                    host=host,
                    background=(args.master or args.num_worker > 1),
                    working_dir=args.working_dir)

        if args.master:
            pickle_name = '{}-{}-{}'.format(args.run_id, strat.name, i)
            run_master(
                args.run_id,
                pickle_name,
                ns,
                working_dir=args.working_dir,)

    # shutdown nameserver
    if args.master:
        ns.shutdown()


if __name__ == "__main__":
    main()

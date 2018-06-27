import os
import argparse
import pickle
import logging
import random
import numpy as np
import yaml
from typing import Optional, NamedTuple

from hpbandster.optimizers import HyperBand, BOHB, RandomSearch
import hpbandster.core.nameserver as hpns
import ConfigSpace as CS

import branin
import config
import util
from worker import BraninWorker
from fidelity_strat import FidelityPropToBudget, FidelityPropToCost


strategies = [
    # FidelityPropToBudget([False, False, False]),
    # FidelityPropToBudget([True, False, False]),
    # FidelityPropToBudget([False, True, False]),
    # FidelityPropToBudget([False, False, True]),
    # FidelityPropToBudget([True, True, False]),
    # FidelityPropToBudget([True, False, True]),
    # FidelityPropToBudget([False, True, True]),
    # FidelityPropToBudget([True, True, True]),
]

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HpBandSter branin toy function example.')
    parser.add_argument('--run-id', help='unique id to identify the HPB run.',
                        default='HPB_branin', type=str)
    parser.add_argument('--config', help='location of the config file.',
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
        cfg: config.ExperimentConfig,
        run: config.RunConfig,
        host: Optional[str] = None,
        background: bool = False) -> None:
    true_y = branin.min()
    cost = branin.build_cost()
    w = BraninWorker(
        true_y,
        run.cost,
        run.strategy,
        cfg.min_budget, cfg.max_budget,
        run_id=run_id, host=host, **run.branin_params)

    assert cfg.working_dir is not None, "Need working_dir to load nameserver credentials."
    w.load_nameserver_credentials(cfg.working_dir)

    w.run(background)


def run_master(run_id: str, pickle_name: str, ns: hpns.NameServer, cfg: config.ExperimentConfig, run: config.RunConfig):
    config_space = build_config_space()
    hb = run.constructor(
        configspace=config_space,
        run_id=run_id,
        min_budget=cfg.min_budget,
        max_budget=cfg.max_budget,
        eta=3,
        host=ns.host,
        nameserver=ns.host,
        nameserver_port=ns.port,
        ping_interval=3600
    )

    res = hb.run(n_iterations=cfg.num_brackets*cfg.num_hb_runs, min_n_workers=1)

    # pickle result here for later analysis
    with open(os.path.join(cfg.working_dir, 'results.{}.pkl'.format(pickle_name)), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown all workers and namespace
    hb.shutdown(shutdown_workers=True)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    args = parse_cli()
    cfg = config.load(args.config)
    logger.info(cfg)

    if not args.master and not args.worker:
        logger.warning("Nothing to do. Please specify --master and/or --worker.")
        return

    # start name server
    if args.master:
        ns = hpns.NameServer(run_id=args.run_id, nic_name=args.nic_name, working_directory=cfg.working_dir)
        ns.start()

    runs = [(i, run) for run in cfg.runs for i in range(cfg.num_runs)]
    for i, (run_id, run) in enumerate(runs):
        print("Start run {}/{}".format(i+1, len(runs)))
        if args.worker:
            host = hpns.nic_name_to_host(args.nic_name)
            for j in range(args.num_worker):
                start_worker(args.run_id, cfg, run, host=host, background=(args.master or args.num_worker > 1))
        if args.master:
            pickle_name = '{}-{}-{}'.format(args.run_id, run.display_name.lower(), run_id+cfg.offset)
            run_master(args.run_id, pickle_name, ns, cfg, run)

    # shutdown nameserver
    if args.master:
        ns.shutdown()


if __name__ == "__main__":
    main()
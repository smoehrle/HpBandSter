import os
import argparse
import pickle
import logging
import random
from typing import Optional

from hpbandster.optimizers import HyperBand, BOHB
import hpbandster.core.nameserver as hpns
import ConfigSpace as CS

from worker import BraninWorker
import util


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


def start_worker(run_id: str,
                 id: Optional[str] = None,
                 nameserver: Optional[str] = None,
                 nameserver_port: Optional[int] = None,
                 working_dir: Optional[str] = None,
                 host: Optional[str] = None,
                 nic_name: Optional[str] = None,
                 background: bool = False):
    def cost(z1: float, z2: float, z3: float) -> float:
        return 0.05 + (z1**3 * 1**2 * 1**1.5)

    x1 = random.uniform(-5, 10)
    x2 = random.uniform(0, 15)
    true_y = BraninWorker.calc_branin(x1, x2)
    w = BraninWorker(true_y, cost,
                     run_id=run_id, host=host,
                     nameserver=nameserver, nameserver_port=nameserver_port)
    if nameserver is None or nameserver_port is None:
        assert working_dir is not None, "Need working_dir to load nameserver credentials."
        w.load_nameserver_credentials(working_dir)

    # run worker in the forground,
    w.run(background)


def start_master(run_id: str, ns: hpns.NameServer, nic_name: str, working_dir: str):
    config_space = build_config_space()
    hb = BOHB(
        configspace=config_space,
        run_id=run_id,
        eta=3,
        min_budget=27,
        max_budget=243,
        host=ns.host,
        nameserver=ns.host,
        nameserver_port=ns.port,
        ping_interval=3600
    )

    res = hb.run(n_iterations=4,
                 min_n_workers=1)

    # pickle result here for later analysis
    with open(os.path.join(working_dir, 'results.{}.pkl'.format(run_id)), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown all workers and namespace
    hb.shutdown(shutdown_workers=True)
    ns.shutdown()


def main():
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    args = parse_cli()
    ns_host, ns_port = None, None

    if args.master:
        ns = hpns.NameServer(run_id=args.run_id, nic_name=args.nic_name,
                             working_directory=args.working_dir)
        ns_host, ns_port = ns.start()

    if args.worker:
        host = hpns.nic_name_to_host(args.nic_name)
        for i in range(args.num_worker):
            start_worker(
                args.run_id, host=host, id=str(i),
                background=(args.master or args.num_worker > 1),
                nameserver=ns_host,
                nameserver_port=ns_port,
                working_dir=args.working_dir,
                nic_name=args.nic_name)

    if args.master:
        start_master(
            args.run_id,
            ns,
            nic_name=args.nic_name,
            working_dir=args.working_dir,)

    if not args.master and not args.worker:
        logger.warning("Nothing to do.")


if __name__ == "__main__":
    main()

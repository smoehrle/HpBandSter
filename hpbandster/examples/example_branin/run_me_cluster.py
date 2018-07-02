import os
import argparse
import pickle
import logging
from typing import Optional

import hpbandster.core.nameserver as hpns

import config
from worker import SimM2FWorker


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='HpBandSter branin toy function example.')
    parser.add_argument('--run-id', help='unique id to identify the HPB run.',
                        default='HPB_branin', type=str)
    parser.add_argument('--task-id', help='task id to identify the run.',
                        type=int)
    parser.add_argument('--last-task-id', help='Id of the last task.',
                        type=int)
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


def start_worker(
        run_id: str,
        cfg: config.ExperimentConfig,
        run: config.RunConfig,
        host: Optional[str] = None,
        background: bool = False) -> None:
    w = SimM2FWorker(
        run.problem,
        run.strategy,
        cfg.max_budget,
        run_id=run_id, host=host)

    assert cfg.working_dir is not None, "Need working_dir to load nameserver credentials."
    w.load_nameserver_credentials(cfg.working_dir)

    w.run(background)


def run_master(run_id: str, pickle_name: str, ns: hpns.NameServer, cfg: config.ExperimentConfig, run: config.RunConfig):
    config_space = run.problem.build_config_space()
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
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    args = parse_cli()
    # Fix nameserver colision on cluster
    if args.task_id:
        args.run_id += str(args.task_id)
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
        if args.task_id is not None:
            if args.last_task_id is None and i != (args.task_id - 1):
                continue
            elif args.last_task_id is not None and (i % args.last_task_id) != (args.task_id - 1):
                continue

        print("Start run {}/{}".format(i + 1, len(runs)))
        if args.worker:
            host = hpns.nic_name_to_host(args.nic_name)
            for _ in range(args.num_worker):
                start_worker(args.run_id, cfg, run, host=host, background=(args.master or args.num_worker > 1))
        if args.master:
            pickle_name = '{}-{}-{}'.format(args.run_id, run.display_name.lower(), run_id+cfg.offset)
            run_master(args.run_id, pickle_name, ns, cfg, run)

    # shutdown nameserver
    if args.master:
        ns.shutdown()


if __name__ == "__main__":
    main()

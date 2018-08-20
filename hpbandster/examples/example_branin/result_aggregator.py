import argparse
import glob
import logging
import os
import pickle
import re
from typing import List

import config
from models import Experiment, Plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AggregatedResults():
    def __init__(self):
        self.runs = dict()
        self.config: Experiment = None

    def load_run(self, config_id, filename):
        if config_id not in self.runs:
            self.runs[config_id] = []

        with open(filename, 'rb') as file_:
            run = pickle.load(file_)

        self.runs[config_id].append(run)

    def find_runs(self, filter_, directory):
        config_id_regex = re.compile(r"results\.{}-(.+)-[0-9]+\.pkl".format(filter_))

        matched_filenames = []

        for filename in glob.glob(os.path.join(directory, "results.{}*pkl".format(filter_))):
            match = config_id_regex.search(filename)
            if match:
                matched_filenames.append(filename)
                logger.debug("Found file: {}".format(filename))
                self.load_run(match.group(1), filename)

        return matched_filenames

    def dump(self, filename):
        with open(filename, 'wb') as fh:
            logger.info("Write file to {}".format(filename))
            pickle.dump(self, fh)


    @staticmethod
    def load(filename):
        if not os.path.isfile(filename):
            raise Exception("Could not find file {}".format(filename))

        with open(filename, 'rb') as file_:
            logger.info("Loading file: {}".format(filename))
            ar = pickle.load(file_)

        # Fix for aro files where config was a dict
        if type(ar.config) == dict:
            c = ar.config
            del c['strategies']
            del c['problems']
            c['runs'] = []
            plot = Plot(**c.pop('plot'))
            wd = os.path.dirname(os.path.abspath(filename))
            ar.config = Experiment(working_dir=wd, job_id='', plot=plot, **c)

        return ar


def main():
    # Setup argparser
    parser = argparse.ArgumentParser(description="HpBandSter result object aggregator.")

    create_parser = _create_parser()
    add_parser = _add_parser()

    # Add actions
    sp = parser.add_subparsers()
    sp_create = sp.add_parser(
        'create',
        parents=[create_parser],
        help='Create an aggregated result object')
    sp_add = sp.add_parser(
        'add',
        parents=[add_parser],
        help='Add results to an existing aggregated result object')

    # Hook subparsers up to functions
    sp_create.set_defaults(func=create)
    sp_add.set_defaults(func=add)

    args = parser.parse_args()
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()


def create(args):
    logger.info("Create new aggregated result object")
    ar = AggregatedResults()

    ar.config = _load_config(args)

    matched_filenames = ar.find_runs(args.filter, args.directory)

    filename = args.out_name if args.out_name else 'aro.{}.pkl'.format(args.filter)
    ar.dump(os.path.join(args.directory, filename))

    if args.clean:
        _remove_files(matched_filenames)


def add(args):
    logger.info("Create new aggregated result object")

    filename = os.path.join(args.directory, args.object)
    ar = AggregatedResults.load(filename)

    matched_filenames = ar.find_runs(args.filter, args.directory)
    ar.dump(filename)

    if args.clean:
        _remove_files(matched_filenames)


def _create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-d',
        '--directory',
        help='Folder to search for result files.',
        type=str,
        default='.')
    parser.add_argument(
        '-f',
        '--filter',
        help='Search for results.<run-filter>-XXX.pkl files.',
        type=str,
        required=True)
    parser.add_argument(
        '-o',
        '--out-name',
        help='Optional output name',
        type=str,
        default=None)
    parser.add_argument(
        '--clean',
        help='Remove results.*.pkl files after aggregating them.',
        action='store_true')
    parser.add_argument(
        '--config',
        help='Name or path relative to --directory for the config. Only necessary' +
             'if there are multiple yml files or the yml file is in another directory',
        default=None)

    return parser


def _add_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-d',
        '--directory',
        help='Folder to search for result files.',
        type=str,
        default='.')
    parser.add_argument(
        '-f',
        '--filter',
        help='Search for results.<run-filter>-XXX.pkl files.',
        type=str,
        required=True)
    parser.add_argument(
        '-o',
        '--object',
        help='Existing aggregated results object',
        type=str,
        required=True)
    parser.add_argument(
        '--clean',
        help='Remove results.*.pkl files after aggregating them.',
        action='store_true')

    return parser


def _load_config(args) -> Experiment:
    if args.config:
        config_ = os.path.join(args.directory, args.config)
    else:
        configs = [f for f in glob.glob(os.path.join(args.directory, "*.yml"))]

        if len(configs) == 0:
            raise Exception("Could not find a *.yml file as config. Please specify one")
        elif len(configs) > 1:
            raise Exception("Multiple *.yml files found. Please specify one")
        config_ = configs[0]

    return config.load(config_, "", False)


def _remove_files(files: List[str]) -> None:
    logger.info("Clean up files")
    for filename in files:
        os.remove(filename)


if __name__ == '__main__':
    main()

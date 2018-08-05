import argparse
import glob
import logging
import os
import pickle
import re

import config

logger = logging.getLogger(__name__)


class AggregatedResults():
    def __init__(self):
        self.runs = dict()
        self.config = dict()

    def add_run(self, config_id, filename):
        if config_id not in self.runs:
            self.runs[config_id] = []

        with open(filename, 'rb') as file_:
            run = pickle.load(file_)

        self.runs[config_id].append(run)


def main():
    # Setup argparser
    parser = argparse.ArgumentParser(description="HpBandSter result object aggregator.")

    create_parser = _create_parser()

    add_parser = argparse.ArgumentParser(add_help=False)

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

    if args.config:
        config_ = os.path.join(args.directory, args.config)
    else:
        configs = [f for f in glob.glob(os.path.join(args.directory, "*.yml"))]

        if len(configs) == 0:
            raise Exception("Could not find a *.yml file as config. Please specify one")
        elif len(configs) > 1:
            raise Exception("Multiple *.yml files found. Please specify one")
        config_ = configs[0]

    ar.config = config.load_yaml(config_)

    config_id_regex = re.compile(r"results\.{}-([^-]+-)[0-9]+\.pkl".format(args.filter))

    matched_filenames = []

    for filename in glob.glob(os.path.join(args.directory, "results.{}*pkl".format(args.filter))):
        match = config_id_regex.search(filename)
        if match:
            if args.clean:
                matched_filenames.append(filename)
            logger.debug("Found file: {}".format(filename))
            ar.add_run(match[1], filename)

    filename = args.out_name if args.out_name else 'aro.{}.pkl'.format(args.filter)
    result_filename = os.path.join(args.directory, filename)
    with open(result_filename, 'wb') as fh:
        logger.info("Write file to {}".format(result_filename))
        pickle.dump(ar, fh)

    if args.clean:
        # Clean up after results have been aggregated successfully
        logger.info("Clean up files")
        for filename in matched_filenames:
            os.remove(filename)


def add(args):
    print("Adding...")


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

if __name__ == '__main__':
    main()

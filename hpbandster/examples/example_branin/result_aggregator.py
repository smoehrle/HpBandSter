import argparse
import glob
import logging
import os
import pickle
import re
from typing import List

import config
import util
from models import Experiment, Plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AggregatedResults():
    def __init__(self):
        self.runs = dict()
        self.config = None

    def load_run(self, config_id, filename):
        if config_id not in self.runs:
            self.runs[config_id] = []

        with open(filename, 'rb') as file_:
            run = pickle.load(file_)

        self.runs[config_id].append(util.extract_result(run, self.config.plot.bigger_is_better))

    def find_runs(self, filter_, directory):
        config_id_regex = re.compile(r"results\.{}-(.+)-[0-9]+\.pkl".format(filter_))

        matched_filenames = []
        cnt = 0

        for filename in glob.glob(os.path.join(directory, "results.{}*pkl".format(filter_))):
            match = config_id_regex.search(filename)
            if match:
                cnt += 1
                matched_filenames.append(filename)
                logger.debug("Found file ({}): {}".format(cnt, filename))
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
    info_parser = _info_parser()
    delete_parser = _delete_parser()
    rename_parser = _rename_parser()

    # Add actions
    sp = parser.add_subparsers()
    sp_create = sp.add_parser(
        'create',
        parents=[create_parser],
        help='Create an aggregated result object')
    sp_add = sp.add_parser(
        'add',
        parents=[add_parser],
        help='Add results')
    sp_info = sp.add_parser(
        'info',
        parents=[info_parser],
        help='Show information')
    sp_delete = sp.add_parser(
        'delete',
        parents=[delete_parser],
        help='Delete one or more config_ids')
    sp_rename = sp.add_parser(
        'rename',
        parents=[rename_parser],
        help='Rename a config_id')


    # Hook subparsers up to functions
    sp_create.set_defaults(func=create)
    sp_add.set_defaults(func=add)
    sp_info.set_defaults(func=info)
    sp_delete.set_defaults(func=delete)
    sp_rename.set_defaults(func=rename)

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


def info(args):
    for obj in args.objects:
        ar = AggregatedResults.load(obj)
        print("Config ids:")
        for config_id in ar.runs.keys():
            print('\t', config_id, ':\t', len(ar.runs[config_id]))


def delete(args):
    ar = AggregatedResults.load(args.object)
    for config_id in args.ids:
        if config_id in ar.runs.keys():
            logger.info("Delete {}".format(config_id))
            del ar.runs[config_id]
        else:
            logger.warning("Config id {} not found".format(config_id))
    ar.dump(args.object)


def rename(args):
    ar = AggregatedResults.load(args.object)

    if args.old_id in ar.runs.keys():
        logger.info("Rename {} to {}".format(args.old_id, args.new_id))

        tmp = ar.runs[args.old_id]
        del ar.runs[args.old_id]
        ar.runs[args.new_id] = tmp
    else:
        logger.warning("Old config id {} not found".format(args.old_id))

    ar.dump(args.object)


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
        help='Name or path relative to the current directory for the config. Only necessary' +
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


def _info_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        'objects',
        metavar='FILE',
        nargs='+',
        help='Aggregated results object')

    return parser


def _delete_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-o',
        '--object',
        help='Aggregated results object',
        type=str,
        required=True)
    parser.add_argument(
        'ids',
        metavar='id',
        nargs='+',
        help='Config ids which should be deleted')

    return parser


def _rename_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-o',
        '--object',
        help='Aggregated results object',
        type=str,
        required=True)
    parser.add_argument(
        '--old_id',
        help='The old config id',
        type=str,
        required=True)
    parser.add_argument(
        '--new_id',
        help='The new config id',
        type=str,
        required=True)

    return parser


def _load_config(args) -> Experiment:
    if args.config:
        config_ = args.config
    else:
        configs = [f for f in glob.glob(os.path.join(args.directory, "*.yml"))]

        if len(configs) == 0:
            raise Exception("Could not find a *.yml file as config. Please specify one")
        elif len(configs) > 1:
            raise Exception("Multiple *.yml files found. Please specify one")
        config_ = configs[0]

    logger.info("Load config: {}".format(config_))
    return config.load(config_, "", False)


def _remove_files(files: List[str]) -> None:
    logger.info("Clean up files")
    for filename in files:
        os.remove(filename)


if __name__ == '__main__':
    main()

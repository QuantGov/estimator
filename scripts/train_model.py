#!/usr/bin/env python

import argparse
import configparser
import logging

from pathlib import Path

from sklearn.externals import joblib as jl

import models


ENCODE_IN = 'utf-8'

log = logging.getLogger(Path(__file__).stem)


def _autoconvert(value):
    """Convert to int or float if possible, otherwise return string"""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def get_model(configfile):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(configfile, encoding=ENCODE_IN)
    model = models.MODELS[config['Model']['name']]['model']
    model.set_params(**{i: _autoconvert(j)
                        for i, j in config['Parameters'].items()})
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_config')
    parser.add_argument('trainers')
    parser.add_argument('labels')
    parser.add_argument('-o', '--outfile')
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_const',
                           const=logging.DEBUG, default=logging.INFO)
    verbosity.add_argument('-q', '--quiet', dest='verbose',
                           action='store_const', const=logging.WARNING)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model = get_model(args.model_config)
    xindex, X = jl.load(args.trainers)
    yindex, label_names, y = jl.load(args.labels)
    assert xindex == yindex
    model.fit(X, y)
    jl.dump((label_names, model), args.outfile)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import random
import logging

import quantgov
import sklearn.preprocessing

from pathlib import Path

from sklearn.externals import joblib as jl

log = logging.getLogger(Path(__file__).stem)


def create_label(streamer):
    """
    Assign a label to a set of documents using a CountVectorizer

    Arguments:
    * streamer: a quantgov.corpora.CorpusStreamer object

    Returns:
    * index:    A tuple of index values for the documents labeled
    * classes:  A tuple of names of classes for which labels are generated
    * labels:   The label or labels for each document. If more than one class
                is being labeled, each element of this tuple will be a tuple
                corresponding to the returned classes object

    """
    label_names = ('randomly_true',)
    labels = tuple(random.choice([True, False]) for doc in streamer)
    return tuple(streamer.index), label_names, labels


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
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
    driver = quantgov.load_driver(args.corpus)
    jl.dump(create_label(driver.get_streamer()), args.outfile)


if __name__ == "__main__":
    main()

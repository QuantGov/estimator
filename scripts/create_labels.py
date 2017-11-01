#!/usr/bin/env python

import argparse
import random
import logging

import quantgov.estimator

from pathlib import Path

log = logging.getLogger(Path(__file__).stem)


def create_label(streamer):
    """
    Assign a label to a set of documents using a CountVectorizer

    Arguments:
    * streamer: a quantgov.corpora.CorpusStreamer object

    Returns: a quantgov.estimator.Labels object

    """
    label_names = ('randomly_true',)
    labels = tuple(random.choice([True, False]) for doc in streamer)

    return quantgov.estimator.Labels(
        index=tuple(streamer.index),
        label_names=label_names,
        labels=labels
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=quantgov.load_driver)
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
    create_label(args.corpus.get_streamer()).save(args.outfile)


if __name__ == "__main__":
    main()

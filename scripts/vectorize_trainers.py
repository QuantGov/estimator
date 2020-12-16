#!/usr/bin/env python
"""
vectorize_trainers.py

Create a term count representation of training documents. Output is a
joblib-pickled 2-tuple of the index of the corpus and the vectors.
"""

import argparse
import logging

import sklearn.feature_extraction
import quantgov
import quantgov.ml

from pathlib import Path
import joblib as jl

ENCODE_IN = 'utf-8'
ENCODE_OUT = 'utf-8'

log = logging.getLogger(Path(__file__).stem)


def vectorize_trainers(streamer):
    """
    Vectorize a set of documents using a CountVectorizer

    Arguments:
    * streamer: a quantgov.corpora.CorpusStreamer object

    Returns:
    * trainers:     a 2-tuple where the first element is the index of all
                    documents and the second is an array holding the vectorized
                    trainers

    * vectorizer:   the fitted vectorizer used to vectorize the trainers

    """
    # See http://scikit-learn.org/stable/modules/feature_extraction.html for
    # details on using CountVectorizer
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    X = vectorizer.fit_transform(doc.text for doc in streamer)
    trainers = quantgov.ml.Trainers(
        index=tuple(streamer.index),
        vectors=X
    )
    return trainers, vectorizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('-o', '--trainers_outfile')
    parser.add_argument('--vectorizer_outfile')
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
    trainers, vectorizer = vectorize_trainers(driver.get_streamer())
    trainers.save(args.trainers_outfile)
    jl.dump(vectorizer, args.vectorizer_outfile)


if __name__ == "__main__":
    main()

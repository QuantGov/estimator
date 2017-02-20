#!/usr/bin/env python

import argparse
import configparser
import logging

import pandas as pd
import sklearn.model_selection

from pathlib import Path
from sklearn.externals import joblib as jl

import models


ENCODE_OUT = 'utf-8'

log = logging.getLogger(Path(__file__).stem)


def evaluate_model(model, X, y, folds, scoring):
    if hasattr(y[0], '__getitem__'):
        cv = sklearn.model_selection.KFold(folds, shuffle=True)
        if '_' not in scoring:
            log.warning("No averaging method specified, assuming macro")
            scoring += '_macro'
    else:
        cv = sklearn.model_selection.StratifiedKFold(folds, shuffle=True)
    gs = sklearn.model_selection.GridSearchCV(
        estimator=model['model'],
        param_grid=model['parameters'],
        cv=cv,
        scoring=scoring,
        n_jobs=1 if model['multiprocessed'] else 0,
        verbose=100,
        refit=False
    )
    gs.fit(X, y)
    return pd.DataFrame(gs.cv_results_)


def evaluate_models(models, X, y, folds, scoring):
    results = pd.concat([
        evaluate_model(model, X, y, folds, scoring).assign(model=name)
        for name, model in models.items()
    ])
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trainers')
    parser.add_argument('labels')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--scoring', default='f1')
    parser.add_argument('-o', '--evaluation_outfile')
    parser.add_argument('--config_outfile', required=True, type=Path)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_const',
                           const=logging.DEBUG, default=logging.INFO)
    verbosity.add_argument('-q', '--quiet', dest='verbose',
                           action='store_const', const=logging.WARNING)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    xindex, X = jl.load(args.trainers)
    yindex, classes, y = jl.load(args.labels)
    assert xindex == yindex
    results = evaluate_models(models.MODELS, X, y, args.folds, args.scoring)
    results.to_csv(args.evaluation_outfile, index=False)
    top = results.loc[results['mean_test_score'].idxmax()]
    config = configparser.ConfigParser()
    config.optionxform = str
    config['Model'] = {'name': top['model']}
    config['Parameters'] = top['params']
    with args.config_outfile.open('w', encoding=ENCODE_OUT) as outf:
        config.write(outf)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import csv
import io
import logging
import sys

import sklearn.pipeline
import quantgov

from pathlib import Path

from sklearn.externals import joblib as jl

ENCODE_OUT = 'utf-8'

log = logging.getLogger(Path(__file__).stem)


def get_pipeline(vectorizer, model):
    return sklearn.pipeline.Pipeline((
        ('vectorizer', vectorizer),
        ('model', model)
    ))


def estimate(vectorizer, model, streamer):
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    yield from zip(streamer.index, pipeline.predict(texts))


def estimate_probability(vectorizer, model, streamer):
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    truecol = list(int(i) for i in model.classes_).index(1)
    predicted = (i[truecol] for i in pipeline.predict_proba(texts))
    yield from zip(streamer.index, predicted)


def estimate_probability_multilabel(vectorizer, model, streamer):
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    truecols = tuple(
        list(int(i) for i in label_classes).index(1)
        for label_classes in model.classes_
    )
    predicted = pipeline.predict_proba(texts)
    for i, docidx in enumerate(streamer.index):
        yield docidx, tuple(label_predictions[i, truecols[j]]
                            for j, label_predictions in enumerate(predicted))


def estimate_probability_multiclass(vectorizer, model, streamer):
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    yield from zip(streamer.index, pipeline.predict_proba(texts))


def estimate_probability_multilabel_multiclass(vectorizer, model, streamer):
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    predicted = pipeline.predict_proba(texts)
    for i, docidx in enumerate(streamer.index):
        yield docidx, tuple(label_predictions[i]
                            for label_predictions in predicted)


def is_multiclass(classes):
    """
    Returns True if values in classes are anything but 1, 0, True, or False,
    otherwise returns False.
    """
    try:
        return len(set(int(i) for i in classes) - {0, 1}) != 0
    except ValueError:
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vectorizer')
    parser.add_argument('model')
    parser.add_argument('target', type=Path)
    parser.add_argument('-o', '--outfile',
                        type=lambda x: open(
                            x, 'w', newline='', encoding=ENCODE_OUT),
                        default=io.TextIOWrapper(
                            sys.stdout.buffer, newline='', encoding=ENCODE_OUT)
                        )
    parser.add_argument('--probability', action='store_true', default=False)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_const',
                           const=logging.DEBUG, default=logging.INFO)
    verbosity.add_argument('-q', '--quiet', dest='verbose',
                           action='store_const', const=logging.WARNING)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    vectorizer = jl.load(args.vectorizer)
    label_names, model = jl.load(args.model)
    driver = quantgov.load_driver(args.target)
    streamer = driver.get_streamer()

    writer = csv.writer(args.outfile)
    if len(label_names) > 1:
        multilabel = True
        multiclass = any(is_multiclass(i) for i in model.classes_)
    else:
        multilabel = False
        multiclass = is_multiclass(model.classes_)

    # TODO: This is very ugly and complicated and should probably be refactored
    if args.probability:
        if multilabel:
            if multiclass:  # Multilabel-multiclass probability
                results = estimate_probability_multilabel_multiclass(
                    vectorizer, model, streamer)
                writer.writerow(driver.index_labels +
                                ('label', 'class', 'probability'))
                writer.writerows(
                    docidx + (label_name, class_name, prediction)
                    for docidx, predictions in results
                    for label_name, label_classes, label_predictions
                    in zip(label_names, model.classes_, predictions)
                    for class_name, prediction
                    in zip(label_classes, label_predictions)
                )
            else:  # Multilabel probability
                results = estimate_probability_multilabel(
                    vectorizer, model, streamer)
                writer.writerow(driver.index_labels + ('label', 'probability'))
                writer.writerows(
                    docidx + (label_name, prediction)
                    for docidx, predictions in results
                    for label_name, prediction in zip(label_names, predictions)
                )
        elif multiclass:  # Multiclass probability
            writer.writerow(driver.index_labels + ('class', 'probability'))
            results = estimate_probability_multiclass(
                vectorizer, model, streamer)
            writer.writerows(
                docidx + (class_name, prediction)
                for docidx, predictions in results
                for class_name, prediction in zip(model.classes_, predictions)
            )
        else:  # Simple probability
            results = estimate_probability(vectorizer, model, streamer)
            writer.writerow(
                driver.index_labels + (label_names[0] + '_prob',))
            writer.writerows(
                docidx + (prediction,) for docidx, prediction in results)
    elif multilabel:  # Multilabel Prediction
        results = estimate(vectorizer, model, streamer)
        writer.writerow(driver.index_labels + ('label', 'prediction'))
        writer.writerows(
            docidx + (label_name, prediction,)
            for docidx, predictions in results
            for label_name, prediction in zip(label_names, predictions)
        )
    else:  # Simple Prediction
        results = estimate(vectorizer, model, streamer)
        writer.writerow(driver.index_labels + label_names)
        writer.writerows(docidx + (prediction,)
                         for docidx, prediction in results)


if __name__ == "__main__":
    main()

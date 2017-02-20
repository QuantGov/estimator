# The QuantGov Estimator

## Official QuantGov Estimators

This repository is for those who would like to create new datasets using the QuantGov platform. If you would like to find data that has been produced using the QuantGov platform, please visit http://www.quantgov.org/data.

This repository contains all official QuantGov estimators, with each estimator stored in its own branch.

## The Generic Estimator

The `master` branch of this repository is the Generic Estimator, which evaluates and trains a Random Forests Classifier. By default, the `create_labels.py` script generates a random label of True or False for every document; you should modify this script to use the label or labels you are actually interested in.

This estimator uses a scikit-learn [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to vectorize training documents as a preprocessing step. In many cases, it will be useful to modify the default parameters; see the Scikit-learn documentation for details. If vectorization will be include information about the final classes, it is necessary to move the vectorization step into the candidate model pipeline for correct cross-validation results.

Candidate models are defined in scripts\models.py. Parameters follow the naming convention for scikit-learn grid search; see the [scikit-learn documentation](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search) for details.

The generic estimator will use the training corpus to exhaustively evaluate each combination of parameters for each candidate model, and output the results to `data/model_evaluation.csv`. The best scoring model will be suggested in the `data/model.config` file, but users can change the parameters or model based on the evaluation results (for example, using the one-standard-error rule).

## Using this Estimator

To use or modify this estimator, clone it using git or download the archive from the [QuantGov Site](http://www.quantgov.org/platform) and unzip it on your computer.

## Requirements

Using this estimator requires Python >= 3.4 and the `make` utility. 

If you are using the Anaconda Python distribution (recommended), navigate to the estimator folder and use the command `conda install --file conda-requirements.txt`, then the command `pip install -r requirements.txt`. If you are on windows, also use the command `conda install --file conda-requirements-windows.txt`, which will install the `make` utility. 

If you are not using Anaconda, use the command `pip install requirements.txt`. You must ensure that `make` is install separately.

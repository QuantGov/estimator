#### ESTIMATOR SNAKEFILE ######################################################
## This file defines a workflow for a QuantGov Estimator as executed by the
## SnakeMake utility. The default estimator uses the labels produced in the
## scripts/create_labels.py script to train a single-label, binary or
## multi-class classifier on the candidate models defined in
## scripts/candidate_models.py


#### PYTHON IMPORTS ###########################################################
## This section imports libraries and defines functionality to be used
## throughout the workflow 

import os.path

def outpath(path):
    """Ensure Cross-Platform functionality for files in subdirectories"""
    return os.path.sep.join(os.path.split(path))


#### SNAKEFILE CONFIGURATION ##################################################
## This section defines snakfile configuration and variables used throughout
## the workflow

configfile: 'config.yaml'

subworkflow trainer_corpus:
    workdir: config['trainer_corpus']


#### DEFAULT RULE #############################################################
## The default rule runs the `evaluate_models` rule

rule default:
    input:
        outpath('data/model_evaluation.csv'),
        outpath('data/model.cfg')


#### Data Preparation #########################################################
## These rules vectorize the trainer documents with the vectorizer defined in 
## scripts/vectorize_trainers.py, and create the labels for training

rule vectorize_trainers:
    input:
        'scripts/vectorize_trainers.py',
        trainer_corpus('corpus_timestamp')
    output:
        vectorizer=outpath('data/vectorizer.pickle'),
        trainers=outpath('data/trainers.pickle')
    shell:
        'python {input} --vectorizer_outfile {output[0]} --trainers_outfile {output[1]}'

rule create_labels:
    input:
        'scripts/create_labels.py',
        config['trainer_corpus']
    output:
        outpath('data/labels.pickle')
    shell:
        'python {input} --outfile {output}'

#### Model Evaluation #########################################################
## This rule evaluates the models defined in scripts/candidate_models.py and
## 

rule evaluate:
    input:
        'scripts/candidate_models.py',
        rules.vectorize_trainers.output.trainers,
        rules.create_labels.output
    output:
        evaluation=outpath('data/model_evaluation.csv'),
        modelcfg=outpath('data/model.cfg')
    shell:
        'quantgov estimator evaluate {input} {output} --folds {config[folds]} --scoring {config[scoring]}'
    
#### Model Training ###########################################################
## This rule trains the model defined in scripts/candidate_models.py and
## configured in data/model.cfg

rule train:
    input:
        'scripts/candidate_models.py',
        rules.evaluate.output.modelcfg,
        rules.vectorize_trainers.output.trainers,
        rules.create_labels.output
    output:
        outpath('data/model.pickle')
    shell:
        'quantgov estimator train {input} --outfile {output}'

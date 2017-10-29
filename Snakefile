import os.path

def outpath(path):
    """Ensure Cross-Platform functionality for files in subdirectories"""
    return os.path.sep.join(os.path.split(path))


configfile: 'config.yaml'

subworkflow trainer_corpus:
    workdir: config['trainer_corpus']


rule default:
    input:
        outpath('data/model_evaluation.csv'),
        outpath('data/model.cfg')


#### Data Preparation #########################################################
##
##

rule vectorize_trainers:
    input:
        'scripts/vectorize_trainers.py',
        trainer_corpus('driver.py')
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
##
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
##
##

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

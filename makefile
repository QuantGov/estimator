# Set to the path of a corpus representing your trainers.
# This value can be overridden. In general, however, you will want to set this
# value in this file so that the estimator is clearly defined.
TRAINER_CORPUS=/path/to/trainer/corpus

# Set to the path of a corpus representing the text to be classified
# This value can be overridden, and in general you will want to do so in a
# quantgov project makefile (see the Quantgov documentation for details).
TARGET_CORPUS=/path/to/target/corpus

# Set to the desired path for result output.
# This value can be overridden, and in general you will want to do so in a
# quantgov project makefile (see the Quantgov documentation for details).
RESULTS=data/results.csv

VERBOSE_FLAG=-v # Set to empty for less info or to -q for quiet

FOLDS = 5 # Folds for cross-validation
SCORING = f1 # A good default choice for classifiers

# Canned recipe for running python scripts. The script should be the first
# dependency, and the other dependencies should be positional arguments in
# order. The script should allow you to specify the output file with a -o flag,
# and to specify verbosity with a -v flag. If you're using a Python script that
# doesn't follow this pattern, you can of course write the recipe directly.
# Additional explicit arguments can be added after the canned recipe if needed.

define py
python $^ -o $@ $(VERBOSE_FLAG)
endef

evaluate: data/model_evaluation.csv
estimate: $(RESULTS)

data/model.pickle: scripts/train_model.py data/model.config data/trainers.pickle data/labels.pickle
	$(py)

data/model.config: data/model_evaluation.csv
data/model_evaluation.csv: scripts/evaluate_models.py data/trainers.pickle data/labels.pickle
	$(py) --config_outfile=data/model.config --folds=$(FOLDS) --scoring=$(SCORING)

scripts/evaluate_models.py scripts/train_model.py: scripts/models.py
	touch $@

data/labels.pickle: scripts/create_labels.py $(TRAINER_CORPUS)/driver.py
	$(py)

data/vectorizer.pickle: data/trainers.pickle
data/trainers.pickle: scripts/vectorize_trainers.py $(TRAINER_CORPUS)/driver.py data/target_corpus.timestamp
	$(py) --vectorizer_outfile data/vectorizer.pickle

$(RESULTS): scripts/estimate.py data/vectorizer.pickle data/model.pickle $(TARGET_CORPUS)/driver.py
	$(MAKE) -C $(TARGET_CORPUS) driver.py
	$(py) $(ESTIMATE_FLAGS)

check_corpus:
	$(MAKE) -C $(TRAINER_CORPUS) driver.py

.PHONY: evaluate estimate check_corpus

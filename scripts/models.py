import sklearn.feature_extraction
import sklearn.ensemble
import sklearn.pipeline

MODELS = {
    'Random Forests': {
        'model': sklearn.pipeline.Pipeline((
            ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
            ('rf', sklearn.ensemble.RandomForestClassifier(n_jobs=-1)),
        )),
        'parameters': {
            'rf__n_estimators': [5, 10, 15],
        },
        'multiprocessed': True
    }
}

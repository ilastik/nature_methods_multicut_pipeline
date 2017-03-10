
class ClassifierParams:
    def __init__(self,
                 classifier_file='',
                 classifier_key=''
                 ):
        self.classifier_file = classifier_file
        self.classifier_key = classifier_key


def path_classification(features, params):

    import pickle
    with open(params.classifier_file) as f:
        classifier = pickle.load(f)[params.classifier_key]

    # from sklearn.ensemble import RandomForestClassifier
    return classifier.predict_proba(features)
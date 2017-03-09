

class PathFeatureParams:
    def __init__(self,
                 features=[
                     'Sum',
                     'Mean',
                     'Variance',
                     'Maximum',
                     'Minimum',
                     'Kurtosis',
                     'Skewness',
                     'Pathlength'
                 ],
                 anisotropy=[1, 1, 10],
                 max_threads=10,
                 feat_list_file='',
                 experiment_key=''
                 ):
        self.features = features
        self.anisotropy = anisotropy
        self.max_threads = max_threads
        self.feat_list_file = feat_list_file
        self.experiment_key = experiment_key

# singleton type
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# singleton class, holding all experiment parameters
class ExperimentSettings(object):
    __metaclass__ = Singleton

    # init the experiment setttings, setting all parameter to their default
    def __init__(self):

        # cache folder for random forest results
        self.rf_cache_folder = None
        # number of threads for all things in parallel, set to max - 1
        import multiprocessing
        self.n_threads = max( multiprocessing.cpu_count() - 1, 1)

        # parameter fo feature calculation
        # anisotropy factor for the filter calculation
        self.anisotropy_factor = 1.
        # Flag for calculating extra 2d feature
        self.use_2d = False

        # paramter for groundtruth projection and learning
        # flag to activate learning only from the xy-edges (for ISBI12)
        self.learn_2d = False
        # flag to ignore certain edges when learning
        self.use_ignore_mask = False
        # flag to learn from fuzzy groundtruth projection
        self.learn_fuzzy = False
        # minimal overlap for positive examples in fuzzy projection
        self.positive_threshold = 0.6
        # maximal overlap for negative examples in fuzzy projection
        self.negative_threshold = 0.4
        # nubmer of trees for random forests
        self.n_trees = 500
        # use different rfs for xy - and z edges
        self.use_2rfs = False

        # parameter for multicuts
        # multicut solver, possible values: "multicut_exact", "multicut_fusionmoves"
        self.solver = "multicut_exact"
        # weighting scheme for edge energies, possible values "none", "xyz", "z", "all"
        self.weighting_scheme = "none"
        # weight for edge energies
        self.weight = 16.
        # beta for the normal (local) multicut
        self.beta_local   = 0.5
        # beta for the lifteed multicut
        self.beta_global   = 0.5
        # seed fraction for fusion moves
        self.seed_fraction = 0.001
        # total number of iterations for fusion moves
        self.num_it = 3000
        # number of iterations without changes after which fusion moves stop
        self.num_it_stop = 20
        self.verbose = False

        # parameters for lifted multicut
        # locacl training slices
        self.nLocalTrainingSlices = 5
        # number of neighbors for the lifted neighborhood
        self.lifted_neighborhood = 4

        self.pAndMapIterations = 1000

        # parameters for resolving false merges
        self.feature_image_filter_names = ["gaussianSmoothing",
                                      "hessianOfGaussianEigenvalues",
                                      "laplacianOfGaussian"]
        self.feature_image_sigmas = [1.6, 4.2, 8.3]
        self.feature_stats = ["Mean", "Variance", "Sum", "Maximum", "Minimum", "Kurtosis", "Skewness"]
        self.paths_penalty_power = 10
        self.paths_avoid_duplicates = True
        self.min_nh_range = 5
        self.max_sample_size = 20
        self.paths_penalty_power = 10
        self.lifted_path_weights_factor = 1.

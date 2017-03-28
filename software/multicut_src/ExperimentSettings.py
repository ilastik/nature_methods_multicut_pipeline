# class, holding all experiment parameters
class ExperimentSettings(object):

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
        # TODO find better options her
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


    # overloadd str to reliably cache this
    def __str__(self):
        members = vars(self)
        # names sorted alphabetically
        names = members.keys()
        # remove irrelevant things
        names.remove("rf_cache_folder")
        names.remove("n_threads")
        names.remove("verbose")
        names.sort()
        string = ","
        # all values in alphabetic order...
        string = string.join( [str(members[n]) for n in names] )
        return str(hash(string))


    # set the lifted neighborhood
    def set_lifted_neighborhood(self, lifted_neighborhood):
        self.lifted_neighborhood = lifted_neighborhood


    # setter functions for the parameters

    def set_rfcache(self, rf_cache_folder):
        self.rf_cache_folder = rf_cache_folder

    def set_nthreads(self, n_threads):
        self.n_threads = n_threads

    def set_weighting_scheme(self, scheme_str):
        assert scheme_str in ("z", "all", "xyz", "none"), scheme_str
        self.weighting_scheme = scheme_str

    def set_solver(self, solver):
        assert solver in ("multicut_exact", "multicut_fusionmoves")
        self.solver = solver

    def set_weight(self, weight):
        assert weight > 0.
        self.weight = weight

    def set_ntrees(self, n_trees):
        assert n_trees > 10
        self.n_trees = n_trees

    def set_ignore_mask(self, use_ignore_mask):
        assert isinstance(use_ignore_mask, bool)
        self.use_ignore_mask = use_ignore_mask

    def set_beta_local(self, beta_local):
        assert beta_local > 0. and beta_local < 1.
        self.beta_local = beta_local

    def set_verbose(self, verbose):
        assert isinstance(verbose, bool)
        self.verbose = verbose

    def set_anisotropy(self, anisotropy_factor):
        assert isinstance(anisotropy_factor, float)
        self.anisotropy_factor = anisotropy_factor

    def set_use2d(self, use_2d):
        assert isinstance(use_2d, bool)
        self.use_2d = use_2d

    def set_learn2d(self, learn_2d):
        assert isinstance(learn_2d, bool)
        self.learn_2d = learn_2d

    def set_fuzzy_learning(self, learn_fuzzy):
        assert isinstance(learn_fuzzy, bool)
        self.learn_fuzzy = learn_fuzzy

    def set_negative_threshold(self, negative_threshold):
        assert isinstance(negative_threshold, float)
        self.negative_threshold = negative_threshold

    def set_positive_threshold(self, positive_threshold):
        assert isinstance(positive_threshold, float)
        self.positive_threshold = positive_threshold

    def set_seed_fraction(self, seed_fraction):
        assert seed_fraction <= 1., str(seed_fraction)
        self.seed_fraction = seed_fraction

    def set_num_it(self, num_it):
        assert isinstance(num_it, int)
        self.num_it = num_it

    def set_num_it_stop(self, num_it_stop):
        assert isinstance(num_it_stop, int)
        self.num_it_stop = num_it_stop

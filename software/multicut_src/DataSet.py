import numpy as np
import vigra
import vigra.graphs as graphs
import os
import h5py
from concurrent import futures

import graph as agraph

from Tools import cacher_hdf5


# TODO Flag that tells us, if we have flat or 3d superpixel
#      -> we can assert this every time we need flat superpix for a specific function
class DataSet(object):

    def __init__(self, meta_folder, ds_name, block_coordinates = None):
        if not os.path.exists(meta_folder):
            os.mkdir(meta_folder)

        self.ds_name = ds_name
        self.cache_folder = os.path.join(meta_folder, self.ds_name)

        if not os.path.exists(self.cache_folder):
            os.mkdir(self.cache_folder)

        # Flag if raw data was added
        self.has_raw = False

        # Number of input data
        self.n_inp = 0

        # shape of input data
        self.shape = None

        # Number of segmentations
        self.n_seg = 0

        self.has_gt = False

        self.is_subvolume = False
        if block_coordinates is not None:
            assert len(block_coordinates) == 6
            self.block_coordinates = block_coordinates
            self.is_subvolume = True

        # cutouts, tesselations and inverse cutouts
        self.cutouts   = []
        self.n_cutouts = 0

        self.inverse_cutouts = {}

        # compression method used
        self.compression = 'gzip'

        # maximal anisotropy factor for filter calculation
        self.aniso_max = 20.

        # gt ids to be ignored for positive training examples
        self.gt_false_splits = set()
        # gt ids to be ignored for negative training examples
        self.gt_false_merges = set()


    def __str__(self):
        return self.ds_name


    def add_false_split_gt_id(self, gt_id):
        self.gt_false_splits.add(gt_id)

    def add_false_merge_gt_id(self, gt_id):
        self.gt_false_merges.add(gt_id)

    #
    # Interface for adding inputs, segmentations and groundtruth
    #

    # add the raw_data
    # expects hdf5 input
    # probably better to do normalization in a way suited to the data!
    def add_raw(self, raw_path, raw_key):
        if self.has_raw:
            raise RuntimeError("Rawdata has already been added")
        raw = vigra.readHDF5(raw_path, raw_key).view(np.ndarray)
        assert len(raw.shape) == 3, "Only 3d data supported"
        # for subvolume make sure that boundaries are included
        if self.is_subvolume:
            p = self.block_coordinates
            assert raw.shape[0] >= p[1] and raw.shape[1] >= p[3] and raw.shape[2] >= p[5]
            raw = raw[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        self.shape = raw.shape
        save_path = os.path.join(self.cache_folder,"inp0.h5")
        vigra.writeHDF5(raw, save_path, "data")
        self.has_raw = True
        self.n_inp = 1


    # add the raw_data from np.array
    def add_raw_from_data(self, raw):
        if self.has_raw:
            raise RuntimeError("Rawdata has already been added")
        assert isinstance(raw, np.ndarray)
        assert len(raw.shape) == 3, "Only 3d data supported"
        # for subvolume make sure that boundaries are included
        if self.is_subvolume:
            p = self.block_coordinates
            assert raw.shape[0] >= p[1] and raw.shape[1] >= p[3] and raw.shape[2] >= p[5]
            raw = raw[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        self.shape = raw.shape
        save_path = os.path.join(self.cache_folder,"inp0.h5")
        vigra.writeHDF5(raw, save_path, "data")
        self.has_raw = True
        self.n_inp = 1


    # add additional input map
    # expects hdf5 input
    def add_input(self, inp_path, inp_key):
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before additional pixmaps")
        pixmap = vigra.readHDF5(inp_path,inp_key)
        if self.is_subvolume:
            p = self.block_coordinates
            assert pixmap.shape[0] >= p[1] and pixmap.shape[1] >= p[3] and pixmap.shape[2] >= p[5]
            pixmap = pixmap[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        assert pixmap.shape[:3] == self.shape, "Pixmap shape " + str(pixmap.shape) + "does not match " + str(self.shape)
        save_path = os.path.join(self.cache_folder, "inp" + str(self.n_inp) + ".h5" )
        vigra.writeHDF5(pixmap, save_path, "data")
        self.n_inp += 1


    # add additional input map
    # expects hdf5 input
    def add_input_from_data(self, pixmap):
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before additional pixmaps")
        assert isinstance(pixmap, np.ndarray)
        if self.is_subvolume:
            p = self.block_coordinates
            assert pixmap.shape[0] >= p[1] and pixmap.shape[1] >= p[3] and pixmap.shape[2] >= p[5]
            pixmap = pixmap[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        assert pixmap.shape == self.shape, "Pixmap shape " + str(pixmap.shape) + "does not match " + str(self.shape)
        save_path = os.path.join(self.cache_folder, "inp" + str(self.n_inp) + ".h5" )
        vigra.writeHDF5(pixmap, save_path, "data")
        self.n_inp += 1


    # return input with inp_id (0 corresponds to the raw data)
    def inp(self, inp_id):
        if inp_id >= self.n_inp:
            raise RuntimeError("Trying to read inp_id " + str(inp_id) + " but there are only " + str(self.n_inp) + " input maps")
        inp_path = os.path.join(self.cache_folder,"inp" + str(inp_id) + ".h5")
        return vigra.readHDF5(inp_path, "data").astype('float32')


    # add segmentation of the volume
    # expects hdf5 input
    def add_seg(self, seg_path, seg_key):
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before adding a segmentation")
        seg = vigra.readHDF5(seg_path, seg_key).astype('uint32')
        if self.is_subvolume:
            p = self.block_coordinates
            assert seg.shape[0] >= p[1] and seg.shape[1] >= p[3] and seg.shape[2] >= p[5]
            seg = seg[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        assert seg.shape == self.shape, "Seg shape " + str(seg.shape) + "does not match " + str(self.shape)
        seg = vigra.analysis.labelVolume(seg)
        seg -= seg.min()
        save_path = os.path.join(self.cache_folder, "seg" + str(self.n_seg) + ".h5")
        vigra.writeHDF5(seg, save_path, "data", compression = self.compression)
        self.n_seg += 1


    # add segmentation of the volume
    # expects hdf5 input
    def add_seg_from_data(self, seg):
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before adding a segmentation")
        assert isinstance(seg, np.ndarray)
        if self.is_subvolume:
            p = self.block_coordinates
            assert seg.shape[0] >= p[1] and seg.shape[1] >= p[3] and seg.shape[2] >= p[5]
            seg = seg[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        assert seg.shape == self.shape, "Seg shape " + str(seg.shape) + "does not match " + str(self.shape)
        seg = seg.astype('uint32')
        seg = vigra.analysis.labelVolume(seg.astype(np.uint32))
        seg -= seg.min()
        save_path = os.path.join(self.cache_folder, "seg" + str(self.n_seg) + ".h5")
        vigra.writeHDF5(seg, save_path, "data", compression = self.compression)
        self.n_seg += 1


    # return segmentation with seg_id
    def seg(self, seg_id):
        if seg_id >= self.n_seg:
            raise RuntimeError("Trying to read seg_id " + str(seg_id) + " but there are only " + str(self.n_seg) + " segmentations")
        seg_path = os.path.join(self.cache_folder,"seg" + str(seg_id) + ".h5")
        return vigra.readHDF5(seg_path, "data")


    # only single gt for now!
    # add grountruth
    def add_gt(self, gt_path, gt_key):
        if self.has_gt:
            raise RuntimeError("Groundtruth has already been added")
        gt = vigra.readHDF5(gt_path, gt_key)
        if self.is_subvolume:
            p = self.block_coordinates
            assert gt.shape[0] >= p[1] and gt.shape[1] >= p[3] and gt.shape[2] >= p[5]
            gt = gt[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        assert gt.shape == self.shape, "GT shape " + str(gt.shape) + "does not match " + str(self.shape)
        # FIXME running a label volume might be helpful sometimes, but it can mess up the split and merge ids!
        #gt = vigra.analysis.labelVolumeWithBackground(gt.astype(np.uint32))
        save_path = os.path.join(self.cache_folder,"gt.h5")
        vigra.writeHDF5(gt, save_path, "data", compression = self.compression)
        self.has_gt = True


    # only single gt for now!
    # add grountruth
    def add_gt_from_data(self, gt):
        if self.has_gt:
            raise RuntimeError("Groundtruth has already been added")
        assert isinstance(gt, np.ndarray)
        if self.is_subvolume:
            p = self.block_coordinates
            assert gt.shape[0] >= p[1] and gt.shape[1] >= p[3] and gt.shape[2] >= p[5]
            gt = gt[p[0]: p[1], p[2]: p[3], p[4]: p[5]]
        assert gt.shape == self.shape, "GT shape " + str(gt.shape) + "does not match " + str(self.shape)
        gt = vigra.analysis.labelVolumeWithBackground(gt.astype(np.uint32))
        save_path = os.path.join(self.cache_folder,"gt.h5")
        vigra.writeHDF5(gt, save_path, "data", compression = self.compression)
        self.has_gt = True


    # get the groundtruth
    def gt(self):
        if not self.has_gt:
            raise RuntimeError("Need to add groundtruth first!")
        gt_path = os.path.join(self.cache_folder, "gt.h5")
        return vigra.readHDF5(gt_path, "data")


    # calculate the region adjacency graph of seg_id
    def _rag(self, seg_id):

        filename    = "rag_seg" + str(seg_id) + ".h5"
        ragpath = os.path.join(self.cache_folder, filename)
        rag_key  = "rag"
        if not os.path.isfile(ragpath):
            print "Computing RAG for seg_id", str(seg_id)
            grid = graphs.gridGraph(self.shape[0:3])
            _rag  = graphs.regionAdjacencyGraph(grid, self.seg(seg_id))

            print "WRITING IN ", ragpath, rag_key
            _rag.writeHDF5(ragpath, rag_key)

        else:
            #print "Loading RAG for seg_id", str(seg_id), "from HDF5:"
            #print ragpath
            #print rag_key
            _rag = vigra.graphs.loadGridRagHDF5(ragpath, rag_key)
        return _rag


    # get the segments adjacent to the edges for each edge
    @cacher_hdf5()
    def _adjacent_segments(self, seg_id):
        print "Getting segments adjacent to edges from RAG:"
        rag = self._rag(seg_id)

        adjacent_segs = rag.uvIds()
        adjacent_segs = np.sort(adjacent_segs, axis = 1)
        assert adjacent_segs.shape[0] == rag.edgeIds().shape[0]
        return adjacent_segs


    # get the adjacent edges for each edge
    # TODO can't cache this
    #@cacher_hdf5()
    def _adjacent_edges(self, seg_id):
        print "Getting adjacent edges from RAG:"
        rag = self._rag(seg_id)
        adjacent_edges = []
        for edge in rag.edgeIter():
            adj_to_edge = []
            n_1  = rag.u(edge)
            n_2  = rag.v(edge)
            for n in (n_1,n_2):
                for arc in rag.incEdgeIter(n):
                    new_edge = rag.edgeFromArc(arc)
                    if new_edge != edge:
                        adj_to_edge.append(new_edge.id)
            adjacent_edges.append(adj_to_edge)

        assert len(adjacent_edges) == rag.edgeNum, str(len(adjacent_edges)) + " , " + str(rag.edgeNum)
        return adjacent_edges


    #
    # Feature Calculation
    #

    # TODO integrate julian's to the power of 10 ?!
    # this will be ignorant of using a different segmentation
    @cacher_hdf5(ignoreNumpyArrays=True)
    def distance_transform(self, segmentation, penalty_power = 0, anisotropy = [1.,1.,1.]):

        # # if that does what I think it does (segmentation to edge image), we can use vigra...
        # def pixels_at_boundary(image, axes=[1, 1, 1]):
        #    return axes[0] * ((np.concatenate((image[(0,), :, :], image[:-1, :, :]))
        #                       - np.concatenate((image[1:, :, :], image[(-1,), :, :]))) != 0) \
        #           + axes[1] * ((np.concatenate((image[:, (0,), :], image[:, :-1, :]), 1)
        #                         - np.concatenate((image[:, 1:, :], image[:, (-1,), :]), 1)) != 0) \
        #           + axes[2] * ((np.concatenate((image[:, :, (0,)], image[:, :, :-1]), 2)
        #                         - np.concatenate((image[:, :, 1:], image[:, :, (-1,)]), 2)) != 0)
        #
        # anisotropy = np.array(anisotropy).astype(np.float32)
        # image = image.astype(np.float32)
        # # Compute boundaries
        # # FIXME why ?!
        # axes = (anisotropy ** -1).astype(np.uint8)
        # image = pixels_at_boundary(image, axes)

        edge_volume = np.concatenate(
                [vigra.analysis.regionImageToEdgeImage(segmentation[:,:,z])[:,:,None] for z in xrange(segmentation.shape[2])],
                axis = 2)
        dt = vigra.filters.distanceTransform(edge_volume, pixel_pitch=anisotropy, background=True)
        if penalty_power > 0:
            dt = np.power(dt, penalty_power)
        return dt

    # make pixelfilter for the given input.
    # the sigmas are scaled with the anisotropy factor
    # max. anisotropy factor is 20.
    # if it is higher, the features are calculated purely in 2d
    # TODO make sigmas accessible in a clever way
    def make_filters(self,
            inp_id,
            anisotropy_factor,
            filter_names = [ "gaussianSmoothing",
                             "hessianOfGaussianEigenvalues",
                             "laplacianOfGaussian"],
            sigmas = [1.6, 4.2, 8.3],
            use_fastfilters = True
            ):

        assert anisotropy_factor >= 1., "Finer resolution in z-direction is not supported"

        # FIXME dirty hack to calculate features on the ditance trafo
        # FIXME the dt must be pre-computed for this to work
        if inp_id == 'distance_transform':
            fake_seg = np.zeros((10,10))
            inp = self.distance_transform(fake_seg, [1.,1.,anisotropy_factor])
            input_name = 'distance_transform'
        else:
            assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
            inp = self.inp(inp_id)
            input_name = "inp_" + str(inp_id)

        top_folder = os.path.join(self.cache_folder, "filters")
        if not os.path.exists(top_folder):
            os.mkdir(top_folder)

        # determine, how we calculate the pixfeats (2d, pure 3d or 3d scaled with anisotropy)
        # save filters to corresponding path
        calculation_2d = False

        if anisotropy_factor == 1.:
            filter_folder = os.path.join(top_folder, "filters_3d")
        elif anisotropy_factor >= self.aniso_max:
            filter_folder = os.path.join(top_folder, "filters_2d")
            calculation_2d = True
        else:
            filter_folder = os.path.join(top_folder, "filters_" + str(anisotropy_factor) )

        if not os.path.exists(filter_folder):
            os.makedirs(filter_folder)

        if not calculation_2d and anisotropy_factor > 1. and use_fastfilters:
            print "WARNING: Anisotropic feature calculation not supported in fastfilters yet."
            print "Using vigra filters instead."
            use_fastfilters = False

        if use_fastfilters:
            import fastfilters
            filter_names = [".".join( ("fastfilters", filtname) ) for filtname in filter_names]
        else:
            filter_names = [".".join( ("vigra.filters", filtname) ) for filtname in filter_names]

        # update the filter folder to the input
        filter_folder = os.path.join( filter_folder, input_name )
        if not os.path.exists(filter_folder):
            os.mkdir(filter_folder)
        filter_key    = "data"

        # list of paths to the filters, that will be calculated

        return_paths = []

        # for pure 2d calculation, we only take into account the slices individually
        if calculation_2d:
            print "Calculating Filter in 2d"
            for filt_name in filter_names:
                filter = eval(filt_name)
                for sig in sigmas:

                    # check whether this is already there
                    filt_path = os.path.join(filter_folder, filt_name + "_" + str(sig) )
                    return_paths.append(filt_path)

                    if not os.path.exists(filt_path):

                        # TODO set max_workers with ppl param value!
                        with futures.ThreadPoolExecutor(max_workers = 8) as executor:
                            tasks = []
                            for z in xrange(inp.shape[2]):
                                tasks.append( executor.submit(filter, inp[:,:,z], sig ) )

                        res = [task.result() for task in tasks]

                        if res[0].ndim == 2:
                            res = [re[:,:,None] for re in res]
                        elif res[0].ndim == 3:
                            res = [re[:,:,None,:] for re in res]
                        res = np.concatenate( res, axis = 2)
                        assert res.shape[0:2] == self.shape[0:2]
                        vigra.writeHDF5(res, filt_path, filter_key)

        # TODO we should parallelize over the filters here!
        else:
            print "Calculating filter in 3d, with anisotropy factor:", str(anisotropy_factor)
            for filt_name in filter_names:
                filter = eval(filt_name)
                for sig in sigmas:

                    if anisotropy_factor != 1.:
                        sig = (sig, sig, sig / anisotropy_factor)
                    # check whether this is already there
                    filt_path = os.path.join(filter_folder, filt_name + "_" + str(sig) )
                    return_paths.append(filt_path)

                    if not os.path.exists(filt_path):
                        filter_res = filter( inp, sig )
                        assert filter_res.shape[0:2] == self.shape[0:2]
                        vigra.writeHDF5(filter_res,
                                filt_path,
                                filter_key)

        return_paths.sort()
        return return_paths


    # accumulates the given filter over all edges in the
    # filter has to be given in the correct size!
    # Also Median, 0.25-Quantile, 0.75-Quantile, Kurtosis, Skewness
    # we can pass the rag, because loading it for large datasets takes some time...
    def _accumulate_filter_over_edge(self, seg_id, filt, filt_name, rag = None):
        assert len(filt.shape) in (3,4)
        assert filt.shape[0:3] == self.shape
        if rag == None:
            rag = self._rag(seg_id)
        # split multichannel features
        feats_return = []
        names_return = []
        if len(filt.shape) == 3:
            # let RAG do the work
            gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, filt)
            #edgeFeat_mean = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)[:,np.newaxis]
            edgeFeats     = rag.accumulateEdgeStatistics(gridGraphEdgeIndicator)
            feats_return.append(edgeFeats)
            names_return.append("EdgeFeature_" + filt_name + "_mean")
            names_return.append("EdgeFeature_" + filt_name + "_sum")
            names_return.append("EdgeFeature_" + filt_name + "_min")
            names_return.append("EdgeFeature_" + filt_name + "_max")
            names_return.append("EdgeFeature_" + filt_name + "_variance")
            names_return.append("EdgeFeature_" + filt_name + "_skewness")
            names_return.append("EdgeFeature_" + filt_name + "_kurtosis")
            names_return.append("EdgeFeature_" + filt_name + "_0.1quantile")
            names_return.append("EdgeFeature_" + filt_name + "_0.25quantile")
            names_return.append("EdgeFeature_" + filt_name + "_0.5quantile")
            names_return.append("EdgeFeature_" + filt_name + "_0.75quantile")
            names_return.append("EdgeFeature_" + filt_name + "_0.90quantile")
        elif len(filt.shape) == 4:
            for c in range(filt.shape[3]):
                print "Multichannel feature, accumulating channel:", c + 1, "/", filt.shape[3]
                gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(
                        rag.baseGraph, filt[:,:,:,c] )
                #edgeFeat_mean = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)[:,np.newaxis]
                edgeFeats     = rag.accumulateEdgeStatistics(gridGraphEdgeIndicator)
                feats_return.append(edgeFeats)
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_mean")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_sum")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_min")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_max")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_variance")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_skewness")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_kurtosis")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_0.1quantile")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_0.25quantile")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_0.5quantile")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_0.75quantile")
                names_return.append("EdgeFeature_" + filt_name + "_c" + str(c)  + "_0.90quantile")
        return feats_return, names_return


    # Features from different filters, accumulated over the edges
    # hacked in for features from affinity maps
    @cacher_hdf5("feature_folder", True)
    def edge_features_from_affinity_maps(self, seg_id, inp_id, anisotropy_factor):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        assert anisotropy_factor >= 20., "Affinity map features only for 2d filters."

        import fastfilters
        from concurrent import futures

        filter_names = [ "fastfilters.gaussianSmoothing",
                         "fastfilters.hessianOfGaussianEigenvalues",
                         "fastfilters.laplacianOfGaussian"]
        sigmas = [1.6, 4.2, 8.3]

        #inp = self.inp(inp_id)
        #assert inp.ndim == 4, "Need affinity channels"
        #assert inp.shape[3] == 3, "Need 3 affinity channels"

        #inpXY = np.maximum( inp[:,:,:,0], inp[:,:,:,1] )
        #inpZ  = inp[:,:,:,2]

        rag = self._rag(seg_id)

        inpXY = self.inp(inp_id)
        inpZ = self.inp(inp_id+1)

        edge_indications = self.edge_indications(seg_id)
        edge_features = []

        n = 0
        N = len(filter_names) * len(sigmas)

        for fname in filter_names:
            filter_fu = eval(fname)
            for sigma in sigmas:
                print "Accumulating features:", n, "/", N

                # filters for xy channels
                with futures.ThreadPoolExecutor(max_workers = 35 ) as executor:
                    tasks = []
                    for z in xrange(inpXY.shape[2]):
                        tasks.append( executor.submit(filter_fu, inpXY[:,:,z], sigma ) )
                    filtXY = [task.result() for task in tasks]

                if filtXY[0].ndim == 2:
                    filtXY = np.concatenate([re[:,:,None] for re in filtXY], axis = 2)
                elif filtXY[0].ndim == 3:
                    filtXY = np.concatenate([re[:,:,None,:] for re in filtXY], axis = 2)

                # filters for xy channels
                with futures.ThreadPoolExecutor(max_workers = 20 ) as executor:
                    tasks = []
                    for z in xrange(inpZ.shape[2]):
                        tasks.append( executor.submit(filter_fu, inpZ[:,:,z], sigma ) )
                    filtZ = [task.result() for task in tasks]

                if filtZ[0].ndim == 2:
                    filtZ = np.concatenate([re[:,:,None] for re in filtZ], axis = 2)
                elif filtZ[0].ndim == 3:
                    filtZ = np.concatenate([re[:,:,None,:] for re in filtZ], axis = 2)

                # accumulate over the edge
                featsXY, _ = self._accumulate_filter_over_edge(seg_id, filtXY, "", rag)
                featsXY    = np.concatenate(featsXY, axis = 1)
                featsZ, _  = self._accumulate_filter_over_edge(seg_id, filtZ, "", rag)
                featsZ     = np.concatenate(featsZ,  axis = 1)

                feats = np.zeros_like(featsXY)
                feats[edge_indications==1] = featsXY[edge_indications==1]
                feats[edge_indications==0] = featsZ[edge_indications==0]

                edge_features.append(feats)
                n += 1

        edge_features = np.concatenate( edge_features, axis = 1)
        assert edge_features.shape[0] == len( rag.edgeIds() ), str(edge_features.shape[0]) + " , " +str(len( rag.edgeIds() ))

        # remove NaNs
        edge_features = np.nan_to_num(edge_features)

        return edge_features


    # Features from different filters, accumulated over the edges
    @cacher_hdf5("feature_folder", True)
    def edge_features(self, seg_id, inp_id, anisotropy_factor):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        assert anisotropy_factor >= 1., "Finer resolution in z-direction is nor supported"

        # calculate the volume filters for the given input
        if isinstance(self, Cutout):
            filter_paths = self.make_filters(inp_id, anisotropy_factor, self.ancestor_folder)
        else:
            filter_paths = self.make_filters(inp_id, anisotropy_factor)

        filter_key = "data"

        rag = self._rag(seg_id)

        n = 0
        N = len(filter_paths)

        # iterate over all filters and accumulate the edge features
        edge_features = []
        edge_features_names = []
        for path in filter_paths:
            n += 1
            print "Accumulating features:", n, "/", N
            print "From:", path

            filt = vigra.readHDF5(path, filter_key)
            # check whether the shapes match, otherwise cutout the correct shape
            # this happens in cutouts!
            if filt.shape[0:3] != self.shape[0:3]:
                assert self.is_subvolume, "This should only happen in cutouts!"
                p = self.block_coordinates
                o = self.block_offsets
                filt = filt[p[0]+o[0]:p[1]+o[0],p[2]+o[1]:p[3]+o[1],p[4]+o[2]:p[5]+o[2]]
            # now it gets hacky...
            # for InverseCutouts, we have to remove the not covered part from the filter
            if isinstance(self, InverseCutout):
                p = self.cut_coordinates
                filt[p[0]:p[1],p[2]:p[3],p[4]:p[5]] = 0

            # get the name (string magic....)
            filt_name = os.path.split(path)[1][len("fastfilters."):]

            # accumulate over the edge
            feats_acc, names_acc = self._accumulate_filter_over_edge(seg_id, filt,
                    filt_name, rag)
            edge_features.extend(feats_acc)
            edge_features_names.extend(names_acc)

        edge_features = np.concatenate( edge_features, axis = 1)
        assert edge_features.shape[0] == len( rag.edgeIds() ), str(edge_features.shape[0]) + " , " +str(len( rag.edgeIds() ))

        # save the feature names to file
        save_folder = os.path.join(self.cache_folder, "features")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # clip the anisotropy factor
        if anisotropy_factor >= self.aniso_max:
            anisotropy_factor = self.aniso_max
        save_name = "edge_features_" + str(seg_id) + "_" + str(inp_id) + "_" + str(anisotropy_factor) + ".h5"
        save_file = os.path.join( save_folder, save_name)
        vigra.writeHDF5(edge_features_names, save_file, "edge_features_names")

        # remove NaNs
        edge_features = np.nan_to_num(edge_features)

        return edge_features


    # get the name of the edge features
    def edge_features_names(self, seg_id, inp_id, anisotropy_factor):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        self.edge_features(seg_id, inp_id, anisotropy_factor)

        save_folder = os.path.join(self.cache_folder, "features")
        if anisotropy_factor >= self.aniso_max:
            anisotropy_factor = self.aniso_max
        save_name = "edge_features_" + str(seg_id) + "_" + str(inp_id) + "_" + str(anisotropy_factor) + ".h5"
        save_file = os.path.join( save_folder, save_name)
        assert os.path.exists(save_file)

        return vigra.readHDF5(save_file,"edge_features_names")


    # get region statistics with the vigra region feature extractor
    def _region_statistics(self, seg_id, inp_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)

        # list of the region statistics, that we want to extract
        statistics =  [ "Count", "Kurtosis", #"Histogram",
                        "Maximum", "Minimum", "Quantiles",
                        "RegionRadii", "Skewness", "Sum",
                        "Variance", "Weighted<RegionCenter>", "RegionCenter"]

        extractor = vigra.analysis.extractRegionFeatures(
                self.inp(inp_id).astype(np.float32),
                self.seg(seg_id).astype(np.uint32),
                features = statistics )

        return extractor, statistics


    @cacher_hdf5(folder = "feature_folder", ignoreNumpyArrays=True)
    def region_features(self, seg_id, inp_id, uv_ids, lifted_nh):

        import gc

        if lifted_nh:
            print "Computing Lifted Region Features for NH:", lifted_nh
        else:
            print "Computing Region features for local Edges"

        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        region_statistics, region_statistics_names = self._region_statistics(seg_id, inp_id)

        regStats = []
        regStatNames = []

        for regStatName in region_statistics_names[:9]:
            regStat = region_statistics[regStatName]
            if regStat.ndim == 1:
                regStats.append(regStat[:,None])
            else:
                regStats.append(regStat)
            regStatNames.extend([regStatName for _ in xrange(regStats[-1].shape[1])])
        regStats = np.concatenate(regStats, axis=1)

        regCenters = []
        regCenterNames = []
        for regStatName in  region_statistics_names[9:]:
            regCenter = region_statistics[regStatName]
            if regCenter.ndim == 1:
                regCenters.append(regCenter[:,None])
            else:
                regCenters.append(regCenter)
            regCenterNames.extend([regStatName for _ in xrange(regCenters[-1].shape[1])])
        regCenters = np.concatenate(regCenters, axis=1)

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        del region_statistics
        gc.collect()

        fU = regStats[uv_ids[:,0],:]
        fV = regStats[uv_ids[:,1],:]

        allFeat = [
                np.minimum(fU, fV),
                np.maximum(fU, fV),
                np.abs(fU - fV),
                fU + fV
            ]

        feat_names = []
        feat_names.extend(["RegionFeatures_" + name + combine for combine in  ("_min", "_max", "_absdiff", "_sum") for name in regStatNames  ])

        fV = fV.resize((1,1))
        fU = fU.resize((1,1))
        del fU
        del fV
        gc.collect()

        sU = regCenters[uv_ids[:,0],:]
        sV = regCenters[uv_ids[:,1],:]
        allFeat.append( (sU - sV)**2 )

        feat_names.extend(["RegionFeatures_" + name for name in regCenterNames])

        sV = sV.resize((1,1))
        sU = sU.resize((1,1))
        del sU
        del sV
        gc.collect()

        allFeat = np.concatenate(allFeat, axis = 1)

        assert len(feat_names) == allFeat.shape[1], str(len(feat_names)) + " , " + str(allFeat.shape[1])

        # save feature names
        save_folder = os.path.join(self.cache_folder, "features")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_file = os.path.join(save_folder,
            "region_features_" + str(seg_id) + "_" + str(inp_id) + "_" + str(lifted_nh) + ".h5" )
        vigra.writeHDF5(feat_names, save_file, "region_features_names")
        print "writing feat_names to", save_file

        return np.nan_to_num(allFeat)


    # get the names of the region features
    def region_features_names(self, seg_id, inp_id, lifted_nh):

        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        self.region_features(seg_id, inp_id)

        save_folder = os.path.join(self.cache_folder, "features")
        save_file = os.path.join(save_folder,
            "region_features_" + str(seg_id) + "_" + str(inp_id) + "_" + str(lifted_nh) + ".h5" )
        assert os.path.exists(save_file)

        return vigra.readHDF5(save_file,"region_features_names")


    # Find the number of faces (= connected components), that make up the edge
    # Could find some more features based on the ccs
    @cacher_hdf5()
    def edge_connected_components(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        rag = self._rag(seg_id)
        n_edges = rag.edgeNum

        n_ccs = np.zeros(n_edges)

        for edge_id in range( n_edges ):
            edge_coords = rag.edgeCoordinates(edge_id)
            # need to map grid graph coords to normal coords
            edge_coords = np.floor(edge_coords).astype(np.uint32)
            x_min = np.min(edge_coords[:,0])
            y_min = np.min(edge_coords[:,1])
            z_min = np.min(edge_coords[:,2])
            edge_coords[:,0] -= x_min
            edge_coords[:,1] -= y_min
            edge_coords[:,2] -= z_min
            x_max = np.max(edge_coords[:,0])
            y_max = np.max(edge_coords[:,1])
            z_max = np.max(edge_coords[:,2])
            edge_mask = np.zeros( (x_max + 1, y_max + 1, z_max + 1), dtype = np.uint32 )
            # bring edge_coords in np.where format
            edge_coords = (edge_coords[:,0], edge_coords[:,1], edge_coords[:,2])
            edge_mask[edge_coords] = 1
            ccs = vigra.analysis.labelVolumeWithBackground(edge_mask, neighborhood = 26)
            # - 1, because we have to substract for the background label
            n_ccs[edge_id] = len(np.unique(ccs)) - 1

        return n_ccs


    # find the edget-type indications
    # 0 for z-edges, 1 for xy-edges
    @cacher_hdf5()
    def edge_indications(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        rag = self._rag(seg_id)
        n_edges = rag.edgeNum
        edge_indications = np.zeros(n_edges, dtype = 'uint8')
        uv_ids = rag.uvIds()
        # TODO no loops, no no no loops
        for edge_id in range( n_edges ):
            edge_coords = rag.edgeCoordinates(edge_id)
            z_coords = edge_coords[:,2]
            z = np.unique(z_coords)
            if z.size != 1:
                uv = uv_ids[edge_id]
                if not 0 in uv:
                    assert z.size == 1, "Edge indications can only be calculated for flat superpixel" + str(z)
                else:
                    continue
            # check whether we have a z or a xy edge
            if z - int(z) == 0.:
                # xy-edge!
                edge_indications[edge_id] = 1
            else:
                # z-edge!
                edge_indications[edge_id] = 0
        return edge_indications


    # Features from edge_topology
    @cacher_hdf5("feature_folder")
    def topology_features(self, seg_id, use_2d_edges):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert isinstance( use_2d_edges, bool ), type(use_2d_edges)

        if not use_2d_edges:
            n_feats = 1
        else:
            n_feats = 7

        rag = self._rag(seg_id)

        n_edges = rag.edgeNum
        topology_features = np.zeros( (n_edges, n_feats) )

        # length / area of the edge
        edge_lens = rag.edgeLengths()
        assert edge_lens.shape[0] == n_edges
        topology_features[:,0] = edge_lens
        topology_features_names = ["TopologyFeature_EdgeLength"]

        # deactivated for now, because it segfaults for large ds
        # TODO look into this
        ## number of connected components of the edge
        #n_ccs = self.edge_connected_components(seg_id)
        #assert n_ccs.shape[0] == n_edges
        #topology_features[:,1] = n_ccs
        #topology_features_names = ["TopologyFeature_NumFaces"]

        # extra feats for z-edges in 2,5 d
        if use_2d_edges:

            # edge indications
            edge_indications = self.edge_indications(seg_id)
            assert edge_indications.shape[0] == n_edges
            topology_features[:,1] = edge_indications
            topology_features_names.append("TopologyFeature_xy_vs_z_indication")

            # region sizes to build some features
            statistics =  [ "Count", "RegionCenter" ]

            extractor = vigra.analysis.extractRegionFeatures(
                    self.inp(0).astype(np.float32),
                    self.seg(seg_id).astype(np.uint32),
                    features = statistics )

            z_mask = edge_indications == 0

            sizes = extractor["Count"]
            uvIds = self._adjacent_segments(seg_id)
            sizes_u = sizes[ uvIds[:,0] ]
            sizes_v = sizes[ uvIds[:,1] ]
            # union = size_up + size_dn - intersect
            unions  = sizes_u + sizes_v - edge_lens
            # Union features
            topology_features[:,2][z_mask] = unions[z_mask]
            topology_features_names.append("TopologyFeature_union")
            # IoU features
            topology_features[:,3][z_mask] = edge_lens[z_mask] / unions[z_mask]
            topology_features_names.append("TopologyFeature_intersectionoverunion")

            # segment shape features
            seg_coordinates = extractor["RegionCenter"]
            len_bounds      = np.zeros(rag.nodeNum)
            # TODO no loop ?! or CPP
            # iterate over the nodes, to get the boundary length of each node
            for n in rag.nodeIter():
                node_z = seg_coordinates[n.id][2]
                for arc in rag.incEdgeIter(n):
                    edge = rag.edgeFromArc(arc)
                    edge_c = rag.edgeCoordinates(edge)
                    # only edges in the same slice!
                    if edge_c[0,2] == node_z:
                        len_bounds[n.id] += edge_lens[edge.id]
            # shape feature = Area / Circumference
            shape_feats_u = sizes_u / len_bounds[uvIds[:,0]]
            shape_feats_v = sizes_v / len_bounds[uvIds[:,1]]
            # combine w/ min, max, absdiff
            topology_features[:,4][z_mask] = np.minimum(
                    shape_feats_u[z_mask], shape_feats_v[z_mask])
            topology_features[:,5][z_mask] = np.maximum(
                    shape_feats_u[z_mask], shape_feats_v[z_mask])
            topology_features[:,6][z_mask] = np.absolute(
                    shape_feats_u[z_mask] - shape_feats_v[z_mask])
            topology_features_names.append("TopologyFeature_shapeSegment_min")
            topology_features_names.append("TopologyFeature_shapeSegment_max")
            topology_features_names.append("TopologyFeature_shapeSegment_absdiff")

            # edge shape features
            # this is too hacky, don't use it for now !
            #edge_bounds = np.zeros(rag.edgeNum)
            #adjacent_edges = self._adjacent_edges(seg_id)
            ## TODO no loop or CPP
            #for edge in rag.edgeIter():
            #    edge_coords = rag.edgeCoordinates(edge)
            #    edge_coords_up = np.ceil(edge_coords)
            #    #edge_coords_dn = np.floor(edge_coords)
            #    edge_z = edge_coords[0,2]
            #    for adj_edge_id in adjacent_edges[edge.id]:
            #        adj_coords = rag.edgeCoordinates(adj_edge_id)
            #        # only consider this edge, if it is in the same slice
            #        if adj_coords[0,2] == edge_z:
            #            # find the overlap and add it to the boundary
            #            #adj_coords_up = np.ceil(adj_coords)
            #            adj_coords_dn = np.floor(adj_coords)
            #            # overlaps (set magic...)
            #            ovlp0 = np.array(
            #                    [x for x in set(tuple(x) for x in edge_coords_up[:,:2])
            #                        & set(tuple(x) for x in adj_coords_dn[:,:2])] )
            #            #print edge_coords_up
            #            #print adj_coords_dn
            #            #print ovlp0
            #            #quit()
            #            #ovlp1 = np.array(
            #            #        [x for x in set(tuple(x) for x in edge_coords_dn[:,:2])
            #            #            & set(tuple(x) for x in adj_coords_up[:,:2])])
            #            #assert len(ovlp0) == len(ovlp1), str(len(ovlp0)) + " , " + str(len(ovlp1))
            #            edge_bounds[edge.id] += len(ovlp0)

            ## shape feature = Area / Circumference
            #topology_features[:,7][z_mask] = edge_lens[z_mask] / edge_bounds[z_mask]
            #topology_features_names.append("TopologyFeature_shapeEdge")

        save_folder = os.path.join(self.cache_folder, "features")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_name = "topology_features_" + str(seg_id) + "_" + str(use_2d_edges) + ".h5"
        save_file = os.path.join(save_folder, save_name )
        vigra.writeHDF5(topology_features_names, save_file, "topology_features_names")

        topology_features[np.isinf(topology_features)] = 0.
        topology_features[np.isneginf(topology_features)] = 0.
        topology_features = np.nan_to_num(topology_features)

        return topology_features


    # get the names of the region features
    def topology_features_names(self, seg_id, use_2d_edges):

        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        self.topology_features(seg_id, use_2d_edges)

        save_folder = os.path.join(self.cache_folder, "features")
        save_name = "topology_features_" + str(seg_id) + "_" + str(use_2d_edges) + ".h5"
        save_file = os.path.join(save_folder, save_name )
        assert os.path.exists(save_file)

        return vigra.readHDF5(save_file,"topology_features_names")


    # safely combine features
    # TODO loading the rag may consume some time and concatenate should also catch
    # non-matching shapes, so we could get rid of the asserts
    def combine_features(self, feat_list, seg_id):
        n_edges = self._rag(seg_id).edgeNum
        for f in feat_list:
            assert f.shape[0] == n_edges, str(f.shape[0]) + " , " +  str(n_edges)
        return np.concatenate(feat_list, axis = 1)


    # features based on curvature of xy edges
    # FIXME very naive implementation
    @cacher_hdf5("feature_folder")
    def curvature_features(self, seg_id):
        rag = self._rag(seg_id)
        curvature_feats = np.zeros( (rag.edgeNum, 4) )
        edge_ind = self.edge_indications(seg_id)
        for edge in xrange(rag.edgeNum):
            if edge_ind[edge] == 0:
                continue
            coords = rag.edgeCoordinates(edge)[:,:-1]
            try:
                dx_dt = np.gradient(coords[:,0])
            except IndexError as e:
                #print coords
                continue
            dy_dt = np.gradient(coords[:,1])
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)

            # curvature implemented after:
            # http://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
            curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt + dy_dt * dy_dt)**1.5

            curvature_feats[edge,0] = np.mean(curvature)
            curvature_feats[edge,1] = np.min(curvature)
            curvature_feats[edge,2] = np.max(curvature)
            curvature_feats[edge,3] = np.std(curvature)

        return np.nan_to_num(curvature_feats)


    #
    # Groundtruth projection
    #

    # get the edge labeling from dense groundtruth
    @cacher_hdf5()
    def edge_gt(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert self.has_gt

        rag = self._rag(seg_id)
        node_gt, _ = rag.projectBaseGraphGt( self.gt().astype(np.uint32) )
        assert node_gt.shape[0] == rag.nodeNum, str(node_gt.shape[0]) + " , " +  str(rag.nodeNum)

        uv_ids = self._adjacent_segments(seg_id)
        u_gt = node_gt[ uv_ids[:,0] ]
        v_gt = node_gt[ uv_ids[:,1] ]

        assert u_gt.shape == v_gt.shape
        assert u_gt.shape[0] == rag.edgeNum

        edge_gt = np.zeros( rag.edgeNum )

        # look, where the adjacent nodes are different
        edge_gt[(u_gt != v_gt)] = 1.
        edge_gt[(u_gt == v_gt)] = 0.

        return edge_gt


    # get edge gt from thresholding the overlaps
    # edges with ovlp > positive_threshold are taken as positive trainging examples
    # edges with ovlp < negative_threshold are taken as negative training examples
    @cacher_hdf5()
    def edge_gt_fuzzy(self, seg_id, positive_threshold, negative_threshold):
        assert positive_threshold > 0.5, str(positive_threshold)
        assert negative_threshold < 0.5, str(negative_threshold)
        print negative_threshold, positive_threshold
        edge_overlaps = self.edge_overlaps(seg_id)
        edge_gt_fuzzy = 0.5 * np.ones( edge_overlaps.shape )
        edge_gt_fuzzy[edge_overlaps > positive_threshold] = 1.
        edge_gt_fuzzy[edge_overlaps < negative_threshold] = 0.

        return edge_gt_fuzzy


    # get edge overlaps
    # with values from 0 (= no overlap) to 1 (= max overlap)
    @cacher_hdf5()
    def edge_overlaps(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert self.has_gt

        edge_overlaps = agraph.candidateSegToRagSeg(
                self.seg(seg_id).astype('uint32'),
                self.gt().astype('uint32'),
                self._adjacent_segments(seg_id).astype(np.uint64))

        return edge_overlaps


    # return mask that hides edges that lie between 2 superpixel
    # which are projected to an ignore label
    # -> we don t want to learn on these!
    @cacher_hdf5()
    def ignore_mask(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert self.has_gt
        #need the node gt to determine the gt val of superpixel
        rag = self._rag(seg_id)
        node_gt, _ = rag.projectBaseGraphGt( self.gt().astype(np.uint32) )
        assert node_gt.shape[0] == rag.nodeNum, str(node_gt.shape[0]) + " , " +  str(rag.nodeNum)
        uv_ids = self._adjacent_segments(seg_id)
        ignore_mask = np.zeros( rag.edgeNum, dtype = bool)
        for edge_id in xrange(rag.edgeNum):
            n0 = uv_ids[edge_id][0]
            n1 = uv_ids[edge_id][1]
            # if both superpixel have ignore label in the gt
            # block them in our mask
            if node_gt[n0] in self.gt_false_splits or node_gt[n1] in self.gt_false_splits:
                if node_gt[n0] != node_gt[n1]:
                    ignore_mask[edge_id] = True
            if node_gt[n0] in self.gt_false_merges and node_gt[n1] in self.gt_false_merges:
                ignore_mask[edge_id] = True

        print "IGNORE MASK NONZEROS:", np.sum(ignore_mask)
        return ignore_mask


    # return mask that hides edges that lie between 2 superpixel for lifted edges
    # which are projected to an ignore label
    # -> we don t want to learn on these!
    @cacher_hdf5(ignoreNumpyArrays=True)
    def lifted_ignore_mask(self, seg_id, liftedNh, liftedUvs):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert self.has_gt
        #need the node gt to determine the gt val of superpixel
        rag = self._rag(seg_id)
        node_gt, _ = rag.projectBaseGraphGt( self.gt().astype(np.uint32) )
        assert node_gt.shape[0] == rag.nodeNum, str(node_gt.shape[0]) + " , " +  str(rag.nodeNum)

        numEdges = liftedUvs.shape[0]

        ignore_mask = np.zeros( numEdges, dtype = bool)
        for edge_id in xrange(numEdges):
            n0 = liftedUvs[edge_id][0]
            n1 = liftedUvs[edge_id][1]
            # if both superpixel have ignore label in the gt
            # block them in our mask
            if node_gt[n0] in self.gt_false_splits or node_gt[n1] in self.gt_false_splits:
                if node_gt[n0] != node_gt[n1]:
                    ignore_mask[edge_id] = True
            if node_gt[n0] in self.gt_false_merges and node_gt[n1] in self.gt_false_merges:
                ignore_mask[edge_id] = True

        print "IGNORE MASK NONZEROS:", np.sum(ignore_mask)
        return ignore_mask


    # get the projection of the gt to the segmentation
    @cacher_hdf5()
    def seg_gt(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert self.has_gt

        rag = self._rag(seg_id)
        seg = self.seg(seg_id)
        node_gt, _ = rag.projectBaseGraphGt( self.gt().astype(np.uint32) )
        assert node_gt.shape[0] == rag.nodeNum, str(node_gt.shape[0]) + " , " +  str(rag.nodeNum)

        seg_gt = rag.projectLabelsToBaseGraph(node_gt)
        assert seg_gt.shape == self.shape

        return seg_gt.astype(np.uint32)


    # get the projection of a multicut result to the segmentation
    def project_mc_result(self, seg_id, mc_node):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        rag = self._rag(seg_id)
        assert mc_node.shape[0] == rag.nodeNum, str(mc_node.shape[0]) + " , " + str(rag.nodeNum)

        mc_seg = rag.projectLabelsToBaseGraph(mc_node.astype(np.uint32))
        assert mc_seg.shape == self.shape

        return mc_seg.astype(np.uint32)


    #
    # Convenience functions for Cutouts and Tesselation
    #


    # make a cutout of the given block shape
    # need to update the ds in the MetaSet after this!
    def make_cutout(self, block_coordinates, block_offsets = [0,0,0]):
        assert self.has_raw, "Need at least raw data to make a cutout"
        assert len(block_coordinates) == 6
        assert block_coordinates[1] <= self.shape[0] and block_coordinates[3] <= self.shape[1] and block_coordinates[5] <= self.shape[2], str(block_coordinates) + " , " + str(self.shape)

        cutout_name = self.ds_name + "_cutout_" + str(self.n_cutouts)
        ancestor_folder = self.cache_folder
        if isinstance(self, Cutout):
            ancestor_folder = self.ancestor_folder
        cutout = Cutout(self.cache_folder, cutout_name, block_coordinates, ancestor_folder, block_offsets)

        # copy all inputs, segs and the gt to cutuout
        for inp in range(self.n_inp):
            inp_path = os.path.join(self.cache_folder,"inp" + str(inp) + ".h5")
            if isinstance(self, Cutout):
                inp_path = self.inp_path[inp]
            if inp == 0:
                cutout.add_raw(inp_path)
            else:
                cutout.add_input(inp_path)

        for seg_id in range(self.n_seg):
            seg_path = os.path.join(self.cache_folder,"seg" + str(seg_id) + ".h5")
            cutout.add_seg(seg_path, "data")

        if self.has_gt:
            gt_path = os.path.join(self.cache_folder,"gt.h5")
            cutout.add_gt(gt_path, "data")

        for false_merge_gt in self.gt_false_merges:
            cutout.add_false_merge_gt_id(false_merge_gt)

        for false_split_gt in self.gt_false_splits:
            cutout.add_false_split_gt_id(false_split_gt)

        self.n_cutouts += 1
        self.cutouts.append(cutout)


    def get_cutout(self, cutout_id):
        assert cutout_id < self.n_cutouts, str(cutout_id) + " , " + str(self.n_cutouts)
        return self.cutouts[cutout_id]


    def make_inverse_cutout(self, cut_id):
        assert self.has_raw, "Need at least raw data to make a cutout"
        assert cut_id < self.n_cutouts, "Cutout was not done yet!"
        assert not cut_id in self.inverse_cutouts.keys(), "Inverse Cutout is already there!"
        inv_name = self.ds_name + "_invcut_" + str(cut_id)
        ancestor_folder = self.cache_folder
        if isinstance(self, Cutout):
            ancestor_folder = self.ancestor_folder

        cut_coordinates = self.get_cutout(cut_id).block_coordinates

        inv_cut = InverseCutout(self.cache_folder, inv_name,
                cut_id, cut_coordinates, self.shape, ancestor_folder)

        # copy all inputs, segs and the gt to the inverse cutout
        for inp in range(self.n_inp):
            inp_path = os.path.join(self.cache_folder,"inp" + str(inp) + ".h5")
            if inp == 0:
                inv_cut.add_raw(inp_path)
            else:
                inv_cut.add_input(inp_path)

        for seg_id in range(self.n_seg):
            seg_path = os.path.join(self.cache_folder,"seg" + str(seg_id) + ".h5")
            inv_cut.add_seg(seg_path)

        if self.has_gt:
            gt_path = os.path.join(self.cache_folder,"gt.h5")
            inv_cut.add_gt(gt_path)

        self.inverse_cutouts[cut_id] = inv_cut


    def get_inverse_cutout(self, cut_id):
        assert cut_id in self.inverse_cutouts.keys(), "InverseCutout not produced yet"
        return self.inverse_cutouts[cut_id]



#cutout from a given Dataset, used for cutouts and tesselations
#calls the cache of the parent dataset for inp, seg, gt and filtercalls the cache of the parent dataset for inp, seg, gt and filter
class Cutout(DataSet):

    def __init__(self, meta_folder, ds_name, block_coordinates, ancestor_folder, block_offsets):
        super(Cutout, self).__init__(meta_folder, ds_name, block_coordinates )

        self.inp_path = []
        self.shape = (self.block_coordinates[1] - self.block_coordinates[0],
                self.block_coordinates[3] - self.block_coordinates[2],
                self.block_coordinates[5] - self.block_coordinates[4])

        self.block_offsets = block_offsets

        # this is the cache folder of the "oldest ancestor",
        # i.e. of the top dataset that is not a cutout or tesselation
        # we need it for make_filters
        self.ancestor_folder = ancestor_folder

        # we need to copy the ignore masks


    # fot the inputs, we dont need to cache everythin again, however for seg and gt we have to, because otherwise the segmentations are not consecutive any longer

    # add path to the raw data from original ds
    # expects hdf5 input
    def add_raw(self, raw_path):
        if self.has_raw:
            raise RuntimeError("Rawdata has already been added")
        assert os.path.exists(raw_path), raw_path
        shape = vigra.readHDF5(raw_path, "data").shape
        assert len(shape) == 3, "Only 3d data supported"
        # for subvolume make sure that boundaries are included
        p = self.block_coordinates
        assert shape[0] >= p[1] and shape[1] >= p[3] and shape[2] >= p[5]
        self.inp_path.append(raw_path)
        self.has_raw = True
        self.n_inp = 1


    # add path to input from original ds
    # expects hdf5 input
    def add_input(self, inp_path):
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before additional pixmaps")
        shape = vigra.readHDF5(inp_path, "data").shape
        p = self.block_coordinates
        assert shape[0] >= p[1] and shape[1] >= p[3] and shape[2] >= p[5]
        self.inp_path.append(inp_path)
        self.n_inp += 1


    # return input with inp_id (0 corresponds to the raw data)
    def inp(self, inp_id):
        if inp_id >= self.n_inp:
            raise RuntimeError("Trying to read inp_id " + str(inp_id) + " but there are only " + str(self.n_inp) + " input maps")
        inp_path = self.inp_path[inp_id]
        p = self.block_coordinates
        o = self.block_offsets
        return vigra.readHDF5(inp_path, "data")[p[0]+o[0]:p[1]+o[0],p[2]+o[1]:p[3]+o[1],p[4]+o[2]:p[5]+o[2]]


    # seg and gt can't be reimplemented that way, because they need to be connected!

    # we get the paths to the filters of the top dataset
    def make_filters(self, inp_id, anisotropy_factor, ancestor_folder):
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        assert anisotropy_factor >= 1., "Finer resolution in z-direction is nor supported"

        top_ds_folder = self.cache_folder
        # keep splitting the path, until we get to the meta folder
        # then we know, that we have reached the cache folder for the parent dataset
        while top_ds_folder != ancestor_folder:
            top_ds_folder, sub_folder = os.path.split(top_ds_folder)
        filter_folder = os.path.join(top_ds_folder, "filters")

        # determine, how we calculate the pixfeats (2d, pure 3d or 3d scaled with anisotropy)
        # save filters to corresponding path
        calculation_2d = False

        if anisotropy_factor == 1.:
            filter_folder = os.path.join(filter_folder, "filters_3d")
        elif anisotropy_factor >= self.aniso_max:
            filter_folder = os.path.join(filter_folder, "filters_2d")
            calculation_2d = True
        else:
            filter_folder = os.path.join(filter_folder, "filters_" + str(anisotropy_factor) )

        filter_folder = os.path.join(filter_folder,"inp_" + str(inp_id))

        assert os.path.exists(filter_folder), "Call make_filters of the parent DataSet, before calling it in the cutout!"

        # get all the files in the filter folder
        filter_paths = []
        for file in os.listdir(filter_folder):
            filter_paths.append( os.path.join(filter_folder,file) )

        # sort to make this consistent!
        filter_paths.sort()

        return filter_paths


# the inverse of a cutout
# implemented for crossvalidation mc style
class InverseCutout(Cutout):

    def __init__(self, meta_folder, inv_cut_name, cut_id,
            cut_coordinates, vol_shape, ancestor_folder):
        self.cut_id = cut_id
        self.cut_coordinates = cut_coordinates
        self.shape = vol_shape

        self.cache_folder = os.path.join(meta_folder, inv_cut_name)
        self.ds_name = inv_cut_name
        assert not os.path.exists(self.cache_folder), "This InverseCutout already exists"
        os.mkdir(self.cache_folder)
        self.ancestor_folder = ancestor_folder

        # have to set this to be consistent with top classes
        self.is_subvolume = False
        self.aniso_max = 20.
        self.compression = 'gzip'

        # we cant call the init of Cutout, so we have to redefinde these
        self.has_raw   = False
        self.n_inp     = 0
        self.inp_paths = []

        self.n_seg = 0
        self.has_gt = 0


    def add_raw(self, raw_path):
        assert not self.has_raw, "Rawdata has already been added!"
        with h5py.File(raw_path) as f:
            h5_ds = f["data"]
            shape = h5_ds.shape
        assert shape[0] >= self.shape[0] and shape[1] >= self.shape[1] and shape[2] >= self.shape[2]
        self.inp_paths.append(raw_path)
        self.n_inp = 1
        self.has_raw = True


    def add_input(self, inp_path):
        assert self.has_raw, "Add Rawdata first!"
        with h5py.File(inp_path) as f:
            h5_ds = f["data"]
            shape = h5_ds.shape
        assert shape[0] >= self.shape[0] and shape[1] >= self.shape[1] and shape[2] >= self.shape[2]
        self.inp_paths.append(inp_path)
        self.n_inp += 1


    def inp(self, inp_id):
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        inp = vigra.readHDF5(self.inp_paths[inp_id],"data")
        p = self.cut_coordinates
        inp[p[0]:p[1],p[2]:p[3],p[4]:p[5]] = 0
        return inp


    def add_seg(self, seg_path):
        assert self.has_raw, "Add Rawdata first!"
        seg = vigra.readHDF5(seg_path, "data")
        shape = seg.shape
        assert shape[0] >= self.shape[0] and shape[1] >= self.shape[1] and shape[2] >= self.shape[2]

        # zero  is reserved for the 'empty' part of the volume
        if 0 in seg:
            seg += 1

        p = self.cut_coordinates
        seg[p[0]:p[1],p[2]:p[3],p[4]:p[5]] = 0
        seg = vigra.analysis.labelVolume(seg)
        seg -= 1

        save_path = os.path.join(self.cache_folder, "seg" + str(self.n_seg) + ".h5")
        vigra.writeHDF5(seg, save_path, "data", compression = self.compression)
        self.n_seg += 1


    def seg(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        save_path = os.path.join(self.cache_folder, "seg" + str(seg_id) + ".h5")
        return vigra.readHDF5(save_path, "data")


    def add_gt(self, gt_path):
        assert not self.has_gt, "GT already exists!"
        gt = vigra.readHDF5(gt_path, "data")
        assert gt.shape[0] >= self.shape[0] and gt.shape[1] >= self.shape[1] and gt.shape[2] >= self.shape[2]
        p = self.cut_coordinates
        gt[p[0]:p[1],p[2]:p[3],p[4]:p[5]] = 0
        gt = vigra.analysis.labelVolumeWithBackground(gt.astype(np.uint32))
        save_path = os.path.join(self.cache_folder, "gt" + ".h5")
        vigra.writeHDF5(gt, save_path, "data", compression = self.compression)
        self.has_gt = True


    def gt(self):
        assert self.has_gt
        save_path = os.path.join(self.cache_folder, "gt" + ".h5")
        gt = vigra.readHDF5(save_path, "data")
        return gt


    # returns ids of the edges that are artificially introduced
    # by the inverse cutout
    @cacher_hdf5()
    def get_artificial_edges(self,seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        artificial_edge_ids = []
        uv_ids = self._adjacent_segments(seg_id)
        for edge_id in xrange(uv_ids.shape[0]):
            # the zero label is reserved for the region not covered by this inv cutout
            # so all edges linking to it are introduced by the cutout
            if uv_ids[edge_id,0] == 0 or uv_ids[edge_id,1] == 0:
                artificial_edge_ids.append(edge_id)
        return artificial_edge_ids


    # in addition to the 2 ignore labels, we also ignore all edges
    # with the artificial boundaries
    @cacher_hdf5()
    def ignore2ignorers(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert self.has_gt
        #need the node gt to determine the gt val of superpixel
        rag = self._rag(seg_id)
        node_gt, _ = rag.projectBaseGraphGt( self.gt().astype(np.uint32) )
        assert node_gt.shape[0] == self._rag(seg_id).nodeNum, str(node_gt.shape[0]) + " , " +  str(rag.nodeNum)
        uv_ids = self._adjacent_segments(seg_id)
        artificial_edges = self.get_artificial_edges(seg_id)
        ignore_mask = np.ones( rag.edgeNum, dtype = bool)
        for edge_id in xrange(rag.edgeNum):
            n0 = uv_ids[edge_id][0]
            n1 = uv_ids[edge_id][1]
            # if both superpixel have ignore label in the gt
            # block them in our mask
            # or if this edge is artificial
            if (node_gt[n0] == 0 and node_gt[n1] == 0) or edge_id in artificial_edges:
                ignore_mask[edge_id] = False
        return ignore_mask


    # there's a bunch of methods, that cant be called
    # from inverse cutout, cause they don't make sense!
    def make_cutout(self, block_coordinates):
        raise AttributeError("Can't be called for InverseCutout")

    def get_cutout(self, cutout_id):
        raise AttributeError("Can't be called for InverseCutout")

    def make_inverse_cutout(self, block_coordinates):
        raise AttributeError("Can't be called for InverseCutout")

    def get_inverse_cutout(self, cutout_id):
        raise AttributeError("Can't be called for InverseCutout")

    def make_tesselation(self, block_shape, n_blocks):
        raise AttributeError("Can't be called for InverseCutout")

    def get_tesselation(self, tesselation_id):
        raise AttributeError("Can't be called for InverseCutout")


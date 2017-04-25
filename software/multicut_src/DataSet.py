import numpy as np
import vigra
import vigra.graphs as graphs
import os
import shutil
import h5py
from concurrent import futures
import itertools
import cPickle as pickle

from tools import cacher_hdf5, cache_name, find_matching_row_indices
from ExperimentSettings import ExperimentSettings

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty # conda version build with cplex
    except ImportError:
        try:
            import nifty_wit_gurobi as nifty # conda version build with gurobi
        except ImportError:
            raise ImportError("No valid nifty version was found.")

# this can be used in 2 different ways:
# ds_name = None: -> called with cache folder and loads from there
# ds_name = string -> called with meta_folder and descends into cache folder
def load_dataset(meta_folder, ds_name = None):
    assert os.path.exists(meta_folder), meta_folder
    if ds_name is None:
        cache_folder = meta_folder
    else:
        cache_folder = os.path.join(meta_folder, ds_name)
        assert os.path.exists(cache_folder)
    ds_obj_path = os.path.join(cache_folder, 'ds_obj.pkl')
    assert os.path.exists(ds_obj_path), ds_obj_path
    with open(ds_obj_path) as f:
        return pickle.load(f)


def is_inp_name(file_name):
    if file_name.startswith('seg'):
        return True
    elif file_name.startswith('inp'):
        return True
    elif file_name.startswith('gt'):
        return True
    elif file_name.startswith('defect_mask'):
        return True
    else:
        return False


# TODO use nifty cgp features instead + add curvature
def _topology_features_impl(rag, seg, edge_indications, edge_lens):
    extra_features = np.zeros( (rag.edgeNum, 6), dtype = 'float32' )

    # edge indications and mask for z edges
    extra_features[:,0] = edge_indications

    extractor = vigra.analysis.extractRegionFeatures(
            np.zeros_like(seg, dtype = 'float32'),
            seg,
            features = ['Count', 'RegionCenter'] )

    sizes = extractor["Count"]
    uv_ids = np.sort(rag.uvIds(), axis = 1)
    sizes_u = sizes[ uv_ids[:,0] ]
    sizes_v = sizes[ uv_ids[:,1] ]

    # Features: Union + Intersection over Union
    unions  = sizes_u + sizes_v - edge_lens
    extra_features[:,1] = unions
    extra_features[:,2] = edge_lens / unions

    # Segment shape features
    seg_coordinates = extractor["RegionCenter"]
    len_bounds      = np.zeros(rag.nodeNum, dtype = 'float32')
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
    shape_feats_u = sizes_u / len_bounds[uv_ids[:,0]]
    shape_feats_v = sizes_v / len_bounds[uv_ids[:,1]]
    # combine w/ min, max, absdiff
    extra_features[:,3] = np.minimum(shape_feats_u, shape_feats_v)
    extra_features[:,4] = np.maximum(shape_feats_u, shape_feats_v)
    extra_features[:,5] = np.absolute(shape_feats_u - shape_feats_v)

    # the feature names
    extra_names = [ 'TopologyFeatures_' + name for name in \
            ['indication', 'union', 'IoU', 'shapeSegment_min', 'shapeSegment_max', 'shapeSegment_absdiff'] ]

    extra_features[np.isinf(extra_features)] = 0.
    extra_features[np.isneginf(extra_features)] = 0.
    extra_features = np.nan_to_num(extra_features)

    return extra_features, extra_names


class DataSet(object):

    def __init__(self, meta_folder, ds_name):
        if not os.path.exists(meta_folder):
            os.mkdir(meta_folder)

        self.ds_name = ds_name
        self.cache_folder = os.path.join(meta_folder, self.ds_name)

        if not os.path.exists(self.cache_folder):
            os.mkdir(self.cache_folder)

        # paths to external input data
        self.external_gt_path   = None
        self.external_inp_paths = []
        self.external_seg_paths = []
        # segmentation mask -> learning + inference will be restriced to
        # superpixels in this mask
        self.external_seg_mask_path = None
        self.external_defect_mask_path = None

        # additional flags
        self.has_raw = False
        self._has_defects = False

        # keys to external data
        self.external_raw_key  = None
        self.external_gt_key   = None
        self.external_inp_keys = []
        self.external_seg_keys = []
        self.external_seg_mask_key = None
        self.exrernal_defect_mask_key = None

        # shape of input data
        self.shape = None

        # paths to cutouts
        self.cutouts = []

        # gt ids to be ignored for positive training examples
        self.gt_false_splits = set()
        # gt ids to be ignored for negative training examples
        self.gt_false_merges = set()


    def __str__(self):
        return self.ds_name

    #
    # Properties to check for inp-data etc.
    #

    @property
    def has_defects(self):
        return self._has_defects if self.external_defect_mask_path != None else False

    @property
    def has_gt(self):
        return self.external_gt_path != None

    @property
    def n_inp(self):
        return len(self.external_inp_paths)

    @property
    def n_seg(self):
        return len(self.external_seg_paths)

    @property
    def has_seg_mask(self):
        return self.external_seg_mask_path != None

    @property
    def n_cutouts(self):
        return len(self.cutouts)

    #
    # Saving the dataset, and clearing caches
    #

    def save(self):
        obj_save_path = os.path.join(self.cache_folder, 'ds_obj.pkl')
        with open(obj_save_path, 'w') as f:
            pickle.dump(self, f)

    def clear_all_caches(self):
        self.clear_inputs()
        self.clear_regular_caches()
        for inp_id in range(self.n_inp):
            self.clear_filters(inp_id)

    # clear all external inputs
    def clear_inputs(self):
        cache_names = os.listdir(self.cache_folder)
        for nn in cache_names:
            pp = os.path.join(self.cache_folder, nn)
            if os.path.isdir(pp):
                continue
            elif is_inp_name(nn):
                os.reomve(pp)
            else:
                continue
        for cut_id in range(self.n_cutouts):
            cut = self.get_cutout(cut_id)
            cut.clear_inputs()

    # clear cache that is not external
    def clear_regular_caches(self):
        cache_names = os.listdir(self.cache_folder)
        for nn in cache_names:
            pp = os.path.join(self.cache_folder, nn)
            if os.path.isdir(pp):
                if nn == 'features':
                    shutil.rmtree(pp)
                else:
                    continue
            elif not is_inp_name(nn):
                os.remove(pp)
            else:
                continue
        for cut_id in range(self.n_cutouts):
            cut = self.get_cutout(cut_id)
            cut.clear_regular_caches()
        self.save()

    def clear_filters(self, inp_id):
        filter_folder = os.path.join(self.cache_folder, 'filters')
        if os.path.exists(filter_folder):
            for sub_folder in os.listdir(filter_folder):
                subsub = os.path.join(filter_folder, sub_folder)
                # This will only work for inp_ids <= 9
                for inp_folder in os.listdir(subsub):
                    if inp_folder[-1] == inp_id:
                        shutil.rmtree(os.path.join(subsub, inp_folder))

    #
    # replace input data that was already added
    #

    def replace_raw(self, new_path, new_key, clear_cache = True):
        assert self.has_raw
        internal_raw_path = os.path.join(self.cache_folder, 'inp0.h5')
        if os.path.exists(internal_raw_path):
            os.remove(internal_raw_path)
        self.external_inp_paths[0] = new_path
        self.external_inp_keys[0]  = new_key
        if clear_cache:
            self.clear_regular_caches()
            self.clear_filters(0)
        self.save()
        # we need to make the cache to be compatible with cutouts that load it
        self.inp(0)

    def replace_raw_from_data(self, new_data, clear_cache = True):
        assert self.has_raw
        internal_raw_path = os.path.join(self.cache_folder, 'inp0.h5')
        if os.path.exists(internal_raw_path):
            os.remove(internal_raw_path)
        vigra.writeHDF5(new_data, internal_raw_path, 'data')
        self.external_inp_paths[0] = internal_raw_path
        self.external_inp_keys[0]  = 'data'
        if clear_cache:
            self.clear_regular_caches()
            self.clear_filters(0)
        self.save()

    def replace_inp(self, inp_id, new_path, new_key):
        assert inp_id > 0, "Use replace raw instead"
        assert inp_id < self.n_inp
        internal_inp_path = os.path.join(self.cache_folder, 'inp%i.h5' % inp_id)
        if os.path.exists(internal_inp_path):
            os.remove(internal_inp_path)
        self.external_inp_paths[inp_id] = new_path
        self.external_inp_keys[inp_id]  = new_key
        self.clear_regular_caches()
        self.clear_filters(inp_id)
        self.save()
        # we need to make the cache to be compatible with cutouts that load it
        self.inp(inp_id)

    def replace_inp_from_data(self, inp_id, new_data, clear_cache = True):
        assert inp_id > 0, "Use replace raw instead"
        assert inp_id < self.n_inp
        internal_inp_path = os.path.join(self.cache_folder, 'inp%i.h5' % inp_id)
        if os.path.exists(internal_inp_path):
            os.remove(internal_inp_path)
        vigra.writeHDF5(new_data, internal_inp_path, 'data')
        self.external_inp_paths[inp_id] = internal_inp_path
        self.external_inp_keys[inp_id]  = 'data'
        if clear_cache:
            self.clear_regular_caches()
            self.clear_filters(inp_id)
        self.save()

    def replace_seg(self, seg_id, new_path, new_key, clear_cache = True):
        assert seg_id < self.n_seg
        internal_seg_path = os.path.join(self.cache_folder, 'seg%i.h5' % seg_id)
        if os.path.exists(internal_seg_path):
            os.remove(internal_seg_path)
        self.external_seg_paths[seg_id] = new_path
        self.external_seg_keys[seg_id]  = new_key
        if clear_cache:
            self.clear_regular_caches()
        self.save()
        # we need to make the cache to be compatible with cutouts that load it
        self.seg(seg_id)

    def replace_seg_from_data(self, seg_id, new_data, clear_cache = True):
        assert seg_id < self.n_seg
        internal_seg_path = os.path.join(self.cache_folder, 'seg%i.h5' % seg_id)
        if os.path.exists(internal_seg_path):
            os.remove(internal_seg_path)
        vigra.writeHDF5(new_data, internal_seg_path, 'data', compression = ExperimentSettings().compression)
        self.external_seg_paths[seg_id] = internal_seg_path
        self.external_seg_keys[seg_id]  = 'data'
        if clear_cache:
            self.clear_regular_caches()
        self.save()

    def replace_gt(self, new_path, new_key, clear_cache = True):
        assert self.has_gt
        internal_gt_path = os.path.join(self.cache_folder, 'gt.h5')
        if os.path.exists(internal_gt_path):
            os.remove(internal_gt_path)
        self.external_gt_path = new_path
        self.external_gt_key  = new_key
        if clear_cache:
            self.clear_regular_caches()
        self.save()
        # we need to make the cache to be compatible with cutouts that load it
        self.gt()

    def replace_gt_from_data(self, new_data, clear_cache = True):
        assert self.has_gt
        internal_gt_path = os.path.join(self.cache_folder, 'gt.h5')
        if os.path.exists(internal_gt_path):
            os.remove(internal_gt_path)
        vigra.writeHDF5(new_data, internal_gt_path, 'data', compression = ExperimentSettings().compression)
        self.external_gt_path = internal_gt_path
        self.external_gt_key  = 'data'
        if clear_cache:
            self.clear_regular_caches()
        self.save()

    def replace_seg_mask(self, new_path, new_key, clear_cache = True):
        assert self.has_seg_mask
        internal_mask_path = os.path.join(self.cache_folder, 'seg_mask.h5')
        if os.path.exists(internal_mask_path):
            os.remove(internal_mask_path)
        self.external_mask_path = new_path
        self.external_mask_key  = new_key
        if clear_cache:
            self.clear_regular_caches()
        self.save()
        # we need to make the cache to be compatible with cutouts that load it
        self.seg_mask()

    def replace_seg_mask_from_data(self, new_data, clear_cache = True):
        assert self.has_seg_mask
        internal_mask_path = os.path.join(self.cache_folder, 'seg_mask.h5')
        if os.path.exists(internal_mask_path):
            os.remove(internal_mask_path)
        vigra.writeHDF5(new_data, internal_mask_path, 'data', compression = ExperimentSettings().compression)
        self.external_mask_path = internal_mask_path
        self.external_mask_key  = 'data'
        if clear_cache:
            self.clear_regular_caches()
        self.save()

    def replace_defect_mask(self, new_path, new_key):
        assert self.external_defect_mask_path != None
        internal_mask_path = os.path.join(self.cache_folder, 'defect_mask.h5')
        if os.path.exists(internal_mask_path):
            os.remove(internal_mask_path)
        self.external_mask_path = new_path
        self.external_mask_key  = new_key
        self.clear_regular_caches()
        self.save()
        # we need to make the cache to be compatible with cutouts that load it
        self.defect_mask()

    def replace_defect_mask_from_data(self, new_data, clear_cache = True):
        assert self.has_seg_mask
        internal_mask_path = os.path.join(self.cache_folder, 'defect_mask.h5')
        if os.path.exists(internal_mask_path):
            os.remove(internal_mask_path)
        vigra.writeHDF5(new_data, internal_mask_path, 'data', compression = ExperimentSettings().compression)
        self.external_mask_path = internal_mask_path
        self.external_mask_key  = 'data'
        if clear_cache:
            self.clear_regular_caches()
        self.save()

    #
    # Masking
    #

    def add_false_split_gt_id(self, gt_id):
        self.gt_false_splits.add(gt_id)
        self.save()

    def add_false_merge_gt_id(self, gt_id):
        self.gt_false_merges.add(gt_id)
        self.save()

    #
    # Adding and loading external input data
    #

    # FIXME: Doesn't work for internal paths as key (e.g., key = 'a/b/data')
    def _check_input(self, path, key):
        assert os.path.exists(path), path
        with h5py.File(path) as f:
            assert key in f.keys(), "%s, %s" % (key, f.keys())

    def _check_shape(self, path, key):
        with h5py.File(path) as f:
            assert f[key].shape == self.shape, "%s: %s, %s" % (path, str(f[key].shape), str(self.shape) )

    def add_raw(self, raw_path, raw_key):
        if self.has_raw:
            raise RuntimeError("Rawdata has already been added")
        self._check_input(raw_path, raw_key)
        with h5py.File(raw_path) as f:
            assert len(f[raw_key].shape) == 3, "Only 3d data supported"
            if not isinstance(self, Cutout):
                self.shape = f[raw_key].shape
        self.external_inp_paths.append(raw_path)
        self.external_inp_keys.append(raw_key)
        self.has_raw = True
        self.save()

    def add_raw_from_data(self, raw):
        assert isinstance(raw, np.ndarray)
        if self.has_raw:
            raise RuntimeError("Rawdata has already been added")
        self.shape = raw.shape
        internal_raw_path = os.path.join(self.cache_folder, 'inp0.h5')
        internal_inp_path = os.path.join(self.cache_folder, 'inp%i.h5' % self.n_inp)
        vigra.writeHDF5(raw, internal_raw_path, 'data')
        self.external_inp_paths.append(internal_raw_path)
        self.external_inp_keys.append('data')
        self.has_raw = True
        self.save()

    def add_input(self, inp_path, inp_key):
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before additional pixmaps")
        self._check_input(inp_path, inp_key)
        if not isinstance(self, Cutout): # don't check for cutouts
            self._check_shape(inp_path, inp_key)
        self.external_inp_paths.append(inp_path)
        self.external_inp_keys.append(inp_key)
        self.save()

    def add_input_from_data(self, pixmap):
        assert isinstance(pixmap, np.ndarray)
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before additional pixmaps")
        assert pixmap.shape[:3] == self.shape, "%s, %s" % (str(pixmap.shape), str(self.shape))
        internal_inp_path = os.path.join(self.cache_folder, 'inp%i.h5' % self.n_inp)
        vigra.writeHDF5(pixmap, internal_inp_path, 'data')
        self.external_inp_paths.append(internal_inp_path)
        self.external_inp_keys.append('data')
        self.save()

    # FIXME TODO we don't really need to cache the raw data at all, but keep this for legacy for now
    def inp(self, inp_id):
        assert inp_id < self.n_inp, "Trying to read inp_id %i but there are only %i input maps" % (inp_id, self.n_inp)
        internal_inp_path = os.path.join(self.cache_folder, 'inp%i.h5' % inp_id)
        if os.path.exists(internal_inp_path): # this is already cached
            return vigra.readHDF5(internal_inp_path, "data").astype('float32')
        else: # this is cached for the first time, load from external data
            external_path = self.external_inp_paths[inp_id]
            external_key = self.external_inp_keys[inp_id]
            inp_data = vigra.readHDF5(external_path, external_key)
            vigra.writeHDF5(inp_data, internal_inp_path, 'data')
            return inp_data.astype('float32')

    def _process_seg(self, seg):
        if isinstance(self, Cutout):
            seg = seg[self.bb]
        assert seg.shape == self.shape, "Seg shape " + str(seg.shape) + "does not match " + str(self.shape)
        if self.has_seg_mask:
            print "Cutting segmentation mask from seg"
            mask = self.seg_mask()
            assert ExperimentSettings().ignore_seg_value == 0, "Only zero ignore value supported for now" # TODO change once we allow more general values
            seg[ np.logical_not(mask) ] = ExperimentSettings().ignore_seg_value
            seg, _, _ = vigra.analysis.relabelConsecutive( seg.astype('uint32'),
                    start_label = 1,
                    keep_zeros = True)
        else:
            seg, _, _ = vigra.analysis.relabelConsecutive(seg.astype('uint32'), start_label = 0, keep_zeros = False)
        return seg

    def add_seg(self, seg_path, seg_key):
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before adding a segmentation")
        self._check_input(seg_path, seg_key)
        self.external_seg_paths.append(seg_path)
        self.external_seg_keys.append(seg_key)
        self.save()

    def add_seg_from_data(self, seg):
        assert isinstance(seg, np.ndarray)
        if not self.has_raw:
            raise RuntimeError("Add Rawdata before adding a segmentation")
        internal_seg_path = os.path.join(self.cache_folder, 'seg%i.h5' % self.n_seg)
        seg = self._process_seg(seg)
        vigra.writeHDF5(seg, internal_seg_path, 'data', compression = ExperimentSettings().compression)
        self.external_seg_paths.append(internal_seg_path)
        self.external_seg_keys.append('data')
        self.save()

    def seg(self, seg_id):
        assert seg_id < self.n_seg, "Trying to read seg_id %i but there are only %i segmentations" % (seg_ind, self.n_seg)
        internal_seg_path = os.path.join(self.cache_folder,"seg%i.h5" % seg_id)
        if os.path.exists(internal_seg_path):
            return vigra.readHDF5(internal_seg_path, "data")
        else:
            seg = vigra.readHDF5(self.external_seg_paths[seg_id], self.external_seg_keys[seg_id])
            seg = self._process_seg(seg)
            vigra.writeHDF5(seg, internal_seg_path, 'data', compression = ExperimentSettings().compression)
            self.save()
            return seg

    def _process_gt(self, gt):
        if isinstance(self, Cutout):
            gt = gt[self.bb]
        assert gt.shape == self.shape, "GT shape " + str(gt.shape) + "does not match " + str(self.shape)
        # FIXME running a label volume might be helpful sometimes, but it can mess up the split and merge ids!
        # also messes up defects in cremi...
        #gt = vigra.analysis.labelVolumeWithBackground(gt.astype(np.uint32))
        #gt -= gt.min()
        return gt

    # only single gt for now!
    # add grountruth
    def add_gt(self, gt_path, gt_key):
        if self.has_gt:
            raise RuntimeError("Groundtruth has already been added")
        self._check_input(gt_path, gt_key)
        self.external_gt_path = gt_path
        self.external_gt_key  = gt_key
        self.save()


   # add grountruth
    def add_gt_from_data(self, gt):
        assert isinstance(gt, np.ndarray)
        if self.has_gt:
            raise RuntimeError("Groundtruth has already been added")
        gt = self._process_gt(gt)
        internal_gt_path = os.path.join(self.cache_folder, 'gt.h5')
        vigra.writeHDF5(gt, internal_gt_path, 'data', compression = ExperimentSettings().compression)
        self.external_gt_path = internal_gt_path
        self.external_gt_key  = 'data'
        self.save()


    # get the groundtruth
    def gt(self):
        if not self.has_gt:
            raise RuntimeError("Need to add groundtruth first!")
        internal_gt_path = os.path.join(self.cache_folder, 'gt.h5')
        if os.path.exists(internal_gt_path):
            return vigra.readHDF5(internal_gt_path, "data")
        else:
            gt = vigra.readHDF5(self.external_gt_path, self.external_gt_key)
            gt = self._process_gt(gt)
            vigra.writeHDF5(gt, internal_gt_path, 'data', compression = ExperimentSettings().compression)
            return gt


    def add_seg_mask(self, mask_path, mask_key):
        assert self.has_raw
        assert not self.has_seg_mask
        self._check_input(mask_path, mask_key)
        if not isinstance(self, Cutout): # don't check for cutouts
            self._check_shape(mask_path, mask_key)
        self.external_seg_mask_path = mask_path
        self.external_seg_mask_key  = mask_key
        self.save()


    def add_seg_mask_from_data(self, mask):
        assert self.has_raw
        assert not self.has_seg_mask
        if not isinstance(self, Cutout): # don't check for cutouts
            assert mask.shape == self.shape
        internal_mask_path = os.path.join(self.cache_folder, 'seg_mask.h5')
        vigra.writeHDF5(mask, internal_mask_path, 'data', compression = ExperimentSettings().compression)
        self.external_seg_mask_path = internal_mask_path
        self.external_seg_mask_key  = 'data'
        self.save()


    def seg_mask(self):
        assert self.has_seg_mask
        internal_mask_path = os.path.join(self.cache_folder, 'seg_mask.h5')
        if os.path.exists(internal_mask_path):
            return vigra.readHDF5(internal_mask_path, 'data')
        else:
            mask = vigra.readHDF5(self.external_seg_mask_path, self.external_seg_mask_key)
            assert all( np.unique(mask) == np.array([0,1]) ), str(np.unique(mask))
            if isinstance(self, Cutout):
                mask = mask[self.bb]
                assert mask.shape == self.shape, str(mask.shape) + " , " + str(self.shape)
            vigra.writeHDF5(mask, internal_mask_path, 'data', compression = ExperimentSettings().compression)
            return mask


    def add_defect_mask(self, mask_path, mask_key):
        assert self.external_defect_mask_path == None
        self._check_input(mask_path, mask_key)
        if not isinstance(self, Cutout): # don't check for cutouts
            self._check_shape(mask_path, mask_key)
        self.external_defect_mask_path = mask_path
        self.external_defect_mask_key  = mask_key
        self.save()


    def add_defect_mask_from_data(self, mask):
        assert self.external_defect_mask_path == None
        if not isinstance(self, Cutout): # don't check for cutouts
            assert mask.shape == self.shape
        internal_mask_path = os.path.join(self.cache_folder, 'defect_mask.h5')
        vigra.writeHDF5(mask, internal_mask_path, 'data', compression = ExperimentSettings().compression)
        self.external_defect_mask_path = internal_mask_path
        self.external_defect_mask_key  = 'data'
        self.save()


    def defect_mask(self):
        assert self.external_defect_mask_path != None
        internal_mask_path = os.path.join(self.cache_folder, 'defect_mask.h5')
        if os.path.exists(internal_mask_path):
            return vigra.readHDF5(internal_mask_path, 'data')
        else:
            mask = vigra.readHDF5(self.external_defect_mask_path, self.external_defect_mask_key)
            assert all( np.unique(mask) == np.array([0,1]) ), str(np.unique(mask))
            if isinstance(self, Cutout):
                mask = mask[self.bb]
                assert mask.shape == self.shape, str(mask.shape) + " , " + str(self.shape)
            vigra.writeHDF5(mask, internal_mask_path, 'data', compression = ExperimentSettings().compression)
            return mask


    #
    # General functionality
    #

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


    # TODO use nifty rag for everything, for this, need to
    # -> change all inputs (2.5 data) from xyz to zyx, for not having to transpose (though need to transpose for vigra functions then)
    # seg can be passed as an optional argument to avoid loading it
    def nifty_rag(self, seg_id, seg = None):
        save_path = os.path.join(self.cache_folder, "nifty_rag%i" % seg_id)
        if seg is None:
            seg = self.seg(seg_id)
        if not os.path.exists(save_path):
            # FIXME don't need transposing once all the inputs have the correct shape
            seg_ = np.ascontiguousarray(seg.transpose((2,1,0)))
            rag = nifty.graph.rag.gridRag(seg_, numberOfThreads = ExperimentSettings().n_threads)
            serialization = rag.serialize()
            vigra.writeHDF5(serialization, save_path, 'data')
        else:
            # FIXME don't need transposing once all the inputs have the correct shape
            seg_ = np.ascontiguousarray(seg.transpose((2,1,0)))
            serialization = vigra.readHDF5(save_path, 'data')
            rag = nifty.graph.rag.gridRag(seg_,
                    serialization = serialization,
                    numberOfThreads = ExperimentSettings().n_threads)
        return rag

    @cacher_hdf5()
    def nifty_to_vigra(self,seg_id):
        nifty_rag = self.nifty_rag(seg_id)
        uvs_nifty = nifty_rag.uvIds()
        uvs_vigra = self._adjacent_segments(seg_id)
        assert uvs_nifty.shape == uvs_vigra.shape
        matches = find_matching_row_indices(uvs_nifty, uvs_vigra)
        assert len(matches) == len(uvs_nifty)
        return matches[:,0]

    @cacher_hdf5()
    def vigra_to_nifty(self,seg_id):
        nifty_rag = self.nifty_rag(seg_id)
        uvs_nifty = nifty_rag.uvIds()
        uvs_vigra = self._adjacent_segments(seg_id)
        assert uvs_nifty.shape == uvs_vigra.shape
        matches = find_matching_row_indices(uvs_vigra, uvs_nifty)
        assert len(matches) == len(uvs_nifty)
        return matches[:,0]

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


    # TODO replace is_2d_stacked with flag from ds, once this is implemented
    # calculates the eccentricity centers for given seg_id
    @cacher_hdf5()
    def eccentricity_centers(self, seg_id, is_2d_stacked):
        seg = self.seg(seg_id)
        if is_2d_stacked: # if we have a stacked segmentation, we can parallelize over the slices

            # calculate the centers for a 2d slice
            def centers_2d(z):
                seg_z = seg[:,:,z]
                min_label = seg_z.min()
                # eccentricity centers expect a consecutive labeling -> we only return the relevant part
                centers = vigra.filters.eccentricityCenters(seg_z)[min_label:]
                return [cent + (z,) for cent in centers] # extend by z coordinate

            with futures.ThreadPoolExecutor(max_workers = ExperimentSettings().n_threads) as executor:
                tasks = [executor.submit(centers_2d, z) for z in xrange(seg.shape[2])]
                centers = [t.result() for t in tasks]
                # return flattened list
            centers_list = list(itertools.chain(*centers))
            n_segs = seg.max() + 1
            assert len(centers_list) == n_segs, "%i, %i" % (len(centers_list), n_segs)
            return centers_list
        else:
            return vigra.filters.eccentricityCenters(seg)


    #
    # Feature Calculation
    #

    # make pixelfilter for the given input.
    # the sigmas are scaled with the anisotropy factor
    # max. anisotropy factor is 20.
    # if it is higher, the features are calculated purely in 2d
    def make_filters(self,
            inp_id,
            anisotropy_factor,
            filter_names = [ "gaussianSmoothing",
                             "hessianOfGaussianEigenvalues",
                             "laplacianOfGaussian"],
            sigmas = [1.6, 4.2, 8.3]
            ):

        assert anisotropy_factor >= 1., "Finer resolution in z-direction is not supported"
        print "Calculating filters for input id:", inp_id

        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        input_name = "inp_" + str(inp_id)

        top_folder = os.path.join(self.cache_folder, "filters")
        if not os.path.exists(top_folder):
            os.mkdir(top_folder)

        # determine, how we calculate the pixfeats (2d, pure 3d or 3d scaled with anisotropy)
        # save filters to corresponding path
        calculation_2d = False

        if anisotropy_factor == 1.:
            filter_folder = os.path.join(top_folder, "filters_3d")
        elif anisotropy_factor >= ExperimentSettings().aniso_max:
            filter_folder = os.path.join(top_folder, "filters_2d")
            calculation_2d = True
        else:
            filter_folder = os.path.join(top_folder, "filters_" + str(anisotropy_factor) )

        if not os.path.exists(filter_folder):
            os.makedirs(filter_folder)

        if not calculation_2d and anisotropy_factor > 1.:
            print "Anisotropic feature calculation not supported in fastfilters yet."
            print "Using vigra filters instead."
            filter_names = [".".join( ("vigra.filters", filtname) ) for filtname in filter_names]
        else:
            filter_names = [".".join( ("fastfilters", filtname) ) for filtname in filter_names]

        # update the filter folder to the input
        filter_folder = os.path.join( filter_folder, input_name )
        if not os.path.exists(filter_folder):
            os.mkdir(filter_folder)
        filter_key    = "data"

        # list of paths to the filters, that will be calculated
        return_paths = []

        # first pass over the paths to check if we have to compute anything
        filter_and_sigmas_to_compute = []
        for filt_name in filter_names:
            for sig in sigmas:
                # check whether this is already there
                filt_path = os.path.join(filter_folder, "%s_%f" % (filt_name, sig) )
                return_paths.append(filt_path)
                if not os.path.exists(filt_path):
                    filter_and_sigmas_to_compute.append((filt_name, sig))

        if filter_and_sigmas_to_compute:
            import fastfilters # very weird, if we built nifty with debug, this causes a segfault oO
            inp = self.inp(inp_id)

            def _calc_filter_2d(filter_fu, sig, filt_path):
                filt_name = os.path.split(filt_path)[1].split(".")[-2].split('_')[0] # some string gymnastics to recover the name
                is_singlechannel = True if filt_name != "hessianOfGaussianEigenvalues" else False
                f_shape = inp.shape if is_singlechannel else inp.shape + (2,)
                chunks  = ( min(512,inp.shape[0]), min(512,inp.shape[2]), 1 ) if is_singlechannel else ( min(512,inp.shape[0]), min(512,inp.shape[2]), 1, 2)
                filter_res = np.zeros(f_shape, dtype = 'float32')
                for z in xrange(inp.shape[2]):
                    filter_res[:,:,z] = filter_fu(inp[:,:,z], sig)
                with h5py.File(filt_path) as f:
                    f.create_dataset(filter_key, data = filter_res, chunks = chunks)


            def _calc_filter_3d(filter_fu, sig, filt_path):
                filter_res = filter_fu( inp, sig )
                with h5py.File(filt_path) as f:
                    f.create_dataset(filter_key, data = filter_res, chunks = True)

            if calculation_2d:
                print "Calculating Filter in 2d"
                _calc_filter = _calc_filter_2d
            else:
                print "Calculating filter in 3d, with anisotropy factor:", str(anisotropy_factor)
                if anisotropy_factor != 1.:
                    sigmas = [(sig, sig, sig / anisotropy_factor) for sig in sigmas]
                _calc_filter = _calc_filter_3d

            #n_workers = 1
            n_workers = ExperimentSettings().n_threads
            with futures.ThreadPoolExecutor(max_workers = n_workers) as executor:
                tasks = []
                for filt_name, sigma in filter_and_sigmas_to_compute:
                    filter_fu = eval(filt_name)
                    filt_path = os.path.join(filter_folder, "%s_%f" % (filt_name, sigma) )
                    tasks.append( executor.submit(_calc_filter, filter_fu, sig, filt_path) )
                res = [t.result() for t in tasks]

        return_paths.sort()
        return return_paths


    # accumulates the given filter over all edges
    def _accumulate_filter_over_edge(self, seg_id, filt, filt_name, rag = None):
        assert len(filt.shape) in (3,4)
        assert filt.shape[0:3] == self.shape, "%s, %s" % (str(filt.shape), str(self.shape))

        # suffixes for the feature names in the correct order
        suffixes = ["mean", "sum", "min", "max", "variance", "skewness", "kurtosis",
                "0.1quantile", "0.25quantile", "0.5quantile", "0.75quantile", "0.90quantile"]

        if rag == None:
            rag = self._rag(seg_id)

        feats_return = []
        names_return = []
        if len(filt.shape) == 3:
            gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, filt)
            feats_return.append(rag.accumulateEdgeStatistics(gridGraphEdgeIndicator))
            names_return.extend( [ "_".join(["EdgeFeature", filt_name, suffix ]) for suffix in suffixes ] )
        elif len(filt.shape) == 4:
            for c in range(filt.shape[3]):
                print "Multichannel feature, accumulating channel:", c + 1, "/", filt.shape[3]
                gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(
                        rag.baseGraph, filt[:,:,:,c] )
                feats_return.append(rag.accumulateEdgeStatistics(gridGraphEdgeIndicator))
                names_return.extend( [ "_".join(["EdgeFeature", filt_name, "c%i" % c, suffix ]) for suffix in suffixes ] )
        return np.concatenate(feats_return, axis = 1), names_return


    # accumulates the given filter over all edges
    def _accumulate_filter_over_edge_with_nifty(
            self,
            seg_id,
            filt,
            filt_name,
            z_direction,
            rag = None):

        assert filt.ndim in (3,4)
        assert filt.shape[0:3] == self.shape, "%s, %s" % (str(filt.shape), str(self.shape))

        # suffixes for the feature names in the correct order
        suffixes = ["mean", "variance", "skewness", "kurtosis","min",
                "0.1quantile", "0.25quantile", "0.5quantile", "0.75quantile", "0.90quantile","max"]

        # TODO don't need to transpose once we use nifty as default and transpose all inps
        # nifty shapes are reversed
        if filt.ndim == 3:
            filt = filt.transpose((2,1,0))
        elif filt.ndim == 4:
            filt = filt.transpose((2,1,0,3))
        filt = np.ascontiguousarray(filt)

        n_threads = ExperimentSettings().n_threads
        #n_threads = 1

        if rag == None:
            rag = self.nifty_rag(seg_id)
        # split multichannel features
        feats_return = []
        names_return = []
        if len(filt.shape) == 3:
            min_val = filt.min()
            max_val = filt.max()
            feats_return.append(
                    nifty.graph.rag.accumulateEdgeFeaturesFlat(rag, filt, min_val, max_val, z_direction, n_threads) )
            names_return.extend( [ "_".join(["EdgeFeature", filt_name, suffix ]) for suffix in suffixes ] )
        elif len(filt.shape) == 4:
            for c in range(filt.shape[3]):
                print "Multichannel feature, accumulating channel:", c + 1, "/", filt.shape[3]
                filt_c = filt[...,c]
                min_val = filt_c.min()
                max_val = filt_c.max()
                feats_return.append(
                        nifty.graph.rag.accumulateEdgeFeaturesFlat(rag, filt_c, min_val, max_val, z_direction, n_threads) )
                names_return.extend( [ "_".join(["EdgeFeature", filt_name, "c%i" % c, suffix ]) for suffix in suffixes ] )
        return np.concatenate(feats_return, axis = 1), names_return


    # filters from affinity maps for xy and z edges
    # z-direction determines how the z features from z edges are accumulated:
    # 0 -> features are accumulated both from z and z + 1
    # 1 -> features are accumulated only from z
    # 2 -> features are accumulated only from z + 1
    @cacher_hdf5("feature_folder", cache_edgefeats=True)
    def edge_features_from_affinity_maps(self, seg_id, inp_ids, anisotropy_factor, z_direction):
        assert z_direction in (0,1,2)
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert anisotropy_factor >= 20., "Affinity map features only for 2d filters."

        assert len(inp_ids) == 2
        assert inp_ids[0] < self.n_inp
        assert inp_ids[1] < self.n_inp

        rag = self.nifty_rag(seg_id)
        vigra_order = self.nifty_to_vigra(seg_id)
        assert len(vigra_order) == rag.numberOfEdges

        paths_xy = self.make_filters(inp_ids[0], anisotropy_factor)
        paths_z  = self.make_filters(inp_ids[1], anisotropy_factor)

        edge_indications = self.edge_indications(seg_id)
        edge_features = []

        N = len(paths_xy)
        for ii, path_xy in enumerate(paths_xy):
            print "Accumulating features:", ii, "/", N
            path_z = paths_z[ii]

            # accumulate over the xy channel
            with h5py.File(path_xy) as f:
                filtXY = f['data'][self.bb] if isinstance(self, Cutout) else f['data'][:]
            print "computing XY from", path_xy
            featsXY, _ = self._accumulate_filter_over_edge_with_nifty(seg_id, filtXY, "", z_direction, rag)
            # project back to vigra graph order
            featsXY = featsXY[vigra_order]

            # accumulate over the z channel
            with h5py.File(path_z) as f:
                filtZ = f['data'][self.bb] if isinstance(self, Cutout) else f['data'][:]
            print "computing Z from", path_z
            featsZ, _  = self._accumulate_filter_over_edge_with_nifty(seg_id, filtZ, "", z_direction, rag)
            # project back to nifty graph order
            featsZ = featsZ[vigra_order]

            # merge the feats
            featsXY[edge_indications==0] = featsZ[edge_indications==0]
            edge_features.append(featsXY)

        edge_features = np.concatenate( edge_features, axis = 1)
        assert edge_features.shape[0] == rag.numberOfEdges, "%i, %i" % (edge_features.shape[0], rag.numberOfEdges)
        return np.nan_to_num(edge_features)


    # Features from different filters, accumulated over the edges
    @cacher_hdf5("feature_folder", True)
    def edge_features(self, seg_id, inp_id, anisotropy_factor):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        assert anisotropy_factor >= 1., "Finer resolution in z-direction is nor supported"

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

            # load the precomputed filter from file
            with h5py.File(path) as f:
                filt = f[filter_key][self.bb] if isinstance(self, Cutout) else f[filter_key][:]

            # accumulate over the edge
            feats_acc, names_acc = self._accumulate_filter_over_edge(
                    seg_id,
                    filt,
                    os.path.split(path)[1],
                    rag)
            edge_features.extend(feats_acc)
            edge_features_names.extend(names_acc)

        edge_features = np.concatenate( edge_features, axis = 1)
        assert edge_features.shape[0] == len( rag.edgeIds() ), str(edge_features.shape[0]) + " , " +str(len( rag.edgeIds() ))

        # save the feature names to file
        save_file = cache_name('edge_features', 'feature_folder', False, True, self, seg_id, inp_id, anisotropy_factor)
        vigra.writeHDF5(edge_features_names, save_file, "edge_features_names")

        return np.nan_to_num(edge_features)


    # get the name of the edge features
    def edge_features_names(self, seg_id, inp_id, anisotropy_factor):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)
        self.edge_features(seg_id, inp_id, anisotropy_factor)
        save_file = cache_name('edge_features', 'feature_folder', False, True, self, seg_id, inp_id, anisotropy_factor)
        assert os.path.exists(save_file)
        return vigra.readHDF5(save_file,"edge_features_names")


    # get region statistics with the vigra region feature extractor
    @cacher_hdf5(folder = "feature_folder")
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

        node_features = np.concatenate(
            [extractor[stat_name][:,None].astype('float32') if extractor[stat_name].ndim == 1 else extractor[stat_name].astype('float32') for stat_name in statistics],
            axis = 1)

        reg_stat_names = list(itertools.chain.from_iterable(
            [ [stat_name for _ in xrange(extractor[stat_name].shape[1])] if extractor[stat_name].ndim>1 else [stat_name] for stat_name in statistics[:9] ] ))

        reg_center_names = list(itertools.chain.from_iterable(
            [ [stat_name for _ in xrange(extractor[stat_name].shape[1])] for stat_name in statistics[9:] ] ))

        # this is the number of node feats that are combined with min, max, sum, absdiff
        # in conrast to center feats, which are combined with euclidean distance
        n_stat_feats = 17 # magic_nu...

        save_path = cache_name("_region_statistics", "feature_folder", False, False, self, seg_id, inp_id)

        vigra.writeHDF5(node_features[:,:n_stat_feats], save_path, "region_statistics")
        vigra.writeHDF5(reg_stat_names, save_path, "region_statistics_names")
        vigra.writeHDF5(node_features[:,n_stat_feats:], save_path, "region_centers")
        vigra.writeHDF5(reg_center_names, save_path, "region_center_names")

        return statistics


    @cacher_hdf5(folder = "feature_folder", ignoreNumpyArrays=True)
    def region_features(self, seg_id, inp_id, uv_ids, lifted_nh):

        import gc

        if lifted_nh:
            print "Computing Lifted Region Features for NH:", lifted_nh
        else:
            print "Computing Region features for local Edges"

        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert inp_id < self.n_inp, str(inp_id) + " , " + str(self.n_inp)

        # make sure the region statistics are calcuulated
        self._region_statistics(seg_id, inp_id)
        region_statistics_path = cache_name("_region_statistics", "feature_folder", False, False, self, seg_id, inp_id)

        # if we have a segmentation mask, we don't calculate features for uv-ids which
        # include 0 (== everything outside of the mask)
        # otherwise the ram consumption for the lmc can blow up...
        if self.has_seg_mask:
            where_uv = (uv_ids != ExperimentSettings().ignore_seg_value).all(axis = 1)
            # for lifted edges assert that no ignore segments are in lifted uvs
            if lifted_nh:
                assert np.sum(where_uv) == where_uv.size
            else:
                uv_ids = uv_ids[where_uv]

        # compute feature from region statistics
        regStats = vigra.readHDF5(region_statistics_path, 'region_statistics')
        regStatNames = vigra.readHDF5(region_statistics_path, 'region_statistics_names')

        fU = regStats[uv_ids[:,0],:]
        fV = regStats[uv_ids[:,1],:]

        allFeat = [
                np.minimum(fU, fV),
                np.maximum(fU, fV),
                np.abs(fU - fV),
                fU + fV
            ]

        feat_names = []
        feat_names.extend(
                ["RegionFeatures_" + name + combine for combine in  ("_min", "_max", "_absdiff", "_sum") for name in regStatNames  ])

        # we actively delete stuff we don't need to free memory
        # because this may become memory consuming for lifted edges
        fV = fV.resize((1,1))
        fU = fU.resize((1,1))
        regStats = regStats.resize((1,))
        del fU
        del fV
        del regStats
        gc.collect()

        # compute features from region centers
        regCenters = vigra.readHDF5(region_statistics_path, 'region_centers')
        regCenterNames = vigra.readHDF5(region_statistics_path, 'region_center_names')

        sU = regCenters[uv_ids[:,0],:]
        sV = regCenters[uv_ids[:,1],:]
        allFeat.append( (sU - sV)**2 )

        feat_names.extend(["RegionFeatures_" + name for name in regCenterNames])

        sV = sV.resize((1,1))
        sU = sU.resize((1,1))
        regCenters = regCenters.resize((1,1))
        del sU
        del sV
        del regCenters
        gc.collect()

        allFeat = np.nan_to_num(
                np.concatenate(allFeat, axis = 1) )

        assert allFeat.shape[0] == uv_ids.shape[0]
        assert len(feat_names) == allFeat.shape[1], str(len(feat_names)) + " , " + str(allFeat.shape[1])

        # if we have excluded the ignore segments before, we need to reintroduce
        # them now to keep edge numbering consistent
        if self.has_seg_mask and not lifted_nh:
            where_ignore = np.logical_not( where_uv )
            n_ignore = np.sum(where_ignore)
            newFeat = np.zeros(
                    (allFeat.shape[0] + n_ignore, allFeat.shape[1]),
                    dtype = 'float32' )
            newFeat[where_uv] = allFeat
            allFeat = newFeat
            assert allFeat.shape[0] == uv_ids.shape[0] + n_ignore

        # save feature names
        save_folder = os.path.join(self.cache_folder, "features")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_file = os.path.join(save_folder,
            "region_features_" + str(seg_id) + "_" + str(inp_id) + "_" + str(lifted_nh) + ".h5" )
        vigra.writeHDF5(feat_names, save_file, "region_features_names")
        print "writing feat_names to", save_file

        return allFeat


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


    @cacher_hdf5()
    def node_z_coord(self, seg_id):
        rag = self._rag(seg_id)
        labels = rag.labels
        labels = labels.squeeze()
        nz = np.zeros(rag.maxNodeId +1, dtype='uint32')
        for z in range(labels.shape[2]):
            lz = labels[:,:,z]
            nz[lz] = z
        return nz


    # find the edge-type indications
    # 0 for z-edges, 1 for xy-edges
    @cacher_hdf5()
    def edge_indications(self, seg_id):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)

        # TODO test this and use instead
        # need to include some of the checks from below, too
        #node_z = self.node_z_coord(seg_id)
        #uv_ids = self._adjacent_segments(seg_id)
        #z_u = node_z[uv_ids[:,0]]
        #z_u = node_z[uv_ids[:,1]]
        ## xy edges (same z coordinate are) set to 1
        ## z  edges (different z coordinates) set to 0
        #return (z_u != z_v).astype('uint8')

        rag = self._rag(seg_id)
        n_edges = rag.edgeNum
        edge_indications = np.zeros(n_edges, dtype = 'uint8')
        uv_ids = rag.uvIds()

        # loop over the edges and check whether they are xy or z by checking the edge coords
        for edge_id in xrange(n_edges):
            edge_coords = rag.edgeCoordinates(edge_id)
            z = np.unique(edge_coords[:,2])
            if z.size > 1:
                uv = uv_ids[edge_id]
                if ExperimentSettings().ignore_seg_value in uv: # the ignore segment can be in multiple slices for flat sp
                    continue
                else:
                    assert False, "Edge indications can only be calculated for flat superpixel" + str(z)
            z = z[0]
            # check whether we have a z (-> 0) or a xy edge (-> 1)
            edge_indications[edge_id] = 1 if (z - int(z) == 0.) else 0

        return edge_indications


    # Features from edge_topology
    @cacher_hdf5("feature_folder")
    def topology_features(self, seg_id, use_2d_edges):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert isinstance( use_2d_edges, bool ), type(use_2d_edges)

        rag = self._rag(seg_id)

        # length / area of the edge
        topo_feats      = rag.edgeLengths()[:,None].astype('float32')
        topo_feat_names = ["TopologyFeatures_EdgeLengths"]

        # extra feats for z-edges in 2,5 d
        if use_2d_edges:
            extra_feats, extra_names = _topology_features_impl(
                    rag,
                    self.seg(seg_id),
                    self.edge_indications(seg_id),
                    topo_feats.squeeze() )
            topo_feats = np.concatenate([topo_feats, extra_feats], axis = 1)
            topo_feat_names.extend(extra_names)

        save_file = cache_name('topology_features', 'feature_folder', False, False, self, seg_id, use_2d_edges)
        vigra.writeHDF5(topo_feat_names, save_file, "topology_features_names")

        return topo_feats


    # get the names of the region features
    def topology_features_names(self, seg_id, use_2d_edges):
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        self.topology_features(seg_id, use_2d_edges)
        save_file = cache_name('topology_features', 'feature_folder', False, False, self, seg_id, use_2d_edges)
        return vigra.readHDF5(save_file,"topology_features_names")

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
        # this fails for non-consecutive gt, which however should not be a problem
        #assert node_gt.shape[0] == rag.nodeNum, str(node_gt.shape[0]) + " , " +  str(rag.nodeNum)

        uv_ids = self._adjacent_segments(seg_id)
        u_gt = node_gt[ uv_ids[:,0] ]
        v_gt = node_gt[ uv_ids[:,1] ]

        assert u_gt.shape == v_gt.shape
        assert u_gt.shape[0] == rag.edgeNum
        return (u_gt != v_gt).astype('uint8')


    # get edge gt from thresholding the overlaps
    # edges with ovlp > positive_threshold are taken as positive trainging examples
    # edges with ovlp < negative_threshold are taken as negative training examples
    @cacher_hdf5()
    def edge_gt_fuzzy(self, seg_id, positive_threshold, negative_threshold):
        assert positive_threshold > 0.5, str(positive_threshold)
        assert negative_threshold < 0.5, str(negative_threshold)
        uv_ids = self._adjacent_segments(seg_id)
        overlaps = nifty.groundtruth.Overlap(
                uv_ids.max(),
                self.seg(seg_id),
                self.gt() )
        edge_overlaps = overlaps.differentOverlaps(uv_ids)
        edge_gt_fuzzy = 0.5 * np.ones( edge_overlaps.shape, dtype = 'float32')
        edge_gt_fuzzy[edge_overlaps > positive_threshold] = 1.
        edge_gt_fuzzy[edge_overlaps < negative_threshold] = 0.
        return edge_gt_fuzzy


    # return mask that hides edges that lie between 2 superpixel
    # which are projected to an ignore label
    # -> we don t want to learn on these!
    @cacher_hdf5(ignoreNumpyArrays=True)
    def ignore_mask(self, seg_id, uv_ids, with_defects = False): # with defects only  for caching
        assert seg_id < self.n_seg, str(seg_id) + " , " + str(self.n_seg)
        assert self.has_gt
        #need the node gt to determine the gt val of superpixel
        rag = self._rag(seg_id)
        node_gt, _ = rag.projectBaseGraphGt( self.gt().astype(np.uint32) )
        assert node_gt.shape[0] == rag.nodeNum, str(node_gt.shape[0]) + " , " +  str(rag.nodeNum)
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
    def lifted_ignore_mask(self, seg_id, liftedNh, liftedUvs, with_defects = False): # with defects only for caching
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
    # Convenience functions for Cutouts
    #

    # make a cutout of the given block shape
    # need to update the ds in the MetaSet after this!
    def make_cutout(self, start, stop):
        assert self.has_raw, "Need at least raw data to make a cutout"
        assert len(start) == 3
        assert len(stop)  == 3
        assert start[0] < stop[0] and start[1] < stop[1] and start[2] < stop[2]
        assert stop[0] <= self.shape[0] and stop[1] <= self.shape[1] and stop[2] <= self.shape[2], "%s, %s" % (str(self.shape), str(stop))

        cutout_name = self.ds_name + "_cutout_" + str(self.n_cutouts)

        # if we are making a cutout of a cutout, we must adjust the parent ds and block offset
        parent_ds     = self.parent_ds.cache_folder if isinstance(self, Cutout) else self.cache_folder
        if isinstance(self, Cutout):
            for dd in xrange(3):
                start[dd] += self.start[dd]
                stop[dd]  += self.stop[dd]

        cutout = Cutout(self.cache_folder, cutout_name, start, stop, parent_ds)

        # copy all inputs, segs and the gt to cutuout
        for inp_id in range(self.n_inp):
            inp_path = os.path.join(self.cache_folder, 'inp%i.h5' % inp_id)
            if not os.path.exists(inp_path): # make sure that the cache was generated
                self.inp(inp_id)
            if inp_id == 0:
                cutout.add_raw(inp_path, 'data')
            else:
                cutout.add_input(inp_path, 'data')

        # check if we have a seg mask and copy
        if self.has_seg_mask:
            mask_path = os.path.join(self.cache_folder,"seg_mask.h5")
            if not os.path.exists(mask_path):
                self.seg_mask()
            cutout.add_seg_mask(mask_path, 'data')

        for seg_id in range(self.n_seg):
            seg_path = os.path.join(self.cache_folder,"seg" + str(seg_id) + ".h5")
            if not os.path.exists(seg_path):
                self.seg(seg_id)
            cutout.add_seg(seg_path, "data")

        if self.has_gt:
            gt_path = os.path.join(self.cache_folder,"gt.h5")
            if not os.path.exists(gt_path):
                self.gt()
            cutout.add_gt(gt_path, "data")

        for false_merge_gt in self.gt_false_merges:
            cutout.add_false_merge_gt_id(false_merge_gt)

        for false_split_gt in self.gt_false_splits:
            cutout.add_false_split_gt_id(false_split_gt)

        if self.external_defect_mask_path != None:
            mask_path = os.path.join(self.cache_folder,"defect_mask.h5")
            if not os.path.exists(mask_path):
                self.defect_mask()
            cutout.add_defect_mask(mask_path, "data")
        cutout.save()

        self.cutouts.append(cutout_name)
        self.save()


    def get_cutout(self, cutout_id):
        assert cutout_id < self.n_cutouts, str(cutout_id) + " , " + str(self.n_cutouts)
        return load_dataset(self.cache_folder, self.cutouts[cutout_id])


#cutout from a given Dataset, used for cutouts and tesselations
#calls the cache of the parent dataset for inp, seg, gt and filtercalls the cache of the parent dataset for inp, seg, gt and filter
class Cutout(DataSet):

    def __init__(self, cache_folder, ds_name, start, stop, parent_ds_folder):
        super(Cutout, self).__init__(cache_folder, ds_name)

        self.start = start
        self.stop  = stop
        self.shape = (self.stop[0] - self.start[0],
                      self.stop[1] - self.start[1],
                      self.stop[2] - self.start[2])

        # the actual bounding box for easy cutouts of data
        self.bb = np.s_[
                self.start[0]:self.stop[0],
                self.start[1]:self.stop[1],
                self.start[2]:self.stop[2]
                ]

        self.parent_ds_folder = parent_ds_folder


    # fot the inputs, we dont need to cache everythin again, however for seg and gt we have to, because otherwise the segmentations are not consecutive any longer
    # seg and gt can't be reimplemented that way, because they need to be connected!

    # we need to overload this, to not cache the raw data again, but read the one from the parent dataset
    def inp(self, inp_id):
        if inp_id >= self.n_inp:
            raise RuntimeError("Trying to read inp_id " + str(inp_id) + " but there are only " + str(self.n_inp) + " input maps")
        inp_path = self.external_inp_paths[inp_id]
        inp_key  = self.external_inp_keys[inp_id]
        with h5py.File(inp_path) as f:
            return f[inp_key][self.bb]


    # we get the paths to the filters of the top dataset
    def make_filters(self,
            inp_id,
            anisotropy_factor,
            filter_names = [ "gaussianSmoothing",
                             "hessianOfGaussianEigenvalues",
                             "laplacianOfGaussian"],
            sigmas = [1.6, 4.2, 8.3]
            ):

        # call make filters of the parent dataset
        parent_ds = load_dataset(self.parent_ds_folder)
        return parent_ds.make_filters(inp_id, anisotropy_factor, filter_names, sigmas)
